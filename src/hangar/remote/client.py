import logging
import os
import tempfile
import time
from typing import Tuple, Sequence

import blosc
import grpc
import lmdb
import numpy as np
from tqdm import tqdm

from . import chunks
from . import hangar_service_pb2
from . import hangar_service_pb2_grpc
from .header_manipulator_client_interceptor import header_adder_interceptor
from .. import constants as c
from ..context import Environments
from ..txnctx import TxnRegister
from ..backends import BACKEND_ACCESSOR_MAP, backend_decoder
from ..records import commiting
from ..records import hashs
from ..records.hashmachine import ndarray_hasher_tcode_0
from ..records import hash_data_db_key_from_raw_key
from ..records import queries
from ..records import summarize
from ..utils import set_blosc_nthreads

set_blosc_nthreads()

logger = logging.getLogger(__name__)


class HangarClient(object):
    """Client which connects and handles data transfer to the hangar server.

    Parameters
    ----------
    envs : Environments
        environment handles to manage all required calls to the local
        repostory state.
    address : str
        IP:PORT where the hangar server can be reached.
    auth_username : str, optional, kwarg-only
        credentials to use for authentication.
    auth_password : str, optional, kwarg-only, by default ''.
        credentials to use for authentication, by default ''.
    wait_for_ready : bool, optional, kwarg-only, be default True.
        If the client should wait before erring for a short period of time
        while a server is `UNAVAILABLE`, typically due to it just starting up
        at the time the connection was made
    wait_for_read_timeout : float, optional, kwarg-only, by default 5.
        If `wait_for_ready` is True, the time in seconds which the client should
        wait before raising an error. Must be positive value (greater than 0)
    """

    def __init__(self,
                 envs: Environments,
                 address: str,
                 *,
                 auth_username: str = '',
                 auth_password: str = '',
                 wait_for_ready: bool = True,
                 wait_for_ready_timeout: float = 5):

        self.env: Environments = envs
        self.address: str = address
        self.wait_ready: bool = wait_for_ready
        self.wait_ready_timeout: float = abs(wait_for_ready_timeout + 0.001)

        self.channel: grpc.Channel = None
        self.stub: hangar_service_pb2_grpc.HangarServiceStub = None
        self.header_adder_int = header_adder_interceptor(auth_username, auth_password)

        self.cfg: dict = {}
        self._rFs: BACKEND_ACCESSOR_MAP = {}

        for backend, accessor in BACKEND_ACCESSOR_MAP.items():
            if accessor is not None:
                self._rFs[backend] = accessor(
                    repo_path=self.env.repo_path,
                    schema_shape=None,
                    schema_dtype=None)
                self._rFs[backend].open(mode='r')

        self._setup_client_channel_config()

    def _setup_client_channel_config(self):
        """get grpc client configuration from server and setup channel and stub for use.
        """
        tmp_insec_channel = grpc.insecure_channel(self.address)
        tmp_channel = grpc.intercept_channel(tmp_insec_channel, self.header_adder_int)
        tmp_stub = hangar_service_pb2_grpc.HangarServiceStub(tmp_channel)
        t_init, t_tot = time.time(), 0
        while t_tot < self.wait_ready_timeout:
            try:
                request = hangar_service_pb2.GetClientConfigRequest()
                response = tmp_stub.GetClientConfig(request)
                self.cfg['push_max_nbytes'] = int(response.config['push_max_nbytes'])
                self.cfg['optimization_target'] = response.config['optimization_target']

                enable_compression = response.config['enable_compression']
                if enable_compression == 'NoCompression':
                    compression_val = grpc.Compression.NoCompression
                elif enable_compression == 'Deflate':
                    compression_val = grpc.Compression.Deflate
                elif enable_compression == 'Gzip':
                    compression_val = grpc.Compression.Gzip
                else:
                    compression_val = grpc.Compression.NoCompression
                self.cfg['enable_compression'] = compression_val

            except grpc.RpcError as err:
                if not (err.code() == grpc.StatusCode.UNAVAILABLE) and (self.wait_ready is True):
                    logger.error(err)
                    raise err
            else:
                break
            time.sleep(0.05)
            t_tot = time.time() - t_init
        else:
            err = ConnectionError(f'Server did not connect after: {self.wait_ready_timeout} sec.')
            logger.error(err)
            raise err

        tmp_channel.close()
        tmp_insec_channel.close()
        configured_channel = grpc.insecure_channel(
            self.address,
            options=[('grpc.optimization_target', self.cfg['optimization_target'])],
            compression=self.cfg['enable_compression'])
        self.channel = grpc.intercept_channel(configured_channel, self.header_adder_int)
        self.stub = hangar_service_pb2_grpc.HangarServiceStub(self.channel)

    def close(self):
        """Close reader file handles and the GRPC channel connection, invalidating this instance.
        """
        for backend_accessor in self._rFs.values():
            backend_accessor.close()
        self.channel.close()

    def ping_pong(self) -> str:
        """Ping server to ensure that connection is working

        Returns
        -------
        str
            Should be value 'PONG'
        """
        request = hangar_service_pb2.PingRequest()
        response: hangar_service_pb2.PingReply = self.stub.PING(request)
        return response.result

    def push_branch_record(self, name: str, head: str
                           ) -> hangar_service_pb2.PushBranchRecordReply:
        """Create a branch (if new) or update the server branch HEAD to new commit.

        Parameters
        ----------
        name : str
            branch name to be pushed
        head : str
            commit hash to update the server head to

        Returns
        -------
        hangar_service_pb2.PushBranchRecordReply
            code indicating success, message with human readable info
        """
        rec = hangar_service_pb2.BranchRecord(name=name, commit=head)
        request = hangar_service_pb2.PushBranchRecordRequest(rec=rec)
        response = self.stub.PushBranchRecord(request)
        return response

    def fetch_branch_record(self, name: str
                            ) -> hangar_service_pb2.FetchBranchRecordReply:
        """Get the latest head commit the server knows about for a given branch

        Parameters
        ----------
        name : str
            name of the branch to query on the server

        Returns
        -------
        hangar_service_pb2.FetchBranchRecordReply
            rec containing name and head commit if branch exists, along with
            standard error proto if it does not exist on the server.
        """
        rec = hangar_service_pb2.BranchRecord(name=name)
        request = hangar_service_pb2.FetchBranchRecordRequest(rec=rec)
        response = self.stub.FetchBranchRecord(request)
        return response

    def push_commit_record(self, commit: str, parentVal: bytes, specVal: bytes,
                           refVal: bytes
                           ) -> hangar_service_pb2.PushBranchRecordReply:
        """Push a new commit reference to the server.

        Parameters
        ----------
        commit : str
            hash digest of the commit to send
        parentVal : bytes
            lmdb ref parentVal of the commit
        specVal : bytes
            lmdb ref specVal of the commit
        refVal : bytes
            lmdb ref refVal of the commit

        Returns
        -------
        hangar_service_pb2.PushBranchRecordReply
            standard error proto
        """
        cIter = chunks.clientCommitChunkedIterator(commit=commit,
                                                   parentVal=parentVal,
                                                   specVal=specVal,
                                                   refVal=refVal)
        response = self.stub.PushCommit(cIter)
        return response

    def fetch_commit_record(self, commit: str) -> Tuple[str, bytes, bytes, bytes]:
        """get the refs for a commit digest

        Parameters
        ----------
        commit : str
            digest of the commit to retrieve the references for

        Returns
        -------
        Tuple[str, bytes, bytes, bytes]
            ['commit hash', 'parentVal', 'specVal', 'refVal']
        """
        request = hangar_service_pb2.FetchCommitRequest(commit=commit)
        replies = self.stub.FetchCommit(request)
        for idx, reply in enumerate(replies):
            if idx == 0:
                refVal = bytearray(reply.total_byte_size)
                specVal = reply.record.spec
                parentVal = reply.record.parent
                offset = 0
            size = len(reply.record.ref)
            refVal[offset: offset + size] = reply.record.ref
            offset += size

        if reply.error.code != 0:
            logger.error(reply.error)
            return False
        return (commit, parentVal, specVal, refVal)

    def fetch_schema(self, schema_hash: str) -> Tuple[str, bytes]:
        """get the schema specification for a schema hash

        Parameters
        ----------
        schema_hash : str
            schema hash to retrieve from the server

        Returns
        -------
        Tuple[str, bytes]
            ['schema hash', 'schemaVal']
        """
        schema_rec = hangar_service_pb2.SchemaRecord(digest=schema_hash)
        request = hangar_service_pb2.FetchSchemaRequest(rec=schema_rec)
        reply = self.stub.FetchSchema(request)
        if reply.error.code != 0:
            logger.error(reply.error)
            return False

        schemaVal = reply.rec.blob
        return (schema_hash, schemaVal)

    def push_schema(self, schema_hash: str,
                    schemaVal: bytes) -> hangar_service_pb2.PushSchemaReply:
        """push a schema hash record to the remote server

        Parameters
        ----------
        schema_hash : str
            hash digest of the schema being sent
        schemaVal : bytes
            ref value of the schema representation

        Returns
        -------
        hangar_service_pb2.PushSchemaReply
            standard error proto indicating success
        """
        rec = hangar_service_pb2.SchemaRecord(digest=schema_hash,
                                              blob=schemaVal)
        request = hangar_service_pb2.PushSchemaRequest(rec=rec)
        response = self.stub.PushSchema(request)
        return response

    def fetch_data(self, schema_hash: str,
                   digests: Sequence[str]) -> Sequence[Tuple[str, np.ndarray]]:
        """Fetch data hash digests for a particular schema.

        As the total size of the data to be transferred isn't known before this
        operation occurs, if more tensor data digests are requested then the
        Client is configured to allow in memory at a time, only a portion of the
        requested digests will actually be materialized. The received digests
        are listed as the return value of this function, be sure to check that
        all requested digests have been received!

        Parameters
        ----------
        schema_hash : str
            hash of the schema each of the digests is associated with
        digests : Sequence[str]
            iterable of data digests to receive

        Returns
        -------
        Sequence[Tuple[str, np.ndarray]]
            iterable containing 2-tuples' of the hash digest and np.ndarray data.

        Raises
        ------
        RuntimeError
            if received digest != requested or what was reported to be sent.
        """
        try:
            raw_digests = c.SEP_LST.join(digests).encode()
            cIter = chunks.tensorChunkedIterator(
                buf=raw_digests, uncomp_nbytes=len(raw_digests), itemsize=1,
                pb2_request=hangar_service_pb2.FetchDataRequest)

            replies = self.stub.FetchData(cIter)
            for idx, reply in enumerate(replies):
                if idx == 0:
                    uncomp_nbytes, comp_nbytes = reply.uncomp_nbytes, reply.comp_nbytes
                    dBytes, offset = bytearray(comp_nbytes), 0
                size = len(reply.raw_data)
                if size > 0:
                    dBytes[offset:offset + size] = reply.raw_data
                    offset += size
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                logger.info(rpc_error.details())
            else:
                logger.error(rpc_error.details())
                raise rpc_error

        uncompBytes = blosc.decompress(dBytes)
        if uncomp_nbytes != len(uncompBytes):
            raise RuntimeError(f'uncomp_nbytes: {uncomp_nbytes} != received {comp_nbytes}')
        received_data = []
        unpacked_records = chunks.deserialize_record_pack(uncompBytes)
        for record in unpacked_records:
            data = chunks.deserialize_record(record)
            received_hash = ndarray_hasher_tcode_0(data.array)
            if received_hash != data.digest:
                raise RuntimeError(f'MANGLED! got: {received_hash} != requested: {data.digest}')
            received_data.append((received_hash, data.array))
        return received_data

    def push_data(self, schema_hash: str, digests: Sequence[str],
                  pbar: tqdm = None) -> hangar_service_pb2.PushDataReply:
        """Given a schema and digest list, read the data and send to the server

        Parameters
        ----------
        schema_hash : str
            hash of the digest schemas
        digests : Sequence[str]
            iterable of digests to be read in and sent to the server
        pbar : tqdm, optional
            progress bar instance to be updated as the operation occurs, by default None

        Returns
        -------
        hangar_service_pb2.PushDataReply
            standard error proto indicating success

        Raises
        ------
        KeyError
            if one of the input digests does not exist on the client
        rpc_error
            if the server received corrupt data
        """
        try:
            specs = []
            hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
            for digest in digests:
                hashKey = hash_data_db_key_from_raw_key(digest)
                hashVal = hashTxn.get(hashKey, default=False)
                if not hashVal:
                    raise KeyError(f'No hash record with key: {hashKey}')
                be_loc = backend_decoder(hashVal)
                specs.append((digest, be_loc))
        finally:
            TxnRegister().abort_reader_txn(self.env.hashenv)

        try:
            totalSize, records = 0, []
            for k in self._rFs.keys():
                self._rFs[k].__enter__()
            responses = []
            for digest, spec in specs:
                arr = self._rFs[spec.backend].read_data(spec)
                record = chunks.serialize_record(arr, digest, schema_hash)
                records.append(record)
                totalSize += len(record)
                if (totalSize >= self.cfg['push_max_nbytes']) or (len(records) > 2000):
                    # send tensor pack when >= configured max nbytes occupied in memory
                    pbar.update(len(records))
                    pack = chunks.serialize_record_pack(records)
                    cIter = chunks.tensorChunkedIterator(
                        buf=pack, uncomp_nbytes=len(pack), itemsize=1,
                        pb2_request=hangar_service_pb2.PushDataRequest)
                    response = self.stub.PushData.future(cIter)
                    responses.append(response)
                    totalSize = 0
                    records = []
        except grpc.RpcError as rpc_error:
            logger.error(rpc_error.with_traceback())
            raise rpc_error
        finally:
            for k in self._rFs.keys():
                self._rFs[k].__exit__()
            if totalSize > 0:
                # finish sending all remaining tensors if max size has not been hit.
                pack = chunks.serialize_record_pack(records)
                cIter = chunks.tensorChunkedIterator(
                    buf=pack, uncomp_nbytes=len(pack), itemsize=arr.itemsize,
                    pb2_request=hangar_service_pb2.PushDataRequest)
                response = self.stub.PushData.future(cIter)
                responses.append(response)
        for fut in responses:
            last = fut.result()
        return last

    def fetch_find_missing_commits(self, branch_name):

        c_commits = commiting.list_all_commits(self.env.refenv)
        branch_rec = hangar_service_pb2.BranchRecord(name=branch_name)
        request = hangar_service_pb2.FindMissingCommitsRequest()
        request.commits.extend(c_commits)
        request.branch.CopyFrom(branch_rec)
        reply = self.stub.FetchFindMissingCommits(request)
        return reply

    def push_find_missing_commits(self, branch_name):
        branch_commits = summarize.list_history(
            refenv=self.env.refenv,
            branchenv=self.env.branchenv,
            branch_name=branch_name)
        branch_rec = hangar_service_pb2.BranchRecord(
            name=branch_name, commit=branch_commits['head'])

        request = hangar_service_pb2.FindMissingCommitsRequest()
        request.commits.extend(branch_commits['order'])
        request.branch.CopyFrom(branch_rec)
        reply = self.stub.PushFindMissingCommits(request)
        return reply

    def fetch_find_missing_hash_records(self, commit):

        all_hashs = hashs.HashQuery(self.env.hashenv).list_all_hash_keys_raw()
        all_hashs_raw = [chunks.serialize_ident(digest, '') for digest in all_hashs]
        raw_pack = chunks.serialize_record_pack(all_hashs_raw)
        pb2_func = hangar_service_pb2.FindMissingHashRecordsRequest
        cIter = chunks.missingHashRequestIterator(commit, raw_pack, pb2_func)
        responses = self.stub.FetchFindMissingHashRecords(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        raw_idents = chunks.deserialize_record_pack(uncompBytes)
        idents = [chunks.deserialize_ident(raw) for raw in raw_idents]
        return idents

    def push_find_missing_hash_records(self, commit, tmpDB: lmdb.Environment = None):

        if tmpDB is None:
            with tempfile.TemporaryDirectory() as tempD:
                tmpDF = os.path.join(tempD, 'test.lmdb')
                tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
                commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
                c_hashs_schemas = queries.RecordQuery(tmpDB).data_hash_to_schema_hash()
                c_hashes = list(set(c_hashs_schemas.keys()))
                tmpDB.close()
        else:
            c_hashs_schemas = queries.RecordQuery(tmpDB).data_hash_to_schema_hash()
            c_hashes = list(set(c_hashs_schemas.keys()))

        c_hashs_raw = [chunks.serialize_ident(digest, '') for digest in c_hashes]
        raw_pack = chunks.serialize_record_pack(c_hashs_raw)
        pb2_func = hangar_service_pb2.FindMissingHashRecordsRequest
        cIter = chunks.missingHashRequestIterator(commit, raw_pack, pb2_func)

        responses = self.stub.PushFindMissingHashRecords(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        s_missing_raw = chunks.deserialize_record_pack(uncompBytes)
        s_mis_hsh = [chunks.deserialize_ident(raw).digest for raw in s_missing_raw]
        s_mis_hsh_sch = [(s_hsh, c_hashs_schemas[s_hsh]) for s_hsh in s_mis_hsh]
        return s_mis_hsh_sch

    def fetch_find_missing_schemas(self, commit):
        c_schemaset = set(hashs.HashQuery(self.env.hashenv).list_all_schema_digests())
        c_schemas = list(c_schemaset)

        request = hangar_service_pb2.FindMissingSchemasRequest()
        request.commit = commit
        request.schema_digests.extend(c_schemas)

        response = self.stub.FetchFindMissingSchemas(request)
        return response

    def push_find_missing_schemas(self, commit, tmpDB: lmdb.Environment = None):

        if tmpDB is None:
            with tempfile.TemporaryDirectory() as tempD:
                tmpDF = os.path.join(tempD, 'test.lmdb')
                tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
                commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
                c_schemaset = set(queries.RecordQuery(tmpDB).schema_hashes())
                c_schemas = list(c_schemaset)
                tmpDB.close()
        else:
            c_schemaset = set(queries.RecordQuery(tmpDB).schema_hashes())
            c_schemas = list(c_schemaset)

        request = hangar_service_pb2.FindMissingSchemasRequest()
        request.commit = commit
        request.schema_digests.extend(c_schemas)

        response = self.stub.PushFindMissingSchemas(request)
        return response
