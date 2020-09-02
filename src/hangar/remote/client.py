import concurrent.futures
import logging
import os
import tempfile
import time
from threading import Lock
from typing import Tuple, Sequence, List, Iterable, TYPE_CHECKING

import blosc
import grpc
import lmdb
from tqdm import tqdm

from . import chunks, hangar_service_pb2, hangar_service_pb2_grpc
from .header_manipulator_client_interceptor import header_adder_interceptor
from .. import constants as c
from ..backends import BACKEND_ACCESSOR_MAP, backend_decoder
from ..context import Environments
from ..records import commiting, hashs, hash_data_db_key_from_raw_key, queries, summarize
from ..records.hashmachine import hash_func_from_tcode
from ..txnctx import TxnRegister
from ..utils import set_blosc_nthreads, calc_num_threadpool_workers

if TYPE_CHECKING:
    from .content import DataWriter


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
    wait_for_ready_timeout : float, optional, kwarg-only, by default 5.
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
        self.data_writer_lock = Lock()

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
            options=[
                ('grpc.optimization_target', self.cfg['optimization_target']),
                ("grpc.keepalive_time_ms", 1000 * 60 * 1),
                ("grpc.keepalive_timeout_ms", 1000 * 10),
                ("grpc.http2_min_sent_ping_interval_without_data_ms", 1000 * 10),
                ("grpc.http2_max_pings_without_data", 0),
                ("grpc.keepalive_permit_without_calls", 1),
            ],
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

    def fetch_data(
            self,
            origins: Sequence[hangar_service_pb2.DataOriginReply],
            datawriter_cm: 'DataWriter',
            schema: str,
            pbar: 'tqdm'
    ) -> Sequence[str]:
        """Fetch data hash digests for a particular schema.

        As the total size of the data to be transferred isn't known before this
        operation occurs, if more tensor data digests are requested then the
        Client is configured to allow in memory at a time, only a portion of the
        requested digests will actually be materialized. The received digests
        are listed as the return value of this function, be sure to check that
        all requested digests have been received!

        Parameters
        ----------
        origins : Sequence[hangar_service_pb2.DataOriginReply],
        datawriter_cm : 'DataWriter',
        schema : str,
        pbar : 'tqdm'

        Returns
        -------
        Sequence[str]

        Raises
        ------
        RuntimeError
            if received digest != requested or what was reported to be sent.

         client.fetch_data(origins, DW_CM, schema, pbar):
            _ = DW_CM.data(schema, data_digest=returned_digest, data=returned_data)
        """

        def fetch_write_data_parallel(
                pb: 'hangar_service_pb2.DataOriginReply',
                dw_cm: 'DataWriter',
                schema: str,
                lock: 'Lock'
        ) -> str:
            requested_uri = pb.uri
            request = hangar_service_pb2.FetchDataRequest(uri=requested_uri)
            replies = self.stub.FetchData(request)
            for idx, reply in enumerate(replies):
                if idx == 0:
                    dBytes = bytearray(reply.nbytes)
                    offset = 0
                    if reply.uri != requested_uri:
                        raise ValueError(f'requested uri: {requested_uri}, returned: {reply.uri}')
                size = len(reply.raw_data)
                if size > 0:
                    dBytes[offset:offset + size] = reply.raw_data
                    offset += size

            if pb.compression is True:
                codex = pb.compression_opts['id']
                if codex == 'blosc':
                    returned_raw = blosc.decompress(dBytes)
                else:
                    raise ValueError(f'compression id: {codex}')
            else:
                returned_raw = dBytes

            dtype_code = pb.data_type
            returned_data = chunks.deserialize_data(dtype_code, returned_raw)
            hash_func = hash_func_from_tcode(str(dtype_code))
            received_hash = hash_func(returned_data)
            if received_hash != pb.digest:
                raise RuntimeError(f'MANGLED! got: {received_hash} != requested: {pb.digest}')
            with lock:
                written_digest = dw_cm.data(
                    schema, data_digest=received_hash, data=returned_data)
            return written_digest

        saved_digests = []
        nWorkers = calc_num_threadpool_workers()
        with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
            futures = [executor.submit(fetch_write_data_parallel,
                pb, datawriter_cm, schema, self.data_writer_lock) for pb in origins]
            for future in concurrent.futures.as_completed(futures):
                saved_digests.append(future.result())
                pbar.update(1)
        return saved_digests

    def fetch_data_origin(self, digests: Sequence[str]) -> List[hangar_service_pb2.DataOriginReply]:

        def origin_request_iter(digests: Sequence[str]):
            for digest in digests:
                yield hangar_service_pb2.DataOriginRequest(digest=digest)

        requestIter = origin_request_iter(digests)
        replies = self.stub.FetchFindDataOrigin(requestIter)

        output = []
        for reply in replies:
            output.append(reply)
        return output

    def push_find_data_origin(self, digests):
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

    def push_data_begin_context(self):
        request = hangar_service_pb2.PushBeginContextRequest()
        reply = self.stub.PushBeginContext(request)
        return reply

    def push_data_end_context(self):
        request = hangar_service_pb2.PushEndContextRequest()
        reply = self.stub.PushEndContext(request)
        return reply

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
        CONFIG_COMPRESSION_IS_DESIRED = True
        try:
            specs = {}
            request_stack = []
            hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
            for digest in digests:
                hashKey = hash_data_db_key_from_raw_key(digest)
                hashVal = hashTxn.get(hashKey, default=False)
                if not hashVal:
                    raise KeyError(f'No hash record with key: {hashKey}')

                be_loc = backend_decoder(hashVal)
                specs[digest] = be_loc  # saving for later so no recompute cost

                if be_loc.backend in ['01', '00', '10']:
                    dtype = hangar_service_pb2.DataType.NP_ARRAY
                elif be_loc.backend == '30':
                    dtype = hangar_service_pb2.DataType.STR
                elif be_loc.backend == '31':
                    dtype = hangar_service_pb2.DataType.BYTES
                else:
                    raise TypeError(be_loc)

                _request = hangar_service_pb2.PushFindDataOriginRequest(
                    data_type=dtype,
                    digest=digest,
                    compression_is_desired=CONFIG_COMPRESSION_IS_DESIRED)
                request_stack.append(_request)
        finally:
            TxnRegister().abort_reader_txn(self.env.hashenv)

        def request_stack_iterator(request_stack):
            for request in request_stack:
                yield request

        requestIter = request_stack_iterator(request_stack)
        replies: Iterable[hangar_service_pb2.PushFindDataOriginReply]
        replies = self.stub.PushFindDataOrigin(requestIter)

        try:
            for k in self._rFs.keys():
                self._rFs[k].__enter__()

            def push_request_iterator(raw, uri, data_type, schema_hash):
                push_request = hangar_service_pb2.PushDataRequest(
                    uri=uri,
                    nbytes=len(raw),
                    data_type=data_type,
                    schema_hash=schema_hash)
                for raw_chunk in chunks.chunk_bytes(raw):
                    push_request.raw_data = raw_chunk
                    yield push_request

            def push_data_parallel(reply):
                be_loc = specs[reply.digest]
                data = self._rFs[be_loc.backend].read_data(be_loc)
                _, raw_data = chunks.serialize_data(data)

                if reply.compression_expected is True:
                    compressed_record = blosc.compress(
                        raw_data, clevel=3, cname='blosclz', shuffle=blosc.NOSHUFFLE)
                else:
                    compressed_record = raw_data

                if be_loc.backend in ['01', '00', '10']:
                    dtype = hangar_service_pb2.DataType.NP_ARRAY
                elif be_loc.backend == '30':
                    dtype = hangar_service_pb2.DataType.STR
                elif be_loc.backend == '31':
                    dtype = hangar_service_pb2.DataType.BYTES
                else:
                    raise TypeError(be_loc)

                pushDataIter = push_request_iterator(compressed_record, reply.uri, dtype, schema_hash)
                push_data_response = self.stub.PushData(pushDataIter)
                return push_data_response

            nWorkers = calc_num_threadpool_workers()
            with concurrent.futures.ThreadPoolExecutor(max_workers=nWorkers) as executor:
                push_futures = tuple((executor.submit(push_data_parallel, reply) for reply in replies))
                for future in concurrent.futures.as_completed(push_futures):
                    _ = future.result()
                    pbar.update(1)

        except grpc.RpcError as rpc_error:
            logger.error(rpc_error)
            raise rpc_error

        finally:
            for k in self._rFs.keys():
                self._rFs[k].__exit__()

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
