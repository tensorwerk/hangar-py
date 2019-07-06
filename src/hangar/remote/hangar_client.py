import logging
import hashlib
import io
import os
import tempfile
import time

import blosc
import grpc
import lmdb
import msgpack
import numpy as np

from . import chunks
from . import hangar_service_pb2
from . import hangar_service_pb2_grpc
from .header_manipulator_client_interceptor import header_adder_interceptor
from .. import constants as c
from ..context import Environments, TxnRegister
from ..backends.selection import BACKEND_ACCESSOR_MAP, backend_decoder
from ..records import commiting
from ..records import hashs
from ..records import heads
from ..records import parsing
from ..records import queries
from ..records import summarize
from ..utils import set_blosc_nthreads

set_blosc_nthreads()

logger = logging.getLogger(__name__)


class HangarClient(object):
    '''Client which connects and handles data transfer to the hangar server.

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
        If the client should wait before erroring for a short period of time
        while a server is `UNAVAILABLE`, typically due to it just starting up
        at the time the connection was made
    wait_for_read_timeout : float, optional, kwarg-only, by default 5.
        If `wait_for_ready` is True, the time in seconds which the client should
        wait before raising an error. Must be positive value (greater than 0)
    '''

    def __init__(self,
                 envs: Environments, address: str, *,
                 auth_username: str = '', auth_password: str = '',
                 wait_for_ready: bool = True, wait_for_ready_timeout: float = 5):

        self.cfg = {}
        self._rFs = {}
        self.env = envs
        self.address: str = address
        self.channel: grpc.Channel = None
        self.wait_ready = wait_for_ready
        self.wait_ready_timeout = abs(wait_for_ready_timeout + 0.001)
        self.stub: hangar_service_pb2_grpc.HangarServiceStub = None
        self.header_adder_int = header_adder_interceptor(auth_username, auth_password)

        for backend, accessor in BACKEND_ACCESSOR_MAP.items():
            if accessor is not None:
                self._rFs[backend] = accessor(
                    repo_path=self.env.repo_path,
                    schema_shape=None,
                    schema_dtype=None)
                self._rFs[backend].open(mode='r')

        self._setup_client_channel_config()

    def _setup_client_channel_config(self):
        '''get grpc client configuration from server and setup channel and stub for use.
        '''
        tmp_insec_channel = grpc.insecure_channel(self.address)
        tmp_channel = grpc.intercept_channel(tmp_insec_channel, self.header_adder_int)
        tmp_stub = hangar_service_pb2_grpc.HangarServiceStub(tmp_channel)

        t_init, t_tot = time.time(), 0
        while t_tot < self.wait_ready_timeout:
            try:
                request = hangar_service_pb2.GetClientConfigRequest()
                response = tmp_stub.GetClientConfig(request)
            except grpc.RpcError as err:
                if not (err.code() == grpc.StatusCode.UNAVAILABLE) and (self.wait_ready is True):
                    logger.error(err)
                    raise err
            else:
                break
            logger.debug(f'Wait-for-ready: {self.wait_ready}, time elapsed: {t_tot}')
            time.sleep(0.05)
            t_tot = time.time() - t_init
        else:
            err = TimeoutError(f'Server did not connect after: {self.wait_ready_timeout} sec.')
            logger.error(err)
            raise err

        self.cfg['push_max_nbytes'] = int(response.config['push_max_nbytes'])
        self.cfg['enable_compression'] = bool(int(response.config['enable_compression']))
        self.cfg['optimization_target'] = response.config['optimization_target']

        tmp_channel.close()
        tmp_insec_channel.close()
        insec_channel = grpc.insecure_channel(
            self.address,
            options=[('grpc.default_compression_algorithm', self.cfg['enable_compression']),
                     ('grpc.optimization_target', self.cfg['optimization_target'])])
        self.channel = grpc.intercept_channel(insec_channel, self.header_adder_int)
        self.stub = hangar_service_pb2_grpc.HangarServiceStub(self.channel)

    def close(self):
        for backend_accessor in self._rFs.values():
            backend_accessor.close()
        self.channel.close()

    def ping_pong(self):
        request = hangar_service_pb2.PingRequest()
        response = self.stub.PING(request)
        return response.result

    def push_branch_record(self, name, head):
        rec = hangar_service_pb2.BranchRecord(name=name, commit=head)
        request = hangar_service_pb2.PushBranchRecordRequest(rec=rec)
        response = self.stub.PushBranchRecord(request)
        return response

    def fetch_branch_record(self, name):
        rec = hangar_service_pb2.BranchRecord(name=name)
        request = hangar_service_pb2.FetchBranchRecordRequest(rec=rec)
        response = self.stub.FetchBranchRecord(request)
        return response

    def push_commit_record(self, commit, parentVal, specVal, refVal):
        cIter = chunks.clientCommitChunkedIterator(commit=commit,
                                                   parentVal=parentVal,
                                                   specVal=specVal,
                                                   refVal=refVal)
        response = self.stub.PushCommit(cIter)
        return response

    def fetch_commit_record(self, commit):
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

    def fetch_schema(self, schema_hash):
        schema_rec = hangar_service_pb2.SchemaRecord(digest=schema_hash)
        request = hangar_service_pb2.FetchSchemaRequest(rec=schema_rec)
        reply = self.stub.FetchSchema(request)
        if reply.error.code != 0:
            logger.error(reply.error)
            return False

        schemaVal = reply.rec.blob
        return (schema_hash, schemaVal)

    def push_schema(self, schema_hash, schemaVal):
        rec = hangar_service_pb2.SchemaRecord(digest=schema_hash, blob=schemaVal)
        request = hangar_service_pb2.PushSchemaRequest(rec=rec)
        response = self.stub.PushSchema(request)
        return response

    def fetch_data(self, schema_hash, digests):

        totalSize, buf = 0, io.BytesIO()
        packer = msgpack.Packer(use_bin_type=True)
        packed_digests = map(packer.pack, digests)
        for p in packed_digests:
            buf.write(p)
            totalSize += len(p)

        try:
            cIter = chunks.tensorChunkedIterator(
                buf=buf, uncomp_nbytes=totalSize, itemsize=1,
                pb2_request=hangar_service_pb2.FetchDataRequest)
            replies = self.stub.FetchData(cIter)
            for idx, reply in enumerate(replies):
                if idx == 0:
                    uncomp_nbytes, comp_nbytes = reply.uncomp_nbytes, reply.comp_nbytes
                    dBytes, offset = bytearray(comp_nbytes), 0
                size = len(reply.raw_data)
                if size > 0:
                    dBytes[offset: offset + size] = reply.raw_data
                    offset += size
        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                logger.info(rpc_error.details())
            else:
                logger.error(rpc_error.details())
                raise rpc_error

        uncompBytes = blosc.decompress(dBytes)
        if uncomp_nbytes != len(uncompBytes):
            raise RuntimeError(f'uncomp_nbytes: {uncomp_nbytes} != recieved {comp_nbytes}')
        buff = io.BytesIO(uncompBytes)
        unpacker = msgpack.Unpacker(
            buff, use_list=True, raw=False, max_buffer_size=1_000_000_000)

        recieved_data = []
        for data in unpacker:
            hdigest, dShape, dTypeN, ddBytes = data
            tensor = np.frombuffer(ddBytes, dtype=np.typeDict[dTypeN]).reshape(dShape)
            recieved_hash = hashlib.blake2b(tensor.tobytes(), digest_size=20).hexdigest()
            if recieved_hash != hdigest:
                raise RuntimeError(f'MANGLED! got: {recieved_hash} != requested: {hdigest}')
            recieved_data.append((recieved_hash, tensor))
        return recieved_data

    def push_data(self, schema_hash, digests):

        totalSize, buf = 0, io.BytesIO()
        packer = msgpack.Packer(use_bin_type=True)
        hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
        try:
            for digest in digests:
                hashKey = parsing.hash_data_db_key_from_raw_key(digest)
                hashVal = hashTxn.get(hashKey, default=False)
                if not hashVal:
                    raise KeyError(f'No hash record with key: {hashKey}')

                spec = backend_decoder(hashVal)
                arr = self._rFs[spec.backend].read_data(spec)
                p = packer.pack((digest, schema_hash, arr.shape, arr.dtype.num, arr.tobytes()))
                totalSize += len(p)
                buf.write(p)
                # only send a group of tensors <= Max Size so that the server
                # does not run out of RAM for large repos
                if totalSize >= self.cfg['push_max_nbytes']:
                    cIter = chunks.tensorChunkedIterator(
                        buf=buf, uncomp_nbytes=totalSize, itemsize=arr.itemsize,
                        pb2_request=hangar_service_pb2.PushDataRequest)
                    response = self.stub.PushData(cIter)
                    totalSize = 0
                    buf.close()
                    buf = io.BytesIO()

        except grpc.RpcError as rpc_error:
            logger.error(rpc_error)
            raise rpc_error

        finally:
            # finish sending all remaining tensors if max size has not been hit.
            if totalSize > 0:
                cIter = chunks.tensorChunkedIterator(
                    buf=buf, uncomp_nbytes=totalSize, itemsize=arr.itemsize,
                    pb2_request=hangar_service_pb2.PushDataRequest)
                response = self.stub.PushData(cIter)
                buf.close()
            TxnRegister().abort_reader_txn(self.env.hashenv)

        return response

    def fetch_label(self, digest):

        rec = hangar_service_pb2.HashRecord(digest=digest)
        request = hangar_service_pb2.FetchLabelRequest(rec=rec)
        reply = self.stub.FetchLabel(request)

        uncompBlob = blosc.decompress(reply.blob)
        recieved_hash = hashlib.blake2b(uncompBlob, digest_size=20).hexdigest()
        if recieved_hash != digest:
            raise RuntimeError(f'recieved_hash: {recieved_hash} != digest: {digest}')
        return (recieved_hash, uncompBlob)

    def push_label(self, digest, labelVal):

        rec = hangar_service_pb2.HashRecord(digest=digest)
        request = hangar_service_pb2.PushLabelRequest(rec=rec)
        request.blob = blosc.compress(labelVal)
        reply = self.stub.PushLabel(request)
        return reply

    def fetch_find_missing_commits(self, branch_name):

        c_commits = commiting.list_all_commits(self.env.refenv)
        branch_rec = hangar_service_pb2.BranchRecord(name=branch_name)
        request = hangar_service_pb2.FindMissingCommitsRequest()
        request.commits.extend(c_commits)
        request.branch.CopyFrom(branch_rec)
        reply = self.stub.FetchFindMissingCommits(request)
        return reply

    def push_find_missing_commits(self, branch_name):

        branch_head = heads.get_branch_head_commit(self.env.branchenv, branch_name)
        branch_rec = hangar_service_pb2.BranchRecord(name=branch_name, commit=branch_head)
        branch_commits = summarize.list_history(
            refenv=self.env.refenv,
            branchenv=self.env.branchenv,
            branch_name=branch_name)

        request = hangar_service_pb2.FindMissingCommitsRequest()
        request.commits.extend(branch_commits['order'])
        request.branch.CopyFrom(branch_rec)
        reply = self.stub.PushFindMissingCommits(request)
        return reply

    def fetch_find_missing_hash_records(self, commit):

        all_hashs = hashs.HashQuery(self.env.hashenv).list_all_hash_keys_raw()
        pb2_func = hangar_service_pb2.FindMissingHashRecordsRequest
        cIter = chunks.missingHashRequestIterator(commit, all_hashs, pb2_func)
        responses = self.stub.FetchFindMissingHashRecords(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                commit = response.commit
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        missing_hashs = msgpack.unpackb(uncompBytes, raw=False, use_list=False)
        recieved_data = [(digest, schema_hash) for digest, schema_hash in missing_hashs]
        return recieved_data

    def push_find_missing_hash_records(self, commit):

        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            c_hashs_schemas = queries.RecordQuery(tmpDB).data_hash_to_schema_hash()
            c_hashes = list(set(c_hashs_schemas.keys()))
            tmpDB.close()

        pb2_func = hangar_service_pb2.FindMissingHashRecordsRequest
        cIter = chunks.missingHashRequestIterator(commit, c_hashes, pb2_func)
        responses = self.stub.PushFindMissingHashRecords(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                commit = response.commit
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        s_missing_hashs = msgpack.unpackb(uncompBytes, raw=False, use_list=False)
        s_mis_hsh_sch = dict((s_hsh, c_hashs_schemas[s_hsh]) for s_hsh in s_missing_hashs)
        return s_mis_hsh_sch

    def fetch_find_missing_labels(self, commit):
        c_hash_keys = hashs.HashQuery(self.env.labelenv).list_all_hash_keys_db()
        c_hashset = set(map(parsing.hash_meta_raw_key_from_db_key, c_hash_keys))
        c_hashes = list(c_hashset)

        pb2_func = hangar_service_pb2.FindMissingLabelsRequest
        cIter = chunks.missingHashRequestIterator(commit, c_hashes, pb2_func)
        responses = self.stub.FetchFindMissingLabels(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                commit = response.commit
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        missing_hashs = msgpack.unpackb(uncompBytes, raw=False, use_list=False)
        return missing_hashs

    def push_find_missing_labels(self, commit):
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            c_hashset = set(queries.RecordQuery(tmpDB).metadata_hashes())
            c_hashes = list(c_hashset)
            tmpDB.close()

        pb2_func = hangar_service_pb2.FindMissingLabelsRequest
        cIter = chunks.missingHashRequestIterator(commit, c_hashes, pb2_func)
        responses = self.stub.PushFindMissingLabels(cIter)
        for idx, response in enumerate(responses):
            if idx == 0:
                commit = response.commit
                hBytes, offset = bytearray(response.total_byte_size), 0
            size = len(response.hashs)
            hBytes[offset: offset + size] = response.hashs
            offset += size

        uncompBytes = blosc.decompress(hBytes)
        missing_hashs = msgpack.unpackb(uncompBytes, raw=False, use_list=False)
        return missing_hashs

    def fetch_find_missing_schemas(self, commit):
        c_schemaset = set(hashs.HashQuery(self.env.hashenv).list_all_schema_keys_raw())
        c_schemas = list(c_schemaset)

        request = hangar_service_pb2.FindMissingSchemasRequest()
        request.commit = commit
        request.schema_digests.extend(c_schemas)

        response = self.stub.FetchFindMissingSchemas(request)
        return response

    def push_find_missing_schemas(self, commit):

        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            c_schemaset = set(queries.RecordQuery(tmpDB).schema_hashes())
            c_schemas = list(c_schemaset)
            tmpDB.close()

        request = hangar_service_pb2.FindMissingSchemasRequest()
        request.commit = commit
        request.schema_digests.extend(c_schemas)

        response = self.stub.PushFindMissingSchemas(request)
        return response
