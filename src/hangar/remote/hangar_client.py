import hashlib
import io
import os
import tempfile
import threading

import blosc
import grpc
import lmdb
import msgpack
import numpy as np
from tqdm.auto import tqdm

from . import chunks
from . import hangar_service_pb2
from . import hangar_service_pb2_grpc
from .header_manipulator_client_interceptor import header_adder_interceptor
from .. import config
from ..context import Environments
from ..context import TxnRegister
from ..hdf5_store import FileHandles
from ..records import commiting
from ..records import hashs
from ..records import heads
from ..records import parsing
from ..records import queries
from ..records import summarize

blosc.set_nthreads(blosc.detect_number_of_cores() - 2)


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
    '''

    def __init__(self,
                 envs: Environments, address: str, *,
                 auth_username: str = '', auth_password: str = ''):

        self.env = envs
        self.fs = FileHandles(repo_path=self.env.repo_path)
        self.fs.open(self.env.repo_path, 'r')

        self.header_adder_int = header_adder_interceptor(auth_username, auth_password)
        self.cfg = {}
        self.address = address
        self.temp_channel = grpc.insecure_channel(self.address)
        self.channel = grpc.intercept_channel(self.temp_channel, self.header_adder_int)
        self.stub = hangar_service_pb2_grpc.HangarServiceStub(self.channel)
        self._setup_client_channel_config()

    def _setup_client_channel_config(self):
        request = hangar_service_pb2.GetClientConfigRequest()
        response = self.stub.GetClientConfig(request)

        self.cfg['push_max_stream_nbytes'] = int(response.config['push_max_stream_nbytes'])
        self.cfg['enable_compression'] = bool(int(response.config['enable_compression']))
        self.cfg['optimization_target'] = response.config['optimization_target']

        self.temp_channel.close()
        self.channel.close()
        self.temp_channel = grpc.insecure_channel(
            self.address,
            options=[('grpc.default_compression_algorithm', self.cfg['enable_compression']),
                     ('grpc.optimization_target', self.cfg['optimization_target'])])
        self.channel = grpc.intercept_channel(self.temp_channel, self.header_adder_int)
        self.stub = hangar_service_pb2_grpc.HangarServiceStub(self.channel)

    def push_branch_record(self, name):
        head = heads.get_branch_head_commit(self.env.branchenv, name)
        rec = hangar_service_pb2.BranchRecord(name=name, commit=head)
        request = hangar_service_pb2.PushBranchRecordRequest(rec=rec)
        response = self.stub.PushBranchRecord(request)
        return response

    def fetch_branch_record(self, name):
        rec = hangar_service_pb2.BranchRecord(name=name)
        request = hangar_service_pb2.FetchBranchRecordRequest(rec=rec)
        response = self.stub.FetchBranchRecord(request)
        return response

    def push_commit_record(self, commit):
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)

        reftxn = TxnRegister().begin_reader_txn(self.env.refenv)
        try:
            commitRefVal = reftxn.get(commitRefKey, default=False)
            commitParentVal = reftxn.get(commitParentKey, default=False)
            commitSpecVal = reftxn.get(commitSpecKey, default=False)
        finally:
            TxnRegister().abort_reader_txn(self.env.refenv)

        cIter = chunks.clientCommitChunkedIterator(commit, commitParentVal, commitSpecVal, commitRefVal)
        response = self.stub.PushCommit(cIter)
        return response

    def fetch_commit_record(self, commit):
        request = hangar_service_pb2.FetchCommitRequest(commit=commit)
        replies = self.stub.FetchCommit(request)
        for idx, reply in enumerate(replies):
            if idx == 0:
                total_size_of_data = reply.total_byte_size
                cRefBytes = bytearray(total_size_of_data)
                specVal = reply.record.spec
                parentVal = reply.record.parent
                offset = 0
            size = len(reply.record.ref)
            cRefBytes[offset: offset + size] = reply.record.ref
            offset += size

        if reply.error.code != 0:
            print(reply.error)
            return False

        commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        refTxn = TxnRegister().begin_writer_txn(self.env.refenv)
        try:
            refTxn.put(commitParentKey, parentVal, overwrite=False)
            refTxn.put(commitRefKey, cRefBytes, overwrite=False)
            refTxn.put(commitSpecKey, specVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.refenv)

        return commit

    def fetch_schema(self, schema_hash):
        schema_rec = hangar_service_pb2.SchemaRecord(digest=schema_hash)
        request = hangar_service_pb2.FetchSchemaRequest(rec=schema_rec)
        reply = self.stub.FetchSchema(request)
        if reply.error.code != 0:
            print(reply.error)
            return False

        schemaVal = reply.rec.blob
        schema_spec = parsing.dataset_record_schema_raw_val_from_db_val(schemaVal)
        sample_array = np.zeros(
            shape=tuple(schema_spec.schema_max_shape),
            dtype=np.typeDict[schema_spec.schema_dtype])

        h = self.fs.create_schema(
            repo_path=self.env.repo_path,
            schema_hash=schema_spec.schema_hash,
            sample_array=sample_array,
            remote_operation=True)
        h.close()
        self.fs.open(repo_path=self.env.repo_path, mode='a', remote_operation=True)

        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = TxnRegister().begin_writer_txn(self.env.hashenv)
        try:
            hashTxn.put(schemaKey, schemaVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.hashenv)
        return reply.error

    def push_schema(self, schema_hash):
        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)

        hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
        try:
            schemaVal = hashTxn.get(schemaKey, default=False)
        finally:
            TxnRegister().abort_reader_txn(self.env.hashenv)

        if schemaVal:
            rec = hangar_service_pb2.SchemaRecord(digest=schema_hash, blob=schemaVal)
            request = hangar_service_pb2.PushSchemaRequest(rec=rec)
            response = self.stub.PushSchema(request)
            return response
        else:
            print(f'Error: no schema with hash: {schema_hash} exists')
            return False

    def fetch_data(self, digests, fs, fetch_bar, save_bar):

        buf = io.BytesIO()
        packer = msgpack.Packer(use_bin_type=True)
        totalSize = 0
        for digest in digests:
            p = packer.pack(digest)
            buf.write(p)
            totalSize += len(p)

        cIter = chunks.tensorChunkedIterator(
            io_buffer=buf,
            uncomp_nbytes=totalSize,
            pb2_request=hangar_service_pb2.FetchDataRequest)

        try:
            ret = True
            replies = self.stub.FetchData(cIter)
            for idx, reply in enumerate(replies):
                if idx == 0:
                    uncomp_nbytes = reply.uncomp_nbytes
                    comp_nbytes = reply.comp_nbytes
                    dBytes, offset = bytearray(comp_nbytes), 0

                size = len(reply.raw_data)
                if size > 0:
                    dBytes[offset: offset + size] = reply.raw_data
                    offset += size
                    fetch_bar.update(size)

        except grpc.RpcError as rpc_error:
            if rpc_error.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
                ret = 'AGAIN'  # Sentinal indicating not all data was retrieved

        uncompBytes = blosc.decompress(dBytes)
        if uncomp_nbytes != len(uncompBytes):
            msg = f'ERROR: uncomp_nbytes sent: {uncomp_nbytes} != recieved {comp_nbytes}'
            raise RuntimeError(msg)

        buff = io.BytesIO(uncompBytes)
        unpacker = msgpack.Unpacker(
            buff, use_list=True, raw=False, max_buffer_size=1_000_000_000)

        hashTxn = TxnRegister().begin_writer_txn(self.env.hashenv)
        try:
            for data in unpacker:
                hdigest, schema_hash, dShape, dTypeN, ddBytes = data
                tensor = np.frombuffer(ddBytes, dtype=np.typeDict[dTypeN]).reshape(dShape)
                recieved_hash = hashlib.blake2b(tensor.tobytes(), digest_size=20).hexdigest()
                if recieved_hash != hdigest:
                    msg = f'HASH MANGLED, recieved: {recieved_hash} != digest: {hdigest}'
                    raise RuntimeError(msg)

                hdf_instance, hdf_dset, hdf_idx = fs.add_tensor_data(
                    array=tensor,
                    schema_hash=schema_hash,
                    remote_operation=True)
                hashKey = parsing.hash_data_db_key_from_raw_key(hdigest)
                hashVal = parsing.hash_data_db_val_from_raw_val(
                    hdf5_file_schema=schema_hash,
                    hdf5_schema_instance=hdf_instance,
                    hdf5_dataset=hdf_dset,
                    hdf5_dataset_idx=hdf_idx,
                    data_shape=tensor.shape)
                hashTxn.put(hashKey, hashVal)
                save_bar.update(1)
        finally:
            TxnRegister().commit_writer_txn(self.env.hashenv)

        return ret

    def push_data(self, digests):

        totalSize = 0
        buf = io.BytesIO()
        packer = msgpack.Packer(use_bin_type=True)
        hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
        try:
            for digest in tqdm(digests, desc='Push Data'):
                hashKey = parsing.hash_data_db_key_from_raw_key(digest)
                hashVal = hashTxn.get(hashKey, default=False)
                if not hashVal:
                    raise KeyError(f'No hash record with key: {hashKey}')

                hash_val = parsing.hash_data_raw_val_from_db_val(hashVal)
                schema_hash = hash_val.hdf5_file_schema
                data_shape = hash_val.data_shape
                hashSchemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
                schemaVal = hashTxn.get(hashSchemaKey, default=False)
                if not schemaVal:
                    raise KeyError(f'No hash schema key with key: {hashSchemaKey}')

                schema_val = parsing.dataset_record_schema_raw_val_from_db_val(schemaVal)
                dtype_num = schema_val.schema_dtype
                tensor = self.fs.read_data(hashVal=hash_val, mode='r', dtype=dtype_num)

                p = packer.pack((digest, schema_hash, data_shape, dtype_num, tensor.tobytes()))
                totalSize += len(p)
                buf.write(p)

                # only send a group of tensors <= Max Size so that the server does not
                # run out of RAM for large repos
                if totalSize >= self.cfg['push_max_stream_nbytes']:
                    cIter = chunks.tensorChunkedIterator(
                        io_buffer=buf,
                        uncomp_nbytes=totalSize,
                        pb2_request=hangar_service_pb2.PushDataRequest)
                    response = self.stub.PushData(cIter)
                    totalSize = 0
                    buf.close()
                    buf = io.BytesIO()
        finally:
            # finish sending all remaining tensors if max size hash not been hit.
            if totalSize > 0:
                cIter = chunks.tensorChunkedIterator(
                    io_buffer=buf,
                    uncomp_nbytes=totalSize,
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
        try:
            assert recieved_hash == digest
        except AssertionError:
            print(f'recieved_hash: {recieved_hash} != digest: {digest}')
            print(f'reply: {reply}')
            raise

        labelHashKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = TxnRegister().begin_writer_txn(self.env.labelenv)
        try:
            labelTxn.put(labelHashKey, uncompBlob, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.labelenv)
        return reply.error

    def push_label(self, digest):

        rec = hangar_service_pb2.HashRecord(digest=digest)
        request = hangar_service_pb2.PushLabelRequest(rec=rec)
        labelKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = TxnRegister().begin_reader_txn(self.env.labelenv)
        try:
            labelVal = labelTxn.get(labelKey, default=False)
            if labelVal is False:
                print(f'Error: labelval with key: {labelKey} does not exist')
                return False
            else:
                compLabelVal = blosc.compress(labelVal)
                request.blob = compLabelVal
        finally:
            TxnRegister().abort_reader_txn(self.env.labelenv)

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
        return missing_hashs

    def push_find_missing_hash_records(self, commit):

        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            s_hashset = set(queries.RecordQuery(tmpDB).data_hashes())
            s_hashes = list(s_hashset)
            tmpDB.close()

        pb2_func = hangar_service_pb2.FindMissingHashRecordsRequest
        cIter = chunks.missingHashRequestIterator(commit, s_hashes, pb2_func)
        responses = self.stub.PushFindMissingHashRecords(cIter)
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
        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
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

        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            c_schemaset = set(queries.RecordQuery(tmpDB).schema_hashes())
            c_schemas = list(c_schemaset)
            tmpDB.close()

        request = hangar_service_pb2.FindMissingSchemasRequest()
        request.commit = commit
        request.schema_digests.extend(c_schemas)

        response = self.stub.PushFindMissingSchemas(request)
        return response
