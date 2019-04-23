import hashlib
import io
import os
import tempfile
import threading
import time
from concurrent import futures
from os.path import join as pjoin

import blosc
import grpc
import lmdb
import msgpack
import numpy as np

from . import chunks
from . import hangar_service_pb2
from . import hangar_service_pb2_grpc
from .request_header_validator_interceptor import RequestHeaderValidatorInterceptor
from .. import config
from ..context import Environments
from ..context import TxnRegister
from ..records import commiting
from ..records import hashs
from ..records import heads
from ..records import parsing
from ..records import queries
from ..records import summarize

blosc.set_nthreads(blosc.detect_number_of_cores() - 2)


class HangarServer(hangar_service_pb2_grpc.HangarServiceServicer):

    def __init__(self, repo_path, overwrite=False):

        self.env = Environments(repo_path=repo_path)
        try:
            self.env._init_repo(
                user_name='SERVER_USER',
                user_email='SERVER_USER@HANGAR.SERVER',
                remove_old=overwrite)
        except OSError:
            pass

        src_path = pjoin(os.path.dirname(__file__), 'config_server.yml')
        config.ensure_file(src_path, destination=repo_path, comment=False)
        config.refresh(paths=[repo_path])

        self.txnregister = TxnRegister()
        self.repo_path = self.env.repo_path
        self.data_dir = pjoin(self.repo_path, config.get('hangar.repository.data_dir'))

    # -------------------- Client Config --------------------------------------

    def GetClientConfig(self, request, context):

        push_max_stream_nbytes = str(config.get('remote.client.app.push_max_stream_nbytes'))
        enable_compression = config.get('remote.client.options.enable_compression')
        enable_compression = str(1) if enable_compression is True else str(0)
        optimization_target = config.get('remote.client.options.optimization_target')

        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        reply = hangar_service_pb2.GetClientConfigReply(error=err)
        reply.config['push_max_stream_nbytes'] = push_max_stream_nbytes
        reply.config['enable_compression'] = enable_compression
        reply.config['optimization_target'] = optimization_target
        return reply

    # -------------------- Branch Record --------------------------------------

    def FetchBranchRecord(self, request, context):
        branch_name = request.rec.name
        try:
            head = heads.get_branch_head_commit(self.env.branchenv, branch_name)
            rec = hangar_service_pb2.BranchRecord(name=branch_name, commit=head)
            err = hangar_service_pb2.ErrorProto(code=0, message='OK')
            reply = hangar_service_pb2.FetchBranchRecordReply(rec=rec, error=err)
        except ValueError:
            err = hangar_service_pb2.ErrorProto(code=1, message='BRANCH DOES NOT EXIST')
            reply = hangar_service_pb2.FetchBranchRecordReply(error=err)
        return reply

    def PushBranchRecord(self, request, context):
        branch_name = request.rec.name
        commit = request.rec.commit
        branch_names = heads.get_branch_names(self.env.branchenv)
        if branch_name not in branch_names:
            heads.create_branch(self.env.branchenv, branch_name=branch_name, base_commit=commit)
            err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        else:
            current_head = heads.get_branch_head_commit(self.env.branchenv, branch_name)
            if current_head == commit:
                err = hangar_service_pb2.ErrorProto(code=1, message='NO CHANGE TO BRANCH HEAD')
            else:
                heads.set_branch_head_commit(self.env.branchenv, branch_name, commit)
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')

        reply = hangar_service_pb2.PushBranchRecordReply(error=err)
        return reply

    # -------------------------- Commit Record --------------------------------

    def FetchCommit(self, request, context):
        commit = request.commit
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)

        reftxn = self.txnregister.begin_reader_txn(self.env.refenv)
        try:
            commitRefVal = reftxn.get(commitRefKey, default=False)
            commitParentVal = reftxn.get(commitParentKey, default=False)
            commitSpecVal = reftxn.get(commitSpecKey, default=False)
        finally:
            self.txnregister.abort_reader_txn(self.env.refenv)

        if commitRefVal is False:
            err = hangar_service_pb2.ErrorProto(code=1, message='COMMIT DOES NOT EXIST')
            reply = hangar_service_pb2.FetchCommitReply(commit=commit, error=err)
            yield reply
            raise StopIteration()
        else:
            raw_data_chunks = chunks.chunk_bytes(commitRefVal)
            bsize = len(commitRefVal)
            commit_proto = hangar_service_pb2.CommitRecord()
            commit_proto.parent = commitParentVal
            commit_proto.spec = commitSpecVal
            reply = hangar_service_pb2.FetchCommitReply(
                commit=commit,
                total_byte_size=bsize)
            for chunk in raw_data_chunks:
                commit_proto.ref = chunk
                reply.record.CopyFrom(commit_proto)
                yield reply

    def PushCommit(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                commit = request.commit
                refBytes, offset = bytearray(request.total_byte_size), 0
                specVal = request.record.spec
                parentVal = request.record.parent
            size = len(request.record.ref)
            refBytes[offset: offset + size] = request.record.ref
            offset += size

        commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        refTxn = self.txnregister.begin_writer_txn(self.env.refenv)
        try:
            cmtParExists = refTxn.put(commitParentKey, parentVal, overwrite=False)
            cmtRefExists = refTxn.put(commitRefKey, refBytes, overwrite=False)
            cmtSpcExists = refTxn.put(commitSpecKey, specVal, overwrite=False)
        finally:
            self.txnregister.commit_writer_txn(self.env.refenv)

        if cmtParExists is False:
            err = hangar_service_pb2.ErrorProto(code=1, message='COMMIT EXISTS')
        else:
            err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        reply = hangar_service_pb2.PushCommitReply(error=err)
        return reply

    # --------------------- Schema Record -------------------------------------

    def FetchSchema(self, request, context):
        schema_hash = request.rec.digest
        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = self.txnregister.begin_reader_txn(self.env.hashenv)
        try:
            schemaExists = hashTxn.get(schemaKey, default=False)
            if schemaExists is not False:
                print(f'found schema: {schema_hash}')
                rec = hangar_service_pb2.SchemaRecord(digest=schema_hash, blob=schemaExists)
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')
            else:
                print(f'not exists: {schema_hash}')
                rec = hangar_service_pb2.SchemaRecord(digest=schema_hash)
                err = hangar_service_pb2.ErrorProto(code=1, message='DOES NOT EXIST')
        finally:
            self.txnregister.abort_reader_txn(self.env.hashenv)

        reply = hangar_service_pb2.FetchSchemaReply(rec=rec, error=err)
        return reply

    def PushSchema(self, request, context):
        schema_hash = request.rec.digest
        schema_val = request.rec.blob

        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = self.txnregister.begin_writer_txn(self.env.hashenv)
        try:
            newSchema = hashTxn.put(schemaKey, schema_val, overwrite=False)
            if newSchema is True:
                print(f'created new: {schema_val}')
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')
            else:
                print(f'exists: {schema_val}')
                err = hangar_service_pb2.ErrorProto(code=1, message='ALREADY EXISTS')
        finally:
            self.txnregister.commit_writer_txn(self.env.hashenv)

        reply = hangar_service_pb2.PushSchemaReply(error=err)
        return reply

    # ---------------------------- Data ---------------------------------------

    def FetchData(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                uncomp_nbytes = request.uncomp_nbytes
                comp_nbytes = request.comp_nbytes
                dBytes, offset = bytearray(comp_nbytes), 0
            size = len(request.raw_data)
            dBytes[offset: offset + size] = request.raw_data
            offset += size

        uncompBytes = blosc.decompress(dBytes)
        if uncomp_nbytes != len(uncompBytes):
            msg = f'ERROR: uncomp_nbytes sent: {uncomp_nbytes} != recieved {comp_nbytes}'
            err = hangar_service_pb2.ErrorProto(code=1, message=msg)
            reply = hangar_service_pb2.FetchDataReply(error=err)
            yield reply
            raise StopIteration()

        buff = io.BytesIO(uncompBytes)
        unpacker = msgpack.Unpacker(buff, use_list=False, raw=False, max_buffer_size=1_000_000_000)

        totalSize = 0
        buf = io.BytesIO()
        packer = msgpack.Packer(use_bin_type=True)
        hashTxn = self.txnregister.begin_reader_txn(self.env.hashenv)
        fetch_max_stream_nbytes = config.get('remote.server.app.fetch_max_stream_nbytes')
        try:
            for digest in unpacker:
                hashKey = parsing.hash_data_db_key_from_raw_key(digest)
                hashVal = hashTxn.get(hashKey, default=False)
                if hashVal is False:
                    msg = f'HASH DOES NOT EXIST: {hashKey}'
                    err = hangar_service_pb2.ErrorProto(code=1, message=msg)
                    reply = hangar_service_pb2.FetchDataReply(error=err)
                    yield reply
                    raise StopIteration()
                else:
                    schema_hash, fname = hashVal.decode().split(' ', 1)
                    tensor = np.load(fname)

                p = packer.pack((
                    digest,
                    schema_hash,
                    tensor.shape,
                    tensor.dtype.num,
                    tensor.tobytes()))
                buf.seek(totalSize)
                buf.write(p)
                totalSize += len(p)

                # only send a group of tensors <= Max Size so that the server does not
                # run out of RAM for large repos
                if totalSize >= fetch_max_stream_nbytes:
                    err = hangar_service_pb2.ErrorProto(code=0, message='OK')
                    cIter = chunks.tensorChunkedIterator(
                        io_buffer=buf,
                        uncomp_nbytes=totalSize,
                        pb2_request=hangar_service_pb2.FetchDataReply,
                        err=err)
                    yield from cIter
                    time.sleep(0.1)
                    msg = 'HANGAR REQUESTED RETRY: developer enforced limit on returned '\
                          'raw data size to prevent memory overload of user system.'
                    context.set_details(msg)
                    context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                    err = hangar_service_pb2.ErrorProto(code=1, message=msg)
                    yield hangar_service_pb2.FetchDataReply(error=err, raw_data=b'')
                    raise StopIteration()

        except StopIteration:
            totalSize = 0

        finally:
            # finish sending all remaining tensors if max size hash not been hit.
            if totalSize > 0:
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')
                cIter = chunks.tensorChunkedIterator(
                    io_buffer=buf,
                    uncomp_nbytes=totalSize,
                    pb2_request=hangar_service_pb2.FetchDataReply,
                    err=err)
                yield from cIter
            buf.close()
            self.txnregister.abort_reader_txn(self.env.hashenv)

    def PushData(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                uncomp_nbytes = request.uncomp_nbytes
                comp_nbytes = request.comp_nbytes
                dBytes, offset = bytearray(comp_nbytes), 0
            size = len(request.raw_data)
            dBytes[offset: offset + size] = request.raw_data
            offset += size

        uncompBytes = blosc.decompress(dBytes)
        if uncomp_nbytes != len(uncompBytes):
            msg = f'ERROR: uncomp_nbytes sent: {uncomp_nbytes} != recieved {comp_nbytes}'
            err = hangar_service_pb2.ErrorProto(code=1, message=msg)
            reply = hangar_service_pb2.PushDataReply(error=err)
            return reply

        buff = io.BytesIO(uncompBytes)
        unpacker = msgpack.Unpacker(buff, use_list=False, raw=False, max_buffer_size=1_000_000_000)
        hashTxn = self.txnregister.begin_writer_txn(self.env.hashenv)
        try:
            for data in unpacker:
                digest, schema_hash, dShape, dTypeN, dBytes = data
                tensor = np.frombuffer(dBytes, dtype=np.typeDict[dTypeN]).reshape(dShape)
                recieved_hash = hashlib.blake2b(tensor.tobytes(), digest_size=20).hexdigest()
                if recieved_hash != digest:
                    msg = f'HASH MANGLED, recieved: {recieved_hash} != digest: {digest}'
                    err = hangar_service_pb2.ErrorProto(code=1, message=msg)
                    reply = hangar_service_pb2.PushDataReply(error=err)
                    return reply

                hashKey = parsing.hash_data_db_key_from_raw_key(digest)
                hashdir = os.path.join(self.data_dir, digest[:2])
                fname = os.path.join(hashdir, f'{digest}.npz')
                hashVal = f'{schema_hash} {fname}'.encode()
                if not os.path.isdir(hashdir):
                    os.makedirs(hashdir)

                noPreviousHash = hashTxn.put(hashKey, hashVal, overwrite=False)
                if noPreviousHash:
                    try:
                        with open(fname, 'xb') as fh:
                            np.save(fh, tensor)
                    except FileExistsError:
                        hashTxn.delete(hashKey)
                        msg = f'DATA FILE EXISTS BUT HASH NOT RECORDED: {hashKey}'
                        err = hangar_service_pb2.ErrorProto(code=1, message=msg)
                        reply = hangar_service_pb2.PushDataReply(error=err)
                        return reply
                else:
                    msg = f'HASH EXISTS: {hashKey}'
                    err = hangar_service_pb2.ErrorProto(code=1, message=msg)
                    reply = hangar_service_pb2.PushDataReply(error=err)
                    return reply
        finally:
            self.txnregister.commit_writer_txn(self.env.hashenv)

        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        reply = hangar_service_pb2.PushDataReply(error=err)
        return reply

    # ----------------------------- Label Data --------------------------------

    def FetchLabel(self, request, context):
        digest = request.rec.digest
        digest_type = request.rec.type
        rec = hangar_service_pb2.HashRecord(digest=digest, type=digest_type)
        reply = hangar_service_pb2.FetchLabelReply(rec=rec)

        labelKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = self.txnregister.begin_reader_txn(self.env.labelenv)
        try:
            labelVal = labelTxn.get(labelKey, default=False)
            if labelVal is False:
                msg = f'DOES NOT EXIST: labelval with key: {labelKey}'
                err = hangar_service_pb2.ErrorProto(code=1, message=msg)
            else:
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')
                compLabelVal = blosc.compress(labelVal)
                reply.blob = compLabelVal
        finally:
            self.txnregister.abort_reader_txn(self.env.labelenv)

        reply.error.CopyFrom(err)
        return reply

    def PushLabel(self, request, context):
        digest = request.rec.digest

        uncompBlob = blosc.decompress(request.blob)
        recieved_hash = hashlib.blake2b(uncompBlob, digest_size=20).hexdigest()
        try:
            assert recieved_hash == digest
            err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        except AssertionError:
            msg = f'HASH MANGED: recieved_hash: {recieved_hash} != digest: {digest}'
            err = hangar_service_pb2.ErrorProto(code=1, message=msg)
            reply = hangar_service_pb2.PushLabelReply(error=err)
            return reply

        labelHashKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = self.txnregister.begin_writer_txn(self.env.labelenv)
        try:
            succ = labelTxn.put(labelHashKey, uncompBlob, overwrite=False)
            if succ is False:
                msg = f'HASH ALREADY EXISTS: {labelHashKey}'
                err = hangar_service_pb2.ErrorProto(code=1, message=msg)
            else:
                err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        finally:
            self.txnregister.commit_writer_txn(self.env.labelenv)

        reply = hangar_service_pb2.PushLabelReply(error=err)
        return reply

    # ------------------------ Fetch Find Missing -----------------------------------

    def FetchFindMissingCommits(self, request, context):
        c_branch_name = request.branch.name
        c_ordered_commits = request.commits

        try:
            s_history = summarize.list_history(
                refenv=self.env.refenv,
                branchenv=self.env.branchenv,
                branch_name=c_branch_name)
        except ValueError:
            msg = f'BRANCH NOT EXIST. Name: {c_branch_name}'
            err = hangar_service_pb2.ErrorProto(code=1, message=msg)

        s_orderset = set(s_history['order'])
        c_orderset = set(c_ordered_commits)
        c_missing = list(s_orderset.difference(c_orderset))   # only difference to PushFindMissingCommits

        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        if len(c_missing) == 0:
            brch = hangar_service_pb2.BranchRecord(name=c_branch_name, commit=s_history['head'])
            reply = hangar_service_pb2.FindMissingCommitsReply(branch=brch, error=err)
        else:
            brch = hangar_service_pb2.BranchRecord(name=c_branch_name, commit=s_history['head'])
            reply = hangar_service_pb2.FindMissingCommitsReply(branch=brch, error=err)
            reply.commits.extend(c_missing)

        return reply

    def PushFindMissingCommits(self, request, context):
        c_branch_name = request.branch.name
        c_head_commit = request.branch.commit
        c_ordered_commits = request.commits

        s_commits = commiting.list_all_commits(self.env.refenv)
        s_orderset = set(s_commits)
        c_orderset = set(c_ordered_commits)
        s_missing = list(c_orderset.difference(s_orderset))  # only difference to FetchFindMissingCommits

        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        if len(s_missing) == 0:
            brch = hangar_service_pb2.BranchRecord(name=c_branch_name, commit=c_head_commit)
            reply = hangar_service_pb2.FindMissingCommitsReply(branch=brch, error=err)
        else:
            brch = hangar_service_pb2.BranchRecord(name=c_branch_name, commit=c_head_commit)
            reply = hangar_service_pb2.FindMissingCommitsReply(branch=brch, error=err)
            reply.commits.extend(s_missing)

        return reply

    def FetchFindMissingHashRecords(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                commit = request.commit
                hBytes, offset = bytearray(request.total_byte_size), 0
            size = len(request.hashs)
            hBytes[offset: offset + size] = request.hashs
            offset += size
        uncompBytes = blosc.decompress(hBytes)
        c_hashset = set(msgpack.unpackb(uncompBytes, raw=False, use_list=False))

        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            s_hashes = set(queries.RecordQuery(tmpDB).data_hashes())
            tmpDB.close()

        c_missing = list(s_hashes.difference(c_hashset))
        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        response_pb = hangar_service_pb2.FindMissingHashRecordsReply
        cIter = chunks.missingHashIterator(commit, c_missing, err, response_pb)
        yield from cIter

    def PushFindMissingHashRecords(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                commit = request.commit
                hBytes, offset = bytearray(request.total_byte_size), 0
            size = len(request.hashs)
            hBytes[offset: offset + size] = request.hashs
            offset += size
        uncompBytes = blosc.decompress(hBytes)
        c_hashset = set(msgpack.unpackb(uncompBytes, raw=False, use_list=False))

        s_hashset = set(hashs.HashQuery(self.env.hashenv).list_all_hash_keys_raw())

        s_missing = list(c_hashset.difference(s_hashset))
        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        response_pb = hangar_service_pb2.FindMissingHashRecordsReply
        cIter = chunks.missingHashIterator(commit, s_missing, err, response_pb)
        yield from cIter

    def FetchFindMissingLabels(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                commit = request.commit
                hBytes, offset = bytearray(request.total_byte_size), 0
            size = len(request.hashs)
            hBytes[offset: offset + size] = request.hashs
            offset += size
        uncompBytes = blosc.decompress(hBytes)
        c_hashset = set(msgpack.unpackb(uncompBytes, raw=False, use_list=False))

        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            s_hashes = set(queries.RecordQuery(tmpDB).metadata_hashes())
            tmpDB.close()

        c_missing = list(s_hashes.difference(c_hashset))
        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        response_pb = hangar_service_pb2.FindMissingLabelsReply
        cIter = chunks.missingHashIterator(commit, c_missing, err, response_pb)
        yield from cIter

    def PushFindMissingLabels(self, request_iterator, context):
        for idx, request in enumerate(request_iterator):
            if idx == 0:
                commit = request.commit
                hBytes, offset = bytearray(request.total_byte_size), 0
            size = len(request.hashs)
            hBytes[offset: offset + size] = request.hashs
            offset += size
        uncompBytes = blosc.decompress(hBytes)
        c_hashset = set(msgpack.unpackb(uncompBytes, raw=False, use_list=True))
        s_hash_keys = list(hashs.HashQuery(self.env.labelenv).list_all_hash_keys_db())
        s_hashes = map(parsing.hash_meta_raw_key_from_db_key, s_hash_keys)
        s_hashset = set(s_hashes)

        s_missing = list(c_hashset.difference(s_hashset))
        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        response_pb = hangar_service_pb2.FindMissingLabelsReply
        cIter = chunks.missingHashIterator(commit, s_missing, err, response_pb)
        yield from cIter

    def FetchFindMissingSchemas(self, request, context):
        commit = request.commit
        c_schemas = set(request.schema_digests)

        LMDB_CONFIG = config.get('hangar.lmdb')
        with tempfile.TemporaryDirectory() as tempD:
            tmpDF = os.path.join(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.env.refenv, tmpDB, commit)
            s_schemas = set(queries.RecordQuery(tmpDB).schema_hashes())
            tmpDB.close()

        c_missing = list(s_schemas.difference(c_schemas))
        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        reply = hangar_service_pb2.FindMissingSchemasReply(commit=commit, error=err)
        reply.schema_digests.extend(c_missing)
        return reply

    def PushFindMissingSchemas(self, request, context):
        commit = request.commit
        c_schemas = set(request.schema_digests)

        s_schemas = set(hashs.HashQuery(self.env.hashenv).list_all_schema_keys_raw())
        s_missing = list(c_schemas.difference(s_schemas))

        err = hangar_service_pb2.ErrorProto(code=0, message='OK')
        reply = hangar_service_pb2.FindMissingSchemasReply(commit=commit, error=err)
        reply.schema_digests.extend(s_missing)
        return reply


def serve(hangar_path, overwrite=False):
    '''Start serving the GRPC server. Should only be called once.

    Raises:
        e: critical error from one of the workers.
    '''

    # ------------------- Configure Server ------------------------------------

    src_path = pjoin(os.path.dirname(__file__), 'config_server.yml')
    dest_path = pjoin(hangar_path, config.get('hangar.repository.hangar_server_dir_name'))
    config.ensure_file(src_path, destination=dest_path, comment=False)
    config.refresh(paths=[dest_path])

    enable_compression = config.get('remote.server.grpc.options.enable_compression')
    optimization_target = config.get('remote.server.grpc.options.optimization_target')
    channel_address = config.get('remote.server.grpc.channel_address')
    max_thread_pool_workers = config.get('remote.server.grpc.max_thread_pool_workers')
    max_concurrent_rpcs = config.get('remote.server.grpc.max_concurrent_rpcs')

    admin_restrict_push = config.get('remote.server.admin.restrict_push')
    admin_username = config.get('remote.server.admin.username')
    admin_password = config.get('remote.server.admin.password')
    msg = 'PERMISSION ERROR: PUSH OPERATIONS RESTRICTED FOR CALLER'
    code = grpc.StatusCode.PERMISSION_DENIED
    interc = RequestHeaderValidatorInterceptor(
        admin_restrict_push, admin_username, admin_password, code, msg)

    # ---------------- Start the thread pool for the grpc server --------------

    grpc_thread_pool = futures.ThreadPoolExecutor(
        max_workers=max_thread_pool_workers,
        thread_name_prefix='grpc_thread_pool')
    server = grpc.server(
        thread_pool=grpc_thread_pool,
        maximum_concurrent_rpcs=max_concurrent_rpcs,
        options=[('grpc.default_compression_algorithm', enable_compression),
                 ('grpc.optimization_target', optimization_target)],
        interceptors=(interc,))

    # ------------------- Start the GRPC server -------------------------------

    hangserv = HangarServer(dest_path, overwrite)
    hangar_service_pb2_grpc.add_HangarServiceServicer_to_server(hangserv, server)
    server.add_insecure_port(channel_address)
    server.start()

    print('started')
    try:
        while True:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        print('stopped')
        server.stop(0)


if __name__ == '__main__':
    workdir = os.getcwd()
    print(workdir)
    serve(workdir)
