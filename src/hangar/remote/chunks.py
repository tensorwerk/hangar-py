import math
import struct
from typing import NamedTuple, List

import blosc
import numpy as np

from . import hangar_service_pb2
from ..utils import set_blosc_nthreads

set_blosc_nthreads()


def chunk_bytes(bytesData):
    """Slice a bytestring into subelements and store the data in a list

    Arguments
    ---------
        bytesData : bytes
            bytestring buffer of the array data

    Yields
    ------
    bytes
        data split into 32kb chunk sizes.
    """
    chunkSize = 32_000
    numIters = math.ceil(len(bytesData) / chunkSize)
    currentStart = 0
    currentEnd = chunkSize
    for i in range(numIters):
        yield bytesData[currentStart:currentEnd]
        currentStart += chunkSize
        currentEnd += chunkSize


def clientCommitChunkedIterator(commit: str, parentVal: bytes, specVal: bytes,
                                refVal: bytes) -> hangar_service_pb2.PushCommitRequest:
    """Generator splitting commit specs into chunks sent from client to server

    Parameters
    ----------
    commit : str
        commit hash which is being sent
    parentVal : bytes
        bytes representing the commits immediate parents
    specVal : bytes
        bytes representing the commit message/user specifications
    refVal : bytes
        bytes containing all records stored in the repository

    Yields
    ------
    hangar_service_pb2.PushCommitRequest
        Chunked generator of the PushCommitRequest protobuf.
    """
    commit_proto = hangar_service_pb2.CommitRecord(
        parent=parentVal,
        spec=specVal)
    byteSize = len(refVal)
    chunkIterator = chunk_bytes(refVal)
    for refChunk in chunkIterator:
        commit_proto.ref = refChunk
        request = hangar_service_pb2.PushCommitRequest(
            commit=commit,
            total_byte_size=byteSize,
            record=commit_proto)
        yield request


def tensorChunkedIterator(buf, uncomp_nbytes, itemsize, pb2_request, *, err=None):

    compBytes = blosc.compress(
        buf, clevel=3, cname='lz4', typesize=1, shuffle=blosc.NOSHUFFLE)

    request = pb2_request(
        comp_nbytes=len(compBytes),
        uncomp_nbytes=uncomp_nbytes,
        error=err)
    chunkIterator = chunk_bytes(compBytes)
    for dchunk in chunkIterator:
        request.raw_data = dchunk
        yield request


def missingHashIterator(commit, hash_bytes, err, pb2_func):
    comp_bytes = blosc.compress(
        hash_bytes, cname='zlib', clevel=3, typesize=1, shuffle=blosc.SHUFFLE)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes),
        error=err)

    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        rpc_method.hashs = bchunk
        yield rpc_method


def missingHashRequestIterator(commit, hash_bytes, pb2_func):
    comp_bytes = blosc.compress(
        hash_bytes, cname='zlib', clevel=3, typesize=1, shuffle=blosc.SHUFFLE)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes))

    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        rpc_method.hashs = bchunk
        yield rpc_method


# ------------------------ serialization formats -------------------------


ArrayIdent = NamedTuple('ArrayIdent', [
    ('digest', str),
    ('schema', str)
])

ArrayRecord = NamedTuple('ArrayRecord', [
    ('array', np.ndarray),
    ('digest', str),
    ('schema', str),
])


def serialize_arr(arr: np.ndarray) -> bytes:
    """
    dtype_num ndim dim1_size dim2_size ... dimN_size array_bytes
    """
    domain = struct.pack(f'<II', arr.dtype.num, arr.ndim)
    dims = [struct.pack(f'<I', dim) for dim in arr.shape]
    return b''.join([domain, *dims, arr.tobytes()])


def deserialize_arr(raw: bytes) -> np.ndarray:
    dtnum, ndim = struct.unpack('<II', raw[:8])
    dtype = np.typeDict[dtnum]
    dataStart = 8 + (ndim * 4)
    shape = struct.unpack(f'<{ndim}I', raw[8:dataStart])
    arr = np.frombuffer(raw, dtype=dtype, offset=dataStart).reshape(shape)
    return arr


def serialize_ident(digest: str, schema: str) -> bytes:
    """
    len_digest digest_str len_schema schema_str
    """
    ident_digest = struct.pack(f'<I{len(digest)}s', len(digest), digest.encode())
    ident_schema = struct.pack(f'<I{len(schema)}s', len(schema), schema.encode())
    return b''.join([ident_digest, ident_schema])


def deserialize_ident(raw: bytes) -> ArrayIdent:
    digestLen = struct.unpack('<I', raw[:4])[0]
    digestEnd = 4 + (digestLen * 1)
    schemaStart = 4 + digestEnd
    schemaLen = struct.unpack('<I', raw[digestEnd:schemaStart])[0]
    digest = struct.unpack(f'<{digestLen}s', raw[4:digestEnd])[0].decode()
    schema = struct.unpack(f'<{schemaLen}s', raw[schemaStart:])[0].decode()
    return ArrayIdent(digest, schema)


def serialize_record(arr: np.ndarray, digest: str, schema: str) -> bytes:
    """
    len_raw_ident len_raw_arr raw_ident, raw_arr
    """
    raw_arr = serialize_arr(arr)
    raw_ident = serialize_ident(digest, schema)
    record = struct.pack(f'<2Q', len(raw_ident), len(raw_arr))
    return b''.join([record, raw_ident, raw_arr])


def deserialize_record(raw: bytes) -> ArrayRecord:
    identStart = 16  # 2 * 8 bytes
    identLen, arrLen = struct.unpack(f'<2Q', raw[:identStart])
    identEnd = identStart + identLen
    arrEnd = identEnd + arrLen

    arr = deserialize_arr(raw[identEnd:arrEnd])
    ident = deserialize_ident(raw[identStart:identEnd])
    return ArrayRecord(arr, ident.digest, ident.schema)


def serialize_record_pack(records: List[bytes]) -> bytes:
    """
    num_records len_rec1 raw_rec1 len_rec2 raw_rec2 ... len_recN raw_recN
    """
    raw_num_records = struct.pack(f'<I', len(records))
    raw_records = [b''.join([struct.pack(f'<Q', len(rec)), rec]) for rec in records]
    return b''.join([raw_num_records, *raw_records])


def deserialize_record_pack(raw: bytes) -> List[bytes]:
    numRecords = struct.unpack(f'<I', raw[:4])[0]
    cursorPos, recs = 4, []
    for i in range(numRecords):
        lenRec = struct.unpack(f'<Q', raw[cursorPos:cursorPos+8])[0]
        recs.append(raw[cursorPos+8:cursorPos+8+lenRec])
        cursorPos += (8 + lenRec)
    return recs
