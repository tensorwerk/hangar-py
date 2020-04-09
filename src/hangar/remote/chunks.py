import math
import struct
from typing import NamedTuple, List, Union, Tuple

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


def tensorChunkedIterator(buf, uncomp_nbytes, pb2_request, *, err=None):

    compBytes = blosc.compress(
        buf, clevel=3, cname='blosclz', shuffle=blosc.NOSHUFFLE)

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


class DataIdent(NamedTuple):
    digest: str
    schema: str


class DataRecord(NamedTuple):
    data: Union[np.ndarray, str, bytes]
    digest: str
    schema: str


def _serialize_arr(arr: np.ndarray) -> bytes:
    """
    dtype_num ndim dim1_size dim2_size ... dimN_size array_bytes
    """
    raw = struct.pack(
        f'<bb{len(arr.shape)}i{arr.nbytes}s',
        arr.dtype.num, arr.ndim, *arr.shape, arr.tobytes()
    )
    return raw


def _deserialize_arr(raw: bytes) -> np.ndarray:
    dtnum, ndim = struct.unpack('<bb', raw[0:2])
    end = 2 + (4 * ndim)
    arrshape = struct.unpack(f'<{ndim}i', raw[2:end])
    arr = np.frombuffer(raw, dtype=np.typeDict[dtnum], offset=end).reshape(arrshape)
    return arr


def _serialize_str(data: str) -> bytes:
    """
    data_bytes
    """
    return data.encode()


def _deserialize_str(raw: bytes) -> str:
    return raw.decode()


def _serialize_bytes(data: bytes) -> bytes:
    """
    data_bytes
    """
    return data


def _deserialize_bytes(data: bytes) -> bytes:
    return data


def serialize_ident(digest: str, schema: str) -> bytes:
    """
    len_digest len_schema digest_str schema_str
    """
    raw = struct.pack(
        f'<hh{len(digest)}s{len(schema)}s',
        len(digest), len(schema), digest.encode(), schema.encode()
    )
    return raw


def deserialize_ident(raw: bytes) -> DataIdent:
    digestLen, schemaLen = struct.unpack('<hh', raw[:4])
    rawdigest, rawschema = struct.unpack(f'<{digestLen}s{schemaLen}s', raw[4:])
    digest = rawdigest.decode()
    schema = rawschema.decode()
    return DataIdent(digest, schema)


def serialize_data(data: Union[np.ndarray, str, bytes]) -> Tuple[int, bytes]:
    if isinstance(data, np.ndarray):
        return (0, _serialize_arr(data))
    elif isinstance(data, str):
        return (2, _serialize_str(data))
    elif isinstance(data, bytes):
        return (3, _serialize_bytes(data))
    else:
        raise TypeError(type(data))


def deserialize_data(dtype_code: int, raw_data: bytes) -> Union[np.ndarray, str, bytes]:
    if dtype_code == 0:
        return _deserialize_arr(raw_data)
    elif dtype_code == 2:
        return _deserialize_str(raw_data)
    elif dtype_code == 3:
        return _deserialize_bytes(raw_data)
    else:
        raise ValueError(f'dtype_code unknown {dtype_code}')


def serialize_record(data: Union[np.ndarray, str, bytes], digest: str, schema: str) -> bytes:
    """
    dtype_code len_raw_ident len_raw_data raw_ident, raw_data
    """
    dtype_code, raw_data = serialize_data(data)
    raw_ident = serialize_ident(digest, schema)
    raw = struct.pack(
        f'<b2Q{len(raw_ident)}s{len(raw_data)}s',
        dtype_code, len(raw_ident), len(raw_data), raw_ident, raw_data
    )
    return raw


def deserialize_record(raw: bytes) -> DataRecord:
    identStart = 17  # 1 + 2 * 8 bytes
    dtype_code, identLen, dataLen = struct.unpack(f'<b2Q', raw[:identStart])
    identEnd = identStart + identLen
    arrEnd = identEnd + dataLen
    arr = deserialize_data(dtype_code, raw[identEnd:arrEnd])
    ident = deserialize_ident(raw[identStart:identEnd])
    return DataRecord(arr, ident.digest, ident.schema)


def serialize_record_pack(records: List[bytes]) -> bytes:
    """
    num_records len_rec1 raw_rec1 len_rec2 raw_rec2 ... len_recN raw_recN
    """
    raw_num_records = struct.pack(f'<i', len(records))
    raw_records = [b''.join([struct.pack(f'<Q', len(rec)), rec]) for rec in records]
    return b''.join([raw_num_records, *raw_records])


def deserialize_record_pack(raw: bytes) -> List[bytes]:
    numRecords = struct.unpack(f'<i', raw[:4])[0]
    cursorPos, recs = 4, []
    for i in range(numRecords):
        lenRec = struct.unpack(f'<Q', raw[cursorPos:cursorPos+8])[0]
        recs.append(raw[cursorPos+8:cursorPos+8+lenRec])
        cursorPos += (8 + lenRec)
    return recs
