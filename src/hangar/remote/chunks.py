import math

import blosc
import msgpack

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
        if numIters == 1:
            yield bytesData
        elif i == 0:
            yield bytesData[:currentEnd]
        elif i == int(numIters - 1):
            yield bytesData[currentStart:]
        else:
            yield bytesData[currentStart:currentEnd]
        currentStart = currentStart + chunkSize
        currentEnd = currentEnd + chunkSize


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
    commit_proto = hangar_service_pb2.CommitRecord()
    commit_proto.parent = parentVal
    commit_proto.spec = specVal
    byteSize = len(refVal)
    chunkIterator = chunk_bytes(refVal)
    request = hangar_service_pb2.PushCommitRequest(commit=commit, total_byte_size=byteSize)
    for refChunk in chunkIterator:
        commit_proto.ref = refChunk
        request.record.CopyFrom(commit_proto)
        yield request


def tensorChunkedIterator(buf, uncomp_nbytes, itemsize, pb2_request, *, err=None):

    buf.seek(0)
    compBytes = blosc.compress(
        buf.getbuffer(), clevel=5, cname='blosclz', typesize=itemsize, shuffle=blosc.SHUFFLE)

    request = pb2_request(
        comp_nbytes=len(compBytes),
        uncomp_nbytes=uncomp_nbytes,
        error=err)
    chunkIterator = chunk_bytes(compBytes)
    for dchunk in chunkIterator:
        request.raw_data = dchunk
        yield request


def missingHashIterator(commit, hashes, err, pb2_func):
    hash_bytes = msgpack.packb(hashes)
    comp_bytes = blosc.compress(
        hash_bytes, cname='blosclz', clevel=3, typesize=1, shuffle=blosc.SHUFFLE)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes),
        error=err)

    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        rpc_method.hashs = bchunk
        yield rpc_method


def missingHashRequestIterator(commit, hashes, pb2_func):
    hash_bytes = msgpack.packb(hashes)
    comp_bytes = blosc.compress(
        hash_bytes, cname='blosclz', clevel=3, typesize=1, shuffle=blosc.SHUFFLE)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes))

    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        rpc_method.hashs = bchunk
        yield rpc_method
