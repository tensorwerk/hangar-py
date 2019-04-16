import math

import blosc
import msgpack

from . import hangar_service_pb2

blosc.set_nthreads(blosc.detect_number_of_cores() - 2)


def chunk_bytes(bytesData):
    '''Slice a bytestring into subelements and store the data in a list

    Args:
        bytesData (bytes): bytestring buffer of the array data

    Returns:
        list: chunks of the bytestring seperated into elements of the configured
        size.

    TODO:
        Make chunks yield in a generator.
    '''
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


def clientCommitChunkedIterator(commit, parentVal, specVal, refVal):
    commit_proto = hangar_service_pb2.CommitRecord()
    commit_proto.parent = parentVal
    commit_proto.spec = specVal
    totalByteSize = len(refVal)
    chunkIterator = chunk_bytes(refVal)
    request = hangar_service_pb2.PushCommitRequest(commit=commit, total_byte_size=totalByteSize)
    for refChunk in chunkIterator:
        commit_proto.ref = refChunk
        request.record.CopyFrom(commit_proto)
        yield request


def tensorChunkedIterator(io_buffer, uncomp_nbytes, pb2_request, *, err=None):

    io_buffer.seek(0)
    compBytes = blosc.compress(
        io_buffer.getbuffer(),
        typesize=8,
        cname='lz4',
        clevel=9,
        shuffle=blosc.SHUFFLE)

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
    comp_bytes = blosc.compress(hash_bytes, cname='lz4', clevel=7, typesize=8)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes),
        error=err)

    offset = 0
    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        size = len(bchunk)
        rpc_method.hashs = bchunk
        offset += size
        yield rpc_method


def missingHashRequestIterator(commit, hashes, pb2_func):
    hash_bytes = msgpack.packb(hashes)
    comp_bytes = blosc.compress(hash_bytes, cname='lz4', clevel=7, typesize=8)

    rpc_method = pb2_func(
        commit=commit,
        total_byte_size=len(comp_bytes))

    offset = 0
    chunkIterator = chunk_bytes(comp_bytes)
    for bchunk in chunkIterator:
        size = len(bchunk)
        rpc_method.hashs = bchunk
        offset += size
        yield rpc_method
