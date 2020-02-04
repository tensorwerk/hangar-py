# decoding methods to convert from byte string -> spec struct.

from .specs cimport HDF5_01_DataHashSpec, \
    HDF5_00_DataHashSpec, \
    NUMPY_10_DataHashSpec, \
    LMDB_30_DataHashSpec, \
    REMOTE_50_DataHashSpec


cdef HDF5_01_DataHashSpec HDF5_01_Parser(str inp):
    cdef str fmt, uid, cksum, dset, dset_idx
    cdef tuple shape_tup
    cdef list shape_list = []
    cdef int dataset_idx_int
    cdef unsigned char i, c, cc
    cdef unsigned char n = len(inp)
    cdef HDF5_01_DataHashSpec res

    c = 0
    cc = 0
    for i in range(n):
        if inp[i] == ':':
            if cc == 0:
                fmt = inp[c:i]
            elif cc == 1:
                uid = inp[c:i]
            elif cc == 2:
                cksum = inp[c:i]
            elif cc == 3:
                dset = inp[c:i]
            elif cc == 4:
                dset_idx = inp[c:i]
            c = i + 1
            cc = cc + 1
    shape_vs = inp[c:n]

    c = 0
    n = len(shape_vs)
    for i in range(n):
        if shape_vs[i] == ' ':
            shape_list.append(int(shape_vs[c:i]))
            c = i + 1
    if shape_vs[c:n] != '':
        shape_list.append(int(shape_vs[c:]))

    shape_tup = tuple(shape_list)
    dataset_idx_int = int(dset_idx)
    res = HDF5_01_DataHashSpec(fmt, uid, cksum, dset, dataset_idx_int, shape_tup)
    return res


cdef HDF5_00_DataHashSpec HDF5_00_Parser(str inp):
    cdef str fmt, uid, cksum, dset, dset_idx
    cdef tuple shape_tup
    cdef list shape_list = []
    cdef int dataset_idx_int
    cdef unsigned char i, c, cc
    cdef unsigned char n = len(inp)
    cdef HDF5_00_DataHashSpec res

    c = 0
    cc = 0
    for i in range(n):
        if inp[i] == ':':
            if cc == 0:
                fmt = inp[c:i]
            elif cc == 1:
                uid = inp[c:i]
            elif cc == 2:
                cksum = inp[c:i]
            elif cc == 3:
                dset = inp[c:i]
            elif cc == 4:
                dset_idx = inp[c:i]
            c = i + 1
            cc = cc + 1
    shape_vs = inp[c:n]

    c = 0
    n = len(shape_vs)
    for i in range(n):
        if shape_vs[i] == ' ':
            shape_list.append(int(shape_vs[c:i]))
            c = i + 1
    if shape_vs[c:n] != '':
        shape_list.append(int(shape_vs[c:]))

    shape_tup = tuple(shape_list)
    dataset_idx_int = int(dset_idx)
    res = HDF5_00_DataHashSpec(fmt, uid, cksum, dset, dataset_idx_int, shape_tup)
    return res


cdef NUMPY_10_DataHashSpec NUMPY_10_Parser(str inp):
    cdef str fmt, uid, cksum, collection_idx
    cdef tuple shape_tup
    cdef list shape_list = []
    cdef int collection_idx_int
    cdef unsigned char i, c, cc
    cdef unsigned char n = len(inp)
    cdef NUMPY_10_DataHashSpec res

    c = 0
    cc = 0
    for i in range(n):
        if inp[i] == ':':
            if cc == 0:
                fmt = inp[c:i]
            elif cc == 1:
                uid = inp[c:i]
            elif cc == 2:
                cksum = inp[c:i]
            elif cc == 3:
                collection_idx = inp[c:i]
            c = i + 1
            cc = cc + 1
    shape_vs = inp[c:n]

    c = 0
    n = len(shape_vs)
    for i in range(n):
        if shape_vs[i] == ' ':
            shape_list.append(int(shape_vs[c:i]))
            c = i + 1
    if shape_vs[c:n] != '':
        shape_list.append(int(shape_vs[c:]))

    shape_tup = tuple(shape_list)
    collection_idx_int = int(collection_idx)
    res = NUMPY_10_DataHashSpec(fmt, uid, cksum, collection_idx_int, shape_tup)
    return res


cdef LMDB_30_DataHashSpec LMDB_30_Parser(str inp):
    cdef str fmt, uid, row_idx, checksum
    cdef unsigned char i, c, cc
    cdef unsigned char n = len(inp)
    cdef LMDB_30_DataHashSpec res

    c = 0
    cc = 0
    for i in range(n):
        if inp[i] == ':':
            if cc == 0:
                fmt = inp[c:i]
            elif cc == 1:
                uid = inp[c:i]
            elif cc == 2:
                row_idx = inp[c:i]
            c = i + 1
            cc = cc + 1
    checksum = inp[c:n]

    res = LMDB_30_DataHashSpec(fmt, uid, row_idx, checksum)
    return res



cdef REMOTE_50_DataHashSpec REMOTE_50_Parser(str inp):
    cdef str fmt, schema_hash
    cdef unsigned char i, c
    cdef unsigned char n = len(inp)
    cdef REMOTE_50_DataHashSpec res

    c = 0
    for i in range(n):
        if inp[i] == ':':
            fmt = inp[c:i]
            c = i + 1
    schema_hash = inp[c:]
    res = REMOTE_50_DataHashSpec(fmt, schema_hash)
    return res


cpdef object backend_decoder(bytes inp):
    cdef str backend, inp_str
    inp_str = inp.decode('utf-8')
    backend = inp_str[:2]
    if backend == '00':
        return HDF5_00_Parser(inp_str)
    elif backend == '01':
        return HDF5_01_Parser(inp_str)
    elif backend == '10':
        return NUMPY_10_Parser(inp_str)
    elif backend == '30':
        return LMDB_30_Parser(inp_str)
    elif backend == '50':
        return REMOTE_50_Parser(inp_str)
    else:
        raise ValueError(f'unknown backend type for input str {inp_str}')
