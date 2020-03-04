# header files for spec containers

cdef class HDF5_01_DataHashSpec:

    cdef readonly str backend
    cdef readonly str uid
    cdef readonly str checksum
    cdef readonly str dataset
    cdef readonly int dataset_idx
    cdef readonly tuple shape


cdef class HDF5_00_DataHashSpec:

    cdef readonly str backend
    cdef readonly str uid
    cdef readonly str checksum
    cdef readonly str dataset
    cdef readonly int dataset_idx
    cdef readonly tuple shape


cdef class NUMPY_10_DataHashSpec:

    cdef readonly str backend
    cdef readonly str uid
    cdef readonly str checksum
    cdef readonly int collection_idx
    cdef readonly tuple shape


cdef class LMDB_30_DataHashSpec:

    cdef readonly str backend
    cdef readonly str uid
    cdef readonly str row_idx
    cdef readonly  str checksum


cdef class REMOTE_50_DataHashSpec:

    cdef readonly str backend
    cdef readonly str schema_hash
