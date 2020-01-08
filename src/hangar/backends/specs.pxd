# header files for spec containers

cdef class HDF5_01_DataHashSpec:

    cdef public str backend
    cdef public str uid
    cdef public str checksum
    cdef public str dataset
    cdef public unsigned int dataset_idx
    cdef public tuple shape


cdef class HDF5_00_DataHashSpec:

    cdef public str backend
    cdef public str uid
    cdef public str checksum
    cdef public str dataset
    cdef public unsigned int dataset_idx
    cdef public tuple shape


cdef class NUMPY_10_DataHashSpec:

    cdef public str backend
    cdef public str uid
    cdef public str checksum
    cdef public unsigned int collection_idx
    cdef public tuple shape


cdef class REMOTE_50_DataHashSpec:

    cdef public str backend
    cdef public str schema_hash
