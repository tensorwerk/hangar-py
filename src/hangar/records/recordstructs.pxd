# header file for record containers

cdef class CompatibleData:
    cdef readonly bint compatible
    cdef readonly str reason


cdef class ColumnSchemaKey:
    cdef readonly str column
    cdef readonly str layout
    cpdef readonly str digest


cdef class FlatColumnDataKey:
    cdef readonly str column
    cdef str _sample
    cdef bint _s_int


cdef class NestedColumnDataKey:
    cdef readonly str column
    cdef str _sample, _subsample
    cdef bint _s_int, _ss_int


cdef class DataRecordVal:
    cdef readonly str digest


cdef class MetadataRecordKey:
    cdef str _key
    cdef bint _k_int
