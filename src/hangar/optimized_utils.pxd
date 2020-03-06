
cdef class SizedDict(dict):
    cdef int _maxsize
    cdef object _stack
    cdef dict _data
    cdef int _stack_size
