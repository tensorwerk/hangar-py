
cdef class SizedDict(dict):
    cdef public int _maxsize
    cdef public object _stack
    cdef public dict _data
    cdef public int _stack_size
