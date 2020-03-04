
cdef class SizedDict(dict):
    cdef public int _maxsize
    cdef public object _stack
