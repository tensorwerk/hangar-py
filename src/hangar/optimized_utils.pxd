"""
Portions of this code have been taken and modified from the "cytoolz" project.

URL:      https://github.com/pytoolz/cytoolz
File:     cytoolz/dicttoolz.pyd
Commit:   b66732f7f51937e85f5112481baf9db9c97b2ad2
Accessed: 05 APR 2020

CyToolz License
-------------------------------------------------------------------------------
License: New BSD
URL:     https://github.com/pytoolz/cytoolz/blob/b66732f7f51937e85f5112481baf9db9c97b2ad2/LICENSE.txt
"""
from cpython.ref cimport PyObject

cdef class SizedDict(dict):
    cdef public int _maxsize
    cdef public object _stack
    cdef public dict _data
    cdef public int _stack_size

cpdef object is_iterable(object x)

cpdef object is_ordered_sequence(object x)

cpdef int find_next_prime(int N)

ctypedef int (*f_map_next)(object p, Py_ssize_t *ppos, PyObject* *pkey, PyObject* *pval) except -1

# utility functions to perform iteration over dicts or generic mapping
cdef class _iter_mapping:
    cdef object it
    cdef object cur

cdef f_map_next get_map_iter(object d, PyObject* *ptr) except NULL

cdef int PyMapping_Next(object p, Py_ssize_t *ppos, PyObject* *pkey, PyObject* *pval) except -1

cpdef object valfilter(object predicate, object d, object factory=*)

cpdef object valfilterfalse(object predicate, object d, object factory=*)
