from cpython.bytes cimport (PyBytes_GET_SIZE,
                            PyBytes_AS_STRING,
                            PyBytes_Size,
                            PyBytes_FromString,
                            PyBytes_FromStringAndSize)
from cpython.float cimport PyFloat_FromDouble
from cpython.long cimport PyLong_FromLong

from cpython.ref cimport (Py_INCREF, Py_DECREF, PyTypeObject)

from libc.stdio cimport (FILE, stdout)
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport (uint8_t, int8_t,
                          uint16_t, int16_t,
                          uint32_t, int32_t,
                          uint64_t, int64_t,
                          uintptr_t)
from libc.stddef cimport ptrdiff_t

from libc cimport limits

cdef extern from "Python.h":
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    object PyUnicode_FromString(const char *u)
