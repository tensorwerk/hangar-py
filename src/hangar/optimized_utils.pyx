"""
Portions of this code have been taken and modified from the "cytoolz" project.

URL:      https://github.com/pytoolz/cytoolz
File:     cytoolz/dicttoolz.pyx
Commit:   b66732f7f51937e85f5112481baf9db9c97b2ad2
Accessed: 05 APR 2020

CyToolz License
-------------------------------------------------------------------------------
License: New BSD
URL:     https://github.com/pytoolz/cytoolz/blob/b66732f7f51937e85f5112481baf9db9c97b2ad2/LICENSE.txt
"""
from cpython.dict cimport PyDict_CheckExact
from cpython.ref cimport PyObject, Py_DECREF, Py_INCREF, Py_XDECREF

# Locally defined bindings that differ from `cython.cpython` bindings
from .external_cpython cimport PyDict_Next_Compat, PtrIter_Next
from collections import deque


__all__ = ['valfilter', 'valfilterfalse', 'find_next_prime', 'is_iterable',
           'is_ordered_sequence', 'SizedDict']


cdef class SizedDict:
    """Sized dictionary"""

    def __init__(self, int maxsize=1000):
        self._data = dict()
        self._maxsize = maxsize
        self._stack = deque()
        self._stack_size = 0

    @property
    def maxsize(self):
        return self._maxsize

    def __repr__(self):
        return repr(self._data)

    def __contains__(self, key):
        """Return True if d has a key key, else False."""
        cdef bint res
        res = key in self._data
        return res

    def __getitem__(self, key):
        """Return the item of d with key key. Raises a KeyError if key
        is not in the map.
        """
        return self._data[key]

    def get(self, key, default=None):
        """Return the value for key if key is in the dictionary, else default.

        If default is not given, it defaults to None, so that this method
        never raises a KeyError.
        """
        return self._data.get(key, default)

    def __len__(self):
        """Return the number of items in the dictionary d.
        """
        return self._stack_size

    def __iter__(self):
        """Return an iterator over the keys of the dictionary.

        This is a shortcut for iter(d.keys()).
        """
        return iter(self.keys())

    def __setitem__(self, key, value):
        """Set d[key] to value
        """
        if self._stack_size >= self._maxsize:
            k_pop = self._stack.popleft()
            del self._data[k_pop]
            self._stack_size = self._stack_size - 1
        self._stack.append(key)
        self._data[key] = value
        self._stack_size = self._stack_size + 1

    def __delitem__(self, key):
        """Remove d[key] from d. Raises a KeyError if key is not in the map.
        """
        del self._data[key]
        self._stack.remove(key)
        self._stack_size = self._stack_size - 1

    def keys(self):
        """Return a new view of the dictionary’s keys."""
        return self._data.keys()

    def values(self):
        """Return a new view of the dictionary’s values."""
        return self._data.values()

    def items(self):
        """Return a new view of the dictionary’s items (``(key, value)`` pairs).
        """
        return self._data.items()

    def clear(self):
        """Remove all items from the dictionary.
        """
        self._stack.clear()
        self._data.clear()
        self._stack_size = 0

    def pop(self, key, default=None):
        """If key is in the dictionary, remove it and return its value,
        else return default.

        If default is not given and key is not in the dictionary, a KeyError is raised.
        """
        cdef bint has_default

        has_default = not bool(default is None)
        if key in self._data:
            val = self._data.pop(key)
            self._stack.remove(key)
            self._stack_size = self._stack_size - 1
        elif has_default:
            val = default
        return val

    def popitem(self):
        """Remove and return a (key, value) pair from the dictionary.
        Pairs are returned in LIFO order.

        popitem() is useful to destructively iterate over a dictionary,
        as often used in set algorithms. If the dictionary is empty, calling
        popitem() raises a KeyError.
        """
        cdef object lifo_key, lifo_val
        lifo_key = self._stack.pop()
        lifo_val = self._data.pop(lifo_key)
        self._stack_size = self._stack_size - 1
        return lifo_key, lifo_val

    def update(self, other):
        """Update the dictionary with the key/value pairs from other, overwriting
        existing keys. Return None.

        update() accepts either another dictionary object or an iterable of
        key/value pairs (as tuples or other iterables of length two). If keyword
        arguments are specified, the dictionary is then updated with those
        key/value pairs: d.update(red=1, blue=2).
        """
        if not isinstance(other, dict):
            other = dict(other)
        for k, v in other.items():
            self[k] = v

    def setdefault(self, key, default=None):
        """If key is in the dictionary, return its value. If not, insert key
        with a value of default and return default. default defaults to None.
        """
        try:
            return self._data[key]
        except KeyError:
            self[key] = default
            return default


cpdef object is_iterable(object x):
    """Is x iterable?
    
    >>> is_iterable([1, 2, 3])
    True
    >>> is_iterable('abc')
    True
    >>> is_iterable(5)
    False
    """
    try:
        iter(x)
        return True
    except TypeError:
        pass
    return False


cpdef object is_ordered_sequence(object x):
    """Is x an ordered sequence? (list, tuple)
    
    >>> is_ordered_sequence([1, 2, 3])
    True
    >>> is_ordered_sequence('abc')
    False
    >>> is_ordered_sequence({4, '3', 2})
    False
    """
    if isinstance(x, list) or isinstance(x, tuple):
        return True
    return False


cdef bint _is_prime(int n):
    cdef int i

    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i != 0:
            i += 2
        else:
            return False
    return True


cpdef int find_next_prime(int N):
    """Find next prime >= N

    Parameters
    ----------
    N : int
        Starting point to find the next prime >= N.

    Returns
    -------
    int
        the next prime found after the number N
    """

    if N < 3:
        return 2
    if N % 2 == 0:
        N += 1
    for n in range(N, 2 * N, 2):
        if _is_prime(n):
            return n


cdef class _iter_mapping:
    """ Keep a handle on the current item to prevent memory clean up too early"""
    def __cinit__(self, object it):
        self.it = it
        self.cur = None

    def __iter__(self):
        return self

    def __next__(self):
        self.cur = next(self.it)
        return self.cur


cdef int PyMapping_Next(object p, Py_ssize_t *ppos, PyObject* *pkey, PyObject* *pval) except -1:
    """Mimic "PyDict_Next" interface, but for any mapping"""
    cdef PyObject *obj
    obj = PtrIter_Next(p)
    if obj is NULL:
        return 0
    pkey[0] = <PyObject*>(<object>obj)[0]
    pval[0] = <PyObject*>(<object>obj)[1]
    Py_XDECREF(obj)  # removing this results in memory leak
    return 1


cdef f_map_next get_map_iter(object d, PyObject* *ptr) except NULL:
    """Return function pointer to perform iteration over object returned in ptr.
    The returned function signature matches "PyDict_Next".  If ``d`` is a dict,
    then the returned function *is* PyDict_Next, so iteration wil be very fast.
    The object returned through ``ptr`` needs to have its reference count
    reduced by one once the caller "owns" the object.
    This function lets us control exactly how iteration should be performed
    over a given mapping.  The current rules are:
    1) If ``d`` is exactly a dict, use PyDict_Next
    2) If ``d`` is subtype of dict, use PyMapping_Next.  This lets the user
       control the order iteration, such as for ordereddict.
    3) If using PyMapping_Next, iterate using ``iteritems`` if possible,
       otherwise iterate using ``items``.
    """
    cdef object val
    cdef f_map_next rv
    if PyDict_CheckExact(d):
        val = d
        rv = &PyDict_Next_Compat
    elif hasattr(d, 'iteritems'):
        val = _iter_mapping(iter(d.iteritems()))
        rv = &PyMapping_Next
    else:
        val = _iter_mapping(iter(d.items()))
        rv = &PyMapping_Next
    Py_INCREF(val)
    ptr[0] = <PyObject*>val
    return rv


cpdef object valfilter(object predicate, object d, object factory=dict):
    """
    Filter items in dictionary by value
    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}
    See Also:
        keyfilter
        itemfilter
        valmap
    """
    cdef:
        object rv
        f_map_next f
        PyObject *obj
        PyObject *pkey
        PyObject *pval
        Py_ssize_t pos = 0

    rv = factory()
    f = get_map_iter(d, &obj)
    d = <object>obj
    Py_DECREF(d)
    while f(d, &pos, &pkey, &pval):
        if predicate(<object>pval):
            rv[<object>pkey] = <object>pval
    return rv


cpdef object valfilterfalse(object predicate, object d, object factory=dict):
    """ Filter items in dictionary by values which are false.

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilterfalse(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
    """
    cdef:
        object rv
        f_map_next f
        PyObject *obj
        PyObject *pkey
        PyObject *pval
        Py_ssize_t pos = 0

    rv = factory()
    f = get_map_iter(d, &obj)
    d = <object>obj
    Py_DECREF(d)
    while f(d, &pos, &pkey, &pval):
        if not predicate(<object>pval):
            rv[<object>pkey] = <object>pval
    return rv



