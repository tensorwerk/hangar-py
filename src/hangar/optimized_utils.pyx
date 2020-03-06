
from collections import deque


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
