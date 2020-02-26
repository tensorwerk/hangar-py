from collections import deque


cdef class SizedDict(dict):
    """Sized dictionary"""

    def __init__(self, int maxsize=1000):
        dict.__init__(self)
        self._maxsize = maxsize
        self._stack = deque()

    def __setitem__(self, name, value):
        cdef int size_stack = len(self._stack)

        if size_stack >= self._maxsize:
            k_pop = self._stack.popleft()
            dict.__delitem__(self, k_pop)
        self._stack.append(name)
        dict.__setitem__(self, name, value)

    def __delitem__(self, name):
        cdef int idx = 0
        cdef int size_stack = len(self._stack)

        dict.__delitem__(self, name)
        for idx in range(size_stack):
            key = self._stack[idx]
            if key == name:
                break
        del self._stack[idx]

    def update(self, other):
        if isinstance(other, dict):
            for k, v in other.items():
                self.__setitem__(k, v)
        else:
            for k, v in other:
                self.__setitem__(k, v)

    def setdefault(self, name, default=None):
        try:
            return self.__getitem__(name)
        except KeyError:
            self.__setitem__(name, default)
            return default
