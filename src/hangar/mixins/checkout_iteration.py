

class CheckoutDictIteration:
    """Mixin class for checkout objects which mock common iter methods

    Methods
    -------
    __len__
    __contains__
    __iter__
    keys
    values
    items
    """

    def __len__(self):
        """Returns number of columns in the checkout.
        """
        self._verify_alive()
        return len(self.columns)

    def __contains__(self, key):
        """Determine if some column name (key) exists in the checkout.
        """
        self._verify_alive()
        return bool(key in self.columns)

    def __iter__(self):
        """Iterate over column keys"""
        self._verify_alive()
        return iter(self.columns)

    def keys(self):
        """Generator yielding the name (key) of every column
        """
        self._verify_alive()
        yield from self.columns.keys()

    def values(self):
        """Generator yielding accessor object of every column
        """
        self._verify_alive()
        yield from self.columns.values()

    def items(self):
        """Generator yielding tuple of (name, accessor object) of every column
        """
        self._verify_alive()
        yield from self.columns.items()
