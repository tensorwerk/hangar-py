
cdef class CompatibleData:
    """Bool recording if data `compatible` and if False the rejection `reason`.
    """

    def __init__(self, bint compatible, str reason):
        self.compatible = compatible
        self.reason = reason

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'compatible={self.compatible}, '
                f'reason="{self.reason}")')

    def __iter__(self):
        for attr in ['compatible', 'reason']:
            yield getattr(self, attr)



cdef class ColumnSchemaKey:
    """Record listing `column` name and `layout` type.
    """

    def __init__(self, str column, str layout):
        self.column = column
        self.layout = layout

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'column={self.column}, '
                f'reason="{self.layout}")')

    def __iter__(self):
        for attr in ['column', 'layout']:
            yield getattr(self, attr)



cdef class FlatColumnDataKey:
    """Record listing `column` & `sample` name along with `layout` property
    """

    def __init__(self, str column, str sample):
        self.column = column
        self._sample = sample
        self._s_int = True if sample[0] == '#' else False

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'column={self.column}, '
                f'sample={f"{self.sample if self._s_int else repr(self.sample)}"})')

    def __iter__(self):
        for attr in ['column', 'sample']:
            yield getattr(self, attr)

    @property
    def sample(self):
        if self._s_int:
            return int(self._sample[1:])
        else:
            return self._sample

    @property
    def layout(self):
        return 'flat'



cdef class NestedColumnDataKey:
    """Record listing `column`, `sample`, & `subsample` name along with `layout` property
    """

    def __init__(self, str column, str sample, str subsample):
        self.column = column
        self._sample = sample
        self._subsample = subsample
        self._s_int = True if sample[0] == '#' else False
        self._ss_int = True if subsample[0] == '#' else False

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'column={self.column}, '
                f'sample={f"{self.sample if self._s_int else repr(self.sample)}"}, '
                f'subsample={f"{self.subsample if self._ss_int else repr(self.subsample)}"})')

    def __iter__(self):
        for attr in ['column', 'sample', 'subsample']:
            yield getattr(self, attr)

    @property
    def sample(self):
        if self._s_int:
            return int(self._sample[1:])
        else:
            return self._sample

    @property
    def subsample(self):
        if self._ss_int:
            return int(self._subsample[1:])
        else:
            return self._subsample

    @property
    def layout(self):
        return 'nested'


cdef class DataRecordVal:

    def __init__(self, str digest):
        self.digest = digest

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'digest={repr(self.digest)})')

    def __iter__(self):
        for attr in ['digest']:
            yield getattr(self, attr)


cdef class MetadataRecordKey:
    """Record listing `key` of metadata
    """

    def __init__(self, str key):
        self._key = key
        self._k_int = True if key[0] == '#' else False

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'key={f"{self.key if self._k_int else repr(self.key)}"})')

    def __iter__(self):
        for attr in ['key']:
            yield getattr(self, attr)

    @property
    def key(self):
        if self._k_int:
            return int(self._key[1:])
        else:
            return self._key
