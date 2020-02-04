# memory efficient container classes for data backends specs.
# Allow for attribute access similar to named tuples.

cdef class HDF5_01_DataHashSpec:

    def __init__(self, str backend, str uid, str checksum, str dataset,
                 int dataset_idx, tuple shape):

        self.backend = backend
        self.uid = uid
        self.checksum = checksum
        self.dataset = dataset
        self.dataset_idx = dataset_idx
        self.shape = shape

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'backend="{self.backend}", '
                f'uid="{self.uid}", '
                f'checksum="{self.checksum}", '
                f'dataset="{self.dataset}", '
                f'dataset_idx={self.dataset_idx}, '
                f'shape={self.shape})')

    def __iter__(self):
        for attr in ['backend', 'uid', 'checksum', 'dataset', 'dataset_idx', 'shape']:
            yield getattr(self, attr)

    @property
    def islocal(self):
        return True


cdef class HDF5_00_DataHashSpec:

    def __init__(self, str backend, str uid, str checksum,
                 str dataset, int dataset_idx, tuple shape):

        self.backend = backend
        self.uid = uid
        self.checksum = checksum
        self.dataset = dataset
        self.dataset_idx = dataset_idx
        self.shape = shape

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'backend="{self.backend}", '
                f'uid="{self.uid}", '
                f'checksum="{self.checksum}", '
                f'dataset="{self.dataset}", '
                f'dataset_idx={self.dataset_idx}, '
                f'shape={self.shape})')

    def __iter__(self):
        for attr in ['backend', 'uid', 'checksum', 'dataset', 'dataset_idx', 'shape']:
            yield getattr(self, attr)

    @property
    def islocal(self):
        return True


cdef class NUMPY_10_DataHashSpec:

    def __init__(self, str backend, str uid, str checksum,
                 int collection_idx, tuple shape):

        self.backend = backend
        self.uid = uid
        self.checksum = checksum
        self.collection_idx = collection_idx
        self.shape = shape

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'backend="{self.backend}", '
                f'uid="{self.uid}", '
                f'checksum="{self.checksum}", '
                f'collection_idx={self.collection_idx}, '
                f'shape={self.shape})')

    def __iter__(self):
        for attr in ['backend', 'uid', 'checksum', 'collection_idx', 'shape']:
            yield getattr(self, attr)

    @property
    def islocal(self):
        return True


cdef class LMDB_30_DataHashSpec:

    def __init__(self, str backend, str uid, str row_idx, str checksum):

        self.backend = backend
        self.uid = uid
        self.row_idx = row_idx
        self.checksum = checksum


    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'backend="{self.backend}", '
                f'uid="{self.uid}", '
                f'row_idx={self.row_idx}, '
                f'checksum="{self.checksum}")')

    def __iter__(self):
        for attr in ['backend', 'uid', 'row_idx', 'checksum']:
            yield getattr(self, attr)

    @property
    def islocal(self):
        return True


cdef class REMOTE_50_DataHashSpec:

    def __init__(self, str backend, str schema_hash):

        self.backend = backend
        self.schema_hash = schema_hash

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'backend="{self.backend}", '
                f'schema_hash="{self.schema_hash}")')

    def __iter__(self):
        for attr in ['backend', 'schema_hash']:
            yield getattr(self, attr)

    @property
    def islocal(self):
        return False
