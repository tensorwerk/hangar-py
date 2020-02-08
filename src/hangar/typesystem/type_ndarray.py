import numpy as np

from ..columns.column_parsers import CompatibleData
from ..records.hashmachine import array_hash_digest
from .type_column import ColumnBase
from .typesystem import OneOf, String, OptionalString, SizedIntegerTuple, OptionalDict


@OneOf(['variable_shape', 'fixed_shape'])
class NdarraySchemaType(String):
    pass


class NdarraySchemaBase(ColumnBase):
    _schema_type = NdarraySchemaType()

    def __init__(self, shape, dtype, backend=None, backend_options=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._shape = shape
        if not isinstance(dtype, str):
            dtype = np.dtype(dtype).name
        self._dtype = dtype
        self._backend = backend
        self._backend_options = backend_options
        self._schema_attributes.extend(
            ['_schema_type', '_shape', '_dtype', '_backend', '_backend_options'])

    def backend_from_heuristics(self):
        # uncompressed numpy memmap data is most appropriate for data whose shape is
        # likely small tabular row data (CSV or such...)
        if (len(self._shape) == 1) and (self._shape[0] < 400):
            backend = '10'
        # hdf5 is the default backend for larger array sizes.
        elif (len(self._shape) == 1) and (self._shape[0] <= 10_000_000):
            backend = '00'
        # on fixed arrays sized arrays apply optimizations.
        elif self._schema_type == 'fixed_shape':
            backend = '01'
        else:
            backend = '00'
        self._backend = backend

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    @property
    def backend(self):
        return self._backend

    @property
    def backend_options(self):
        return self._backend_options

    def data_hash_digest(self, data: np.ndarray, *, tcode='0') -> str:
        return array_hash_digest(data, tcode=tcode)

    def change_backend(self, backend, backend_options=None):
        old_backend = self._backend
        old_backend_options = self._backend_options
        try:
            self._backend = backend
            self._backend_options = backend_options
            # del and reset beopts object to reverify input correctness.
            del self._beopts
            self._backend_options = self._beopts.backend_options
        except (TypeError, ValueError) as e:
            del self._beopts
            self._backend = old_backend
            self._backend_options = old_backend_options
            self._backend_options = self._beopts.backend_options
            raise e from None


@OneOf(['00', '01', '10', '50', None])
class NdarrayFixedShapeBackends(OptionalString):
    pass


class NdarrayFixedShape(NdarraySchemaBase):
    _shape = SizedIntegerTuple(size=31)
    _dtype = String()
    _backend = NdarrayFixedShapeBackends()
    _backend_options = OptionalDict()

    def __init__(self, *args, **kwargs):
        if 'column_type' in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(column_type='ndarray', *args, **kwargs)

        if 'schema_type' in kwargs:
            self._schema_type = kwargs['schema_type']
        else:
            self._schema_type = 'fixed_shape'

        if self.backend is None:
            self.backend_from_heuristics()
        self._backend_options = self._beopts.backend_options

    def verify_data_compatible(self, data):
        compatible = True
        reason = ''

        if not isinstance(data, np.ndarray):
            compatible = False
            reason = f'`data` argument type: {type(data)} != `np.ndarray`'
        elif data.dtype != self._dtype:
            compatible = False
            reason = f'dtype: {data.dtype.name} != aset: {self._dtype}.'
        elif not data.flags.c_contiguous:
            compatible = False
            reason = f'`data` must be "C" contiguous array.'
        elif data.shape != self._shape:
            compatible = False
            reason = f'data shape {data.shape} != fixed schema {self._shape}'

        res = CompatibleData(compatible, reason)
        return res


@OneOf(['00', '10', '50', None])
class NdarrayVariableShapeBackends(OptionalString):
    pass


class NdarrayVariableShape(NdarraySchemaBase):
    _shape = SizedIntegerTuple(size=31)
    _dtype = String()
    _backend = NdarrayVariableShapeBackends()
    _backend_options = OptionalDict()

    def __init__(self, *args, **kwargs):
        if 'column_type' in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(column_type='ndarray', *args, **kwargs)

        if 'schema_type' in kwargs:
            self._schema_type = kwargs['schema_type']
        else:
            self._schema_type = 'variable_shape'

        if self.backend is None:
            self.backend_from_heuristics()
        self._backend_options = self._beopts.backend_options

    def verify_data_compatible(self, data):
        compatible = True
        reason = ''

        if not isinstance(data, np.ndarray):
            compatible = False
            reason = f'`data` argument type: {type(data)} != `np.ndarray`'
        elif data.dtype != self._dtype:
            compatible = False
            reason = f'dtype: {data.dtype.name} != aset: {self._dtype}.'
        elif not data.flags.c_contiguous:
            compatible = False
            reason = f'`data` must be "C" contiguous array.'
        elif data.ndim != len(self._shape):
            compatible = False
            reason = f'data rank {data.ndim} != aset rank {len(self._shape)}'
        elif not all([(dim > maxdim) for dim, maxdim in zip(data.shape, self._shape)]):
            compatible = False
            reason = f'shape {data.shape} exceeds schema max {self._shape}'

        res = CompatibleData(compatible, reason)
        return res
