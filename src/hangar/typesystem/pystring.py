from .base import ColumnBase
from .descriptors import OneOf, Descriptor, String, OptionalString, OptionalDict
from ..records import CompatibleData
from ..utils import is_ascii


@OneOf(['<class\'str\'>'])
class StringDType(Descriptor):
    pass


SERIAL_DTYPE_TO_OBJ = {
    '<class\'str\'>': str,
}


@OneOf(['variable_shape'])
class StringSchemaType(String):
    pass


@OneOf(['str'])
class StrColumnType(String):
    pass


@OneOf(['2'])
class DataHasherTcode(String):
    pass


class StringSchemaBase(ColumnBase):
    _schema_type = StringSchemaType()
    _column_type = StrColumnType()
    _data_hasher_tcode = DataHasherTcode()

    def __init__(
            self,
            dtype,
            backend=None,
            backend_options=None,
            *args, **kwargs
    ):
        if 'data_hasher_tcode' not in kwargs:
            kwargs['data_hasher_tcode'] = '2'
        super().__init__(*args, **kwargs)

        if backend_options is not None and backend is None:
            raise ValueError(
                '`backend_options` cannot be set if `backend` is not also provided.')

        if not isinstance(dtype, str):
            dtype = repr(dtype).replace(' ', '')

        self._dtype = dtype
        self._backend = backend
        self._backend_options = backend_options
        self._schema_attributes.extend(
            ['_schema_type', '_dtype', '_backend', '_backend_options']
        )

    def backend_from_heuristics(self):
        self._backend = '30'

    @property
    def schema_type(self):
        return self._schema_type

    @property
    def dtype(self):
        return SERIAL_DTYPE_TO_OBJ[self._dtype]

    @property
    def backend(self):
        return self._backend

    @property
    def backend_options(self):
        return self._backend_options

    def data_hash_digest(self, data: str) -> str:
        return self._data_hasher_func(data)

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


@OneOf(['30', '50', None])
class StringVariableShapeBackends(OptionalString):
    pass


@OneOf(['variable_shape'])
class VariableShapeSchemaType(String):
    pass


class StringVariableShape(StringSchemaBase):
    _dtype = StringDType()
    _backend = StringVariableShapeBackends()
    _backend_options = OptionalDict()
    _schema_type = VariableShapeSchemaType()

    def __init__(self, *args, **kwargs):
        if 'column_type' in kwargs:
            super().__init__(*args, **kwargs)
        else:
            super().__init__(column_type='str', *args, **kwargs)

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

        if not isinstance(data, str) or not is_ascii(data):
            compatible = False
            reason = f'data {data} not valid. Must be ascii-only str'

        res = CompatibleData(compatible, reason)
        return res
