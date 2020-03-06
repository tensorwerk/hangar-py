from .descriptors import OneOf, String, checkedmeta
from ..records import hash_func_from_tcode


@OneOf(['flat', 'nested'])
class ColumnLayout(String):
    pass


@OneOf(['str', 'ndarray'])
class ColumnDType(String):
    pass


@OneOf(['1'])
class SchemaHasherTcode(String):
    pass


class ColumnBase(metaclass=checkedmeta):
    _column_layout = ColumnLayout()
    _column_type = ColumnDType()
    _schema_hasher_tcode = SchemaHasherTcode()

    def __init__(
            self,
            column_layout,
            column_type,
            data_hasher_tcode,
            schema_hasher_tcode=None,
            *args, **kwargs
    ):
        if schema_hasher_tcode is None:
            schema_hasher_tcode = '1'

        self._column_layout = column_layout
        self._column_type = column_type
        self._schema_hasher_tcode = schema_hasher_tcode
        self._data_hasher_tcode = data_hasher_tcode
        self._schema_attributes = [
            '_column_layout',
            '_column_type',
            '_schema_hasher_tcode',
            '_data_hasher_tcode',
        ]
        self._schema_hasher_func = hash_func_from_tcode(self._schema_hasher_tcode)
        self._data_hasher_func = hash_func_from_tcode(self._data_hasher_tcode)
        self._hidden_be_opts = None

    @property
    def _beopts(self):
        from ..backends import BACKEND_OPTIONS_MAP
        if self._hidden_be_opts is None:
            self._hidden_be_opts = BACKEND_OPTIONS_MAP[self.backend](
                backend_options=self.backend_options,
                dtype=self.dtype,
                shape=(self.shape if hasattr(self, '_shape') else None))
        return self._hidden_be_opts

    @_beopts.deleter
    def _beopts(self):
        self._hidden_be_opts = None

    @_beopts.setter
    def _beopts(self, backend_options):
        from ..backends import BACKEND_OPTIONS_MAP
        self._hidden_be_opts = BACKEND_OPTIONS_MAP[self.backend](
            backend_options=backend_options,
            dtype=self.dtype,
            shape=(self.shape if hasattr(self, '_shape') else None))

    @property
    def column_layout(self):
        return self._column_layout

    @property
    def column_type(self):
        return self._column_type

    @property
    def schema_hasher_tcode(self):
        return self._schema_hasher_tcode

    @property
    def schema(self):
        schema_dict = {}
        public_attr_names = [attr.lstrip('_') for attr in self._schema_attributes]
        for attr in public_attr_names:
            schema_dict[attr] = getattr(self, f'_{attr}')
        return schema_dict

    def schema_hash_digest(self):
        return self._schema_hasher_func(self.schema)

    def backend_from_heuristics(self, *args, **kwargs):
        raise NotImplementedError

    def verify_data_compatible(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def data_hasher_tcode(self):
        return self._data_hasher_tcode

    def data_hash_digest(self, *args, **kwargs):
        raise NotImplementedError

