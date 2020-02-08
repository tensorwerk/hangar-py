from .typesystem import checkedmeta, OneOf, String
from ..records.hashmachine import schema_hash_digest
from ..backends import BACKEND_OPTIONS_MAP


@OneOf(['flat', 'nested'])
class ColumnLayout(String):
    pass


@OneOf(['str', 'ndarray'])
class ColumnDType(String):
    pass


class ColumnBase(metaclass=checkedmeta):
    _column_layout = ColumnLayout()
    _column_type = ColumnDType()

    def __init__(self, column_layout, column_type, *args, **kwargs):
        self._column_layout = column_layout
        self._column_type = column_type
        self._schema_attributes = ['_column_layout', '_column_type']
        self.__beopts = None

    @property
    def _beopts(self):
        if not self.__beopts:
            self.__beopts = BACKEND_OPTIONS_MAP[self.backend](
                backend_options=self.backend_options,
                dtype=self.dtype)
        return self.__beopts

    @property
    def column_layout(self):
        return self._column_layout

    @property
    def column_type(self):
        return self._column_type

    @property
    def _schema(self):
        schema_dict = {}
        public_attr_names = [attr.lstrip('_') for attr in self._schema_attributes]
        for attr in public_attr_names:
            schema_dict[attr] = getattr(self, f'_{attr}')
        return schema_dict

    def _schema_hash_digest(self, *, tcode='1'):
        return schema_hash_digest(self._schema, tcode=tcode)

