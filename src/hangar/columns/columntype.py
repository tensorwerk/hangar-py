from .columntype_string import StringType
from .columntype_array import ArrayType

class ColumnType:

    _allowed_dtypes = ['str', 'ndarray']

    def __init__(self):
        self.ColumnTypeClassMap = {
            'str': StringType(),
            'ndarray': ArrayType(),
        }

    @property
    def allowed_dtypes(self):
        return self._allowed_dtypes

    def specifier(self, coltype, *args, **kwargs):
        if not self.isvalid(coltype):
            raise ValueError(coltype)

        res = {'column_type': coltype}
        propogator = self.ColumnTypeClassMap[coltype]
        res.update(propogator.specifier(*args, **kwargs))
        return res

    def isvalid(self, coltype):
        return coltype in self.allowed_dtypes


