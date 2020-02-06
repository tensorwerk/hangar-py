from .columntype_array import ArrayType
from .columntype_string import StringType


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
        return coltype in self._allowed_dtypes


class ColumnLayout:
    _allowed_layouts = ['flat', 'nested']

    @property
    def allowed_layouts(self):
        return self._allowed_layouts

    def specifier(self, column_layout, *args, **kwargs):
        if not self.isvalid(column_layout):
            raise ValueError(column_layout)
        return {'column_layout': column_layout}

    def isvalid(self, layout):
        return layout in self._allowed_layouts


class ColumnSpec:
    _allowed_types = ColumnType().allowed_dtypes
    _allowed_layouts = ColumnLayout().allowed_layouts

    @property
    def allowed_types(self):
        return self._allowed_types

    @property
    def allowed_layouts(self):
        return self._allowed_layouts

    def specifier(self, collayout, coltype, *args, **kwargs):
        if not self.isvalid(collayout, coltype):
            raise ValueError(collayout, coltype)

        res = {}
        res.update(ColumnLayout().specifier(column_layout=collayout, *args, **kwargs))
        res.update(ColumnType().specifier(coltype=coltype, *args, **kwargs))
        return res

    def isvalid(self, column_layout, column_type):
        return all([column_layout in self.allowed_layouts,
                    column_type in self.allowed_types])


def spec_allowed_backends(spec):
    """Generate list of allowed backends for a schema specification dict
    """
    coltype = ColumnType()
    rec_column_type = spec['column_type']
    schematype = coltype.ColumnTypeClassMap[rec_column_type]

    rec_schema_type = spec['schema_type']
    schema = schematype.SchemaNameClassMap[rec_schema_type]
    return schema.allowed_backends

