from ..backends import BACKEND_OPTIONS_MAP, BACKEND_CAPABILITIES_MAP
from ..utils import valfilter


class SchemaVariableShape:

    _allowed_backends = ['30', '50']
    _local_backends = ['30']
    _remote_backends = ['50']

    def __init__(self):
        pass

    @property
    def allowed_backends(self):
        return self._allowed_backends

    @property
    def local_backends(self):
        return self._local_backends

    @property
    def remote_backends(self):
        return self._remote_backends

    def isvalid(self, backend, options):
        if backend not in self._allowed_backends:
            return False

        if not isinstance(options, dict):
            return False

        for opt, val in options.items():
            if opt not in self.fields:
                return False
            elif val not in self._permitted_values[opt]:
                return False

        for field in self.required_fields:
            if field not in options:
                return False

        return True

    def specifier(self, backend, options):
        if not self.isvalid(backend, options):
            raise ValueError(backend, options)
        return {
            'backend': backend,
            **options
        }


SchemaNameClassMap = {
    'variable_shape': SchemaVariableShape,
}


class StringType:

    _coltype = 'str'
    _allowed_schema = ['variable_shape']

    def __init__(self):
        pass

    @property
    def column_dtype(self):
        return self._coltype

    @property
    def allowed_schema(self):
        return self._allowed_schema

    @property
    def default(self):
        return {'schema': 'variable_shape'}

    def specifier(self, schema):
        if not self.isvalid(schema):
            raise ValueError(schema)
        return {'schema': schema}

    def isvalid(self, schema):
        return schema in self.allowed_schema
