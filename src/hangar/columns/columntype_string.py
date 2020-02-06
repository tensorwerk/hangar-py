from ..backends import BACKEND_OPTIONS_MAP, BACKEND_CAPABILITIES_MAP


class SchemaVariableShape:
    _allowed_backends = ['30', '50']

    def __init__(self):
        self.BackendCapabilities = {
            be: BACKEND_CAPABILITIES_MAP[be]() for be in self._allowed_backends}
        self.BackendOptions = {
            be: BACKEND_OPTIONS_MAP[be]() for be in self._allowed_backends}

    @property
    def allowed_backends(self):
        return self._allowed_backends

    def specifier(self, dtype, *args, **kwargs):
        if 'backend' in kwargs:
            backend = kwargs['backend']
        elif 'backend_options' in kwargs:
            raise ValueError(f'options set without specifying backend.')
        else:
            backend = '30'

        if not self.isvalid(backend, dtype):
            raise ValueError(backend, dtype)

        if 'backend_options' in kwargs:
            backend_options = kwargs['backend_options']
            if not self.BackendOptions[backend].isvalid(backend_options):
                raise ValueError(backend_options)
        else:
            backend_options = self.BackendOptions[backend].default

        return {'backend': backend,
                'backend_options': backend_options,
                'dtype': dtype}

    def isvalid(self, backend, dtype):
        return ((backend in self._allowed_backends) and (
                    dtype in self.BackendCapabilities[backend].allowed_dtypes))


class StringType:
    _allowed_schemas = ['variable_shape']

    def __init__(self):
        self.SchemaNameClassMap = {
            'variable_shape': SchemaVariableShape()
        }

    @property
    def allowed_schemas(self):
        return self._allowed_schemas

    def specifier(self, schema_type, *args, **kwargs):
        if not self.isvalid(schema_type):
            raise ValueError(schema_type)

        res = {'schema_type': schema_type}
        propogator = self.SchemaNameClassMap[schema_type]
        res.update(propogator.specifier(*args, **kwargs))
        return res

    def isvalid(self, schema):
        return schema in self.allowed_schemas
