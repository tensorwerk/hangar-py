from ..backends import BACKEND_OPTIONS_MAP, BACKEND_CAPABILITIES_MAP
import numpy as np


class SchemaVariableShape:

    _allowed_backends = ['00', '01', '10', '50']
    _local_backends = ['00', '01', '10']
    _remote_backends = ['50']

    def __init__(self):
        self.BackendCapabilities = {
            be: BACKEND_CAPABILITIES_MAP[be]() for be in self._allowed_backends}
        self.BackendOptions = {
            be: BACKEND_OPTIONS_MAP[be]() for be in self._allowed_backends}

    @property
    def allowed_backends(self):
        return self._allowed_backends

    @property
    def local_backends(self):
        return self._local_backends

    @property
    def remote_backends(self):
        return self._remote_backends

    def specifier(self, shape, dtype, *args, **kwargs):
        if 'backend' not in kwargs:
            if 'backend_options' in kwargs:
                raise ValueError(f'options set without specifying backend.')
            else:
                backend = '00'
        else:
            backend = kwargs['backend']

        if not self.isvalid(backend, shape, dtype):
            raise ValueError(backend, shape, dtype)

        if 'backend_options' in kwargs:
            backend_options = kwargs['backend_options']
            if not self.BackendOptions[backend].isvalid(backend_options):
                raise ValueError(backend_options)
        else:
            backend_options = self.BackendOptions[backend].default

        return {'shape': tuple(shape),
                'dtype_num': np.dtype(dtype).num,
                'backend': backend,
                'backend_options': backend_options}

    def isvalid(self, backend, shape, dtype):
        if backend not in self._allowed_backends:
            return False

        if dtype not in self.BackendCapabilities[backend].allowed_dtypes:
            return False

        if not isinstance(shape, (list, tuple)):
            return False
        if not all([isinstance(i, int) for i in shape]):
            return False
        if len(shape) > 31:
            return False
        return True


class SchemaFixedShape:

    _allowed_backends = ['00', '10', '50']
    _local_backends = ['00', '10']
    _remote_backends = ['50']

    def __init__(self):
        self.BackendCapabilities = {
            be: BACKEND_CAPABILITIES_MAP[be]() for be in self._allowed_backends}
        self.BackendOptions = {
            be: BACKEND_OPTIONS_MAP[be]() for be in self._allowed_backends}

    @property
    def allowed_backends(self):
        return self._allowed_backends

    @property
    def local_backends(self):
        return self._local_backends

    @property
    def remote_backends(self):
        return self._remote_backends

    def specifier(self, shape, dtype, *args, **kwargs):
        if 'backend' not in kwargs:
            if 'backend_options' in kwargs:
                raise ValueError(f'options set without specifying backend.')
            else:
                backend = '00'
        else:
            backend = kwargs['backend']

        if not self.isvalid(backend, shape, dtype):
            raise ValueError(backend, shape, dtype)

        if 'backend_options' in kwargs:
            backend_options = kwargs['backend_options']
            if not self.BackendOptions[backend].isvalid(backend_options):
                raise ValueError(backend_options)
        else:
            backend_options = self.BackendOptions[backend].default

        return {'shape': tuple(shape),
                'dtype_num': np.dtype(dtype).num,
                'backend': backend,
                'backend_options': backend_options}

    def isvalid(self, backend, shape, dtype):
        if backend not in self._allowed_backends:
            return False

        if dtype not in self.BackendCapabilities[backend].allowed_dtypes:
            return False

        if not isinstance(shape, (list, tuple)):
            return False
        if not all([isinstance(i, int) for i in shape]):
            return False
        if len(shape) > 31:
            return False
        return True


class ArrayType:

    _coltype = 'ndarray'
    _allowed_schema = ['variable_shape', 'fixed_shape']

    def __init__(self):
        self.SchemaNameClassMap = {
            'variable_shape': SchemaVariableShape(),
            'fixed_shape': SchemaFixedShape(),
        }

    @property
    def column_dtype(self):
        return self._coltype

    @property
    def allowed_schemas(self):
        return self._allowed_schema

    def specifier(self, schema_type,  *args, **kwargs):
        if not self.isvalid(schema_type):
            raise ValueError(schema_type)

        res = {'schema_type': schema_type}
        propogator = self.SchemaNameClassMap[schema_type]
        res.update(propogator.specifier(*args, **kwargs))
        return res

    def isvalid(self, schema_type):
        return schema_type in self._allowed_schema
