import pytest
import numpy as np

from hangar.typesystem import BytesVariableShape


class TestInvalidValues:

    @pytest.mark.parametrize('coltype', ['ndarray', np.ndarray, 32, {'foo': 'bar'}, ascii])
    def test_column_type_must_be_str(self, coltype):
        with pytest.raises(ValueError):
            BytesVariableShape(dtype=bytes, column_layout='flat', column_type=coltype)

    @pytest.mark.parametrize('collayout', ['f', 'n', None, 32, {'foo': 'bar'}, ascii])
    def test_column_layout_must_be_valid_value(self, collayout):
        with pytest.raises(ValueError):
            BytesVariableShape(dtype=bytes, column_layout=collayout)

    @pytest.mark.parametrize('backend', ['00', 24, {'31': '31'}, ('31',), ['50', ], ascii, 'None'])
    def test_variable_shape_backend_code_valid_value(self, backend):
        with pytest.raises(ValueError):
            BytesVariableShape(dtype=bytes, column_layout='flat', backend=backend)

    @pytest.mark.parametrize('opts', ['val', [], (), [('k', 'v')], 10, ({'k': 'v'},), ascii])
    def test_backend_options_must_be_dict_or_nonetype(self, opts):
        with pytest.raises(TypeError):
            BytesVariableShape(dtype=bytes, column_layout='flat', backend='31', backend_options=opts)

    def test_backend_must_be_specified_if_backend_options_provided(self):
        with pytest.raises(ValueError):
            BytesVariableShape(dtype=bytes, column_layout='flat', backend_options={})

    @pytest.mark.parametrize('schema_type', ['fixed_shape', True, 'str', np.uint8, 3, ascii])
    def test_variable_shape_must_have_variable_shape_schema_type(self, schema_type):
        with pytest.raises(ValueError):
            BytesVariableShape(dtype=bytes, column_layout='flat', schema_type=schema_type)


# ----------------------- Fixtures for Valid Schema ---------------------------


@pytest.fixture(params=['nested', 'flat'], scope='class')
def column_layout(request):
    return request.param


@pytest.fixture(params=['31'], scope='class')
def backend(request):
    return request.param


@pytest.fixture(params=[{}], scope='class')
def backend_options(request):
    return request.param


@pytest.fixture(scope='class')
def valid_schema(column_layout, backend, backend_options):
    schema = BytesVariableShape(
        dtype=bytes, column_layout=column_layout, backend=backend, backend_options=backend_options)
    return schema


class TestValidSchema:

    @pytest.mark.parametrize('data', [
        b'hello', b'world how are you?', b'\n what\'s up',
        b'loob!', b'lol',
        (b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x00\x8c\x08__main__"
         b"\x94\x8c\x07testobj\x94\x93\x94)\x81\x94}\x94\x8c\x04name\x94Nsb.")
    ])
    def test_valid_data(self, valid_schema, data):
        res = valid_schema.verify_data_compatible(data)
        assert res.compatible is True
        assert res.reason == ''

    def test_data_over_2MB_size_not_allowed(self, valid_schema):
        data = ''.join(['a' for _ in range(2_000_001)]).encode()
        res = valid_schema.verify_data_compatible(data)
        assert res.compatible is False



