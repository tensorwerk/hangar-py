import pytest
import numpy as np


from hangar.typesystem import StringVariableShape


class TestInvalidValues:

    @pytest.mark.parametrize('coltype', ['ndarray', np.ndarray, 32, {'foo': 'bar'}, ascii])
    def test_column_type_must_be_str(self, coltype):
        with pytest.raises(ValueError):
            StringVariableShape(dtype=str, column_layout='flat', column_type=coltype)

    @pytest.mark.parametrize('collayout', ['f', 'n', None, 32, {'foo': 'bar'}, ascii])
    def test_column_layout_must_be_valid_value(self, collayout):
        with pytest.raises(ValueError):
            StringVariableShape(dtype=str, column_layout=collayout)

    @pytest.mark.parametrize('backend', ['00', 24, {'30': '30'}, ('30',), ['50',], ascii, 'None'])
    def test_variable_shape_backend_code_valid_value(self, backend):
        with pytest.raises(ValueError):
            StringVariableShape(dtype=str, column_layout='flat', backend=backend)

    @pytest.mark.parametrize('opts', ['val', [], (), [('k', 'v')], 10, ({'k': 'v'},), ascii])
    def test_backend_options_must_be_dict_or_nonetype(self, opts):
        with pytest.raises(TypeError):
            StringVariableShape(dtype=str, column_layout='flat', backend='30', backend_options=opts)

    def test_backend_must_be_specified_if_backend_options_provided(self):
        with pytest.raises(ValueError):
            StringVariableShape(dtype=str, column_layout='flat', backend_options={})

    @pytest.mark.parametrize('schema_type', ['fixed_shape', True, 'str', np.uint8, 3, ascii])
    def test_variable_shape_must_have_variable_shape_schema_type(self, schema_type):
        with pytest.raises(ValueError):
            StringVariableShape(dtype=str, column_layout='flat', schema_type=schema_type)


