import pytest
import numpy as np


from hangar.typesystem import NdarrayFixedShape, NdarrayVariableShape


class TestInvalidValues:

    @pytest.mark.parametrize('shape,expected_exc', [
        [tuple(range(32)), ValueError],
        [(1.2, 2), TypeError],
        [[1, 2], TypeError],
        ['shouldntwork', TypeError],
    ])
    def test_shape_not_tuple_of_int_less_than_32_dims(self, shape, expected_exc):
        with pytest.raises(expected_exc):
            NdarrayFixedShape(shape=shape, dtype=np.uint8, column_layout='flat')
        with pytest.raises(expected_exc):
            NdarrayVariableShape(shape=shape, dtype=np.uint8, column_layout='flat')

    @pytest.mark.parametrize(
        'coltype', ['str', str, 'notvalid', None, 32, 3.5, {'foo': 'bar'}, ascii])
    def test_column_type_must_be_ndarray(self, coltype):
        with pytest.raises(ValueError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout='flat', column_type=coltype)
        with pytest.raises(ValueError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout='flat', column_type=coltype)

    @pytest.mark.parametrize(
        'collayout', ['f', 'n', 'notvalid', None, 32, 3.5, {'foo': 'bar'}, ascii])
    def test_column_layout_must_be_valid_value(self, collayout):
        with pytest.raises(ValueError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout=collayout)
        with pytest.raises(ValueError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout=collayout)

    @pytest.mark.parametrize(
        'backend', ['30', 24, {'10': '10'}, ('00',), ['50', ], ascii, 'None'])
    def test_fixed_shape_backend_code_valid_value(self, backend):
        with pytest.raises(ValueError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend=backend)

    @pytest.mark.parametrize(
        'backend', ['30', '01', 24, {'10': '10'}, ('00',), ['50', ], ascii, 'None'])
    def test_variable_shape_backend_code_valid_value(self, backend):
        with pytest.raises(ValueError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend=backend)

    @pytest.mark.parametrize(
        'opts', ['val', [], (), [('key', 'val')], 10, ({'key': 'val'},), ascii])
    def test_backend_options_must_be_dict_or_nonetype(self, opts):
        with pytest.raises(TypeError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend='00', backend_options=opts)
        with pytest.raises(TypeError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend='00', backend_options=opts)

    def test_backend_must_be_specified_if_backend_options_provided(self):
        with pytest.raises(ValueError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend_options={})
        with pytest.raises(ValueError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout='flat', backend_options={})

    @pytest.mark.parametrize(
        'schema_type', ['fixed_shape', True, 'str', np.uint8, 3, ascii])
    def test_variable_shape_must_have_variable_shape_schema_type(self, schema_type):
        with pytest.raises(ValueError):
            NdarrayVariableShape(shape=(1,), dtype=np.uint8, column_layout='flat', schema_type=schema_type)

    @pytest.mark.parametrize(
        'schema_type', ['variable_shape', True, 'str', np.uint8, 3, ascii])
    def test_fixed_shape_must_have_fixed_shape_schema_type(self, schema_type):
        with pytest.raises(ValueError):
            NdarrayFixedShape(shape=(1,), dtype=np.uint8, column_layout='flat', schema_type=schema_type)

