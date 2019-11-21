import pytest
import numpy as np

import string

from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from hypothesis.extra import numpy as npst


st_valid_names = st.text(
    min_size=1, max_size=16, alphabet=string.ascii_letters + string.digits + '_-.')
st_valid_ints = st.integers(min_value=0, max_value=999_999)
st_valid_keys = st.one_of(st_valid_ints, st_valid_names)


valid_arrays_fixed = npst.arrays(np.float32,
                                 shape=(10, 10, 10),
                                 fill=st.floats(min_value=-10,
                                                max_value=10,
                                                allow_nan=False,
                                                allow_infinity=False,
                                                width=32),
                                 elements=st.floats(min_value=-10,
                                                    max_value=10,
                                                    allow_nan=False,
                                                    allow_infinity=False,
                                                    width=32))


@given(key=st_valid_keys, val=valid_arrays_fixed)
@settings(max_examples=100, deadline=200.0, suppress_health_check=[HealthCheck.too_slow])
def test_arrayset_fixed_key_values(key, val, fixed_shape_repo_co_float32):
    co = fixed_shape_repo_co_float32
    assert co.arraysets['writtenaset'].variable_shape is False
    co.arraysets['writtenaset'][key] = val
    out = co.arraysets['writtenaset'][key]
    assert out.dtype == val.dtype
    assert out.shape == val.shape
    assert np.allclose(out, val)


valid_shapes_var = npst.array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=10)
valid_arrays_var_float32 = npst.arrays(np.float32,
                                       shape=valid_shapes_var,
                                       fill=st.floats(min_value=-10,
                                                      max_value=10,
                                                      allow_nan=False,
                                                      allow_infinity=False,
                                                      width=32),
                                       elements=st.floats(min_value=-10,
                                                          max_value=10,
                                                          allow_nan=False,
                                                          allow_infinity=False,
                                                          width=32))


@given(key=st_valid_keys, val=valid_arrays_var_float32)
@settings(max_examples=100, deadline=200.0, suppress_health_check=[HealthCheck.too_slow])
def test_arrayset_variable_shape_float32(key, val, variable_shape_repo_co_float32):
    co = variable_shape_repo_co_float32
    assert co.arraysets['writtenaset'].variable_shape is True
    co.arraysets['writtenaset'][key] = val
    out = co.arraysets['writtenaset'][key]
    assert out.dtype == val.dtype
    assert out.shape == val.shape
    assert np.allclose(out, val)


valid_arrays_var_uint8 = npst.arrays(np.uint8,
                                     shape=valid_shapes_var,
                                     elements=st.integers(min_value=0, max_value=255),
                                     fill=st.integers(min_value=0, max_value=255))


@given(key=st_valid_keys, val=valid_arrays_var_uint8)
@settings(max_examples=100, deadline=200.0, suppress_health_check=[HealthCheck.too_slow])
def test_arrayset_variable_shape_uint8(key, val, variable_shape_repo_co_uint8):
    co = variable_shape_repo_co_uint8
    assert co.arraysets['writtenaset'].variable_shape is True
    co.arraysets['writtenaset'][key] = val
    out = co.arraysets['writtenaset'][key]
    assert out.dtype == val.dtype
    assert out.shape == val.shape
    assert np.allclose(out, val)