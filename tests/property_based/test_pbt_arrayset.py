import pytest
import numpy as np

from conftest import variable_shape_backend_params, fixed_shape_backend_params

import string
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from hypothesis.extra import numpy as npst

from hangar import Repository


# ------------------------ Fixture Setup ------------------------------


added_samples = set()


@pytest.fixture(params=fixed_shape_backend_params)
def fixed_shape_repo_co_float32_aset_flat(managed_tmpdir, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples = set(sample_key)
    global added_samples
    added_samples = set()
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo_obj.checkout(write=True)
    co.arraysets.init_arrayset(name='writtenaset',
                               shape=(10, 10, 10),
                               dtype=np.float32,
                               variable_shape=False,
                               backend_opts=request.param,
                               contains_subsamples=False)
    yield co
    co.close()
    repo_obj._env._close_environments()


@pytest.fixture(params=variable_shape_backend_params)
def variable_shape_repo_co_float32_aset_flat(managed_tmpdir, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples = set(sample_key)
    global added_samples
    added_samples = set()
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo_obj.checkout(write=True)
    co.arraysets.init_arrayset(name='writtenaset',
                               shape=(10, 10, 10),
                               dtype=np.float32,
                               variable_shape=True,
                               backend_opts=request.param,
                               contains_subsamples=False)
    yield co
    co.close()
    repo_obj._env._close_environments()


@pytest.fixture(params=variable_shape_backend_params)
def variable_shape_repo_co_uint8_aset_flat(managed_tmpdir, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples = set(sample_key)
    global added_samples
    added_samples = set()
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo_obj.checkout(write=True)
    co.arraysets.init_arrayset(name='writtenaset',
                               shape=(10, 10, 10),
                               dtype=np.uint8,
                               variable_shape=True,
                               backend_opts=request.param,
                               contains_subsamples=False)
    yield co
    co.close()
    repo_obj._env._close_environments()


# ----------------------------- Test Generation ------------------------------


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
def test_arrayset_fixed_key_values(key, val, fixed_shape_repo_co_float32_aset_flat):
    global added_samples

    co = fixed_shape_repo_co_float32_aset_flat
    assert co.arraysets['writtenaset'].variable_shape is False
    co.arraysets['writtenaset'][key] = val
    added_samples.add(key)
    assert len(co.arraysets['writtenaset']) == len(added_samples)

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
def test_arrayset_variable_shape_float32(key, val, variable_shape_repo_co_float32_aset_flat):
    global added_samples

    co = variable_shape_repo_co_float32_aset_flat
    assert co.arraysets['writtenaset'].variable_shape is True
    co.arraysets['writtenaset'][key] = val
    added_samples.add(key)
    assert len(co.arraysets['writtenaset']) == len(added_samples)

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
def test_arrayset_variable_shape_uint8(key, val, variable_shape_repo_co_uint8_aset_flat):
    global added_samples

    co = variable_shape_repo_co_uint8_aset_flat
    assert co.arraysets['writtenaset'].variable_shape is True
    co.arraysets['writtenaset'][key] = val
    added_samples.add(key)
    assert len(co.arraysets['writtenaset']) == len(added_samples)

    out = co.arraysets['writtenaset'][key]
    assert out.dtype == val.dtype
    assert out.shape == val.shape
    assert np.allclose(out, val)
