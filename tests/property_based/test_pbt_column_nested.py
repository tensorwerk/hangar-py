from collections import defaultdict

import pytest
import numpy as np

from conftest import (
    variable_shape_backend_params,
    fixed_shape_backend_params,
    str_variable_shape_backend_params
)

import string
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from hypothesis.extra import numpy as npst

from hangar import Repository


# ------------------------ Fixture Setup ------------------------------


added_samples_subsamples = defaultdict(set)


@pytest.fixture(params=fixed_shape_backend_params, scope='class')
def fixed_shape_repo_co_float32_aset_nested(classrepo, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples_subsamples[sample_key] = set(subsample_keys)
    global added_samples_subsamples
    added_samples_subsamples = defaultdict(set)
    co = classrepo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset',
                          shape=(5, 5, 5),
                          dtype=np.float32,
                          variable_shape=False,
                          contains_subsamples=True,
                          backend=request.param)
    yield co
    co.reset_staging_area()
    co.close()


@pytest.fixture(params=variable_shape_backend_params, scope='class')
def variable_shape_repo_co_float32_aset_nested(classrepo, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples_subsamples[sample_key] = set(subsample_keys)
    global added_samples_subsamples
    added_samples_subsamples = defaultdict(set)
    co = classrepo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset',
                          shape=(5, 5, 5),
                          dtype=np.float32,
                          variable_shape=True,
                          contains_subsamples=True,
                          backend=request.param)
    yield co
    co.reset_staging_area()
    co.close()


@pytest.fixture(params=variable_shape_backend_params, scope='class')
def variable_shape_repo_co_uint8_aset_nested(classrepo, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples_subsamples[sample_key] = set(subsample_keys)
    global added_samples_subsamples
    added_samples_subsamples = defaultdict(set)
    co = classrepo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset',
                          shape=(5, 5, 5),
                          dtype=np.uint8,
                          variable_shape=True,
                          contains_subsamples=True,
                          backend=request.param)
    yield co
    co.reset_staging_area()
    co.close()


@pytest.fixture(params=str_variable_shape_backend_params, scope='class')
def variable_shape_repo_co_str_aset_nested(classrepo, request) -> Repository:
    # needed because fixtures don't reset between each hypothesis run
    # tracks added_samples = set(sample_key)
    global added_samples_subsamples
    added_samples_subsamples = defaultdict(set)
    co = classrepo.checkout(write=True)
    co.add_str_column(name='strcolumn',
                      contains_subsamples=True,
                      backend=request.param)
    yield co
    co.reset_staging_area()
    co.close()


# -------------------------- Test Generation ---------------------------------
# Test cases are encapsulated in a classes (and fixture functions are set to
# "class" level scope in order to handle a warning introduced in hypothesis
# version 5.6.0 - 2020-02-29
#
# From release notes:
# > This release adds an explicit warning for tests that are both decorated with @given(...)
#   and request a function-scoped pytest fixture, because such fixtures are only executed once
#   for all Hypothesis test cases and that often causes trouble. See issue #377
#   (https://github.com/HypothesisWorks/hypothesis/issues/377)
#
# However, this is actually the intended behavior for hangar, since we ant to reuse
# the same repo/checkout across all of the test case inputs that hypothesis generates


st_valid_names = st.text(
    min_size=1, max_size=16, alphabet=string.ascii_letters + string.digits + '_-.')
st_valid_ints = st.integers(min_value=0, max_value=999_999)
st_valid_keys = st.one_of(st_valid_ints, st_valid_names)

valid_arrays_fixed = npst.arrays(np.float32,
                                 shape=(5, 5, 5),
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


class TestColumn1:

    @given(key=st_valid_keys, subkey=st_valid_keys, val=valid_arrays_fixed)
    @settings(max_examples=200, deadline=None)
    def test_arrayset_fixed_key_values_nested(self, key, subkey, val, fixed_shape_repo_co_float32_aset_nested):
        global added_samples_subsamples

        added_samples_subsamples[key].add(subkey)

        co = fixed_shape_repo_co_float32_aset_nested
        col = co.columns['writtenaset']
        assert col.schema_type == 'fixed_shape'
        assert col.contains_subsamples is True
        col[key] = {subkey: val}
        out = col[key][subkey]

        assert len(col) == len(added_samples_subsamples)
        assert len(col[key]) == len(added_samples_subsamples[key])
        assert out.dtype == val.dtype
        assert out.shape == val.shape
        assert np.allclose(out, val)


valid_shapes_var = npst.array_shapes(min_dims=3, max_dims=3, min_side=1, max_side=5)
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


class TestColumn2:

    @given(key=st_valid_keys, subkey=st_valid_keys, val=valid_arrays_var_float32)
    @settings(max_examples=200, deadline=None)
    def test_arrayset_variable_shape_float32_nested(self, key, val, subkey, variable_shape_repo_co_float32_aset_nested):
        global added_samples_subsamples

        co = variable_shape_repo_co_float32_aset_nested
        col = co.columns['writtenaset']
        assert col.schema_type == 'variable_shape'
        assert col.contains_subsamples is True
        col[key] = {subkey: val}
        out = col[key][subkey]
        added_samples_subsamples[key].add(subkey)

        assert len(col) == len(added_samples_subsamples)
        assert len(col[key]) == len(added_samples_subsamples[key])
        assert out.dtype == val.dtype
        assert out.shape == val.shape
        assert np.allclose(out, val)


valid_arrays_var_uint8 = npst.arrays(np.uint8,
                                     shape=valid_shapes_var,
                                     elements=st.integers(min_value=0, max_value=255),
                                     fill=st.integers(min_value=0, max_value=255))


class TestColumn3:

    @given(key=st_valid_keys, subkey=st_valid_keys, val=valid_arrays_var_uint8)
    @settings(max_examples=200, deadline=None)
    def test_arrayset_variable_shape_uint8_nested(self, key, val, subkey, variable_shape_repo_co_uint8_aset_nested):
        global added_samples_subsamples

        co = variable_shape_repo_co_uint8_aset_nested
        col = co.columns['writtenaset']
        assert col.schema_type == 'variable_shape'
        assert col.contains_subsamples is True
        col[key] = {subkey: val}
        out = col[key][subkey]
        added_samples_subsamples[key].add(subkey)

        assert len(col) == len(added_samples_subsamples)
        assert len(col[key]) == len(added_samples_subsamples[key])
        assert out.dtype == val.dtype
        assert out.shape == val.shape
        assert np.allclose(out, val)


ascii_characters = st.characters(min_codepoint=0, max_codepoint=127)
ascii_text_stratagy = st.text(alphabet=ascii_characters, min_size=0, max_size=500)


class TestStrColumn:

    @given(key=st_valid_keys, subkey=st_valid_keys, val=ascii_text_stratagy)
    @settings(max_examples=200, deadline=None)
    def test_str_column_variable_shape_nested(self, key, subkey, val, variable_shape_repo_co_str_aset_nested):
        global added_samples_subsamples

        co = variable_shape_repo_co_str_aset_nested
        col = co.columns['strcolumn']
        assert col.schema_type == 'variable_shape'
        assert col.contains_subsamples is True

        col[key] = {subkey: val}
        out = col[key][subkey]
        added_samples_subsamples[key].add(subkey)

        assert len(col) == len(added_samples_subsamples)
        assert len(col[key]) == len(added_samples_subsamples[key])
        assert out == val
