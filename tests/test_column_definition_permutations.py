from collections import defaultdict
from functools import partial
import secrets
import string

import pytest
import numpy as np


def assert_equal(expected, actual):
    if isinstance(expected, str):
        assert expected == actual
    elif isinstance(expected, np.ndarray):
        assert np.allclose(expected, actual)
        assert expected.dtype == actual.dtype
    else:
        raise TypeError(f'unknown type of data {type(expected)}')


def ndarray_generate_data_fixed_shape(shape, dtype, low=0, high=255):
    arr = np.random.randint(low, high, size=shape, dtype=dtype)
    return arr


def ndarray_generate_data_variable_shape(shape, dtype, low=0, high=255):
    arr_dims = []
    for dim in shape:
        valid_dim_shapes = [i for i in range(1, dim + 1)]
        dimsize = secrets.choice(valid_dim_shapes)
        arr_dims.append(dimsize)
    arr_dims = tuple(arr_dims)
    arr = np.random.randint(low, high, size=arr_dims, dtype=dtype)
    return arr


def str_generate_data_variable_shape(
        length=20, *, _ALPHABET=''.join([string.ascii_letters, string.digits, string.punctuation, ' '])
):
    tokens = [secrets.choice(_ALPHABET) for i in range(length)]
    res = ''.join(tokens)
    return res


column_settings = {
    'ndarray': {
        'fixed_shape': ['00', '01', '10'],
        'variable_shape': ['00', '10'],
    },
    'str': {
        'variable_shape': ['30']
    }
}


column_data_generators = {
    'ndarray': {
        'fixed_shape': ndarray_generate_data_fixed_shape,
        'variable_shape': ndarray_generate_data_variable_shape,
    },
    'str': {
        'variable_shape': str_generate_data_variable_shape
    }
}


column_layouts = {
    'ndarray': ['flat', 'nested'],
    'str': ['flat', 'nested']
}


def add_data_to_column(col, data_gen, nsamples, nsubsamples=None):
    column_data = {}
    for samp in range(nsamples):
        if nsubsamples is None:
            data = data_gen()
            column_data[samp] = data
            col[samp] = data
        else:
            column_data[samp] = {}
            for subsamp in range(nsubsamples):
                data = data_gen()
                column_data[samp][subsamp] = data
            col[samp] = column_data[samp]
    return column_data


@pytest.fixture(params=[1, 3])
def num_samples_gen(request):
    return request.param


@pytest.fixture(params=[1, 3])
def num_subsamples_gen(request):
    return request.param


@pytest.fixture()
def column_permutation_repo(repo, num_samples_gen, num_subsamples_gen):
    co = repo.checkout(write=True)
    nsamp = num_samples_gen
    nsubs = num_subsamples_gen

    column_name_partials = {}
    column_data_copy = defaultdict(dict)
    shape = (4, 4)
    dtype = np.uint8
    for col_dtype, schema_settings in column_settings.items():
        for layout in column_layouts[col_dtype]:
            for schema_type, valid_backends in schema_settings.items():
                for backend in valid_backends:
                    name = f'{col_dtype}_{layout}_{schema_type}_{backend}'
                    generator = column_data_generators[col_dtype][schema_type]
                    has_subs = False if layout == 'flat' else True
                    is_var = False if schema_type == 'fixed_shape' else True

                    if col_dtype == 'ndarray':
                        col = co.add_ndarray_column(name,
                                                       shape=shape,
                                                       dtype=dtype,
                                                       variable_shape=is_var,
                                                       contains_subsamples=has_subs,
                                                       backend=backend)
                        data_partial = partial(generator, shape, dtype)
                        if layout == 'flat':
                            column_data_copy[name] = add_data_to_column(col, data_partial, nsamp)
                        elif layout == 'nested':
                            column_data_copy[name] = add_data_to_column(col, data_partial, nsamp, nsubs)
                        else:
                            raise ValueError(f'invalid layout {layout}')
                    elif col_dtype == 'str':
                        col = co.add_str_column(name, contains_subsamples=has_subs, backend=backend)
                        data_partial = partial(generator)
                        if layout == 'flat':
                            column_data_copy[name] = add_data_to_column(col, data_partial, nsamp)
                        elif layout == 'nested':
                            column_data_copy[name] = add_data_to_column(col, data_partial, nsamp, nsubs)
                    else:
                        raise ValueError(f'column dtype {col_dtype} invalid')

                    column_name_partials[name] = data_partial

    co.commit('first')
    co.close()
    yield repo, column_data_copy, column_name_partials


@pytest.fixture(params=[True, False])
def column_permutations_read_write_checkout(request, column_permutation_repo):
    repo, column_data, column_data_partials = column_permutation_repo
    co = repo.checkout(write=request.param)
    yield co, column_data, column_data_partials
    co.close()


@pytest.fixture()
def column_permutations_write_checkout(column_permutation_repo):
    repo, column_data, column_data_partials = column_permutation_repo
    co = repo.checkout(write=True)
    yield co, column_data, column_data_partials
    co.close()


def test_read_data_from_column_permutations(column_permutations_read_write_checkout):
    co, column_data, column_data_partials = column_permutations_read_write_checkout

    assert len(co.columns) == len(column_data)
    for column_name, column_samples in column_data.items():
        assert column_name in co.columns
        col = co[column_name]
        assert len(column_samples) == len(col)

        for sample_key, sample_value in column_samples.items():
            if not isinstance(sample_value, dict):
                recorded = col[sample_key]
                assert_equal(sample_value, recorded)
            else:
                col_samp = col[sample_key]
                assert len(sample_value) == len(col_samp)
                for subsample_key, subsample_value in sample_value.items():
                    recorded = col_samp[subsample_key]
                    assert_equal(subsample_value, recorded)


def test_write_data_to_column_permutations(
        column_permutations_write_checkout, num_samples_gen, num_subsamples_gen
):
    co, column_data, column_data_partials = column_permutations_write_checkout

    for column_name in co.columns:
        col = co[column_name]
        data_gen = column_data_partials[column_name]
        if col.column_layout == 'flat':
            for samp in range(num_samples_gen):
                data = data_gen()
                column_data[column_name][str(samp)] = data
                col[str(samp)] = data
        elif col.column_layout == 'nested':
            for samp in range(num_samples_gen):
                column_data[column_name][str(samp)] = {}
                for ssamp in range(num_subsamples_gen):
                    data = data_gen()
                    column_data[column_name][str(samp)][str(ssamp)] = data
                col[str(samp)] = column_data[column_name][str(samp)]
        else:
            raise ValueError(f'unknown layout option {col.column_layout}')

    assert len(co.columns) == len(column_data)
    for column_name, column_samples in column_data.items():
        assert column_name in co.columns
        col = co[column_name]
        assert len(column_samples) == len(col)

        for sample_key, sample_value in column_samples.items():
            if not isinstance(sample_value, dict):
                recorded = col[sample_key]
                assert_equal(sample_value, recorded)
            else:
                col_samp = col[sample_key]
                assert len(sample_value) == len(col_samp)
                for subsample_key, subsample_value in sample_value.items():
                    recorded = col_samp[subsample_key]
                    assert_equal(subsample_value, recorded)


def test_merge_write_data_to_column_permutations(
        column_permutation_repo, num_samples_gen, num_subsamples_gen
):
    repo, column_data, column_data_partials = column_permutation_repo
    repo.create_branch('testbranch')

    # Write new data to master branch
    co = repo.checkout(write=True, branch='master')
    for column_name in co.columns:
        col = co[column_name]
        data_gen = column_data_partials[column_name]
        if col.column_layout == 'flat':
            for samp in range(num_samples_gen):
                data = data_gen()
                column_data[column_name][str(samp)] = data
                col[str(samp)] = data
        elif col.column_layout == 'nested':
            for samp in range(num_samples_gen):
                column_data[column_name][str(samp)] = {}
                for ssamp in range(num_subsamples_gen):
                    data = data_gen()
                    column_data[column_name][str(samp)][str(ssamp)] = data
                col[str(samp)] = column_data[column_name][str(samp)]
        else:
            raise ValueError(f'unknown layout option {col.column_layout}')
    co.commit('commit on master adding data')
    co.close()

    # Write new data to testbranch branch
    co = repo.checkout(write=True, branch='testbranch')
    for column_name in co.columns:
        col = co[column_name]
        data_gen = column_data_partials[column_name]
        if col.column_layout == 'flat':
            for samp in range(num_samples_gen):
                data = data_gen()
                column_data[column_name][f'_{samp}'] = data
                col[f'_{samp}'] = data
        elif col.column_layout == 'nested':
            for samp in range(num_samples_gen):
                column_data[column_name][f'_{samp}'] = {}
                for ssamp in range(num_subsamples_gen):
                    data = data_gen()
                    column_data[column_name][f'_{samp}'][f'_{ssamp}'] = data
                col[f'_{samp}'] = column_data[column_name][f'_{samp}']
        else:
            raise ValueError(f'unknown layout option {col.column_layout}')
    co.commit('commit on master adding data')
    co.close()

    # Merge and check that union of all data added is present
    repo.merge('merge commit', 'master', 'testbranch')

    co = repo.checkout(write=True, branch='master')
    assert len(co.columns) == len(column_data)
    for column_name, column_samples in column_data.items():
        assert column_name in co.columns
        col = co[column_name]
        assert len(column_samples) == len(col)

        for sample_key, sample_value in column_samples.items():
            if not isinstance(sample_value, dict):
                recorded = col[sample_key]
                assert_equal(sample_value, recorded)
            else:
                col_samp = col[sample_key]
                assert len(sample_value) == len(col_samp)
                for subsample_key, subsample_value in sample_value.items():
                    recorded = col_samp[subsample_key]
                    assert_equal(subsample_value, recorded)
    co.close()

