import pytest
import numpy as np
from conftest import fixed_shape_backend_params


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype



@pytest.fixture(scope='class')
def subsample_data_map():
    arr = np.arange(5*7).astype(np.uint16).reshape((5, 7))
    res = {
        'foo': {
            0: arr,
            1: arr + 1,
            2: arr + 2
        },
        2: {
            'bar': arr + 3,
            'baz': arr + 4
        }
    }
    return res


@pytest.fixture(scope='class')
def sample_data_map():
    arr = np.arange(5*7).astype(np.uint16).reshape((5, 7))
    res = {
        0: arr,
        1: arr + 1,
        2: arr + 2,
        'bar': arr + 3,
        'baz': arr + 4,
    }
    return res


@pytest.fixture(params=fixed_shape_backend_params, scope='class')
def backend_param(request):
    return request.param


@pytest.fixture(params=[False, True], scope='class')
def write_enabled(request):
    return request.param


@pytest.fixture(params=[False, True], scope='class')
def contains_subsamples(request):
    return request.param


@pytest.fixture(scope='class')
def initialized_column(
    write_enabled, backend_param, contains_subsamples, classrepo, subsample_data_map, sample_data_map
):
    co = classrepo.checkout(write=True)
    aset = co.add_ndarray_column(f'foo{backend_param}{int(write_enabled)}{int(contains_subsamples)}',
                                    shape=(5, 7), dtype=np.uint16,
                                    backend=backend_param, contains_subsamples=contains_subsamples)
    if contains_subsamples:
        aset.update(subsample_data_map)
    else:
        aset.update(sample_data_map)
    co.commit(f'done {backend_param}{write_enabled}{contains_subsamples}')
    co.close()
    if write_enabled:
        nco = classrepo.checkout(write=True)
        yield nco.columns[f'foo{backend_param}{int(write_enabled)}{int(contains_subsamples)}']
        nco.close()
    else:
        nco = classrepo.checkout()
        yield nco.columns[f'foo{backend_param}{int(write_enabled)}{int(contains_subsamples)}']
        nco.close()


@pytest.fixture(scope='class')
def initialized_column_read_only(backend_param, contains_subsamples, classrepo, subsample_data_map, sample_data_map):
    co = classrepo.checkout(write=True)
    aset = co.add_ndarray_column(f'foo{backend_param}{int(contains_subsamples)}',
                                    shape=(5, 7), dtype=np.uint16,
                                    backend=backend_param, contains_subsamples=contains_subsamples)
    if contains_subsamples:
        aset.update(subsample_data_map)
    else:
        aset.update(sample_data_map)

    digest = co.commit(f'done {backend_param}{contains_subsamples}')
    co.close()
    nco = classrepo.checkout(write=False, commit=digest)
    yield nco.columns[f'foo{backend_param}{int(contains_subsamples)}']
    nco.close()


class TestPickleableColumns:

    def test_is_pickleable(self, initialized_column, sample_data_map, subsample_data_map):
        import pickle

        aset = initialized_column
        if aset.iswriteable:
            with pytest.raises(PermissionError, match='Method "__getstate__" cannot'):
                pickle.dumps(aset, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            pkl = pickle.dumps(aset, protocol=pickle.HIGHEST_PROTOCOL)
            assert isinstance(pkl, bytes)


class TestLoadableColumns:

    def test_is_pickle_is_loadable(self, initialized_column_read_only, sample_data_map, subsample_data_map):
        import pickle

        aset = initialized_column_read_only
        pkl = pickle.dumps(aset, protocol=pickle.HIGHEST_PROTOCOL)
        assert isinstance(pkl, bytes)
        equiv = pickle.loads(pkl)

        if aset.contains_subsamples:
            assert len(aset) == len(subsample_data_map)
            assert len(equiv) == len(subsample_data_map)

            for sample_key, subsample_data in subsample_data_map.items():
                assert sample_key in aset
                assert sample_key in equiv
                aset_sample = aset[sample_key]
                equiv_sample = equiv[sample_key]
                assert len(aset_sample) == len(subsample_data)
                assert len(equiv_sample) == len(subsample_data)

                for subsample_key, expected in subsample_data.items():
                    assert subsample_key in aset_sample
                    assert subsample_key in equiv_sample
                    assert_equal(aset_sample[subsample_key], expected)
                    assert_equal(equiv_sample[subsample_key], expected)
        else:
            assert len(aset) == len(sample_data_map)
            assert len(equiv) == len(sample_data_map)
            for sample_key, expected in sample_data_map.items():
                assert sample_key in aset
                assert sample_key in equiv
                assert_equal(aset[sample_key], expected)
                assert_equal(equiv[sample_key], expected)
        equiv._destruct()
        del equiv
