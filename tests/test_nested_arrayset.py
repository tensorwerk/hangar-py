import pytest
import numpy as np
from conftest import fixed_shape_backend_params


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


@pytest.fixture()
def subsample_data_map():
    arr = np.zeros((5, 7), dtype=np.float64)
    res = {'foo': {0: arr, 1: arr + 1, 2: arr + 2},
           2: {'bar': arr + 3, 'baz': arr + 4}}
    return res


class TestArrayset(object):

    @pytest.mark.parametrize('name', [
        'invalid\n', '\ninvalid', 'inv name', 'inva@lid', 12, ' try', 'andthis ',
        'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
    def test_invalid_asetname(self, repo, randomsizedarray, name):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.arraysets.init_arrayset(name=name, prototype=randomsizedarray,
                                       contains_subsamples=True)
        co.close()

    def test_read_only_mode(self, aset_subsamples_initialized_repo):
        import hangar
        co = aset_subsamples_initialized_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.arraysets.init_arrayset is None
        assert co.arraysets.delete is None
        assert len(co.arraysets['writtenaset']) == 0
        co.close()

    def test_get_arrayset(self, aset_subsamples_initialized_repo, array5by7):
        co = aset_subsamples_initialized_repo.checkout(write=True)
        # getting the arrayset with `get`
        asetOld = co.arraysets.get('writtenaset')
        asetOldPath = asetOld._path
        asetOldAsetn = asetOld._asetn
        asetOldDefaultSchemaHash = asetOld._dflt_schema_hash
        co.metadata['foo'] = 'bar'
        co.commit('this is a commit message')
        co.close()
        co = aset_subsamples_initialized_repo.checkout()

        # getting arrayset with dictionary like style method
        asetNew = co.arraysets['writtenaset']
        assert asetOldPath == asetNew._path
        assert asetOldAsetn == asetNew._asetn
        assert asetOldDefaultSchemaHash == asetNew._dflt_schema_hash
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_remove_arrayset(self, aset_backend, aset_subsamples_initialized_repo):
        co = aset_subsamples_initialized_repo.checkout(write=True)
        co.arraysets.delete('writtenaset')
        with pytest.raises(KeyError):
            co.arraysets.delete('writtenaset')

        co.arraysets.init_arrayset(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                   backend_opts=aset_backend, contains_subsamples=True)
        assert len(co.arraysets) == 1
        co.arraysets.delete('writtenaset')
        co.commit('this is a commit message')
        co.close()

        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.arraysets) == 0

        co.arraysets.init_arrayset(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                   backend_opts=aset_backend, contains_subsamples=True)
        co.commit('this is a commit message')
        co.close()
        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.arraysets) == 1
        del co.arraysets['writtenaset']
        assert len(co.arraysets) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_init_again(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset('aset', prototype=randomsizedarray, backend_opts=aset_backend,
                                   contains_subsamples=True)
        with pytest.raises(LookupError):
            co.arraysets.init_arrayset('aset', prototype=randomsizedarray,
                                       backend_opts=aset_backend, contains_subsamples=True)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_arrayset_with_more_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.arraysets.init_arrayset('aset', shape=shape, dtype=np.int, backend_opts=aset_backend,
                                       contains_subsamples=True)
        shape = [1] * 31
        aset = co.arraysets.init_arrayset('aset1', shape=shape, dtype=np.int,
                                          backend_opts=aset_backend, contains_subsamples=True)
        assert len(aset._schema_max_shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.arraysets.init_arrayset('aset2', shape=shape, dtype=np.int,
                                       backend_opts=aset_backend, contains_subsamples=True)
        co.close()


class TestDataWithFixedSizedArrayset(object):

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_add_single_subsample_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        added = aset.add(1, ('subsample1', arr))
        assert added == 'subsample1'
        added = aset.add(2, {'subsample1': arr})
        assert added == 'subsample1'
        added = aset.add(3, ['subsample1', arr])
        assert added == 'subsample1'
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_add_multiple_subsamples_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        # create sample and append subsamples to it.
        added = aset.add(1, ('subsample1', arr))
        assert added == 'subsample1'
        added = aset.add(1, ('subsample2', arr + 1))
        assert added == 'subsample2'
        added = aset.add(1, [('subsample3', arr + 2), ('subsample4', arr + 3)])
        assert added == ('subsample3', 'subsample4')
        added = aset.add(1, {'subsample5': arr + 4, 'subsample6': arr + 5})
        assert added == ('subsample5', 'subsample6')

        # Add multiple subsamples to new sample in one operation
        added = aset.add(2, [('subsample1', arr + 20), ('subsample2', arr + 30)])
        assert added == ('subsample1', 'subsample2')
        added = aset.add(3, {'subsample1': arr + 40, 'subsample2': arr + 50})
        assert added == ('subsample1', 'subsample2')
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_setitem_single_subsample_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        aset[1] = ('subsample1', arr)
        aset[2] = {'subsample1': arr}
        aset[3] = ['subsample1', arr]
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_setitem_multiple_subsamples_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        # create sample and append subsamples to it.
        aset[1] = ('subsample1', arr)
        aset[1] = ('subsample2', arr + 1)
        aset[1] = [('subsample3', arr + 2), ('subsample4', arr + 3)]
        aset[1] = {'subsample5': arr + 4, 'subsample6': arr + 5}

        # Add multiple subsamples to new sample in one operation
        aset[2] = [('subsample1', arr + 20), ('subsample2', arr + 30)]
        aset[3] = {'subsample1': arr + 40, 'subsample2': arr + 50}
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_update_multiple_subsamples_to_empty_arrayset(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        added = aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            for subsample_name in subsample_data.keys():
                assert (sample_name, subsample_name) in added
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_sample_len_reported_correctly(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        # create sample and append subsamples to it.
        aset[1] = ('subsample1', arr)
        assert len(aset) == 1
        assert len(aset[1]) == 1
        aset[1] = ('subsample2', arr + 1)
        assert len(aset) == 1
        assert len(aset[1]) == 2
        aset[1] = [('subsample3', arr + 2), ('subsample4', arr + 3)]
        assert len(aset) == 1
        assert len(aset[1]) == 4
        aset[1] = {'subsample5': arr + 4, 'subsample6': arr + 5}
        assert len(aset) == 1
        assert len(aset[1]) == 6

        # Add multiple subsamples to new sample in one operation
        aset[2] = [('subsample1', arr + 20), ('subsample2', arr + 30)]
        assert len(aset) == 2
        assert len(aset[1]) == 6
        assert len(aset[2]) == 2
        aset[3] = {'subsample1': arr + 40, 'subsample2': arr + 50}
        assert len(aset) == 3
        assert len(aset[1]) == 6
        assert len(aset[2]) == 2
        assert len(aset[3]) == 2
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_num_subsamples_property_reported_correctly(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8, backend_opts=backend,
                                          contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        # create sample and append subsamples to it.
        aset[1] = ('subsample1', arr)
        assert aset.num_subsamples == 1
        aset[1] = ('subsample2', arr + 1)
        assert aset.num_subsamples == 2
        aset[1] = [('subsample3', arr + 2), ('subsample4', arr + 3)]
        assert aset.num_subsamples == 4
        aset[1] = {'subsample5': arr + 4, 'subsample6': arr + 5}
        assert aset.num_subsamples == 6

        # Add multiple subsamples to new sample in one operation
        aset[2] = [('subsample1', arr + 20), ('subsample2', arr + 30)]
        assert aset.num_subsamples == 8
        aset[3] = {'subsample1': arr + 40, 'subsample2': arr + 50}
        assert aset.num_subsamples == 10
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_returns_object(self, backend, repo, subsample_data_map):
        from hangar.columns import SubsampleWriter

        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample, SubsampleWriter)
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_len_method(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert len(sample) == len(subsample_data)
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_contains_method(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name in subsample_data.keys():
                assert subsample_name in sample
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_sample_property(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.sample == sample_name
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_arrayset_property(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.arrayset == 'foo'
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_contains_remote_references_property(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.contains_remote_references is False
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_subsample_remote_reference_keys_property(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.remote_reference_keys == []
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_get_single_subsample(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample.get(subsample_name)
                assert_equal(res, subsample_value)
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_get_multiple_subsamples(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample.get(list(subsample_data.keys())[:2])
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_getitem_single_subsample(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample[subsample_name]
                assert_equal(res, subsample_value)
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_getitem_multiple_subsamples(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[list(subsample_data.keys())[:2]]
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_get_sample_test_data_property(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.float64, backend_opts=backend,
                                          contains_subsamples=True)
        aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample.data
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])
        co.close()







