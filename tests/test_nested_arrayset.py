"""Tests for the class methods contained in the nested subsample arrayset accessor.

TODO
====
- Variable sized backends as well.
- Operations with context managers
- Picle and unpickle operations
- Reader and writer checkout operations
    - construction from commits with arrayset already existing.
- Integration through the rest of the repos operations (no problems expected)
    - diff / merge
    - remote push / pull
    - CLI?
"""
import pytest
import numpy as np
from conftest import fixed_shape_backend_params


# --------------------------- Setup ------------------------------


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


@pytest.fixture()
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


# ------------------------ Tests ----------------------------------


class TestArraysetSetup:

    @pytest.mark.parametrize('name', [
        'invalid\n', '\ninvalid', 'inv name', 'inva@lid', 12, ' try', 'andthis ',
        'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
    def test_does_not_allow_invalid_arrayset_names(self, repo, randomsizedarray, name):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.arraysets.init_arrayset(name=name, prototype=randomsizedarray, contains_subsamples=True)
        co.close()

    def test_read_only_mode_arrayset_methods_limited(self, aset_subsamples_initialized_repo):
        import hangar
        co = aset_subsamples_initialized_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.arraysets.init_arrayset is None
        assert co.arraysets.delete is None
        assert len(co.arraysets['writtenaset']) == 0
        co.close()

    def test_get_arrayset_in_read_and_write_checkouts(self, aset_subsamples_initialized_repo, array5by7):
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
    def test_delete_arrayset(self, aset_backend, aset_subsamples_initialized_repo):
        co = aset_subsamples_initialized_repo.checkout(write=True)
        co.arraysets.delete('writtenaset')
        assert 'writtenaset' not in co.arraysets
        with pytest.raises(KeyError):
            # cannot delete twice
            co.arraysets.delete('writtenaset')

        # init and immediate delete leaves no trace
        co.arraysets.init_arrayset(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                   backend_opts=aset_backend, contains_subsamples=True)
        assert len(co.arraysets) == 1
        co.arraysets.delete('writtenaset')
        assert len(co.arraysets) == 0
        co.commit('this is a commit message')
        co.close()

        # init arrayset in checkout persists aset records/accessor even if no samples contained
        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.arraysets) == 0
        co.arraysets.init_arrayset(name='writtenaset', shape=(5, 7), dtype=np.float64,
                                   backend_opts=aset_backend, contains_subsamples=True)
        co.commit('this is a commit message')
        co.close()
        co = aset_subsamples_initialized_repo.checkout(write=True)
        assert len(co.arraysets) == 1

        # arrayset can be deleted with via __delitem__ dict style command.
        del co.arraysets['writtenaset']
        assert len(co.arraysets) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_init_same_arrayset_twice_fails_again(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset('aset', prototype=randomsizedarray,
                                   backend_opts=aset_backend, contains_subsamples=True)
        with pytest.raises(LookupError):
            # test if everything is the same as initalized one.
            co.arraysets.init_arrayset('aset', prototype=randomsizedarray,
                                       backend_opts=aset_backend, contains_subsamples=True)
        with pytest.raises(LookupError):
            # test if arrayset container type is different than existing name (no subsamples0
            co.arraysets.init_arrayset('aset', prototype=randomsizedarray,
                                       backend_opts=aset_backend, contains_subsamples=False)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_arrayset_with_invalid_dimension_sizes_shapes(self, aset_backend, repo):
        co = repo.checkout(write=True)

        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            # cannot have zero valued size for any dimension
            co.arraysets.init_arrayset('aset', shape=shape, dtype=np.int,
                                       backend_opts=aset_backend, contains_subsamples=True)

        shape = [1] * 31
        aset = co.arraysets.init_arrayset('aset1', shape=shape, dtype=np.int,
                                          backend_opts=aset_backend, contains_subsamples=True)
        assert len(aset.shape) == 31

        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.arraysets.init_arrayset('aset2', shape=shape, dtype=np.int,
                                       backend_opts=aset_backend, contains_subsamples=True)
        co.close()


# ------------------------------ Add Data Tests --------------------------------------------


class TestAddData:

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_add_single_subsample_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8,
                                          backend_opts=backend, contains_subsamples=True)
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
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8,
                                          backend_opts=backend, contains_subsamples=True)
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
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8,
                                          backend_opts=backend, contains_subsamples=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        aset[1] = ('subsample1', arr)
        aset[2] = {'subsample1': arr}
        aset[3] = ['subsample1', arr]
        co.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_setitem_multiple_subsamples_to_empty_arrayset(self, backend, repo):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8,
                                          backend_opts=backend, contains_subsamples=True)
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
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.uint16,
                                          backend_opts=backend, contains_subsamples=True)
        added = aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            for subsample_name in subsample_data.keys():
                assert (sample_name, subsample_name) in added
        co.close()


# --------------------------- Test Remove Data -------------------------------------


@pytest.fixture(params=fixed_shape_backend_params)
def backend_param(request):
    return request.param


@pytest.fixture(params=[False, True])
def write_enabled(request):
    return request.param


@pytest.fixture()
def initialized_arrayset(backend_param, write_enabled, repo, subsample_data_map):
    co = repo.checkout(write=True)
    aset = co.arraysets.init_arrayset(
        'foo', shape=(5, 7), dtype=np.uint16, backend_opts=backend_param, contains_subsamples=True)
    aset.update(subsample_data_map)
    if not write_enabled:
        co.commit('first')
        co.close()
        co = repo.checkout()
        yield co.arraysets['foo']
    else:
        yield co.arraysets['foo']
    co.close()


@pytest.fixture()
def initialized_arrayset_write_only(backend_param, repo, subsample_data_map):
    co = repo.checkout(write=True)
    aset = co.arraysets.init_arrayset(
        'foo', shape=(5, 7), dtype=np.uint16, backend_opts=backend_param, contains_subsamples=True)
    aset.update(subsample_data_map)
    yield co.arraysets['foo']
    co.close()


class TestRemoveData:

    # --------------------- delete -----------------------------

    def test_delete_single_sample_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset.delete('foo')
        assert res == 'foo'
        assert 'foo' not in aset

    def test_delete_multiple_samples_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset.delete(*['foo', 2])
        assert res == ['foo', 2]
        assert 'foo' not in aset
        assert 2 not in aset

    def test_delitem_single_sample_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        del aset['foo']
        assert 'foo' not in aset

    def test_delete_single_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset['foo'].delete(0)
        assert res == 0
        assert 0 not in aset['foo']

    def test_delitem_single_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        del aset['foo'][0]
        assert 0 not in aset['foo']

    def test_delete_multiple_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset['foo'].delete(*[0, 1])
        assert res == [0, 1]
        assert 0 not in aset['foo']
        assert 1 not in aset['foo']

    # ------------------------ pop ----------------------------

    def test_pop_single_sample_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset.pop('foo')
        assert 'foo' not in aset
        assert isinstance(res, dict)
        assert len(res) == 1
        assert 'foo' in res

        popped_subsample_kvs = res['foo']
        assert isinstance(popped_subsample_kvs, dict)
        assert len(popped_subsample_kvs) == 3
        for expected_k, expected_v in subsample_data_map['foo'].items():
            assert_equal(popped_subsample_kvs[expected_k], expected_v)

    def test_pop_multiple_samples_from_arrayset(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset.pop(*['foo', 2])
        assert 'foo' not in aset
        assert 2 not in aset
        assert isinstance(res, dict)
        assert len(res) == 2
        assert 'foo' in res
        assert 2 in res

        for popped_sample_key, popped_subsample_kvs in res.items():
            assert popped_sample_key in subsample_data_map  # sanity check for test data config
            assert popped_sample_key in ['foo', 2]
            assert isinstance(popped_subsample_kvs, dict)
            assert len(popped_subsample_kvs) == len(subsample_data_map[popped_sample_key])
            for expected_k, expected_v in subsample_data_map[popped_sample_key].items():
                assert_equal(popped_subsample_kvs[expected_k], expected_v)

    def test_pop_single_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset['foo'].pop(0)
        assert 0 not in aset['foo']
        assert isinstance(res, np.ndarray)
        assert_equal(res, subsample_data_map['foo'][0])

    def test_pop_multiple_subsample_from_sample(self, initialized_arrayset_write_only, subsample_data_map):
        aset = initialized_arrayset_write_only
        res = aset['foo'].pop(*[0, 1])
        assert 0 not in aset['foo']
        assert 1 not in aset['foo']

        assert isinstance(res, dict)
        assert 0 in res
        assert 1 in res
        for k, v in res.items():
            assert_equal(subsample_data_map['foo'][k], v)


# ------------------------------ Container Introspection --------------------------------------------


class TestContainerIntrospection:

    def test_get_sample_returns_object(self, initialized_arrayset, subsample_data_map):
        from hangar.columns import SubsampleWriter, SubsampleWriterModifier

        aset = initialized_arrayset
        assert isinstance(aset, SubsampleWriterModifier)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample, SubsampleWriter)

    # -------------------------- test __dunder__ methods ----------------------------------

    def test_get_sample_test_subsample_len_method(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert len(sample) == len(subsample_data)

    def test_get_sample_test_subsample_contains_method(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name in subsample_data.keys():
                assert subsample_name in sample

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_sample_len_reported_correctly(self, backend, repo):
        co = repo.checkout(write=True)
        arr = np.zeros((3, 3), dtype=np.uint8)
        aset = co.arraysets.init_arrayset(
            'foo', prototype=arr, backend_opts=backend, contains_subsamples=True)

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
        aset = co.arraysets.init_arrayset('foo', shape=(3, 3), dtype=np.uint8,
                                          backend_opts=backend, contains_subsamples=True)
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

    # ----------------------------- test property ---------------------------

    def test_get_sample_test_subsample_sample_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.sample == sample_name

    def test_get_sample_test_subsample_arrayset_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.arrayset == 'foo'

    def test_get_sample_test_data_property(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample.data
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_test_subsample_contains_remote_references_property(self, initialized_arrayset, subsample_data_map):
        """TODO: test case where there are actually remote references present.
        """
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.contains_remote_references is False

    def test_get_sample_test_subsample_remote_reference_keys_property(self, initialized_arrayset, subsample_data_map):
        """TODO: test case where there are actually remote references present.
        """
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert sample.remote_reference_keys == []


# ------------------------------ Getting Data --------------------------------------------


class TestGetDataMethods:

    def test_get_sample_missing_key(self, initialized_arrayset):
        aset = initialized_arrayset
        with pytest.raises(KeyError):
            aset.get('doesnotexist')
        with pytest.raises(KeyError):
            aset.get(999_999)

    def test_get_sample_get_single_subsample(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample.get(subsample_name)
                assert_equal(res, subsample_value)

    def test_get_sample_get_single_subsample_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name in subsample_data_map.keys():
            sample = aset.get(sample_name)
            with pytest.raises(KeyError):
                sample.get('doesnotexist')
            with pytest.raises(KeyError):
                sample.get(999_999)

    def test_get_sample_get_multiple_subsamples(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample.get(*list(list(subsample_data.keys())[:2]))
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_get_multiple_subsamples_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            with pytest.raises(KeyError):
                # both keys don't exist
                sample.get(*['doesnotexist', 999_999])
            with pytest.raises(ValueError):
                # only one key doesn't exist, str type
                test_keys = list(subsample_data.keys()).append('doesnotexist')
                sample.get(*test_keys)
            with pytest.raises(ValueError):
                # only one key doesn't exist, int type
                test_keys = list(subsample_data.keys()).append(999_999)
                sample.get(*test_keys)

    def test_get_sample_getitem_single_subsample(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_value in subsample_data.items():
                res = sample[subsample_name]
                assert_equal(res, subsample_value)

    def test_get_sample_getitem_single_subsample_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name in subsample_data_map.keys():
            sample = aset.get(sample_name)
            with pytest.raises(KeyError):
                sample['doesnotexist']
            with pytest.raises(KeyError):
                sample[999_999]

    def test_get_sample_getitem_multiple_subsamples(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            with pytest.raises(ValueError):
                res = sample[list(subsample_data.keys())[:2]]
            res = sample.get(*list(subsample_data.keys())[:2])
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_multiple_subsamples_missing_key(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            real_keys = list(subsample_data.keys())
            with pytest.raises(ValueError):
                # both keys don't exist
                sample[['doesnotexist', 999_999]]
            with pytest.raises(ValueError):
                # only one key doesn't exist, str type
                test_keys = real_keys.append('doesnotexist')
                sample[test_keys]
            with pytest.raises(ValueError):
                # only one key doesn't exist, int type
                test_keys = real_keys.append(999_999)
                sample[test_keys]

    def test_get_sample_getitem_subsamples_with_ellipsis(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[...]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_keys_and_ellipsis_fails(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            existing_subsample_key = next(iter(subsample_data.keys()))
            with pytest.raises(ValueError):
                sample[..., existing_subsample_key]
            with pytest.raises(ValueError):
                sample[..., [existing_subsample_key]]

    def test_get_sample_getitem_subsamples_with_unbound_slice(self, initialized_arrayset, subsample_data_map):
        """unbound slice is ``slice(None) == [:]``"""
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[:]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_bounded_slice(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[0:2]
            assert isinstance(res, dict)
            assert len(res) == 2
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    def test_get_sample_getitem_subsamples_with_out_of_bounds_slice_does_not_fail(
            self, initialized_arrayset, subsample_data_map):
        """Odd python behavior we emulate: out of bounds sequence slicing is allowed.

        Instead of throwing an exception, the slice is treated as if it should just
        go up to the total number of elements in the container. For example:
            [1, 2, 3][0:5] == [1, 2, 3]
        """
        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            res = sample[0:5]
            assert isinstance(res, dict)
            assert len(res) == len(subsample_data)
            for k, v in res.items():
                assert_equal(v, subsample_data[k])

    # -------------------------- dict-style iteration methods ---------------------------

    def test_calling_iter_on_arrayset(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        arrayset_it = iter(aset)  # returns iterator over sample keys
        for sample_name in arrayset_it:
            assert sample_name in aset
            assert sample_name in subsample_data_map

    def test_calling_iter_on_sample_in_arrayset(self, initialized_arrayset, subsample_data_map):
        aset = initialized_arrayset
        arrayset_it = iter(aset)  # returns iterator over sample keys
        for sample_name in arrayset_it:
            assert sample_name in aset
            assert sample_name in subsample_data_map

            sample_it = iter(aset[sample_name])  # returns iterator over subsample keys
            for subsample_name in sample_it:
                assert subsample_name in aset[sample_name]
                assert subsample_name in subsample_data_map[sample_name]

    def test_get_sample_subsample_keys_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.keys(), Iterator)
            res = list(sample.keys())
            for k in res:
                assert k in subsample_data

    def test_get_sample_subsample_keys_method_local_only(self, initialized_arrayset, subsample_data_map):
        """TODO: test case where there are actually remote references present.
        """
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.keys(local=True), Iterator)
            res = list(sample.keys(local=True))
            for k in res:
                assert k in subsample_data

    def test_get_sample_subsample_values_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.values(), Iterator)
            res = list(sample.values())
            for v in res:
                assert any([np.allclose(v, arr) for arr in subsample_data.values()])

    def test_get_sample_subsample_values_method_local_only(self, initialized_arrayset, subsample_data_map):
        """TODO: test case where there are actually remote references present.
        """
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.values(local=True), Iterator)
            res = list(sample.values(local=True))
            for v in res:
                assert any([np.allclose(v, arr) for arr in subsample_data.values()])

    def test_get_sample_subsample_items_method(self, initialized_arrayset, subsample_data_map):
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.items(), Iterator)
            res = list(sample.items())
            for k, v in res:
                assert_equal(v, subsample_data[k])

    def test_get_sample_subsample_items_method_local_only(self, initialized_arrayset, subsample_data_map):
        """TODO: test case where there are actually remote references present.
        """
        from collections.abc import Iterator

        aset = initialized_arrayset
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            assert isinstance(sample.items(local=True), Iterator)
            res = list(sample.items(local=True))
            for k, v in res:
                assert_equal(v, subsample_data[k])


class TestWriteThenReadCheckout:

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_add_data_commit_checkout_read_only_contains_same(self, backend, repo, subsample_data_map):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('foo', shape=(5, 7), dtype=np.uint16,
                                          backend_opts=backend, contains_subsamples=True)
        added = aset.update(subsample_data_map)
        for sample_name, subsample_data in subsample_data_map.items():
            sample = aset.get(sample_name)
            for subsample_name, subsample_val in subsample_data.items():
                assert (sample_name, subsample_name) in added
                assert_equal(sample[subsample_name], subsample_val)
        co.commit('first')
        co.close()

        rco = repo.checkout()
        naset = rco.arraysets['foo']
        for sample_name, subsample_data in subsample_data_map.items():
            sample = naset.get(sample_name)
            for subsample_name, subsample_val in subsample_data.items():
                assert_equal(sample[subsample_name], subsample_val)
        rco.close()
