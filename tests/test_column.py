import pytest
import numpy as np
from conftest import fixed_shape_backend_params, variable_shape_backend_params
from itertools import permutations


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


class TestColumn(object):

    @pytest.mark.parametrize('name', [
        'invalid\n', '\ninvalid', 'inv name', 'inva@lid', 12, ' try', 'andthis ',
        'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
    def test_invalid_column_name(self, repo, randomsizedarray, name):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.add_ndarray_column(name=name, prototype=randomsizedarray)
        with pytest.raises(ValueError):
            co.add_str_column(name=name)
        co.close()

    def test_read_only_mode(self, aset_samples_initialized_repo):
        import hangar
        co = aset_samples_initialized_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        with pytest.raises(AttributeError):
            assert co.add_ndarray_column('foo')
        with pytest.raises(AttributeError):
            assert co.add_str_column('foo')
        with pytest.raises(PermissionError):
            del co.columns['foo']
        with pytest.raises(PermissionError):
            del co.columns['foo']
        assert len(co.columns['writtenaset']) == 0
        co.close()

    def test_get_column(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)

        # getting the column with `get`
        asetOld = co.columns.get('writtenaset')
        asetOldPath = asetOld._path
        asetOldAsetn = asetOld.column
        asetOldDefaultSchemaHash = asetOld._schema.schema_hash_digest()

        asetOld['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()

        # getting column with dictionary like style method
        asetNew = co.columns['writtenaset']
        assert_equal(asetNew['1'], array5by7)
        assert asetOldPath == asetNew._path
        assert asetOldAsetn == asetNew.column
        assert asetOldDefaultSchemaHash == asetNew._schema.schema_hash_digest()
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_remove_column(self, aset_backend, aset_samples_initialized_repo):
        co = aset_samples_initialized_repo.checkout(write=True)
        del co.columns['writtenaset']
        with pytest.raises(KeyError):
            del co.columns['writtenaset']

        co.add_ndarray_column('writtenaset', shape=(5, 7), dtype=np.float64, backend=aset_backend)
        assert len(co.columns) == 1
        del co.columns['writtenaset']
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 0

        co.add_ndarray_column('writtenaset', shape=(5, 7), dtype=np.float64, backend=aset_backend)
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        del co.columns['writtenaset']
        assert len(co.columns) == 0
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_init_again(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.add_ndarray_column('aset', prototype=randomsizedarray, backend=aset_backend)
        with pytest.raises(LookupError):
            co.add_ndarray_column('aset', prototype=randomsizedarray, backend=aset_backend)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_column_with_more_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.add_ndarray_column('aset', shape=shape, dtype=np.int, backend=aset_backend)
        shape = [1] * 31
        aset = co.add_ndarray_column('aset1', shape=shape, dtype=np.int, backend=aset_backend)
        assert len(aset.shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.add_ndarray_column('aset2', shape=shape, dtype=np.int, backend=aset_backend)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_column_with_empty_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.array(1, dtype=np.int64)
        aset = co.add_ndarray_column('aset1', shape=(), dtype=np.int64, backend=aset_backend)
        aset['1'] = arr
        co.commit('this is a commit message')
        aset = co.add_ndarray_column('aset2', prototype=arr)
        aset['1'] = arr
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        aset1 = co.columns['aset1']
        aset2 = co.columns['aset2']
        assert_equal(aset1['1'], arr)
        assert_equal(aset2['1'], arr)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_column_with_int_specifier_as_dimension(self, aset_backend, repo):
        co = repo.checkout(write=True)
        arr = np.arange(10, dtype=np.int64)
        aset = co.add_ndarray_column('aset1', shape=10, dtype=np.int64, backend=aset_backend)
        aset['1'] = arr
        co.commit('this is a commit message')
        arr2 = np.array(53, dtype=np.int64)
        aset = co.add_ndarray_column('aset2', prototype=arr2)
        aset['1'] = arr2
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        aset1 = co.columns['aset1']
        aset2 = co.columns['aset2']
        assert_equal(aset1['1'], arr)
        assert_equal(aset2['1'], arr2)
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_getattr_does_not_raise_permission_error_if_alive(self, aset_backend, write, repo):
        co = repo.checkout(write=True)
        arr = np.arange(10, dtype=np.int64)
        aset = co.add_ndarray_column('aset1', shape=10, dtype=np.int64, backend=aset_backend)
        aset['1'] = arr
        co.commit('hello')
        co.close()
        co = repo.checkout(write=write)
        aset = co.columns['aset1']

        assert hasattr(aset, 'doesnotexist') is False  # does not raise error
        assert hasattr(aset, '_mode') is True
        with pytest.raises(AttributeError):
            assert getattr(aset, 'doesnotexist')
        assert getattr(aset, '_mode') == 'a' if write else 'r'

        co.close()
        with pytest.raises(PermissionError):
            hasattr(aset, 'doesnotexist')
        with pytest.raises(PermissionError):
            hasattr(aset, '_mode')


class TestDataWithFixedSizedColumn(object):

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_column_remote_references_property_with_none(
            self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray
    ):
        co = repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int, backend=aset2_backend)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32, backend=aset3_backend)

        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1[1] = randomsizedarray
            d2[1] = np.ones((2, 2), dtype=np.int)
            d3[1] = np.ones((3, 4), dtype=np.float32)

        assert co.columns.contains_remote_references == {'aset1': False, 'aset2': False, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (), 'aset2': (), 'aset3': ()}
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_column_remote_references_property_with_remotes(
            self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray
    ):
        co = repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int, backend=aset2_backend)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32, backend=aset3_backend)

        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1[1] = randomsizedarray
            d2[1] = np.ones((2, 2), dtype=np.int)
            d3[1] = np.ones((3, 4), dtype=np.float32)

        assert co.columns.contains_remote_references == {'aset1': False, 'aset2': False, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (), 'aset2': (), 'aset3': ()}
        co.commit('hello')
        co.close()
        co = repo.checkout()

        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        co._columns._columns['aset1']._samples[12] = template
        co._columns._columns['aset2']._samples[22] = template

        assert co.columns.contains_remote_references == {'aset1': True, 'aset2': True, 'aset3': False}
        assert co.columns.remote_sample_keys == {'aset1': (12,), 'aset2': (22,), 'aset3': ()}
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_iterating_over(self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int, backend=aset2_backend)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32, backend=aset3_backend)

        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1['1'] = randomsizedarray
            d1['2'] = np.zeros_like(randomsizedarray)
            d1['3'] = np.zeros_like(randomsizedarray) + 5

            d2['1'] = np.ones((2, 2), dtype=np.int)
            d2['2'] = np.ones((2, 2), dtype=np.int) * 5
            d2['3'] = np.zeros((2, 2), dtype=np.int)

            d3['1'] = np.ones((3, 4), dtype=np.float32)
            d3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            d3['3'] = np.zeros((3, 4), dtype=np.float32)

        all_tensors.extend([aset1['1'], aset1['2'], aset1['3']])
        all_tensors.extend([aset2['1'], aset2['2'], aset2['3']])
        all_tensors.extend([aset3['1'], aset3['2'], aset3['3']])

        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname, aset in co.columns.items():
            assert aset._column_name == dname
            for sname, sample in aset.items():
                assert_equal(sample, next(tensors_in_the_order))

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in co.columns.keys():
            for sname in co.columns[dname].keys():
                assert_equal(co.columns[dname][sname], next(tensors_in_the_order))

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for aset in co.columns.values():
            for sample in aset.values():
                assert_equal(sample, next(tensors_in_the_order))
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset3_backend", fixed_shape_backend_params)
    def test_iterating_over_local_only(self, aset1_backend, aset2_backend, aset3_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', shape=(2, 2), dtype=np.int, backend=aset2_backend)
        aset3 = co.add_ndarray_column('aset3', shape=(3, 4), dtype=np.float32, backend=aset3_backend)

        with aset1 as d1, aset2 as d2, aset3 as d3:
            d1['1'] = randomsizedarray
            d1['2'] = np.zeros_like(randomsizedarray)
            d1['3'] = np.zeros_like(randomsizedarray) + 5

            d2['1'] = np.ones((2, 2), dtype=np.int)
            d2['2'] = np.ones((2, 2), dtype=np.int) * 5
            d2['3'] = np.zeros((2, 2), dtype=np.int)

            d3['1'] = np.ones((3, 4), dtype=np.float32)
            d3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            d3['3'] = np.zeros((3, 4), dtype=np.float32)

        all_tensors.extend([aset1['1'], aset1['2'], aset1['3']])
        all_tensors.extend([aset2['1'], aset2['2'], aset2['3']])
        all_tensors.extend([aset3['1'], aset3['2'], aset3['3']])

        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()

        # perform the mock
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        co._columns._columns['aset1']._samples['4'] = template
        co._columns._columns['aset2']._samples['4'] = template

        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname in ['aset1', 'aset2', 'aset3']:
            aset = co.columns[dname]
            count = 0
            for sname, sample in aset.items(local=True):
                count += 1
                assert_equal(sample, next(tensors_in_the_order))
                assert '4' != sname
            assert count == 3

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in ['aset1', 'aset2', 'aset3']:
            aset = co.columns[dname]
            count = 0
            for sname in aset.keys(local=True):
                count += 1
                assert_equal(aset[sname], next(tensors_in_the_order))
                assert '4' != sname
            assert count == 3

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for dname in ['aset1', 'aset2', 'aset3']:
            aset = co.columns[dname]
            count = 0
            for sample in aset.values(local=True):
                count += 1
                assert_equal(sample, next(tensors_in_the_order))
            assert count == 3

        assert list(co['aset1'].keys()) == ['1', '2', '3', '4']
        with pytest.raises((FileNotFoundError, KeyError)):
            list(co['aset1'].values())
        with pytest.raises((FileNotFoundError, KeyError)):
            list(co['aset1'].items())

        assert list(co['aset2'].keys()) == ['1', '2', '3', '4']
        with pytest.raises((FileNotFoundError, KeyError)):
            list(co['aset2'].values())
        with pytest.raises((FileNotFoundError, KeyError)):
            list(co['aset2'].items())

        assert list(co['aset3'].keys()) == ['1', '2', '3']
        assert len(list(co['aset3'].values())) == 3
        assert len(list(co['aset3'].items())) == 3
        co.close()

    def test_get_data(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()
        assert np.allclose(co.columns['writtenaset']['1'], co.columns.get('writtenaset').get('1'), array5by7)
        co.close()

    def test_get_sample_with_default_works(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        res = co.columns['writtenaset'].get('doesnotexist', default=500)
        assert res is 500
        res = co.columns['writtenaset'].get('doesnotexist', 500)
        assert res is 500
        co.close()

    def test_get_multiple_samples_fails(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.columns['writtenaset']['2'] = array5by7 + 1
        co.columns['writtenaset']['3'] = array5by7 + 2
        co.commit('this is a commit message')
        co.close()

        nco = aset_samples_initialized_repo.checkout()
        with pytest.raises(TypeError):
            res = nco.columns['writtenaset'].get(['1', '2'])
        res = nco.columns['writtenaset'].get(('1', '2'))
        assert res is None

        aset = nco.columns['writtenaset']
        with pytest.raises(TypeError):
            res = aset.get(*('1', '2', '3'))
        nco.close()

    def test_getitem_multiple_samples_missing_key(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        nco = aset_samples_initialized_repo.checkout()
        with pytest.raises(KeyError):
            nco.columns['writtenaset'][('1', '2')]
        with pytest.raises(KeyError):
            aset = nco.columns['writtenaset']
            aset[('1', '2')]
        nco.close()

    def test_get_multiple_samples_missing_key(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        nco = aset_samples_initialized_repo.checkout()
        aset = nco.columns['writtenaset']
        res = aset.get(('1', '2'))
        assert res == None
        nco.close()

    def test_add_data_str_keys(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with pytest.raises(KeyError):
            aset['somerandomkey']

        aset['1'] = array5by7
        aset['2'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()
        assert_equal(co.columns['writtenaset']['1'], array5by7)
        assert_equal(co.columns['writtenaset']['2'], array5by7)
        co.close()

    def test_add_data_int_keys(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']

        aset[1] = array5by7
        secondArray = array5by7 + 1
        aset[2] = secondArray
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()
        assert_equal(co.columns['writtenaset'][1], array5by7)
        assert_equal(co.columns['writtenaset'][2], secondArray)
        co.close()

    def test_cannot_add_data_negative_int_key(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with pytest.raises(ValueError):
            aset[-1] = array5by7
        assert len(co.columns['writtenaset']) == 0
        co.close()

    def test_cannot_add_data_float_key(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with pytest.raises(ValueError):
            aset[2.1] = array5by7
        with pytest.raises(ValueError):
            aset[0.0] = array5by7
        assert len(co.columns['writtenaset']) == 0
        co.close()

    def test_add_data_mixed_int_str_keys(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']

        aset[1] = array5by7
        newFirstArray = array5by7 + 1
        aset['1'] = newFirstArray
        secondArray = array5by7 + 2
        aset[2] = secondArray
        thirdArray = array5by7 + 3
        aset['2'] = thirdArray
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()
        assert_equal(co.columns['writtenaset'][1], array5by7)
        assert_equal(co.columns['writtenaset']['1'], newFirstArray)
        assert_equal(co.columns['writtenaset'][2], secondArray)
        assert_equal(co.columns['writtenaset']['2'], thirdArray)
        co.close()

    def test_cannot_add_data_sample_name_longer_than_64_characters(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with pytest.raises(ValueError):
            aset['VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'] = array5by7
        assert len(co.columns['writtenaset']) == 0
        co.close()

    def test_add_with_wrong_argument_order(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        with pytest.raises(ValueError):
            aset[array5by7] = '1'

    def test_update_with_dict_single_item(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = {'foo': array5by7}
        aset.update(data_map)
        assert_equal(aset['foo'], array5by7)

    def test_update_with_dict_multiple_items(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = {
            'foo': array5by7,
            1: array5by7+1
        }
        aset.update(data_map)
        assert_equal(aset['foo'], array5by7)
        assert_equal(aset[1], array5by7+1)

    def test_update_with_list_single_item(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = ['foo', array5by7]
        with pytest.raises(ValueError, match='dictionary update sequence'):
            aset.update(data_map)
        assert 'foo' not in aset

        aset.update((data_map,))  # try again while contained in iterable
        assert_equal(aset['foo'], array5by7)

    def test_update_with_list_multiple_items(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = [
            ('foo', array5by7),
            (1, array5by7+1),
        ]
        aset.update(data_map)
        assert_equal(aset['foo'], array5by7)
        assert_equal(aset[1], array5by7+1)

    def test_update_with_only_kwargs_single_item(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        aset.update(foo=array5by7)
        assert_equal(aset['foo'], array5by7)

    def test_update_with_only_kwargs_multiple_items(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        aset.update(foo=array5by7, bar=array5by7+1)
        assert_equal(aset['foo'], array5by7)
        assert_equal(aset['bar'], array5by7+1)

    def test_update_with_list_and_kwargs(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = [
            ('foo', array5by7),
            (1, array5by7+1),
        ]
        aset.update(data_map, bar=array5by7+2)
        assert_equal(aset['foo'], array5by7)
        assert_equal(aset[1], array5by7+1)
        assert_equal(aset['bar'], array5by7 + 2)

    def test_update_with_dict_and_kwargs(self, aset_samples_initialized_w_checkout, array5by7):
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = {
            'foo': array5by7,
            1: array5by7+1,
        }
        aset.update(data_map, bar=array5by7+2)
        assert_equal(aset['foo'], array5by7)
        assert_equal(aset[1], array5by7+1)
        assert_equal(aset['bar'], array5by7 + 2)

    def test_update_with_dict_and_kwargs_does_not_modify_input_in_calling_scopy(
        self, aset_samples_initialized_w_checkout, array5by7
    ):
        """ensure bug does not revert.

        Had a case where if dict was passed as ``other`` along with kwargs, the operation
        would complete as normally, but when control returned to the caller the original
        dict passed in as ``other`` would have been silently merged with the kwargs.
        """
        aset = aset_samples_initialized_w_checkout.columns['writtenaset']
        data_map = {
            'foo': array5by7,
            1: array5by7+1,
        }
        data_map_before = list(data_map.keys())
        aset.update(data_map, bar=array5by7+2)
        # in bug case, would now observe that data_map would have been
        # silently modified in a method analogous to calling:
        #
        #   ``data_map.update({'bar': np.array})``
        #
        assert list(data_map.keys()) == data_map_before

    @pytest.mark.parametrize('data_map', [
        ['foo', {'bar': np.random.random((5, 7))}],
        ['foo', 'bar', np.random.random((5, 7))],
        [('foo', 'bar', np.random.random((5, 7)))],
        [{('foo', 'bar'): np.random.random((5, 7))}],
        [('foo', 'bar', np.random.random((5, 7)))],
        [('foo', np.random.random((5, 7)), 'bar')],
        [(np.random.random((5, 7)), 'foo', 'bar')],
        [('foo', np.random.random((5, 7)), 'bar'), ('valid', np.random.random((5, 7)))],
        [('valid', np.random.random((5, 7))), ('foo', np.random.random((5, 7)), 'bar')],
        {'foo': np.random.random((5, 7)), 'bar': (np.random.random((5, 7)), np.random.random((5, 7)))},
    ])
    def test_update_with_invalid_data_map_fails(self, aset_samples_initialized_w_checkout, data_map):
        aset = aset_samples_initialized_w_checkout['writtenaset']
        with pytest.raises(ValueError):
            aset.update(data_map)

    @pytest.mark.parametrize('key,value', [
        ['foo', {'bar': np.random.random((5, 7))}],
        ['foo', ('bar', np.random.random((5, 7)))],
        [('foo', 'bar'), np.random.random((5, 7))],
        ['foo', {('foo', 'bar'): np.random.random((5, 7))}],
        ['foo', ('bar', np.random.random((5, 7)))],
        ['foo', (np.random.random((5, 7)), 'bar')],
        [np.random.random((5, 7)), ('foo', 'bar')],
        [('foo', np.random.random((5, 7)), 'bar'), ('valid', np.random.random((5, 7)))],
        [('valid', np.random.random((5, 7))), ('foo', np.random.random((5, 7)), 'bar')],
        [('valid', np.random.random((5, 7))), ('valid2', np.random.random((5, 7)))],
    ])
    def test_setitem_with_invalid_data_map_fails(self, aset_samples_initialized_w_checkout, key, value):
        aset = aset_samples_initialized_w_checkout['writtenaset']
        with pytest.raises(ValueError):
            aset[key] = value

    def test_add_multiple_data_single_commit(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        new_array = np.zeros_like(array5by7)
        co.columns['writtenaset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        aset = co.columns['writtenaset']
        assert len(aset) == 2
        assert list(aset.keys()) == ['1', '2']
        assert_equal(aset['1'], array5by7)
        co.close()

    def test_add_same_data_same_key_does_not_duplicate_hash(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset = co.columns['writtenaset']
        aset['1'] = array5by7
        old_spec = aset._samples['1']
        aset['1'] = array5by7
        new_spec = aset._samples['1']
        assert old_spec == new_spec
        assert len(aset) == 1
        assert len(aset._samples) == 1
        co.close()

    def test_multiple_data_multiple_commit(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        new_array = np.zeros_like(array5by7)
        co.columns['writtenaset']['2'] = new_array
        co.close()

        new_new_array = new_array + 5
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['3'] = new_new_array
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        aset = co.columns['writtenaset']
        assert_equal(aset['1'], array5by7)
        assert_equal(aset['2'], new_array)
        assert_equal(aset['3'], new_new_array)
        co.close()

    def test_added_but_not_commited(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.close()

        with pytest.raises(PermissionError):
            co.commit('this is a commit message')

        co = aset_samples_initialized_repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(KeyError):
            aset['1']
        co.close()

    def test_remove_data(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.columns['writtenaset']['2'] = array5by7 + 1
        co.columns['writtenaset']['3'] = array5by7 + 2
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        del co.columns['writtenaset']['1']
        del co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        with pytest.raises(KeyError):
            co.columns['writtenaset']['1']
        with pytest.raises(KeyError):
            co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        assert_equal(co.columns['writtenaset']['2'], array5by7 + 1)
        co.close()

    def test_remove_data_multiple_items(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.columns['writtenaset']['2'] = array5by7 + 1
        co.columns['writtenaset']['3'] = array5by7 + 2
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        with pytest.raises(KeyError):
            del co.columns['writtenaset'][('1', '3')]
        assert '1' in co.columns['writtenaset']
        assert '3' in co.columns['writtenaset']
        del co.columns['writtenaset']['1']
        del co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        with pytest.raises(KeyError):
            co.columns['writtenaset']['1']
        with pytest.raises(KeyError):
            co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        assert_equal(co.columns['writtenaset']['2'], array5by7 + 1)
        co.close()

    def test_pop_data(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.columns['writtenaset']['2'] = array5by7 + 1
        co.columns['writtenaset']['3'] = array5by7 + 2
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        res = co.columns['writtenaset'].pop('1')
        assert_equal(res, array5by7)

        aset = co.columns['writtenaset']
        res = aset.pop('3')
        assert_equal(res, array5by7 + 2)

        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        with pytest.raises(KeyError):
            co.columns['writtenaset']['1']
        with pytest.raises(KeyError):
            co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        assert_equal(co.columns['writtenaset']['2'], array5by7 + 1)
        co.close()

    def test_pop_data_multiple_items(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        co.columns['writtenaset']['2'] = array5by7 + 1
        co.columns['writtenaset']['3'] = array5by7 + 2
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 3
        with pytest.raises(TypeError):
            co.columns['writtenaset'].pop('1', '3')
        res = co.columns['writtenaset'].pop('1')
        assert_equal(res, array5by7)
        res = co.columns['writtenaset'].pop('3')
        assert_equal(res, array5by7 + 2)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        with pytest.raises(KeyError):
            co.columns['writtenaset']['1']
        with pytest.raises(KeyError):
            co.columns['writtenaset']['3']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        assert_equal(co.columns['writtenaset']['2'], array5by7 + 1)
        co.close()

    def test_remove_all_data(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        new_array = np.zeros_like(array5by7)
        co.columns['writtenaset']['2'] = new_array
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 2
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 2
        del co.columns['writtenaset']['1']
        del co.columns['writtenaset']['2']
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 0

        wset = co.columns['writtenaset']
        del co.columns['writtenaset']

        assert len(co.columns) == 0
        with pytest.raises(KeyError):
            len(co.columns['writtenaset'])
        co.commit('this is a commit message')
        co.close()

        # recreating same and verifying
        co = aset_samples_initialized_repo.checkout(write=True)
        assert len(co.columns) == 0
        co.add_ndarray_column('writtenaset', prototype=array5by7)
        co.columns['writtenaset']['1'] = array5by7
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout()
        assert_equal(co.columns['writtenaset']['1'], array5by7)
        assert len(co.columns) == 1
        assert len(co.columns['writtenaset']) == 1
        co.close()

    def test_remove_data_nonexistant_sample_key_raises(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset']['1'] = array5by7
        new_array = np.zeros_like(array5by7)
        co.columns['writtenaset']['2'] = new_array
        co.columns['writtenaset']['3'] = new_array + 5
        with pytest.raises(KeyError):
            del co.columns['writtenaset']['doesnotexist']
        co.commit('this is a commit message')
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    def test_multiple_columns_single_commit(
            self, aset1_backend, aset2_backend, aset_samples_initialized_repo, randomsizedarray
    ):
        co = aset_samples_initialized_repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', prototype=randomsizedarray, backend=aset2_backend)
        aset1['arr'] = randomsizedarray
        aset2['arr'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = aset_samples_initialized_repo.checkout()
        assert_equal(co.columns['aset1']['arr'], randomsizedarray)
        assert_equal(co.columns['aset2']['arr'], randomsizedarray)
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    def test_prototype_and_shape(self, aset1_backend, aset2_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset1 = co.add_ndarray_column('aset1', prototype=randomsizedarray, backend=aset1_backend)
        aset2 = co.add_ndarray_column('aset2', shape=randomsizedarray.shape, dtype=randomsizedarray.dtype, backend=aset2_backend)

        newarray = np.random.random(randomsizedarray.shape).astype(randomsizedarray.dtype)
        aset1['arr1'] = newarray
        aset2['arr'] = newarray
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert_equal(co.columns['aset1']['arr1'], newarray)
        assert_equal(co.columns['aset2']['arr'], newarray)
        co.close()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=randomsizedarray)
        with pytest.raises(TypeError):
            aset[randomsizedarray]

        aset_no_name = co.add_ndarray_column('aset_no_name', prototype=randomsizedarray)
        added = aset_no_name.append(randomsizedarray)
        assert_equal(next(aset_no_name.values()), randomsizedarray)
        assert_equal(aset_no_name[added], randomsizedarray)
        co.close()

    def test_append_samples(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=randomsizedarray)
        with pytest.raises((ValueError, TypeError)):
            aset[randomsizedarray]

        aset_no_name = co.add_ndarray_column('aset_no_name', prototype=randomsizedarray)
        generated_key = aset_no_name.append(randomsizedarray)
        assert generated_key in aset_no_name
        assert len(aset_no_name) == 1
        assert_equal(aset_no_name[generated_key], randomsizedarray)
        co.close()

    def test_different_data_types_and_shapes(self, repo):
        co = repo.checkout(write=True)
        shape = (2, 3)
        dtype = np.int
        another_dtype = np.float64
        another_shape = (3, 4)
        arr = np.random.random(shape).astype(dtype)
        aset = co.add_ndarray_column('aset', shape=shape, dtype=dtype)
        aset['1'] = arr

        newarr = np.random.random(shape).astype(another_dtype)
        with pytest.raises(ValueError):
            aset['2'] = newarr

        newarr = np.random.random(another_shape).astype(dtype)
        with pytest.raises(ValueError):
            aset['3'] = newarr
        co.close()

    def test_add_sample_with_non_numpy_array_data_fails(self, aset_samples_initialized_repo):
        co = aset_samples_initialized_repo.checkout(write=True)
        with pytest.raises(ValueError, match='`data` argument type'):
            co.columns['writtenaset'][1] = [[1, 2, 3, 4, 5, 6, 7] for i in range(5)]
        co.close()

    def test_add_sample_with_fortran_order_data_fails(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        with pytest.raises(ValueError, match='`data` must be "C" contiguous array.'):
            co.columns['writtenaset'][1] = np.asfortranarray(array5by7)
        co.close()

    def test_add_sample_with_dimension_rank_fails(self, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', shape=(2, 3), dtype=np.float32, variable_shape=True)
        arr = np.random.randn(2, 3, 2).astype(np.float32)
        with pytest.raises(ValueError, match='data rank 3 != aset rank 2'):
            aset[1] = arr
        co.close()

    def test_add_sample_with_dimension_exceeding_max_fails(self, repo):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', shape=(2, 3), dtype=np.float32, variable_shape=True)
        arr = np.random.randn(2, 4).astype(np.float32)
        with pytest.raises(ValueError, match='exceeds schema max'):
            aset[1] = arr
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_writer_context_manager_column_add_sample(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=randomsizedarray, backend=aset_backend)
        with co.columns['aset'] as aset:
            aset['1'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert_equal(co.columns['aset']['1'], randomsizedarray)
        co.close()

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_column_context_manager_aset_sample_add(self, aset_backend, repo, randomsizedarray):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=randomsizedarray, backend=aset_backend)
        with co.columns['aset'] as aset:
            aset['1'] = randomsizedarray
            aset['2'] = randomsizedarray + 1
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert_equal(co.columns['aset']['1'], randomsizedarray)
        assert np.allclose(co.columns['aset'].get('2'), randomsizedarray + 1)
        co.close()

    def test_writer_column_properties_are_correct(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        assert co.columns.iswriteable is True
        d = co.columns['writtenaset']
        assert d.column =='writtenaset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.schema_type == 'fixed_shape'
        assert d.iswriteable is True
        assert d.backend == '01'
        assert isinstance(d.backend_options, dict)
        assert len(d.backend_options) > 0
        assert d.contains_subsamples is False
        assert d.remote_reference_keys == ()
        assert d.contains_remote_references is False
        co.close()

    def test_reader_column_properties_are_correct(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=False)
        assert co.columns.iswriteable is False
        d = co.columns['writtenaset']
        assert d.column =='writtenaset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.schema_type == 'fixed_shape'
        assert d.iswriteable is False
        assert d.backend == '01'
        assert isinstance(d.backend_options, dict)
        assert len(d.backend_options) > 0
        assert d.contains_subsamples is False
        assert d.remote_reference_keys == ()
        assert d.contains_remote_references is False

    def test_iter_column_samples_yields_keys(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.columns['writtenaset'][0] = array5by7
        new_array = np.zeros_like(array5by7)
        co.columns['writtenaset'][1] = new_array
        co.columns['writtenaset'][2] = new_array + 5

        for idx, sname in enumerate(iter(co.columns['writtenaset'])):
            assert sname == idx
        assert idx == 2
        co.close()

    def test_iter_columns_yields_aset_names(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout(write=True)
        for k in iter(co.columns):
            assert k in ['second_aset', 'writtenaset']
        co.close()

    def test_set_item_column_fails(self, aset_samples_initialized_repo):
        co = aset_samples_initialized_repo.checkout(write=True)
        with pytest.raises(AttributeError):
            co.columns['newaset'] = co.columns['writtenaset']
        co.close()


class TestVariableSizedColumn(object):

    @pytest.mark.parametrize(
        'test_shapes,max_shape',
        [[[(2, 5), (1, 10), (10, 1), (5, 2)], (10, 10)],
         [[(10,), (10,)], (10,)],
         [[(3, 3, 3), (27, 1, 1), (1, 27, 1), (1, 1, 27), (3, 9, 1), (9, 3, 1), (1, 3, 9), (1, 9, 3)], (27, 27, 27)]])
    @pytest.mark.parametrize("dtype1", [np.uint8, np.float32, np.int32])
    @pytest.mark.parametrize("dtype2", [np.uint8, np.float32, np.int32])
    @pytest.mark.parametrize('backend1', variable_shape_backend_params)
    @pytest.mark.parametrize('backend2', variable_shape_backend_params)
    def test_write_all_zeros_same_size_different_shape_does_not_store_as_identical_hashs(
        self, repo, test_shapes, max_shape, dtype1, dtype2, backend1, backend2
    ):
        wco = repo.checkout(write=True)
        aset1 = wco.add_ndarray_column('aset1', shape=max_shape, dtype=dtype1, variable_shape=True, backend=backend1)
        aset2 = wco.add_ndarray_column('aset2', shape=max_shape, dtype=dtype2, variable_shape=True, backend=backend2)

        arrdict1, arrdict2 = {}, {}
        for idx, shape in enumerate(test_shapes):
            arr1 = np.zeros(shape, dtype=dtype1)
            arr2 = np.zeros(shape, dtype=dtype2)
            arrdict1[idx] = arr1
            arrdict2[idx] = arr2
            aset1[idx] = arr1
            aset2[idx] = arr2

        for k, v in arrdict1.items():
            # make sure they are good before committed
            res = aset1[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)
        for k, v in arrdict2.items():
            # make sure they are good before committed
            res = aset2[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)

        wco.commit('first')

        for k, v in arrdict1.items():
            # make sure they are good before committed
            res = aset1[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)
        for k, v in arrdict2.items():
            # make sure they are good before committed
            res = aset2[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)

        wco.close()
        rco = repo.checkout()
        naset1 = rco.columns['aset1']
        naset2 = rco.columns['aset2']

        for k, v in arrdict1.items():
            # make sure they are good before committed
            res = naset1[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)
        for k, v in arrdict2.items():
            # make sure they are good before committed
            res = naset2[k]
            assert res.dtype == v.dtype
            assert res.shape == v.shape
            assert_equal(res, v)
        rco.close()

    @pytest.mark.parametrize(
        'test_shapes,shape',
        [[[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
         [[(10,), (1,), (5,)], (10,)],
         [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', variable_shape_backend_params)
    def test_writer_can_create_variable_size_column(
        self, aset_samples_initialized_repo, dtype, test_shapes, shape, backend
    ):
        repo = aset_samples_initialized_repo
        wco = repo.checkout(write=True)
        wco.add_ndarray_column('varaset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        d = wco.columns['varaset']

        arrdict = {}
        for idx, shape in enumerate(test_shapes):
            arr = (np.random.random_sample(shape) * 10).astype(dtype)
            arrdict[str(idx)] = arr
            d[str(idx)] = arr

        for k, v in arrdict.items():
            # make sure they are good before committed
            assert_equal(d[k], v)

        wco.commit('first')

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert_equal(d[k], v)
        wco.close()

    @pytest.mark.parametrize('test_shapes,shape', [
        [[(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
        [[(10,), (1,), (5,)], (10,)],
        [[(100, 100, 100), (100, 100, 1), (100, 1, 100), (1, 100, 100), (1, 1, 1), (34, 6, 3)], (100, 100, 100)]
    ])
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    @pytest.mark.parametrize('backend', variable_shape_backend_params)
    def test_reader_recieves_expected_values_for_variable_size_column(
        self, aset_samples_initialized_repo, dtype, test_shapes, shape, backend
    ):
        repo = aset_samples_initialized_repo
        wco = repo.checkout(write=True)
        wco.add_ndarray_column('varaset', shape=shape, dtype=dtype, variable_shape=True, backend=backend)
        wd = wco.columns['varaset']

        arrdict = {}
        for idx, shape in enumerate(test_shapes):
            arr = (np.random.random_sample(shape) * 10).astype(dtype)
            arrdict[str(idx)] = arr
            wd[str(idx)] = arr

        for k, v in arrdict.items():
            # make sure they are good before committed
            assert_equal(wd[k], v)

        wco.commit('first')
        rco = repo.checkout()
        rd = rco.columns['varaset']

        for k, v in arrdict.items():
            # make sure they can work after commit
            assert_equal(wd[k], v)
            assert_equal(rd[k], v)
        wco.close()
        rco.close()

    @pytest.mark.parametrize('aset_specs', [
        [['aset1', [(10, 10), (1, 10), (2, 2), (3, 5), (1, 1), (10, 1)], (10, 10)],
         ['aset2', [(10,), (1,), (5,)], (10,)]],
        [['aset1', [(100, 100), (1, 100), (20, 20), (30, 50), (1, 10), (10, 1)], (100, 100)],
         ['aset2', [(100,), (1,), (50,)], (100,)]]])
    @pytest.mark.parametrize('backends', permutations(variable_shape_backend_params, 2))
    @pytest.mark.parametrize('dtype', [np.float32, np.uint8])
    def test_writer_reader_can_create_read_multiple_variable_size_column(
        self, aset_samples_initialized_repo, aset_specs, backends, dtype
    ):
        repo = aset_samples_initialized_repo
        wco = repo.checkout(write=True)
        arrdict = {}
        for backend, aset_spec in zip(backends, aset_specs):
            aset_name, test_shapes, max_shape = aset_spec
            wco.add_ndarray_column(
                aset_name, shape=max_shape, dtype=dtype, variable_shape=True, backend=backend)

            arrdict[aset_name] = {}
            for idx, shape in enumerate(test_shapes):
                arr = (np.random.random_sample(shape) * 10).astype(dtype)
                arrdict[aset_name][str(idx)] = arr
                wco.columns[aset_name][str(idx)] = arr

        for aset_k in arrdict.keys():
            for samp_k, v in arrdict[aset_k].items():
                # make sure they are good before committed
                assert_equal(wco.columns[aset_k][samp_k], v)

        wco.commit('first')
        rco = repo.checkout()

        for aset_k in arrdict.keys():
            for samp_k, v in arrdict[aset_k].items():
                # make sure they are good before committed
                assert_equal(wco.columns[aset_k][samp_k], v)
                assert_equal(rco.columns[aset_k][samp_k], v)
        wco.close()
        rco.close()

    def test_writer_column_properties_are_correct(self, aset_samples_var_shape_initialized_repo):
        co = aset_samples_var_shape_initialized_repo.checkout(write=True)
        d = co.columns['writtenaset']
        assert d.column =='writtenaset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.schema_type == 'variable_shape'
        assert d.iswriteable is True
        assert d.backend in variable_shape_backend_params
        assert isinstance(d.backend_options, dict)
        assert d.contains_subsamples is False
        assert d.remote_reference_keys == ()
        assert d.contains_remote_references is False
        co.close()

    def test_reader_column_properties_are_correct(self, aset_samples_var_shape_initialized_repo):
        co = aset_samples_var_shape_initialized_repo.checkout(write=False)
        d = co.columns['writtenaset']
        assert d.column =='writtenaset'
        assert d.dtype == np.float64
        assert np.allclose(d.shape, (10, 10))
        assert d.schema_type == 'variable_shape'
        assert d.iswriteable is False
        assert d.backend in variable_shape_backend_params
        assert isinstance(d.backend_options, dict)
        assert d.contains_subsamples is False
        assert d.remote_reference_keys == ()
        assert d.contains_remote_references is False
        co.close()


class TestMultiprocessColumnReads(object):

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_external_multi_process_pool(self, repo, backend):
        from multiprocessing import get_context

        masterCmtList = []
        co = repo.checkout(write=True)
        co.add_ndarray_column(name='writtenaset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        for cIdx in range(2):
            if cIdx != 0:
                co = repo.checkout(write=True)
            with co.columns['writtenaset'] as d:
                kstart = 20 * cIdx
                for sIdx in range(20):
                    arr = np.random.randn(20, 20).astype(np.float32) * 100
                    sName = str(sIdx + kstart)
                    d[sName] = arr
                    masterSampList.append(arr)
            assert d.backend == backend
            cmt = co.commit(f'master commit number: {cIdx}')
            masterCmtList.append((cmt, list(masterSampList)))
            co.close()

        cmtIdx = 0
        for cmt, sampList in masterCmtList:
            nco = repo.checkout(write=False, commit=cmt)
            ds = nco.columns['writtenaset']
            keys = [str(i) for i in range(20 + (20*cmtIdx))]
            with get_context().Pool(2) as P:
                cmtData = P.map(ds.get, keys)
            for data, sampData in zip(cmtData, sampList):
                assert_equal(data, sampData) is True
            cmtIdx += 1
            nco.close()

    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_external_multi_process_pool_fails_on_write_enabled_checkout(self, repo, backend):
        from multiprocessing import get_context

        co = repo.checkout(write=True)
        co.add_ndarray_column(name='writtenaset', shape=(20, 20), dtype=np.float32, backend=backend)
        with co.columns['writtenaset'] as d:
            for sIdx in range(20):
                d[sIdx] = np.random.randn(20, 20).astype(np.float32) * 100
        assert d.backend == backend
        co.commit(f'master commit number 1')
        co.close()

        nco = repo.checkout(write=True)
        ds = nco.columns['writtenaset']
        keys = [i for i in range(20)]
        with pytest.raises(PermissionError):
            with get_context().Pool(2) as P:
                cmtData = P.map(ds.get, keys)
        nco.close()


    @pytest.mark.parametrize('backend', fixed_shape_backend_params)
    def test_multiprocess_get_succeeds_on_superset_and_subset_of_keys(self, repo, backend):
        from multiprocessing import get_context

        co = repo.checkout(write=True)
        co.add_ndarray_column(name='writtenaset', shape=(20, 20), dtype=np.float32, backend=backend)
        masterSampList = []
        with co.columns['writtenaset'] as d:
            for sIdx in range(20):
                arr = np.random.randn(20, 20).astype(np.float32) * 100
                d[sIdx] = arr
                masterSampList.append(arr)
            assert d.backend == backend
        cmt = co.commit(f'master commit number one')
        co.close()

        nco = repo.checkout(write=False, commit=cmt)
        ds = nco.columns['writtenaset']

        # superset of keys fails
        keys = [i for i in range(24)]
        with get_context().Pool(2) as P:
            cmtData = P.map(ds.get, keys)
        for idx, data in enumerate(cmtData):
            if idx >= 20:
                assert data is None
            else:
                assert_equal(data, masterSampList[idx])

        # subset of keys works
        keys = [i for i in range(10, 20)]
        with get_context().Pool(2) as P:
            cmtData = P.map(ds.get, keys)
        for idx, data in enumerate(cmtData):
            assert_equal(data, masterSampList[10+idx])
        nco.close()

    def test_writer_iterating_over_keys_can_have_additions_made_no_error(self, two_commit_filled_samples_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with aset as ds:
            for idx, k in enumerate(ds.keys()):
                if idx == 0:
                    ds['1232'] = np.random.randn(5, 7).astype(np.float32)
                assert '1232' != k

        added_key_exists_on_later_iteration = False
        for k in aset.keys():
            if k == '1232':
                added_key_exists_on_later_iteration = True
                break
        assert added_key_exists_on_later_iteration is True
        co.close()

    def test_writer_iterating_over_values_can_have_additions_made_no_error(self, two_commit_filled_samples_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        aset = co.columns['writtenaset']
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
            for idx, v in enumerate(ds.values()):
                if idx == 0:
                    ds['1232'] = mysample
                assert not np.allclose(v, mysample)

        added_value_exists_on_later_iteration = False
        for v in aset.values():
            if np.allclose(v, mysample):
                added_value_exists_on_later_iteration = True
                break
        assert added_value_exists_on_later_iteration is True
        co.close()

    def test_writer_iterating_over_items_can_have_additions_made_no_error(self, two_commit_filled_samples_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        aset = co.columns['writtenaset']
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
            for idx, kv in enumerate(ds.items()):
                if idx == 0:
                    ds['1232'] = mysample
                k, v = kv
                assert not np.allclose(v, mysample)
                assert k != '1232'

        added_value_exists_on_later_iteration = False
        for k, v in aset.items():
            if (k == '1232') and np.allclose(v, mysample):
                added_value_exists_on_later_iteration = True
                break
        assert added_value_exists_on_later_iteration is True
        co.close()

    def test_reader_iterating_over_items_can_not_make_additions(self, two_commit_filled_samples_repo):
        # do not want ``RuntimeError dictionary changed size during iteration``

        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=False)
        aset = co.columns['writtenaset']
        mysample = np.random.randn(5, 7).astype(np.float32)
        with aset as ds:
            for idx, kv in enumerate(ds.items()):
                if idx == 0:
                    with pytest.raises(TypeError):
                        ds['1232'] = mysample
                k, v = kv
                assert not np.allclose(v, mysample)
                assert k != '1232'

        assert '1232' not in aset
        co.close()
