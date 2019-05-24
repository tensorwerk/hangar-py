import pytest
import numpy as np


class TestDataset(object):

    def test_invalid_dsetname(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        with pytest.raises(ValueError):
            co.datasets.init_dataset(name='invalid name', prototype=randomsizedarray)

    def test_read_only_mode(self, written_repo):
        import hangar
        co = written_repo.checkout()
        assert isinstance(co, hangar.checkout.ReaderCheckout)
        assert co.datasets.init_dataset is None
        assert co.datasets.remove_dset is None
        assert len(co.datasets['_dset']) == 0
        co.close()

    def test_get_dataset(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)

        # getting the dataset with `get`
        dsetOld = co.datasets.get('_dset')
        dsetOldPath = dsetOld._path
        dsetOldDsetn = dsetOld._dsetn
        dsetOldDefaultSchemaHash = dsetOld._default_schema_hash
        dsetOldSchemaUUID = dsetOld._schema_uuid

        dsetOld.add(array5by7, '1')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()

        # getting dataset with dictionary like style method
        dsetNew = co.datasets['_dset']

        assert np.allclose(dsetNew['1'], array5by7)
        assert dsetOldPath == dsetNew._path
        assert dsetOldDsetn == dsetNew._dsetn
        assert dsetOldDefaultSchemaHash == dsetNew._default_schema_hash
        assert dsetOldSchemaUUID == dsetNew._schema_uuid

    def test_remove_dataset(self, written_repo):
        co = written_repo.checkout(write=True)
        co.datasets.remove_dset('_dset')
        with pytest.raises(KeyError):
            co.datasets.remove_dset('_dset')

        co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float64)
        assert len(co.datasets) == 1
        co.datasets.remove_dset('_dset')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        assert len(co.datasets) == 0

        co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float64)
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout(write=True)
        assert len(co.datasets) == 1
        del co.datasets['_dset']
        assert len(co.datasets) == 0
        co.commit('this is a commit message')
        co.close()

    def test_init_again(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.datasets.init_dataset('dset', prototype=randomsizedarray)
        with pytest.raises(LookupError):
            co.datasets.init_dataset('dset', prototype=randomsizedarray)
        co.close()

    def test_dataset_with_more_dimension(self, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        with pytest.raises(ValueError):
            co.datasets.init_dataset('dset', shape=shape, dtype=np.int)
        shape = [1] * 31
        dset = co.datasets.init_dataset('dset1', shape=shape, dtype=np.int)
        assert len(dset._schema_max_shape) == 31
        shape = [1] * 32
        with pytest.raises(ValueError):
            # maximum tensor rank must be <= 31
            co.datasets.init_dataset('dset2', shape=shape, dtype=np.int)
        co.close()

    def test_dataset_with_empty_dimension(self, repo):
        co = repo.checkout(write=True)
        arr = np.array(1, dtype=np.int64)
        dset = co.datasets.init_dataset('dset1', shape=(), dtype=np.int64)
        dset['1'] = arr
        co.commit('this is a commit message')
        dset = co.datasets.init_dataset('dset2', prototype=arr)
        dset['1'] = arr
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        dset1 = co.datasets['dset1']
        dset2 = co.datasets['dset2']
        assert np.allclose(dset1['1'], arr)
        assert np.allclose(dset2['1'], arr)


class TestDataWithFixedSizedDataset(object):

    def test_iterating_over(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        all_tensors = []
        dset1 = co.datasets.init_dataset('dset1', prototype=randomsizedarray)
        dset2 = co.datasets.init_dataset('dset2', shape=(2, 2), dtype=np.int)
        dset3 = co.datasets.init_dataset('dset3', shape=(3, 4), dtype=np.float32)

        with dset1, dset2, dset3:
            dset1['1'] = randomsizedarray
            dset1['2'] = np.zeros_like(randomsizedarray)
            dset1['3'] = np.zeros_like(randomsizedarray) + 5
            all_tensors.extend([dset1['1'], dset1['2'], dset1['3']])

            dset2['1'] = np.ones((2, 2), dtype=np.int)
            dset2['2'] = np.ones((2, 2), dtype=np.int) * 5
            dset2['3'] = np.zeros((2, 2), dtype=np.int)
            all_tensors.extend([dset2['1'], dset2['2'], dset2['3']])

            dset3['1'] = np.ones((3, 4), dtype=np.float32)
            dset3['2'] = np.ones((3, 4), dtype=np.float32) * 7
            dset3['3'] = np.zeros((3, 4), dtype=np.float32)
            all_tensors.extend([dset3['1'], dset3['2'], dset3['3']])

        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        # iterating over .items()
        tensors_in_the_order = iter(all_tensors)
        for dname, dset in co.datasets.items():
            assert dset._dsetn == dname
            for sname, sample in dset.items():
                assert np.allclose(sample, next(tensors_in_the_order))

        # iterating over .keys()
        tensors_in_the_order = iter(all_tensors)
        for dname in co.datasets.keys():
            for sname in co.datasets[dname].keys():
                assert np.allclose(co.datasets[dname][sname], next(tensors_in_the_order))

        # iterating over .values()
        tensors_in_the_order = iter(all_tensors)
        for dset in co.datasets.values():
            for sample in dset.values():
                assert np.allclose(sample, next(tensors_in_the_order))

    def test_get_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(
            co.datasets['_dset']['1'],
            co.datasets.get('_dset').get('1'),
            array5by7)

    def test_add_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        dset = co.datasets['_dset']
        with pytest.raises(KeyError):
            dset['somerandomkey']

        with pytest.raises(ValueError):
            dset[1] = array5by7
        dset['1'] = array5by7
        dset.add(array5by7, '2')
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(
            co.datasets['_dset']['1'],
            co.datasets['_dset']['2'])

    def test_add_with_wrong_argument_order(self, w_checkout, array5by7):
        dset = w_checkout.datasets['_dset']
        with pytest.raises(ValueError):
            dset.add('1', array5by7)

    def test_multiple_data_single_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dset = co.datasets['_dset']
        assert len(dset) == 2
        assert list(dset.keys()) == ['1', '2']
        assert np.allclose(dset['1'], array5by7)
        co.close()

    def test_multiple_data_multiple_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        co.commit('this is a commit message')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.close()

        new_new_array = new_array + 5
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['3'] = new_new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        dset = co.datasets['_dset']
        assert np.allclose(dset['1'], array5by7)
        assert np.allclose(dset['2'], new_array)
        assert np.allclose(dset['3'], new_new_array)
        co.close()

    def test_added_but_not_commited(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        co.close()

        with pytest.raises(PermissionError):
            co.commit('this is a commit message')

        co = written_repo.checkout()
        dset = co.datasets['_dset']
        with pytest.raises(KeyError):
            dset['1']
        co.close()

    def test_remove_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.datasets['_dset']['3'] = new_array + 5
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.datasets['_dset'].remove('1')
        del co.datasets['_dset']['3']
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            co.datasets['_dset']['1']
        with pytest.raises(KeyError):
            co.datasets['_dset']['3']
        assert len(co.datasets['_dset']) == 1
        assert np.allclose(co.datasets['_dset']['2'], new_array)
        co.close()

    def test_remove_all_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True)
        co.datasets['_dset'].remove('1')
        co.datasets['_dset'].remove('2')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout()
        with pytest.raises(KeyError):
            # removal of all data removes the dataset
            co.datasets['_dset']
        co.close()

        # recreating same and verifying
        co = written_repo.checkout(write=True)
        co.datasets.init_dataset('_dset', prototype=array5by7)
        co.datasets['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datasets['_dset']['1'], array5by7)

    def test_multiple_datasets_single_commit(self, written_repo, randomsizedarray):
        co = written_repo.checkout(write=True)
        dset1 = co.datasets.init_dataset('dset1', prototype=randomsizedarray)
        dset2 = co.datasets.init_dataset('dset2', prototype=randomsizedarray)
        dset1['arr'] = randomsizedarray
        dset2['arr'] = randomsizedarray
        co.commit('this is a commit message')
        co.close()
        co = written_repo.checkout()
        assert np.allclose(co.datasets['dset1']['arr'], randomsizedarray)
        assert np.allclose(co.datasets['dset2']['arr'], randomsizedarray)
        co.close()

    def test_prototype_and_shape(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset1 = co.datasets.init_dataset('dset1', prototype=randomsizedarray)
        dset2 = co.datasets.init_dataset(
            'dset2', shape=randomsizedarray.shape, dtype=randomsizedarray.dtype)
        newarray = np.random.random(randomsizedarray.shape).astype(randomsizedarray.dtype)
        dset1['arr1'] = newarray
        dset2['arr'] = newarray
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.datasets['dset1']['arr1'], newarray)
        assert np.allclose(co.datasets['dset2']['arr'], newarray)
        co.close()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=randomsizedarray)
        with pytest.raises(ValueError):
            dset.add(randomsizedarray)

        dset_no_name = co.datasets.init_dataset(
            'dset_no_name',
            prototype=randomsizedarray,
            samples_are_named=False)
        dset_no_name.add(randomsizedarray)
        assert np.allclose(next(dset_no_name.values()), randomsizedarray)
        co.close()

    def test_different_data_types_and_shapes(self, repo):
        co = repo.checkout(write=True)
        shape = (2, 3)
        dtype = np.int
        another_dtype = np.float64
        another_shape = (3, 4)
        arr = np.random.random(shape).astype(dtype)
        dset = co.datasets.init_dataset('dset', shape=shape, dtype=dtype)
        dset['1'] = arr

        newarr = np.random.random(shape).astype(another_dtype)
        with pytest.raises(ValueError):
            dset['2'] = newarr

        newarr = np.random.random(another_shape).astype(dtype)
        with pytest.raises(ValueError):
            dset['3'] = newarr
        co.close()

    def test_adding_same_data_again_with_same_name(self, repo, array5by7):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=array5by7)
        dset['1'] = array5by7
        with pytest.raises(LookupError):
            # raises before commit
            dset['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(LookupError):
            # raises after commit
            dset['1'] = array5by7
        co.close()
        co = repo.checkout(write=True)
        dset = co.datasets['dset']
        with pytest.raises(LookupError):
            # raises in another checkout
            dset['1'] = array5by7

    def test_writer_context_manager_dataset_add_sample(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=randomsizedarray)
        with co.datasets['dset'] as dset:
            dset.add(randomsizedarray, '1')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert np.allclose(co.datasets['dset']['1'], randomsizedarray)

    def test_writer_context_manager_metadata_add(self, repo):
        co = repo.checkout(write=True)
        with co.metadata as metadata:
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        assert co.metadata['key'] == 'val'

    def test_dataset_context_manager_dset_sample_and_metadata_add(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=randomsizedarray)
        with co.datasets['dset'] as dset:
            dset.add(randomsizedarray, '1')
            co.metadata['hello'] = 'world'
        with co.metadata as metadata:
            newarr = randomsizedarray + 1
            dset['2'] = newarr
            metadata.add('key', 'val')
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        assert np.allclose(co.datasets['dset']['1'], randomsizedarray)
        assert np.allclose(co.datasets['dset'].get('2'), newarr)
        assert co.metadata['key'] == 'val'
        assert co.metadata.get('hello') == 'world'

    def test_bulk_add(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.datasets.init_dataset(
            'dset_no_name1',
            prototype=randomsizedarray,
            samples_are_named=False)
        co.datasets.init_dataset(
            'dset_no_name2',
            prototype=randomsizedarray,
            samples_are_named=False)
        co.commit('this is a commit message')

        # dummy additino with wrong key
        with pytest.raises(KeyError):
            co.datasets.add(
                {
                    'dset_no_name2': randomsizedarray / 255,
                    'dummykey': randomsizedarray
                })
        # making sure above addition did not add partial data
        with pytest.raises(RuntimeError):
            co.commit('this is a commit message')

        # proper addition and verification
        co.datasets.add(
            {
                'dset_no_name1': randomsizedarray,
                'dset_no_name2': randomsizedarray / 255
            })
        co.commit('this is a commit message')
        co.close()

        co = repo.checkout()
        data1 = next(co.datasets['dset_no_name1'].values())
        data2 = next(co.datasets['dset_no_name2'].values())
        assert np.allclose(data1, randomsizedarray)
        assert np.allclose(data2, randomsizedarray / 255)

    def test_writer_dataset_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        d = co.datasets['_dset']
        assert d.name == '_dset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is True
        co.close()

    def test_reader_dataset_properties_are_correct(self, written_repo, array5by7):
        co = written_repo.checkout(write=False)
        d = co.datasets['_dset']
        assert d.name == '_dset'
        assert d.dtype == array5by7.dtype
        assert np.allclose(d.shape, array5by7.shape) is True
        assert d.variable_shape is False
        assert d.named_samples is True
        assert d.iswriteable is False

@pytest.mark.skip(reason='not implemented')
class TestVariableSizedDataset(object):
    pass
