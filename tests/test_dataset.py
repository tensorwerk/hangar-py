import pytest
import numpy as np


class TestDataset:

    def test_read_only_mode(self, written_repo):
        co = written_repo.checkout()
        # TODO: how to check whether it is read only mode
        assert co.datasets.init_dataset is None
        assert co.datasets.remove_dset is None
        assert len(co.datasets['_dset']) == 0

    def test_remove_dataset(self, written_repo):
        co = written_repo.checkout(write=True)
        co.datasets.remove_dset('_dset')
        with pytest.raises(KeyError):
            co.datasets.remove_dset('_dset')
        co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float64)
        assert len(co.datasets) == 1
        co.datasets.remove_dset('_dset')
        co.commit()
        co.close()
        co = written_repo.checkout(write=True)
        co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float64)
        assert len(co.datasets) == 1
        # TODO: removing all datasets removes indexing. test that as well

    def init_again(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        co.datasets.init_dataset('dset', prototype=randomsizedarray)
        # TODO: shouldn't it raise an exception instead of printing
        with pytest.raises(ValueError):
            co.datasets.init_dataset('dset', prototype=randomsizedarray)
        co.close()

    def dataset_with_more_dimension(self, repo):
        co = repo.checkout(write=True)
        shape = (0, 1, 2)
        # TODO: shouldn't it be some other error
        with pytest.raises(ZeroDivisionError):
            co.datasets.init_dataset('dset', shape=shape, dtype=np.int)
        shape = [1] * 31
        dset = co.datasets.init_dataset('dset', shape=shape, dtype=np.int)
        assert len(dset._schema_max_shape) == 31
        shape = [1] * 32
        # TODO: shouldn't it raise a valueerror instead of printing
        with pytest.raises(ValueError):
            co.datasets.init_dataset('dset', shape=shape, dtype=np.int)
        co.close()


class TestDataWithFixedSizedDataset:

    def test_add_data(self, w_checkout, array5by7):
        dset = w_checkout.datasets['_dset']
        # TODO shouldn't this raise keyerror
        dset['somerandomkey']

        # should raise an issue about intiger being used as key
        dset[1] = array5by7

    def test_add_with_wrong_argument_order(self, w_checkout, array5by7):
        # TODO: perhaps use it with written repo fixture
        dset = w_checkout.datasets['_dset']

        # TODO: shouldn't it throw some error
        dset.add('1', array5by7)

    def test_multiple_data_single_commit(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.commit()
        co.close()
        co = written_repo.checkout()
        dset = co.datasets['_dset']
        assert len(dset) == 2
        assert list(dset.keys()) == ['1', '2']
        assert (dset['1'] == array5by7).all()

    def multiple_data_multiple_commit(self, written_repo, array5by7):
        # TODO: enable once #5 is fixed
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        co.commit()
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.close()
        new_new_array = new_array + 5
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['3'] = new_new_array
        co.commit()
        co.close()
        co = written_repo.checkout()
        dset = co.datasets['_dset']
        assert (dset['1'] == array5by7).all()
        assert (dset['2'] == new_array).all()
        assert (dset['3'] == new_new_array).all()

    def test_added_but_not_commited(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        co.close()
        with pytest.raises(PermissionError):
            # TODO: this raise an excpetion
            # AttributeError: 'WriterCheckout' object has no
            #     attribute '_WriterCheckout__writer_lock'
            co.commit()
        co = written_repo.checkout()
        dset = co.datasets['_dset']
        # TODO: Shouldn't it be something else
        assert dset['1'] is False

    def remove_data(self, written_repo, array5by7):
        # TODO: test failing
        # 'DatasetDataWriter' object has no attribute 'TxnRegister'
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.commit()
        co.close()
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].remove('1')
        co.commit()
        co.close()
        co = written_repo.checkout()
        assert co.datasets['_dset']['1'] is False
        assert (co.datasets['_dset']['2'] == new_array).all()

    def remove_all_data(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].add(array5by7, '1')
        new_array = np.zeros_like(array5by7)
        co.datasets['_dset']['2'] = new_array
        co.commit()
        co.close()
        co = written_repo.checkout(write=True)
        co.datasets['_dset'].remove('1')
        # TODO: deleting all elementes supposed to delete dset - Not happening
        # TODO: adding another element and deleting it again raises
        # AttributeError: 'NoneType' object has no attribute 'decode'
        co.datasets['_dset'].remove('2')
        # TODO > check
        # INFO:hangar.checkout:No changes made to the repository. Cannot commit
        co.commit()
        co.close()
        co = written_repo.checkout()
        assert co.datasets['_dset'] is False

    def multiple_datasets_single_commit(
            self, written_repo, randomsizedarray):
        # TODO: enable once #5 is fixed
        co = written_repo.checkout(write=True)
        dset1 = co.datasets.init_dataset('dset1', prototype=randomsizedarray)
        dset2 = co.datasets.init_dataset('dset2', prototype=randomsizedarray)
        dset1['arr'] = randomsizedarray
        dset2['arr'] = randomsizedarray
        co.commit()
        co.close()
        co = written_repo.checkout()
        assert (co.datasets['dset1']['arr'] == randomsizedarray).all()
        assert (co.datasets['dset2']['arr'] == randomsizedarray).all()

    def prototype_and_shape(self, repo, randomsizedarray):
        # TODO: enable once #5 is fixed
        co = repo.checkout(write=True)
        dset1 = co.datasets.init_dataset('dset1', prototype=randomsizedarray)
        dset2 = co.datasets.init_dataset(
            'dset2', shape=randomsizedarray.shape,
            dtype=randomsizedarray.dtype)
        newarray = np.random.random(
            randomsizedarray.shape).astype(dtype=randomsizedarray.dtype)
        dset1['arr1'] = newarray
        dset2['arr'] = newarray
        co.commit()
        co.close()
        co = repo.checkout()
        assert (co.datasets['dset1']['arr'] == newarray).all()
        assert (co.datasets['dset2']['arr'] == newarray).all()

    def test_samples_without_name(self, repo, randomsizedarray):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset(
            'dset', prototype=randomsizedarray)
        # TODO: It should raise error
        dset.add(randomsizedarray)
        dset_no_name = co.datasets.init_dataset(
            'dset_no_name', prototype=randomsizedarray,
            samples_are_named=False)
        dset_no_name.add(randomsizedarray)
        assert (next(dset_no_name.values()) == randomsizedarray).all()
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
        # TODO: shouldn't it raise an excpetion
        dset['2'] = newarr
        newarr = np.random.random(another_shape).astype(dtype)
        # TODO: should raise an error
        dset['3'] = newarr

    def test_bulk_add_names(self):
        pass


class TestVariableSizedDataset:
    pass
