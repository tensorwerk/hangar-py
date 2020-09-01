import sys

import numpy as np
import pytest
from torch.utils.data import DataLoader
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=DeprecationWarning)
    import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from hangar.dataset import make_numpy_dataset
from hangar.dataset import make_torch_dataset
from hangar.dataset import make_tensorflow_dataset
from hangar.dataset.common import HangarDataset


class TestInternalDatasetClass:

    def test_column_without_wrapping_list(self, repo_20_filled_samples, array5by7):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        second_col = co.columns['second_aset']
        dataset = HangarDataset((first_col, second_col))
        key1, key2 = dataset._keys[0]
        assert key1 == key2
        target = array5by7[:] = int(key1)
        assert np.allclose(dataset.index_get(0), target)
        co.close()

    def test_no_column(self):
        with pytest.raises(TypeError):
            HangarDataset([])

    def test_fails_on_write_enabled_columns(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        first_aset = co.columns['writtenaset']
        with pytest.raises(PermissionError):
            HangarDataset((first_aset,))
        co.close()

    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_columns_without_local_data_and_without_key_argument(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        from hangar.backends import backend_decoder

        # perform a mock for nonlocal data
        for k in co._columns._columns['writtenaset']._samples:
            co._columns._columns['writtenaset']._samples[k] = backend_decoder(b'50:daeaaeeaebv')
        col = co.columns['writtenaset']
        with pytest.raises(RuntimeError):
            HangarDataset((col,))

        # perform a mock for nonlocal data
        co = repo.checkout()
        template = backend_decoder(b'50:daeaaeeaebv')
        co._columns._columns['writtenaset']._samples['4'] = template
        col = co.columns['writtenaset']
        dataset = HangarDataset((col,))
        dataset_available_keys = dataset._keys
        assert len(dataset_available_keys) == 19
        assert '4' not in dataset_available_keys
        column_reported_local_keys = list(col.keys(local=True))
        for dset_avail_key in dataset_available_keys:
            assert dset_avail_key in column_reported_local_keys
        assert len(dataset_available_keys) == len(column_reported_local_keys)
        co.close()

    def test_columns_without_common_keys_and_without_key_argument(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout(write=True)
        first_col = co.columns['writtenaset']
        first_col['AnExtraKey'] = first_col['0']
        co.commit('added an extra key')
        co.close()
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        second_col = co.columns['second_aset']
        with pytest.raises(KeyError):
            HangarDataset((first_col, second_col))
        co.close()

    def test_keys_single_column_success(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        keys = ['1', '2', '3']
        dataset = HangarDataset((first_col,), keys=keys)
        assert dataset._keys == keys
        co.close()

    def test_keys_multiple_column_success(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        second_col = co.columns['second_aset']
        keys = [('1', '2'), ('2', '3'), ('3', '4')]
        dataset = HangarDataset((first_col, second_col), keys=keys)
        for i, key in enumerate(keys):
            data = dataset.index_get(i)
            assert np.allclose(data[0], first_col[key[0]])
            assert np.allclose(data[1], second_col[key[1]])
        co.close()

    def test_keys_nested_column_success(self, repo_20_filled_subsamples):
        co = repo_20_filled_subsamples.checkout()
        col1 = co['writtenaset']
        col2 = co['second_aset']
        keys = (((0, ...), (0, 1)), ((1, ...), (1, 4)))
        dataset = HangarDataset([col1, col2], keys=keys)
        data = dataset.index_get(1)
        assert tuple(data[0].keys()) == (4, 5, 6)
        assert np.allclose(data[1], col2[1][4])
        co.close()

    def test_keys_not_valid(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        keys = ['w', 'r', 'o', 'n', 'g']
        dataset = HangarDataset((first_col,), keys=keys)
        with pytest.raises(KeyError):
            dataset.index_get(1)
        co.close()

    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_keys_non_local(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        # perform a mock for nonlocal data
        from hangar.backends import backend_decoder
        template = backend_decoder(b'50:daeaaeeaebv')
        co._columns._columns['writtenaset']._samples['4'] = template

        col = co.columns['writtenaset']
        col_reported_remote_keys = col.remote_reference_keys
        assert col_reported_remote_keys == ('4',)
        assert len(col_reported_remote_keys) == 1
        dataset = HangarDataset((col,), keys=('0', *col_reported_remote_keys))
        with pytest.raises(KeyError):
            # TODO: hangar internal should raise FileNotFoundError?
            dataset.index_get(1)
        co.close()

# ====================================   Numpy    ====================================


@pytest.mark.filterwarnings("ignore:.* experimental method")
class TestNumpyDataset:
    def test_multiple_dataset_batched_loader(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        dset = make_numpy_dataset([first_aset, second_aset], batch_size=6, drop_last=True)
        total_samples = 0
        for dset1, dset2 in dset:
            total_samples += dset1.shape[0]
            assert dset1.shape == (6, 5, 7)
            assert dset2.shape == (6, 5, 7)
        assert total_samples == 18  # drop last is True
        co.close()

    def test_lots_of_data_with_multiple_backend(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        np_dset = make_numpy_dataset([aset], batch_size=10, drop_last=True)
        for data in np_dset:
            assert isinstance(data, np.ndarray)
            assert data.shape == (10, 5, 7)
        co.close()

    def test_shuffle(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']

        unshuffled_dataset = make_numpy_dataset((first_aset,),
                                                keys=[str(i) for i in range(15)],
                                                shuffle=False)
        expected_unshuffled_content = [i for i in range(15)]
        recieved_unshuffled_content = []
        for data in unshuffled_dataset:
            recieved_unshuffled_content.append(int(data[0][0]))
        assert expected_unshuffled_content == recieved_unshuffled_content

        shuffled_dataset = make_numpy_dataset((first_aset,),
                                              keys=[str(i) for i in range(15)],
                                              shuffle=True)
        recieved_shuffled_content = []
        for data in shuffled_dataset:
            recieved_shuffled_content.append(int(data[0][0]))
        assert recieved_shuffled_content != expected_unshuffled_content
        co.close()

    def test_collate_fn(self, repo_20_filled_subsamples):
        co = repo_20_filled_subsamples.checkout()
        col1 = co['writtenaset']
        col2 = co['second_aset']
        keys = (((0, ...), (0, 1)), ((1, ...), (1, 4)))

        dataset = make_numpy_dataset([col1, col2], keys=keys,
                                     shuffle=False, batch_size=2)
        col1data, col2data = next(iter(dataset))
        assert isinstance(col1data, tuple)
        assert isinstance(col2data, np.ndarray)
        assert list(col1data[0].keys()) == [1, 2, 3]
        assert list(col1data[1].keys()) == [4, 5, 6]
        assert np.allclose(col2data, np.stack((col2[0][1], col2[1][4])))

        def collate_fn(data_arr):
            arr1 = []
            arr2 = []
            for elem in data_arr:
                # picking one arbitrary subsample
                k = list(elem[0].keys())[2]
                data1 = elem[0][k]
                data2 = elem[1]
                arr1.append(data1)
                arr2.append(data2)
            return np.stack(arr1), np.stack(arr2)

        dataset = make_numpy_dataset([col1, col2], keys=keys, shuffle=False,
                                     batch_size=2, collate_fn=collate_fn)
        col1data, col2data = next(iter(dataset))
        assert np.allclose(col1data, np.stack((col1[0][3], col1[1][6])))
        assert np.allclose(col2data, np.stack((col2[0][1], col2[1][4])))
        co.close()


# ====================================   PyTorch  ====================================


class TestTorchDataset(object):

    def test_multiple_dataset_loader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        torch_dset = make_torch_dataset([first_aset, second_aset])
        loader = DataLoader(torch_dset, batch_size=6, drop_last=True)
        total_samples = 0
        for dset1, dset2 in loader:
            total_samples += dset1.shape[0]
            assert dset1.shape == (6, 5, 7)
            assert dset2.shape == (6, 5, 7)
        assert total_samples == 18  # drop last is True
        co.close()

    def test_return_as_dict(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        torch_dset = make_torch_dataset([first_aset, second_aset], as_dict=True)
        assert len(torch_dset) == 20
        loader = DataLoader(torch_dset, batch_size=5)
        for sample in loader:
            assert 'writtenaset' in sample.keys()
            assert 'second_aset' in sample.keys()
        co.close()

    def test_lots_of_data_with_multiple_backend(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        torch_dset = make_torch_dataset([aset], as_dict=True)
        loader = DataLoader(torch_dset, batch_size=10, drop_last=True)
        for data in loader:
            assert isinstance(data, dict)
            assert data['aset'].shape == (10, 5, 7)
        co.close()

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="multiprocess workers does not run on windows")
    def test_lots_of_data_with_multiple_backend_multiple_worker_dataloader(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        torch_dset = make_torch_dataset([aset])
        loader = DataLoader(torch_dset, batch_size=10, drop_last=True, num_workers=2)
        for data in loader:
            assert data.shape == (10, 5, 7)
        co.close()

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="multiprocess workers does not run on windows")
    def test_two_aset_loader_two_worker_dataloader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        torch_dset = make_torch_dataset([first_aset, second_aset])
        loader = DataLoader(torch_dset, batch_size=2, drop_last=True, num_workers=2)
        count = 0
        for asets_batch in loader:
            assert isinstance(asets_batch, list)
            assert len(asets_batch) == 2
            assert asets_batch[0].shape == (2, 5, 7)
            assert asets_batch[1].shape == (2, 5, 7)
            assert np.allclose(asets_batch[0], -asets_batch[1])
            count += 1
        assert count == 10
        co.close()


# ==================================== Tensorflow ====================================


class TestTfDataset(object):
    # TODO: Add TF2.0 and 1.0 test cases

    def test_dataset_loader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']

        # multiple datasets
        tf_dset = make_tensorflow_dataset([first_aset, second_aset])
        tf_dset = tf_dset.batch(6)
        for dset1, dset2 in tf_dset.take(2):
            assert dset1.shape == tf.TensorShape((6, 5, 7))
            assert dset2.shape == tf.TensorShape((6, 5, 7))
        co.close()

    def test_variably_shaped(self, aset_samples_var_shape_initialized_repo):
        # Variably shaped test is required since the collation is dependent on
        # the way we return the data from generator
        repo = aset_samples_var_shape_initialized_repo
        co = repo.checkout(write=True)
        aset = co.columns['writtenaset']
        for i in range(5, 10):
            aset[i] = np.random.random((2, i))
        co.commit('added data')
        co.close()

        co = repo.checkout()
        aset = co.columns['writtenaset']
        tf_dset = make_tensorflow_dataset((aset,))
        shape_obj = tf.TensorShape((2, None))
        tf_dset = tf_dset.padded_batch(5, padded_shapes=(shape_obj,))
        for val in tf_dset:
            assert val[0].shape[0] == 5
            assert val[0].shape[1] == 2
            assert 11 > val[0].shape[2] > 4
        co.close()

    def test_lots_of_data_with_multiple_backend(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        tf_dset = make_tensorflow_dataset([aset])
        tf_dset = tf_dset.batch(10)
        for data in tf_dset:
            assert data[0].shape == (10, 5, 7)
        co.close()

    def test_shuffle(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        unshuffled_dataset = make_tensorflow_dataset((first_aset,),
                                                     keys=[str(i) for i in range(15)],
                                                     shuffle=False)
        expected_unshuffled_content = [i for i in range(15)]
        recieved_unshuffled_content = []
        for data in unshuffled_dataset:
            recieved_unshuffled_content.append(int(data[0][0][0]))
        assert expected_unshuffled_content == recieved_unshuffled_content

        shuffled_dataset = make_tensorflow_dataset((first_aset,),
                                                   keys=[str(i) for i in range(15)],
                                                   shuffle=True)
        recieved_shuffled_content = []
        for data in shuffled_dataset:
            recieved_shuffled_content.append(int(data[0][0][0]))
        assert recieved_shuffled_content != expected_unshuffled_content
        co.close()
