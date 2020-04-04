from os.path import join as pjoin
from os import mkdir
import pytest
import sys
import numpy as np
from hangar import Repository


from hangar import make_numpy_dataset
from hangar.dataset.common import HangarDataset

try:
    import torch
    from torch.utils.data import DataLoader
    from hangar import make_torch_dataset
    torchExists = True
except (ImportError, ModuleNotFoundError):
    torchExists = False

try:
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    from hangar import make_tensorflow_dataset
    tfExists = True
except (ImportError, ModuleNotFoundError):
    tfExists = False


class TestInternalDatasetClass:

    def test_column_without_wrapping_list(self, repo_20_filled_samples, array5by7):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        second_col = co.columns['second_aset']
        dataset = HangarDataset((first_col,))
        key = dataset.keys[0]
        target = array5by7[:] = int(key)
        assert np.allclose(dataset[key], target)
        with pytest.raises(TypeError):
            HangarDataset(first_col)

    def test_no_column(self):
        with pytest.raises(ValueError):
            dataset = HangarDataset([])

    def test_fails_on_write_enabled_columns(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        first_aset = co.columns['writtenaset']
        with pytest.raises(TypeError):
            HangarDataset((first_aset,))

    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_columns_without_local_data_and_without_key_argument(self,
                                                                 written_two_cmt_server_repo,
                                                                 managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        repo.remote.fetch_data('origin', branch='master', max_num_bytes=1000)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        dataset = HangarDataset((aset,))
        available_keys = len(dataset.keys)
        all_keys = len(list(aset.keys()))
        assert available_keys != all_keys

    def test_columns_without_common_keys_and_without_key_argument(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout(write=True)
        first_col = co.columns['writtenaset']
        first_col['AnExtraKey'] = first_col['0']
        co.commit('added an extra key')
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        second_col = co.columns['second_aset']
        dataset = HangarDataset((first_col, second_col))
        assert '0' in dataset.keys
        assert 'AnExtraKey' in first_col
        assert 'AnExtraKey' not in dataset.keys

    def test_keys_success(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        keys = ['1', '2', '3']
        dataset = HangarDataset((first_col,), keys=keys)
        assert dataset.keys == keys

    def test_keys_non_common(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_col = co.columns['writtenaset']
        keys = ['w', 'r', 'o', 'n', 'g']
        with pytest.raises(KeyError):
            HangarDataset((first_col,), keys=keys)

    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_keys_non_local(self,
                            written_two_cmt_server_repo,
                            managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        repo.remote.fetch_data('origin', branch='master', max_num_bytes=1000)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        keys = aset.remote_reference_keys
        with pytest.raises(FileNotFoundError):
            HangarDataset((aset,), keys=keys)


# ====================================   Numpy    ====================================
# ====================================================================================


@pytest.mark.filterwarnings("ignore:.* is experimental in the current release")
class TestNumpyDataset:
    def test_warns_experimental(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_aset = co.columns['writtenaset']
        with pytest.warns(UserWarning, match='make_numpy_dataset is experimental'):
            make_numpy_dataset([first_aset])

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
            assert isinstance(data, tuple)
            assert data[0].shape == (10, 5, 7)
        co.close()

    def test_shuffle(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        dataset = make_numpy_dataset((first_aset,), keys=('0', '1', '2', '3', '4'),
                                     shuffle=False)
        one_point_from_each = []
        ordered = [0, 1, 2, 3, 4]
        for data in dataset:
            one_point_from_each.append(int(data[0][0][0]))
        assert one_point_from_each == ordered
        dataset = make_numpy_dataset((first_aset,), keys=('0', '1', '2', '3', '4'),
                                     shuffle=True)
        one_point_from_each = []
        for data in dataset:
            one_point_from_each.append(int(data[0][0][0]))
        assert one_point_from_each != ordered


# ====================================   PyTorch  ====================================
# ====================================================================================


@pytest.mark.skipif(torchExists,
                    reason='pytorch is installed in the test environment.')
def test_no_torch_installed_raises_error_on_dataloader_import():
    with pytest.raises(ImportError):
        from hangar import make_torch_dataset
        make_torch_dataset(None)


@pytest.mark.filterwarnings("ignore:.* is experimental in the current release")
@pytest.mark.skipif(not torchExists,
                    reason='pytorch is not installed in the test environment.')
class TestTorchDataset(object):

    def test_warns_experimental(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_aset = co.columns['writtenaset']
        with pytest.warns(UserWarning, match='make_torch_dataset is experimental'):
            make_torch_dataset([first_aset])

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

    @pytest.mark.xfail(sys.platform == "win32",
                       strict=True,
                       reason="multiprocess workers does not run on windows")
    def test_lots_of_data_with_multiple_backend_multiple_worker_dataloader(self,
                                                                           repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        torch_dset = make_torch_dataset([aset])
        loader = DataLoader(torch_dset, batch_size=10, drop_last=True, num_workers=2)
        for data in loader:
            assert data[0].shape == (10, 5, 7)
        co.close()

    @pytest.mark.xfail(sys.platform == "win32",
                       strict=True,
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

# ==================================== Tensorflow ====================================
# ====================================================================================

@pytest.mark.skipif(tfExists,
                    reason='tensorflow is installed in the test environment.')
def test_no_tf_installed_raises_error_on_dataloader_import():
    with pytest.raises(ImportError):
        from hangar import make_tensorflow_dataset
        make_tensorflow_dataset(None)


@pytest.mark.filterwarnings("ignore:.* is experimental in the current release")
@pytest.mark.skipif(
    not tfExists,
    reason='tensorflow is not installed in the test environment.')
class TestTfDataset(object):
    # TODO: Add TF2.0 and 1.0 test cases

    def test_warns_experimental(self, repo_20_filled_samples):
        co = repo_20_filled_samples.checkout()
        first_aset = co.columns['writtenaset']
        with pytest.warns(UserWarning, match='make_tensorflow_dataset is experimental'):
            make_tensorflow_dataset([first_aset])

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
        dataset = make_tensorflow_dataset((first_aset,), keys=('0', '1', '2', '3', '4'),
                                          shuffle=False)
        one_point_from_each = []
        ordered = [0, 1, 2, 3, 4]
        for data in dataset:
            one_point_from_each.append(int(data[0][0][0]))
        assert one_point_from_each == ordered
        dataset = make_tensorflow_dataset((first_aset,), keys=('0', '1', '2', '3', '4'),
                                          shuffle=True)
        one_point_from_each = []
        for data in dataset:
            one_point_from_each.append(int(data[0][0][0]))
        assert one_point_from_each != ordered
