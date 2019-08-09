import pytest
import numpy as np


try:
    import torch
    from torch.utils.data import DataLoader
    from hangar import make_torch_dataset
    skipTorch = False
except ImportError:
    skipTorch = True


@pytest.mark.skipif(skipTorch is True,
                    reason='pytorch is not installed in the test environment.')
class TestTorchDataLoader(object):

    def test_warns_experimental(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.warns(UserWarning, match='Dataloaders are experimental'):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_warns_arrayset_sample_size_mismatch(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        second_aset = co.arraysets['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.warns(UserWarning, match='Arraysets do not contain equal num samples'):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_multiple_dataset_loader(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        second_aset = co.arraysets['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.raises(ValueError):
            # emtpy list
            make_torch_dataset([])
        with pytest.raises(TypeError):
            # if more than one dataset, those should be in a list/tuple
            make_torch_dataset(first_aset, first_aset)

        with pytest.warns(UserWarning, match='Arraysets do not contain equal num samples'):
            torch_dset = make_torch_dataset([first_aset, second_aset])
        loader = DataLoader(torch_dset, batch_size=6, drop_last=True)
        total_samples = 0
        for dset1, dset2 in loader:
            total_samples += dset1.shape[0]
            assert dset1.shape == (6, 5, 7)
            assert dset2.shape == (6, 5, 7)
        assert total_samples == 18  # drop last is True
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_dataset_loader_fails_with_write_enabled_checkout(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.raises(TypeError):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_keys(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        aset = co.arraysets['_aset']

        # with keys
        keys = ['2', '4', '5', '6', '7', '9', '15', '18', '19']
        bad_tensor0 = aset['0']
        bad_tensor1 = aset['1']
        bad_tensor3 = aset['3']
        bad_tensor8 = aset['8']

        torch_dset = make_torch_dataset(aset, keys=keys)
        loader = DataLoader(torch_dset, batch_size=3)
        total_batches = 0
        for batch in loader:
            assert batch[0].size(0) == 3
            total_batches += 1
            for sample in batch:
                assert not np.allclose(sample, bad_tensor0)
                assert not np.allclose(sample, bad_tensor1)
                assert not np.allclose(sample, bad_tensor3)
                assert not np.allclose(sample, bad_tensor8)
        assert total_batches == 3
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_index_range(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        aset = co.arraysets['_aset']

        # with keys
        bad_tensor0 = aset['0']
        bad_tensor1 = aset['1']

        # with index range
        index_range = slice(2, 20)
        torch_dset = make_torch_dataset(aset, index_range=index_range)
        loader = DataLoader(torch_dset, batch_size=3)
        total_batches = 0
        for batch in loader:
            assert batch[0].size(0) == 3
            total_batches += 1
            for sample in batch:
                assert not np.allclose(sample, bad_tensor0)
                assert not np.allclose(sample, bad_tensor1)
        assert total_batches == 6
        co.close()


    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_field_names(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.raises(ValueError):  # number of dsets and field_names are different
            make_torch_dataset([first_aset, second_aset], field_names=('input',))
        with pytest.raises(TypeError):  # field_names's type is wrong
            make_torch_dataset([first_aset, second_aset], field_names={'input': '', 'target': ''})
        torch_dset = make_torch_dataset([first_aset, second_aset], field_names=('input', 'target'))
        assert hasattr(torch_dset[1], 'input')
        assert hasattr(torch_dset[1], 'target')
        if torch.__version__ > '1.0.1':
            loader = DataLoader(torch_dset, batch_size=5)
            for sample in loader:
                assert hasattr(sample, 'input')
                assert hasattr(sample, 'target')
        co.close()


try:
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()
    from hangar import make_tf_dataset
    skipTF = False
except ImportError:
    skipTF = True


@pytest.mark.skipif(
    skipTF is True,
    reason='tensorflow is not installed in the test environment.')
class TestTfDataLoader(object):

    def test_warns_experimental(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.warns(UserWarning, match='Dataloaders are experimental'):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_wans_arrayset_sample_size_mismatch(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        second_aset = co.arraysets['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.warns(UserWarning, match='Arraysets do not contain equal num samples'):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_dataset_loader(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']

        # multiple datasets
        tf_dset = make_tf_dataset([first_aset, second_aset])
        tf_dset = tf_dset.batch(6)
        for dset1, dset2 in tf_dset.take(2):
            assert dset1.shape == tf.TensorShape((6, 5, 7))
            assert dset2.shape == tf.TensorShape((6, 5, 7))
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_keys(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        aset = co.arraysets['_aset']

        # with keys
        keys = ['2', '4', '5', '6', '7', '9', '15', '18', '19']
        bad_tensor0 = aset['0']
        bad_tensor1 = aset['1']
        bad_tensor3 = aset['3']
        bad_tensor8 = aset['8']

        tf_dset = make_tf_dataset(aset, keys=keys)
        tf_dset = tf_dset.batch(3)
        total_batches = 0
        for dset1 in tf_dset:
            total_batches += 1
            assert dset1[0].shape == tf.TensorShape((3, 5, 7))
            for sample in dset1[0]:
                assert not np.allclose(sample, bad_tensor0)
                assert not np.allclose(sample, bad_tensor1)
                assert not np.allclose(sample, bad_tensor3)
                assert not np.allclose(sample, bad_tensor8)
        assert total_batches == 3
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_index_range(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        aset = co.arraysets['_aset']

        # with keys
        bad_tensor0 = aset['0']
        bad_tensor1 = aset['1']

        # with index range
        index_range = slice(2, 20)
        tf_dset = make_tf_dataset(aset, index_range=index_range)
        tf_dset = tf_dset.batch(3)
        total_batches = 0
        for dset1 in tf_dset:
            total_batches += 1
            assert dset1[0].shape == tf.TensorShape((3, 5, 7))
            for sample in dset1[0]:
                assert not np.allclose(sample, bad_tensor0)
                assert not np.allclose(sample, bad_tensor1)
        assert total_batches == 6
        co.close()


    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_dataset_loader_fails_with_write_enabled_checkout(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        first_aset = co.arraysets['_aset']
        second_aset = co.arraysets['second_aset']
        with pytest.raises(TypeError):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_variably_shaped(self, variable_shape_written_repo):
        # Variably shaped test is required since the collation is dependent on
        # the way we return the data from generator
        repo = variable_shape_written_repo
        co = repo.checkout(write=True)
        aset = co.arraysets['_aset']
        for i in range(5, 10):
            aset[i] = np.random.random((2, i))
        co.commit('added data')
        co.close()

        co = repo.checkout()
        aset = co.arraysets['_aset']
        tf_dset = make_tf_dataset(aset)
        shape_obj = tf.TensorShape((2, None))
        tf_dset = tf_dset.padded_batch(5, padded_shapes=(shape_obj,))
        for val in tf_dset:
            assert val[0].shape[0] == 5
            assert val[0].shape[1] == 2
            assert 11 > val[0].shape[2] > 4
        co.close()
