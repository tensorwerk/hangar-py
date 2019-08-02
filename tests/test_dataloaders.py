import pytest
import numpy as np
import torch
import tensorflow as tf
from torch.utils.data import DataLoader
from hangar.dataloaders import make_torch_dataset, make_tf_dataset


class TestTorchDataLoader(object):

    def test_multiple_dataset_loader(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        first_dset = co.datasets['_dset']
        second_dset = co.datasets['second_dset']
        del second_dset['10']
        co.commit('deleting')
        with pytest.raises(IndexError):
            # emtpy list
            make_torch_dataset([])
        with pytest.raises(TypeError):
            # if more than one dataset, those should be in a list/tuple
            make_torch_dataset(first_dset, first_dset)
        with pytest.raises(RuntimeError):
            # datasets with different length
            make_torch_dataset([first_dset, second_dset])
        torch_dset = make_torch_dataset([second_dset, second_dset])
        loader = DataLoader(torch_dset, batch_size=6, drop_last=True)
        total_samples = 0
        for dset1, dset2 in loader:
            total_samples += dset1.shape[0]
            assert dset1.shape == (6, 5, 7)
            assert dset2.shape == (6, 5, 7)
        assert total_samples == 18  # drop last is True

    def test_with_keys_and_index_range(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        dset = co.datasets['_dset']

        # with keys
        keys = ['2', '4', '7', '9', '15', '18']
        bad_tensor = dset['1']
        torch_dset = make_torch_dataset(dset, keys=keys)
        loader = DataLoader(torch_dset, batch_size=3)
        total_batches = 0
        for batch in loader:
            assert batch[0].size(0) == 3
            total_batches += 1
            for sample in batch[0]:
                assert not np.allclose(sample, bad_tensor)
        assert total_batches == 2

        # with index range
        index_range = slice(10, 16)
        torch_dset = make_torch_dataset(dset, index_range=index_range)
        loader = DataLoader(torch_dset, batch_size=3)
        for batch in loader:
            assert batch[0].size(0) == 3
            for sample in batch[0]:
                assert not np.allclose(sample, bad_tensor)

    def test_field_names(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        first_dset = co.datasets['_dset']
        second_dset = co.datasets['second_dset']
        with pytest.raises(RuntimeError):
            make_torch_dataset([first_dset, second_dset], field_names=('input',))
        torch_dset = make_torch_dataset([first_dset, second_dset], field_names=('input', 'target'))
        assert hasattr(torch_dset[1], 'input')
        assert hasattr(torch_dset[1], 'target')
        if torch.__version__ > '1.0.1':
            loader = DataLoader(torch_dset, batch_size=5)
            for sample in loader:
                assert hasattr(sample, 'input')
                assert hasattr(sample, 'target')

    def test_variably_shaped(self):
        pass


class TestTfDataLoader(object):

    def test_dataset_loader(self, repo_with_20_samples):
        tf.compat.v1.enable_eager_execution()
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        first_dset = co.datasets['_dset']
        second_dset = co.datasets['second_dset']

        # multiple datasets
        tf_dset = make_tf_dataset([first_dset, second_dset])
        tf_dset = tf_dset.batch(6)
        for dset1, dset2 in tf_dset.take(2):
            assert dset1.shape == tf.TensorShape((6, 5, 7))
            assert dset2.shape == tf.TensorShape((6, 5, 7))

    def test_variably_shaped(self):
        pass
