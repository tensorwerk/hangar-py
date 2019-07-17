import pytest
from hangar.dataloaders import TorchLoader


class TestTorchDataLoader(object):
    def test_single_dataset_loader_batchsize_six(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout()
        dset = co.datasets['_dset']
        loader = TorchLoader(dset, batch_size=6)
        total_samples = 0
        for batch in loader:
            total_samples += batch.shape[0]
            assert batch.shape == (6, 5, 7) or batch.shape == (2, 5, 7)  # drop last is False
        assert total_samples == 20

    def test_multiple_dataset_loader(self, repo_with_20_samples):
        repo = repo_with_20_samples
        co = repo.checkout(write=True)
        first_dset = co.datasets['_dset']
        second_dset = co.datasets['second_dset']
        del second_dset[10]
        co.commit('deleting')
        with pytest.raises(TypeError):
            # invalid object
            TorchLoader({})
        with pytest.raises(IndexError):
            # emtpy list
            TorchLoader([])
        with pytest.raises(ValueError):
            # if more than one dataset, those should be in a list/tuple
            TorchLoader(first_dset, first_dset, 6, drop_last=True)
        with pytest.raises(RuntimeError):
            # datasets with different length
            TorchLoader([first_dset, second_dset])
        loader = TorchLoader([first_dset, first_dset], batch_size=6, drop_last=True)
        total_samples = 0
        for dset1, dset2 in loader:
            total_samples += dset1.shape[0]
            assert dset1.shape == (6, 5, 7)
            assert dset2.shape == (6, 5, 7)
        assert total_samples == 18  # drop last is True
