from os.path import join as pjoin
from os import mkdir
import sys
import warnings

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader
with warnings.catch_warnings():
    warnings.simplefilter('ignore', category=DeprecationWarning)
    import tensorflow as tf
    tf.compat.v1.enable_eager_execution()

from hangar import Repository
from hangar import make_torch_dataset
from hangar import make_tf_dataset


class TestTorchDataLoader(object):

    def test_warns_experimental(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.warns(UserWarning, match='Dataloaders are experimental'):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_warns_arrayset_sample_size_mismatch(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        second_aset = co.columns['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.warns(UserWarning, match='Columns do not contain equal number of samples'):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_multiple_dataset_loader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        second_aset = co.columns['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.raises(ValueError):
            # emtpy list
            make_torch_dataset([])
        with pytest.raises(TypeError):
            # if more than one dataset, those should be in a list/tuple
            make_torch_dataset(first_aset, first_aset)

        with pytest.warns(UserWarning, match='Columns do not contain equal number of samples'):
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
    def test_dataset_loader_fails_with_write_enabled_checkout(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.raises(TypeError):
            make_torch_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_keys(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        aset = co.columns['writtenaset']

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
    def test_with_index_range(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        aset = co.columns['writtenaset']

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
    def test_field_names(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.raises(ValueError):  # number of dsets and field_names are different
            make_torch_dataset([first_aset, second_aset], field_names=('input',))
        with pytest.raises(TypeError):  # field_names's type is wrong
            make_torch_dataset([first_aset, second_aset], field_names={'input': '', 'target': ''})
        torch_dset = make_torch_dataset([first_aset, second_aset], field_names=('input', 'target'))
        assert len(torch_dset) == 20
        loader = DataLoader(torch_dset, batch_size=5)
        for sample in loader:
            assert type(sample).__name__ == 'BatchTuple_input_target'
            assert sample._fields == ('input', 'target')
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_lots_of_data_with_multiple_backend(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        torch_dset = make_torch_dataset([aset])
        loader = DataLoader(torch_dset, batch_size=10, drop_last=True)
        for data in loader:
            assert type(data).__name__ == 'BatchTuple_aset'
            assert data.aset.shape == (10, 5, 7)
        co.close()

    @pytest.mark.xfail(sys.platform == "win32",
                       strict=True,
                       reason="multiprocess workers does not run on windows")
    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_lots_of_data_with_multiple_backend_multiple_worker_dataloader(self,
                                                                           repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        torch_dset = make_torch_dataset([aset])
        loader = DataLoader(torch_dset, batch_size=10, drop_last=True, num_workers=2)
        for data in loader:
            assert type(data).__name__ == 'BatchTuple_aset'
            assert data.aset.shape == (10, 5, 7)
        co.close()

    @pytest.mark.xfail(sys.platform == "win32",
                       strict=True,
                       reason="multiprocess workers does not run on windows")
    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_two_aset_loader_two_worker_dataloader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        torch_dset = make_torch_dataset([first_aset, second_aset])
        loader = DataLoader(torch_dset, batch_size=2, drop_last=True, num_workers=2)
        count = 0
        for asets_batch in loader:
            assert type(asets_batch).__name__ == 'BatchTuple_writtenaset_second_aset'
            assert isinstance(asets_batch, tuple)
            assert len(asets_batch) == 2
            assert asets_batch._fields == ('writtenaset', 'second_aset')
            assert asets_batch.writtenaset.shape == (2, 5, 7)
            assert asets_batch.second_aset.shape == (2, 5, 7)
            assert np.allclose(asets_batch.writtenaset, -asets_batch.second_aset)
            count += 1
        assert count == 10

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_no_common_no_local(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(ValueError):
            torch_dset = make_torch_dataset(aset)
        co.close()
        repo._env._close_environments()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_no_common(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(KeyError):
            torch_dset = make_torch_dataset(aset, keys=['1', -1])
        co.close()
        repo._env._close_environments()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_data_unavailable(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(FileNotFoundError):
            torch_dset = make_torch_dataset(aset, keys=['1', '2'])
        co.close()
        repo._env._close_environments()


class TestTfDataLoader(object):

    def test_warns_experimental(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.warns(UserWarning, match='Dataloaders are experimental'):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_wans_arrayset_sample_size_mismatch(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        second_aset = co.columns['second_aset']
        del second_aset['10']
        co.commit('deleting')
        co.close()

        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.warns(UserWarning, match='Columns do not contain equal number of samples'):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_dataset_loader(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']

        # multiple datasets
        tf_dset = make_tf_dataset([first_aset, second_aset])
        tf_dset = tf_dset.batch(6)
        for dset1, dset2 in tf_dset.take(2):
            assert dset1.shape == tf.TensorShape((6, 5, 7))
            assert dset2.shape == tf.TensorShape((6, 5, 7))
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_with_keys(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        aset = co.columns['writtenaset']

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
    def test_with_index_range(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout()
        aset = co.columns['writtenaset']

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
    def test_dataset_loader_fails_with_write_enabled_checkout(self, repo_20_filled_samples):
        repo = repo_20_filled_samples
        co = repo.checkout(write=True)
        first_aset = co.columns['writtenaset']
        second_aset = co.columns['second_aset']
        with pytest.raises(TypeError):
            make_tf_dataset([first_aset, second_aset])
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
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
        tf_dset = make_tf_dataset(aset)
        shape_obj = tf.TensorShape((2, None))
        tf_dset = tf_dset.padded_batch(5, padded_shapes=(shape_obj,))
        for val in tf_dset:
            assert val[0].shape[0] == 5
            assert val[0].shape[1] == 2
            assert 11 > val[0].shape[2] > 4
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    def test_lots_of_data_with_multiple_backend(self, repo_300_filled_samples):
        repo = repo_300_filled_samples
        co = repo.checkout()
        aset = co.columns['aset']
        tf_dset = make_tf_dataset([aset])
        tf_dset = tf_dset.batch(10)
        for data in tf_dset:
            assert data[0].shape == (10, 5, 7)
        co.close()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_no_common_no_local(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(ValueError):
            tf_dset = make_tf_dataset(aset)
        co.close()
        repo._env._close_environments()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_no_common(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(KeyError):
            tf_dset = make_tf_dataset(aset, keys=['1', -1])
        co.close()
        repo._env._close_environments()

    @pytest.mark.filterwarnings("ignore:Dataloaders are experimental")
    @pytest.mark.filterwarnings("ignore:Column.* writtenaset contains `reference-only` samples")
    def test_local_without_data_fails_data_unavailable(self, written_two_cmt_server_repo, managed_tmpdir):
        new_tmpdir = pjoin(managed_tmpdir, 'new')
        mkdir(new_tmpdir)
        server, _ = written_two_cmt_server_repo
        repo = Repository(path=new_tmpdir, exists=False)
        repo.clone('name', 'a@b.c', server, remove_old=True)
        co = repo.checkout()
        aset = co.columns['writtenaset']
        with pytest.raises(FileNotFoundError):
            tf_dset = make_tf_dataset(aset, keys=['1', '2'])
        co.close()
        repo._env._close_environments()
