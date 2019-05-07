import numpy as np
import pytest
import shutil
import platform


class TestCheckout(object):

    def test_two_write_checkouts(self, repo):
        w1_checkout = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            w2_checkout = repo.checkout(write=True)
        w1_checkout.close()

    def test_two_read_checkouts(self, repo, array5by7):
        w_checkout = repo.checkout(write=True)
        dataset_name = 'dset'
        r_ds = w_checkout.datasets.init_dataset(name=dataset_name, prototype=array5by7)
        r_ds['1'] = array5by7
        w_checkout.commit('init')
        r1_checkout = repo.checkout()
        r2_checkout = repo.checkout()
        assert np.allclose(r1_checkout.datasets['dset']['1'], array5by7)
        assert np.allclose(r1_checkout.datasets['dset']['1'], r2_checkout.datasets['dset']['1'])
        r1_checkout.close()
        r2_checkout.close()
        w_checkout.close()

    def test_write_with_r_checkout(self, written_repo, array5by7):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            co.datasets.init_dataset(name='dset', shape=(5, 7), dtype=np.float64)
            co.datasets['_dset']['1'] = array5by7
            co.datasets['_dset'].add('1', array5by7)
        co.close()

    @pytest.mark.skipif(platform.system() == 'Windows',
        reason='Files cannot be removed when process is using them on windows systems')
    def test_write_after_repo_deletion(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        shutil.rmtree(written_repo._repo_path)
        with pytest.raises(OSError):
            co.datasets.init_dataset('dset', prototype=array5by7)
        dset = co.datasets['_dset']
        dset['1'] = array5by7
        # todo: shouldn't it be atleast same as above
        with pytest.raises(FileNotFoundError):
            co.commit()
        co.close()

    def test_writer_dset_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        dsets = co.datasets
        dset = co.datasets['_dset']
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_writer_dset_obj_not_accessible_after_commit_and_close(self, written_repo, array5by7):
        repo = written_repo
        co = repo.checkout(write=True)
        dsets = co.datasets
        dset = co.datasets['_dset']
        dset['1'] = array5by7
        co.commit('hey there')
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dset['1']

    def test_reader_dset_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        dsets = co.datasets
        dset = co.datasets['_dset']
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_writer_metadata_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        md = co.metadata
        co.close()

        with pytest.raises(ReferenceError):
            md.__dict__

    def test_writer_metadata_obj_not_accessible_after_commit_and_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        md = co.metadata
        md['hello'] = 'world'
        co.commit('test commit')
        co.close()

        with pytest.raises(ReferenceError):
            md.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = md['hello']

    def test_reader_metadata_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        md = co.metadata
        co.close()
        with pytest.raises(ReferenceError):
            md.__dict__

    def test_close_read_does_not_invalidate_write_checkout(self, written_repo, array5by7):
        repo = written_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)
        r_co.close()

        with pytest.raises(PermissionError):
            shouldFail = r_co.datasets

        dset = w_co.datasets['_dset']
        dset['1'] = array5by7
        assert np.allclose(w_co.datasets['_dset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_close_write_does_not_invalidate_read_checkout(self, written_repo, array5by7):
        repo = written_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)

        dset = w_co.datasets['_dset']
        dset['1'] = array5by7
        assert np.allclose(w_co.datasets['_dset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            dset.__dict__

        assert '_dset' in r_co.datasets
        assert len(r_co.metadata) == 0

        r_co.close()
        with pytest.raises(PermissionError):
            shouldFail = r_co.datasets



@pytest.mark.skip(reason='not implemented')
class TestBranching(object):

    def test_merge(self):
        pass

    def test_name_conflict(self):
        pass

    def test_delete_branch(self):
        pass

    def test_merge_multiple_checkouts(self):
        pass

    def test_merge_diverged(self):
        pass

    def test_merge_diverged_conflict(self):
        pass


@pytest.mark.skip(reason='not implemented')
class TestTimeTravel(object):
    pass
