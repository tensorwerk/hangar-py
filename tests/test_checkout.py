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
