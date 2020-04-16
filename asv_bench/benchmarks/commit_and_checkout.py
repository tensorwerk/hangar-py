from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
from hangar import Repository


class MakeCommit(object):

    params = (5_000, 20_000, 50_000)
    param_names = ['num_samples']
    processes = 2
    repeat = (2, 4, 20)
    number = 1
    warmup_time = 0

    def setup(self, num_samples):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)
        arr = np.array([0,], dtype=np.uint8)
        try:
            aset = self.co.arraysets.init_arrayset('aset', prototype=arr, backend_opts='10')
        except TypeError:
            aset = self.co.arraysets.init_arrayset('aset', prototype=arr, backend='10')
        except AttributeError:
            aset = self.co.add_ndarray_column('aset', prototype=arr, backend='10')

        with aset as cm_aset:
            for i in range(num_samples):
                arr[:] = i % 255
                cm_aset[i] = arr

    def teardown(self, num_samples):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_commit(self, num_samples):
        self.co.commit('hello')


class CheckoutCommit(object):

    params = (5_000, 20_000, 50_000)
    param_names = ['num_samples']
    processes = 2
    number = 1
    repeat = (2, 4, 20)
    warmup_time = 0

    def setup(self, num_samples):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)
        arr = np.array([0,], dtype=np.uint8)
        try:
            aset = self.co.arraysets.init_arrayset('aset', prototype=arr, backend_opts='10')
        except TypeError:
            aset = self.co.arraysets.init_arrayset('aset', prototype=arr, backend='10')
        except AttributeError:
            aset = self.co.add_ndarray_column('aset', prototype=arr, backend='10')

        with aset as cm_aset:
            for i in range(num_samples):
                arr[:] = i % 255
                cm_aset[i] = arr
        self.co.commit('first')
        self.co.close()
        self.co = None

    def teardown(self, num_samples):
        try:
            self.co.close()
        except PermissionError:
            pass
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_checkout_read_only(self, num_samples):
        self.co = self.repo.checkout(write=False)

    def time_checkout_write_enabled(self, num_samples):
        self.co = self.repo.checkout(write=True)
        self.co.close()
