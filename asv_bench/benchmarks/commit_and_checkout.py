from tempfile import mkdtemp
from shutil import rmtree
import numpy as np
from hangar import Repository


class MakeCommit(object):

    params = [(5_000, 20_000), (5_000, 20_000)]
    param_names = ['num_samples', 'num_metadata']
    processes = 2
    repeat = 2
    number = 1
    warmup_time = 0

    def setup(self, num_samples, num_metadata):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)
        arr = np.array([0,], dtype=np.uint8)
        try:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend_opts='10')
        except TypeError:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend='10')

        with aset as cm_aset:
            for i in range(num_samples):
                arr[:] = i % 255
                cm_aset[i] = arr
        with self.co.metadata as cm_meta:
            for i in range(num_metadata):
                cm_meta[i] = f'{i % 500} data'

    def teardown(self, num_samples, num_metadata):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_commit(self, num_samples, num_metadata):
        self.co.commit('hello')


class CheckoutCommit(object):

    params = [(5_000, 20_000), (5_000, 20_000)]
    param_names = ['num_samples', 'num_metadata']
    processes = 2
    number = 1
    repeat = 2
    warmup_time = 0

    def setup(self, num_samples, num_metadata):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)
        arr = np.array([0,], dtype=np.uint8)
        try:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend_opts='10')
        except TypeError:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend='10')

        with aset as cm_aset:
            for i in range(num_samples):
                arr[:] = i % 255
                cm_aset[i] = arr
        with self.co.metadata as cm_meta:
            for i in range(num_metadata):
                cm_meta[i] = f'{i % 500} data'
        self.co.commit('first')
        self.co.close()
        self.co = None

    def teardown(self, num_samples, num_metadata):
        try:
            self.co.close()
        except PermissionError:
            pass
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_checkout_read_only(self, num_samples, num_metadata):
        self.co = self.repo.checkout(write=False)

    def time_checkout_write_enabled(self, num_samples, num_metadata):
        self.co = self.repo.checkout(write=True)
        self.co.close()