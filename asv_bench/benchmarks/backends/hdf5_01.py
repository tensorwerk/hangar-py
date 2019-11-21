# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from hangar import Repository
from tempfile import mkdtemp
from shutil import rmtree
from hangar.utils import folder_size


class _WriterSuite:

    processes = 2
    repeat = 2
    number = 1
    warmup_time = 0

    def setup(self):

        # self.method
        # self.num_samples
        # self.sample_shape

        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)

        component_arrays = []
        ndims = len(self.sample_shape)
        for idx, shape in enumerate(self.sample_shape):
            layout = [1 for i in range(ndims)]
            layout[idx] = shape
            component = np.hamming(shape).reshape(*layout) * 100
            component_arrays.append(component.astype(np.float32))
        arr = np.prod(component_arrays).astype(np.float32)

        try:
            aset = self.co.arraysets.init_arrayset('aset', prototype=arr, backend_opts='01')
        except ValueError:
            # marks as skipped benchmark for commits which do not have this backend.
            raise NotImplementedError

        if self.method == 'read':
            with aset as cm_aset:
                for i in range(self.num_samples):
                    arr += 1
                    cm_aset[i] = arr
            self.co.commit('first commit')
            self.co.close()
            self.co = self.repo.checkout(write=False)
        else:
            self.arr = arr

    def teardown(self):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def read(self):
        aset = self.co.arraysets['aset']
        ks = list(aset.keys())
        with aset as cm_aset:
            for i in ks:
                arr = cm_aset[i]

    def write(self):
        arr = self.arr
        aset = self.co.arraysets['aset']
        with aset as cm_aset:
            for i in range(self.num_samples):
                arr += 1
                cm_aset[i] = arr

    def size(self):
        return folder_size(self.repo._env.repo_path, recurse=True)


class Write_50by50_100_samples(_WriterSuite):
    method = 'write'
    sample_shape = (50, 50)
    num_samples = 100

    time_write = _WriterSuite.write


class Write_50by50by50by10_100_samples(_WriterSuite):
    method = 'write'
    sample_shape = (50, 50, 50, 10)
    num_samples = 100

    time_write = _WriterSuite.write


# ----------------------------- Reads -----------------------------------------


class Read_50by50_250_samples(_WriterSuite):
    method = 'read'
    sample_shape = (50, 50)
    num_samples = 250

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class Read_50by50by50by10_250_samples(_WriterSuite):
    method = 'read'
    sample_shape = (50, 50, 50, 10)
    num_samples = 250

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'