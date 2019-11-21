# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from hangar import Repository
from tempfile import mkdtemp
from shutil import rmtree
from hangar.utils import folder_size


# ------------------------- fixture functions ----------------------------------


class _WriterSuite:

    params = [('hdf5_00', 'hdf5_01', 'numpy_10'), ((50, 50), (50, 50, 50))]
    param_names = ['backend', 'sample_shape']
    processes = 2
    repeat = 2
    number = 1
    warmup_time = 0

    def setup(self, backend, sample_shape):

        # self.method
        self.backend_code = {
            'numpy_10': '10',
            'hdf5_00': '00',
            'hdf5_01': '01',
        }
        # self.num_samples

        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        self.co = self.repo.checkout(write=True)

        if (backend == 'numpy_10') and (len(sample_shape) > 2):
            # comes after co because teardown needs self.co and self.repo to exist
            raise NotImplementedError

        component_arrays = []
        ndims = len(sample_shape)
        for idx, shape in enumerate(sample_shape):
            layout = [1 for i in range(ndims)]
            layout[idx] = shape
            component = np.hamming(shape).reshape(*layout) * 100
            component_arrays.append(component.astype(np.float32))
        arr = np.prod(component_arrays).astype(np.float32)

        try:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend_opts=self.backend_code[backend])
        except TypeError:
            aset = self.co.arraysets.init_arrayset(
                'aset', prototype=arr, backend=self.backend_code[backend])
        except ValueError:
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

    def teardown(self, backend, sample_shape):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def read(self, backend, sample_shape):
        aset = self.co.arraysets['aset']
        ks = list(aset.keys())
        with aset as cm_aset:
            for i in ks:
                arr = cm_aset[i]

    def write(self, backend, sample_shape):
        arr = self.arr
        aset = self.co.arraysets['aset']
        with aset as cm_aset:
            for i in range(self.num_samples):
                arr += 1
                cm_aset[i] = arr

    def size(self, backend, sample_shape):
        return folder_size(self.repo._env.repo_path, recurse=True)


# ----------------------------- Writes ----------------------------------------


class Write_500_samples(_WriterSuite):
    method = 'write'
    num_samples = 500
    time_write = _WriterSuite.write


# ----------------------------- Reads -----------------------------------------


class Read_2000_samples(_WriterSuite):
    method = 'read'
    num_samples = 2000
    time_read = _WriterSuite.read