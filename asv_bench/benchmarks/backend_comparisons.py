# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
import os
from hangar import Repository
from tempfile import mkdtemp
from shutil import rmtree
from hangar.utils import folder_size


# ------------------------- fixture functions ----------------------------------


class _WriterSuite:

    params = ['hdf5_00', 'hdf5_01', 'numpy_10']
    param_names = ['backend']
    processes = 2
    repeat = (2, 4, 30.0)
    # repeat == tuple (min_repeat, max_repeat, max_time)
    number = 2
    warmup_time = 0

    def setup(self, backend):

        # self.method
        self.current_iter_number = 0
        self.backend_code = {
            'numpy_10': '10',
            'hdf5_00': '00',
            'hdf5_01': '01',
        }
        # self.num_samples

        self.sample_shape = (50, 50, 20)

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
        self.arr = np.prod(component_arrays).astype(np.float32)

        try:
            self.aset = self.co.arraysets.init_arrayset(
                'aset', prototype=self.arr, backend_opts=self.backend_code[backend])
        except TypeError:
            try:
                self.aset = self.co.arraysets.init_arrayset(
                    'aset', prototype=self.arr, backend=self.backend_code[backend])
            except ValueError:
                raise NotImplementedError
        except ValueError:
            raise NotImplementedError
        except AttributeError:
            self.aset = self.co.add_ndarray_column(
                'aset', prototype=self.arr, backend=self.backend_code[backend])

    def teardown(self, backend):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def write(self, backend):
        arr = self.arr
        iter_number = self.current_iter_number
        with self.aset as cm_aset:
            for i in range(self.num_samples):
                arr[iter_number, iter_number, iter_number] += 1
                cm_aset[i] = arr
        self.current_iter_number += 1


# ----------------------------- Writes ----------------------------------------


class Write_50by50by20_300_samples(_WriterSuite):
    method = 'write'
    num_samples = 300
    time_write = _WriterSuite.write


# ----------------------------- Reads -----------------------------------------


class _ReaderSuite:

    params = ['hdf5_00', 'hdf5_01', 'numpy_10']
    param_names = ['backend']
    processes = 2
    repeat = (2, 4, 30.0)
    # repeat == tuple (min_repeat, max_repeat, max_time)
    number = 3
    warmup_time = 0
    timeout = 60

    def setup_cache(self):

        backend_code = {
            'numpy_10': '10',
            'hdf5_00': '00',
            'hdf5_01': '01',
        }

        sample_shape = (50, 50, 10)
        num_samples = 3_000

        repo = Repository(path=os.getcwd(), exists=False)
        repo.init('tester', 'foo@test.bar', remove_old=True)
        co = repo.checkout(write=True)

        component_arrays = []
        ndims = len(sample_shape)
        for idx, shape in enumerate(sample_shape):
            layout = [1 for i in range(ndims)]
            layout[idx] = shape
            component = np.hamming(shape).reshape(*layout) * 100
            component_arrays.append(component.astype(np.float32))
        arr = np.prod(component_arrays).astype(np.float32)

        for backend, code in backend_code.items():
            try:
                co.arraysets.init_arrayset(
                    backend, prototype=arr, backend_opts=code)
            except TypeError:
                try:
                    co.arraysets.init_arrayset(
                        backend, prototype=arr, backend=code)
                except ValueError:
                    pass
            except ValueError:
                pass
            except AttributeError:
                co.add_ndarray_column(backend, prototype=arr, backend=code)

        try:
            col = co.columns
        except AttributeError:
            col = co.arraysets

        with col as asets_cm:
            for aset in asets_cm.values():
                changer = 0
                for i in range(num_samples):
                    arr[changer, changer, changer] += 1
                    aset[i] = arr
                changer += 1
        co.commit('first commit')
        co.close()
        repo._env._close_environments()

    def setup(self, backend):
        self.repo = Repository(path=os.getcwd(), exists=True)
        self.co = self.repo.checkout(write=False)
        try:
            try:
                self.aset = self.co.columns[backend]
            except AttributeError:
                self.aset = self.co.arraysets[backend]
        except KeyError:
            raise NotImplementedError

    def teardown(self, backend):
        self.co.close()
        self.repo._env._close_environments()

    def read(self, backend):
        with self.aset as cm_aset:
            for i in cm_aset.keys():
                arr = cm_aset[i]


class Read_50by50by10_3000_samples(_ReaderSuite):
    method = 'read'
    num_samples = 3000
    time_read = _ReaderSuite.read
