# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from hangar import Repository
from tempfile import mkdtemp
from shutil import rmtree
from hangar.utils import folder_size


# ------------------------- fixture functions ----------------------------------


class _WriterSuite:

    processes = 2
    repeat = 2
    number = 1
    warmup_time = 0

    def setup(self):

        # self.method
        # self.backend
        self.backend_code = {
            'numpy_10': '10',
            'hdf5_00': '00'
        }
        # self.dtype
        self.type_code = {
            'float32': np.float32,
            'uint16': np.uint16,
        }
        # self.num_samples

        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        co = self.repo.checkout(write=True)

        a = np.hamming(100).reshape(100, 1)
        b = np.hamming(100).reshape(1, 100)
        c = np.round(a * b * 1000).astype(self.type_code[self.dtype])
        arr = np.zeros((100, 100), dtype=c.dtype)
        arr[:, :] = c

        try:
            aset = co.arraysets.init_arrayset(
                'aset', prototype=arr, backend_opts=self.backend_code[self.backend])
        except TypeError:
            aset = co.arraysets.init_arrayset(
                'aset', prototype=arr, backend=self.backend_code[self.backend])
        if self.method == 'read':
            with aset as cm_aset:
                for i in range(self.num_samples):
                    arr += 1
                    cm_aset[i] = arr
            co.commit('first commit')
            co.close()
            self.co = self.repo.checkout(write=False)
        else:
            self.arr = arr
            self.co = co

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


# ============================= HDF5_00 =======================================
# ----------------------------- Writes ----------------------------------------


class HDF5_00_UINT16_Write_1000(_WriterSuite):
    method = 'write'
    num_samples = 1000
    backend = 'hdf5_00'
    dtype = 'uint16'

    time_write = _WriterSuite.write


class HDF5_00_UINT16_Write_5000(_WriterSuite):
    method = 'write'
    num_samples = 5000
    backend = 'hdf5_00'
    dtype = 'uint16'

    time_write = _WriterSuite.write


class HDF5_00_FLOAT32_Write_1000(_WriterSuite):
    method = 'write'
    num_samples = 1000
    backend = 'hdf5_00'
    dtype = 'float32'

    time_write = _WriterSuite.write


class HDF5_00_FLOAT32_Write_5000(_WriterSuite):
    method = 'write'
    num_samples = 5000
    backend = 'hdf5_00'
    dtype = 'float32'

    time_write = _WriterSuite.write


# ----------------------------- Reads -----------------------------------------


class HDF5_00_UINT16_Read_1000(_WriterSuite):
    method = 'read'
    num_samples = 1000
    backend = 'hdf5_00'
    dtype = 'uint16'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class HDF5_00_UINT16_Read_5000(_WriterSuite):
    method = 'read'
    num_samples = 5000
    backend = 'hdf5_00'
    dtype = 'uint16'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class HDF5_00_FLOAT32_Read_1000(_WriterSuite):
    method = 'read'
    num_samples = 1000
    backend = 'hdf5_00'
    dtype = 'float32'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class HDF5_00_FLOAT32_Read_5000(_WriterSuite):
    method = 'read'
    num_samples = 5000
    backend = 'hdf5_00'
    dtype = 'float32'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


# ============================ NUMPY_10 =======================================
# ----------------------------- Writes ----------------------------------------


class NUMPY_10_UINT16_Write_1000(_WriterSuite):
    method = 'write'
    num_samples = 1000
    backend = 'numpy_10'
    dtype = 'uint16'

    time_write = _WriterSuite.write


class NUMPY_10_UINT16_Write_5000(_WriterSuite):
    method = 'write'
    num_samples = 5000
    backend = 'numpy_10'
    dtype = 'uint16'

    time_write = _WriterSuite.write


class NUMPY_10_FLOAT32_Write_1000(_WriterSuite):
    method = 'write'
    num_samples = 1000
    backend = 'numpy_10'
    dtype = 'float32'

    time_write = _WriterSuite.write


class NUMPY_10_FLOAT32_Write_5000(_WriterSuite):
    method = 'write'
    num_samples = 5000
    backend = 'numpy_10'
    dtype = 'float32'

    time_write = _WriterSuite.write


# ----------------------------- Reads -----------------------------------------


class NUMPY_10_UINT16_Read_1000(_WriterSuite):
    method = 'read'
    num_samples = 1000
    backend = 'numpy_10'
    dtype = 'uint16'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class NUMPY_10_UINT16_Read_5000(_WriterSuite):
    method = 'read'
    num_samples = 5000
    backend = 'numpy_10'
    dtype = 'uint16'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class NUMPY_10_FLOAT32_Read_1000(_WriterSuite):
    method = 'read'
    num_samples = 1000
    backend = 'numpy_10'
    dtype = 'float32'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'


class NUMPY_10_FLOAT32_Read_5000(_WriterSuite):
    method = 'read'
    num_samples = 5000
    backend = 'numpy_10'
    dtype = 'float32'

    time_read = _WriterSuite.read
    track_repo_size = _WriterSuite.size
    track_repo_size.unit = 'bytes'