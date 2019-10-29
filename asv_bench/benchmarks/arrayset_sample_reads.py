# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from hangar import Repository
from tempfile import mkdtemp
from shutil import rmtree


# ------------------------- fixture functions ----------------------------------


class HDF5_00(object):

    params = [2_000, 5_000, 20_000]
    param_names = ['num_samples']
    processes = 2
    number = 1
    repeat = (2, 4, 60)
    warmup_time = 0

    def setup(self, num_samples):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        co = self.repo.checkout(write=True)

        aint = np.hamming(100).reshape(100, 1)
        bint = np.hamming(100).reshape(1, 100)
        cint = np.round(aint * bint * 1000).astype(np.uint16)
        arrint = np.zeros((100, 100), dtype=cint.dtype)
        arrint[:, :] = cint

        afloat = np.hamming(100).reshape(100, 1).astype(np.float32)
        bfloat = np.hamming(100).reshape(1, 100).astype(np.float32)
        cfloat = np.round(afloat * bfloat * 1000)
        arrfloat = np.zeros((100, 100), dtype=cfloat.dtype)
        arrfloat[:, :] = cfloat
        try:
            aset_int = co.arraysets.init_arrayset(
                'aset_int', prototype=arrint, backend_opts='00')
            aset_float = co.arraysets.init_arrayset(
                'aset_float', prototype=arrfloat, backend_opts='00')
        except TypeError:
            aset_int = co.arraysets.init_arrayset(
                'aset_int', prototype=arrint, backend='00')
            aset_float = co.arraysets.init_arrayset(
                'aset_float', prototype=arrfloat, backend='00')
        with aset_int as cm_aset_int, aset_float as cm_aset_float:
            for i in range(num_samples):
                arrfloat += 1
                arrint += 1
                cm_aset_float[i] = arrfloat
                cm_aset_int[i] = arrint
        co.commit('first commit')
        co.close()
        self.co = self.repo.checkout(write=False)

    def teardown(self, num_samples):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_read_uint16_samples(self, num_samples):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(num_samples):
                arr = cm_aset[i]

    def time_read_float32_samples(self, num_samples):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(num_samples):
                arr = cm_aset[i]


class NUMPY_10(object):

    params = [2_000, 5_000]
    param_names = ['num_samples']
    processes = 2
    number = 1
    repeat = (2, 4, 60)
    warmup_time = 0

    def setup(self, num_samples):
        self.tmpdir = mkdtemp()
        self.repo = Repository(path=self.tmpdir, exists=False)
        self.repo.init('tester', 'foo@test.bar', remove_old=True)
        co = self.repo.checkout(write=True)

        aint = np.hamming(100).reshape(100, 1)
        bint = np.hamming(100).reshape(1, 100)
        cint = np.round(aint * bint * 1000).astype(np.uint16)
        arrint = np.zeros((100, 100), dtype=cint.dtype)
        arrint[:, :] = cint

        afloat = np.hamming(100).reshape(100, 1).astype(np.float32)
        bfloat = np.hamming(100).reshape(1, 100).astype(np.float32)
        cfloat = np.round(afloat * bfloat * 1000)
        arrfloat = np.zeros((100, 100), dtype=cfloat.dtype)
        arrfloat[:, :] = cfloat
        try:
            aset_int = co.arraysets.init_arrayset(
                'aset_int', prototype=arrint, backend_opts='10')
            aset_float = co.arraysets.init_arrayset(
                'aset_float', prototype=arrfloat, backend_opts='10')
        except TypeError:
            aset_int = co.arraysets.init_arrayset(
                'aset_int', prototype=arrint, backend='10')
            aset_float = co.arraysets.init_arrayset(
                'aset_float', prototype=arrfloat, backend='10')
        with aset_int as cm_aset_int, aset_float as cm_aset_float:
            for i in range(num_samples):
                arrfloat += 1
                arrint += 1
                cm_aset_float[i] = arrfloat
                cm_aset_int[i] = arrint
        co.commit('first commit')
        co.close()
        self.co = self.repo.checkout(write=False)

    def teardown(self, num_samples):
        self.co.close()
        self.repo._env._close_environments()
        rmtree(self.tmpdir)

    def time_read_uint16_samples(self, num_samples):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(num_samples):
                arr = cm_aset[i]

    def time_read_float32_samples(self, num_samples):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(num_samples):
                arr = cm_aset[i]