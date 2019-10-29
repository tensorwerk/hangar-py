# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import numpy as np
from hangar import Repository
from os import getcwd


# ------------------------- fixture functions ----------------------------------


class HDF5_00(object):

    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0.00000001

    def setup_cache(self):
        tmpdir = getcwd()
        repo = Repository(path=tmpdir, exists=False)
        repo.init('tester', 'foo@test.bar', remove_old=True)
        co = repo.checkout(write=True)

        aint = np.hamming(250).reshape(250, 1)
        bint = np.hamming(250).reshape(1, 250)
        cint = np.round(aint * bint * 1000).astype(np.uint16)
        arrint = np.zeros((250, 250, 3), dtype=cint.dtype)
        arrint[:, :, 0] = cint
        arrint[:, :, 1] = cint + 1
        arrint[:, :, 2] = cint + 2

        afloat = np.hamming(250).reshape(250, 1).astype(np.float32)
        bfloat = np.hamming(250).reshape(1, 250).astype(np.float32)
        cfloat = np.round(afloat * bfloat * 1000)
        arrfloat = np.zeros((250, 250, 3), dtype=cfloat.dtype)
        arrfloat[:, :, 0] = cfloat
        arrfloat[:, :, 1] = cfloat + 1
        arrfloat[:, :, 2] = cfloat + 2
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
            for i in range(2_000):
                arrfloat += 1
                arrint += 1
                cm_aset_float[i] = arrfloat
                cm_aset_int[i] = arrint
        co.commit('first commit')
        co.close()

    def setup(self):
        tmpdir = getcwd()
        repo = Repository(path=tmpdir, exists=True)
        self.co = repo.checkout(write=False)

    def teardown(self):
        self.co.close()

    def time_read_uint16_2000_samples(self):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def time_read_float32_2000_samples(self):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def peakmem_read_uint16_2000_samples(self):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def peakmem_read_float32_2000_samples(self):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]


class NUMPY_10(object):

    processes = 1
    number = 1
    repeat = 1
    warmup_time = 0.00000001

    def setup_cache(self):
        tmpdir = getcwd()
        repo = Repository(path=tmpdir, exists=False)
        repo.init('tester', 'foo@test.bar', remove_old=True)
        co = repo.checkout(write=True)

        aint = np.hamming(250).reshape(250, 1)
        bint = np.hamming(250).reshape(1, 250)
        cint = np.round(aint * bint * 1000).astype(np.uint16)
        arrint = np.zeros((250, 250, 3), dtype=cint.dtype)
        arrint[:, :, 0] = cint
        arrint[:, :, 1] = cint + 1
        arrint[:, :, 2] = cint + 2

        afloat = np.hamming(250).reshape(250, 1).astype(np.float32)
        bfloat = np.hamming(250).reshape(1, 250).astype(np.float32)
        cfloat = np.round(afloat * bfloat * 1000)
        arrfloat = np.zeros((250, 250, 3), dtype=cfloat.dtype)
        arrfloat[:, :, 0] = cfloat
        arrfloat[:, :, 1] = cfloat + 1
        arrfloat[:, :, 2] = cfloat + 2
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
            for i in range(2_000):
                arrfloat += 1
                arrint += 1
                cm_aset_float[i] = arrfloat
                cm_aset_int[i] = arrint
        co.commit('first commit')
        co.close()

    def setup(self):
        tmpdir = getcwd()
        repo = Repository(path=tmpdir, exists=False)
        self.co = repo.checkout(write=False)

    def teardown(self):
        self.co.close()

    def time_read_uint16_2000_samples(self):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def time_read_float32_2000_samples(self):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def peakmem_read_uint16_2000_samples(self):
        aset = self.co.arraysets['aset_int']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]

    def peakmem_read_float32_2000_samples(self):
        aset = self.co.arraysets['aset_float']
        with aset as cm_aset:
            for i in range(2_000):
                arr = cm_aset[i]
