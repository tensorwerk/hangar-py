"""Local HDF5 Backend Implementation, Identifier: ``HDF5_01``

Backend Identifiers
===================

*  Backend: ``0``
*  Version: ``1``
*  Format Code: ``01``
*  Canonical Name: ``HDF5_01``

Storage Method
==============

*  This module is meant to handle larger datasets which are of fixed size. IO
   and significant compression optimization is achieved by storing arrays at
   their appropriate top level index in the same shape they naturally assume
   and chunking over the entire subarray domain making up a sample (rather than
   having to subdivide chunks when the sample could be variably shaped.)

*  Data is written to specific subarray indexes inside an HDF5 "dataset" in a
   single HDF5 File.

*  In each HDF5 File there are ``COLLECTION_COUNT`` "datasets" (named ``["0" :
   "{COLLECTION_COUNT}"]``). These are referred to as ``"dataset number"``

*  Each dataset is a zero-initialized array of:

   *  ``dtype: {schema_dtype}``; ie ``np.float32`` or ``np.uint8``

   *  ``shape: (COLLECTION_SIZE, *{schema_shape})``; ie ``(500, 10, 10)`` or
      ``(500, 512, 512, 320)``. The first index in the dataset is referred to as a
      ``collection index``.

*  Compression Filters, Chunking Configuration/Options are applied globally for
   all ``datasets`` in a file at dataset creation time.

*  On read and write of all samples the xxhash64_hexdigest is calculated for
   the raw array bytes. This is to ensure that all data in == data out of the
   hdf5 files. That way even if a file is manually edited (bypassing fletcher32
   filter check) we have a quick way to tell that things are not as they should
   be.

Compression Options
===================

Accepts dictionary containing keys

*  ``backend`` == ``"01"``
*  ``complib``
*  ``complevel``
*  ``shuffle``

Blosc-HDF5

*  ``complib`` valid values:

   *  ``'blosc:blosclz'``,
   *  ``'blosc:lz4'``,
   *  ``'blosc:lz4hc'``,
   *  ``'blosc:zlib'``,
   *  ``'blosc:zstd'``

*  ``complevel`` valid values: [0, 9] where 0 is "no compression" and 9 is
   "most compression"

*  ``shuffle`` valid values:

   *  ``None``
   *  ``'none'``
   *  ``'byte'``
   *  ``'bit'``


LZF Filter

*  ``'complib' == 'lzf'``
*  ``'shuffle'`` one of ``[False, None, 'none', True, 'byte']``
*  ``'complevel'`` one of ``[False, None, 'none']``

GZip Filter

*  ``'complib' == 'gzip'``
*  ``'shuffle'`` one of ``[False, None, 'none', True, 'byte']``
*  ``complevel`` valid values: [0, 9] where 0 is "no compression" and 9 is
   "most compression"


Record Format
=============

Fields Recorded for Each Array
------------------------------

*  Format Code
*  File UID
*  xxhash64_hexdigest (ie. checksum)
*  Dataset Number (``0:COLLECTION_COUNT`` dataset selection)
*  Dataset Index (``0:COLLECTION_SIZE`` dataset subarray selection)
*  Subarray Shape

Examples
--------

1)  Adding the first piece of data to a file:

    *  Array shape (Subarray Shape): (10, 10)
    *  File UID: "rlUK3C"
    *  xxhash64_hexdigest: 8067007c0f05c359
    *  Dataset Number: 16
    *  Collection Index: 105

    ``Record Data => "01:rlUK3C:8067007c0f05c359:16:105:10 10"``

1)  Adding to a piece of data to a the middle of a file:

    *  Array shape (Subarray Shape): (20, 2, 3)
    *  File UID: "rlUK3C"
    *  xxhash64_hexdigest: b89f873d3d153a9c
    *  Dataset Number: "3"
    *  Collection Index: 199

    ``Record Data => "01:rlUK3C:b89f873d3d153a9c:8:199:20 2 3"``


Technical Notes
===============

*  The majority of methods not directly related to "chunking" and the "raw data
   chunk cache" are either identical to HDF5_00, or only slightly modified.

*  Files are read only after initial creation/writes. Only a write-enabled
   checkout can open a HDF5 file in ``"w"`` or ``"a"`` mode, and writer
   checkouts create new files on every checkout, and make no attempt to fill in
   unset locations in previous files. This is not an issue as no disk space is
   used until data is written to the initially created "zero-initialized"
   collection datasets

*  On write: Single Writer Multiple Reader (``SWMR``) mode is set to ensure that
   improper closing (not calling ``.close()``) method does not corrupt any data
   which had been previously flushed to the file.

*  On read: SWMR is set to allow multiple readers (in different threads /
   processes) to read from the same file. File handle serialization is handled
   via custom python ``pickle`` serialization/reduction logic which is
   implemented by the high level ``pickle`` reduction ``__set_state__()``,
   ``__get_state__()`` class methods.

*  An optimization is performed in order to increase the read / write
   performance of fixed size datasets. Due to the way that we initialize an
   entire HDF5 file with all datasets pre-created (to the size of the fixed
   subarray shape), and the fact we absolutely know the size / shape /
   access-pattern of the arrays, inefficient IO due to wasted chunk processing
   is not a concern. It is far more efficient for us to completely blow off the
   metadata chunk cache, and chunk each subarray as a single large item item.

   This method of processing tends to have a number of significant effects as
   compared to chunked storage methods:

      1. **Compression rations improve** (by a non-trivial factor). This is
         simply due to the fact that a larger amount of raw data is being passed
         into the compressor at a time. While the exact improvement seen is
         highly dependent on both the data size and compressor used, there
         should be no case where compressing the full tensor uses more disk
         space then chunking the tensor, compressing each chunk individually,
         and then saving each chunk to disk.

      2. **Read performance improves** (so long as a suitable compressor /
         option set was chosen). Instead of issuing (potentially) many read
         requests - one for each chunk - to the storage hardware, signifiantly
         few IOPS are used to retrieve the entire set of compressed raw data
         from disk. Fewer IOPS means much less time waiting on the hard disk.
         Moreso, only a single decompression step is needed to reconstruct
         the numeric array, completly decoupling performance from HDF5's
         ability to parallelize internal filter pipeline operations.

         Additionally, since the entire requested chunk is retrieved in a
         single decompression pipeline run, there is no need for the HDF5 core
         to initialize an intermediate buffer which holds data chunks as each
         decompression operation completes. Futher, by preinitializing an empty
         ``numpy.ndarray`` container and using the low level HDF5
         ``read_direct`` method, the decompressed data buffer is passes
         directly into the returned ``ndarray.__array_interface__.data``
         field with no intermediate copy or processing steps.

      3. **Shuffle filters are favored.**. With much more data to work with in
         a single compression operation, the use of "byte shuffle" filters in
         the compressor spec has been seen to both markedly decrease read time
         and increase compression ratios. Shuffling can significantly reduce
         disk space required to store some piece of data on disk, further
         reducing the time spent waiting on hard disk IO while incuring a
         negligible cost to decompression speed.

   Taking all of these effects into account, there can be up to an order of
   magnitude increase in read performance as compared to the subarray chunking
   strategy employed by the ``HDF5_00`` backend.

*  Like all other backends at the time of writing, only 'C' ordered arrays
   are accepted by this method.
"""
import logging
import math
import os
from collections import ChainMap
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import MutableMapping, Tuple, Optional, Union, Callable

import h5py
import numpy as np

try:
    # hdf5plugin warns if a filter is already loaded.
    _logger = logging.getLogger('hdf5plugin')
    _initialLevel = _logger.getEffectiveLevel()
    _logger.setLevel(logging.ERROR)
    import hdf5plugin
    if 'blosc' not in hdf5plugin.FILTERS:
        raise ImportError(f'BLOSC unavailable via hdf5plugin: {hdf5plugin.FILTERS}')
finally:
    _logger.setLevel(_initialLevel)
from xxhash import xxh64_hexdigest


from .chunk import calc_chunkshape
from .specs import HDF5_01_DataHashSpec
from .. import __version__
from ..optimized_utils import SizedDict
from ..constants import DIR_DATA_REMOTE, DIR_DATA_STAGE, DIR_DATA_STORE, DIR_DATA
from ..op_state import writer_checkout_only, reader_checkout_only
from ..utils import find_next_prime, random_string, set_blosc_nthreads
from ..typesystem import Descriptor, OneOf, DictItems, SizedIntegerTuple, checkedmeta

set_blosc_nthreads()

# ----------------------------- Configuration ---------------------------------

_FmtCode = '01'

# contents of a single hdf5 file
COLLECTION_SIZE = 100
COLLECTION_COUNT = 100
CHUNK_MAX_RDCC_NBYTES = 250_000_000
CHUNK_RDCC_W0 = 0.75

# -------------------------------- Parser Implementation ----------------------


def hdf5_01_encode(uid: str, cksum: str, dset: int, dset_idx: int,
                   shape: Tuple[int]) -> bytes:
    """converts the hdf5 data has spec to an appropriate db value

    Parameters
    ----------
    uid : str
        the file name prefix which the data is written to.
    cksum : int
        xxhash_64.hex_digest checksum of the data bytes in numpy array form.
    dset : int
        collection (ie. hdf5 dataset) name to find this data piece.
    dset_idx : int
        collection first axis index in which this data piece resides.
    shape : Tuple[int]
        shape of the data sample written to the collection idx. ie:
        what subslices of the hdf5 dataset should be read to retrieve
        the sample as recorded.

    Returns
    -------
    bytes
        hash data db value recording all input specifications.
    """
    shape_str = " ".join([str(i) for i in shape])
    return f'01:{uid}:{cksum}:{dset}:{dset_idx}:{shape_str}'.encode()


# ------------------------- Accessor Object -----------------------------------


@DictItems(
    expected_keys_required={'complib': True, 'complevel': True, 'shuffle': True},
    expected_values={
        'complib': ['blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:zlib', 'blosc:zstd'],
        'complevel': [i for i in range(10)],
        'shuffle': [None, 'none', 'byte', 'bit']})
class BloscCompressionOptions(Descriptor):
    pass


@DictItems(
    expected_keys_required={'complib': True, 'complevel': True, 'shuffle': True},
    expected_values={
        'complib': ['gzip'], 'complevel': [i for i in range(10)], 'shuffle': [True, False]})
class GzipCompressionOptions(Descriptor):
    pass


@DictItems(
    expected_keys_required={'complib': True, 'complevel': False, 'shuffle': True},
    expected_values={
        'complib': ['lzf'], 'complevel': ['none', None], 'shuffle': [True, False]})
class LzfCompressionOptions(Descriptor):
    pass


@OneOf(list(map(lambda x: np.dtype(x).name, [
        np.bool, np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16,
        np.int32, np.int64, np.float16, np.float32, np.float64, np.longdouble])))
class AllowedDtypes(Descriptor):
    # Note. np.longdouble since np.float128 not guaranteed to be available on
    # all system. this is a particular issue with some windows numpy builds.
    pass


class HDF5_01_Options(metaclass=checkedmeta):
    _shape = SizedIntegerTuple(size=32)
    _dtype = AllowedDtypes()
    _lzf = LzfCompressionOptions()
    _gzip = GzipCompressionOptions()
    _blosc = BloscCompressionOptions()
    _avail_filters = ('_lzf', '_gzip', '_blosc')

    def __init__(self, backend_options, dtype, shape, *args, **kwargs):
        self._shape = shape
        self._dtype = dtype
        self._selected_filter = None
        if backend_options is None:
            backend_options = self.default_options

        for filter_attr in self._avail_filters:
            with suppress((KeyError, ValueError)):
                setattr(self, filter_attr, backend_options)
                self._selected_filter = filter_attr
                break
        else:  # N.B. for-else loop (ie. "no-break")
            raise ValueError(f'Invalid backend_options {backend_options}')
        self._verify_data_nbytes_larger_than_clib_min()

    def _verify_data_nbytes_larger_than_clib_min(self):
        """blosc clib should not be used if data buffer size < 16 bytes.

        Raises
        ------
        ValueError:
            if the data size is not valid for the clib
        """
        if self._selected_filter in ['_blosc', None]:
            num_items = np.prod(self._shape)
            itemsize = np.dtype(self._dtype).itemsize
            nbytes = itemsize * num_items
            if nbytes <= 16:
                raise ValueError(f'blosc clib requires data buffer size > 16 bytes')

    @property
    def default_options(self):
        if 'blosc' in hdf5plugin.FILTERS:
            try:
                self._verify_data_nbytes_larger_than_clib_min()
                return {'complib': 'blosc:lz4hc', 'complevel': 5, 'shuffle': 'byte'}
            except ValueError:
                pass
        return {'complib': 'lzf', 'complevel': None, 'shuffle': True}

    @property
    def backend_options(self):
        return getattr(self, self._selected_filter)

    @property
    def init_requires(self):
        return ('repo_path', 'schema_shape', 'schema_dtype')


HDF5_01_MapTypes = MutableMapping[str, Union[h5py.File, Callable[[], h5py.File]]]


class HDF5_01_FileHandles(object):
    """Manage HDF5 file handles.

    When in SWMR-write mode, no more than a single file handle can be in the
    "writeable" state. This is an issue where multiple columns may need to
    write to the same column schema.
    """

    def __init__(self, repo_path: Path, schema_shape: tuple, schema_dtype: np.dtype):
        self.path: Path = repo_path
        self.schema_shape: tuple = schema_shape
        self.schema_dtype: np.dtype = schema_dtype
        self._dflt_backend_opts: Optional[dict] = None

        self.rFp: HDF5_01_MapTypes = {}
        self.wFp: HDF5_01_MapTypes = {}
        self.Fp: HDF5_01_MapTypes = ChainMap(self.rFp, self.wFp)
        self.rDatasets = SizedDict(maxsize=100)
        self.wdset: h5py.Dataset = None

        self.mode: Optional[str] = None
        self.hIdx: Optional[int] = None
        self.w_uid: Optional[str] = None
        self.hMaxSize: Optional[int] = None
        self.hNextPath: Optional[int] = None
        self.hColsRemain: Optional[int] = None

        self.STAGEDIR: Path = Path(self.path, DIR_DATA_STAGE, _FmtCode)
        self.REMOTEDIR: Path = Path(self.path, DIR_DATA_REMOTE, _FmtCode)
        self.DATADIR: Path = Path(self.path, DIR_DATA, _FmtCode)
        self.STOREDIR: Path = Path(self.path, DIR_DATA_STORE, _FmtCode)
        self.DATADIR.mkdir(exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.w_uid in self.wFp:
            self.wFp[self.w_uid]['/'].attrs.modify('next_location', (self.hNextPath, self.hIdx))
            self.wFp[self.w_uid]['/'].attrs.modify('collections_remaining', self.hColsRemain)
            self.wFp[self.w_uid].flush()

    @reader_checkout_only
    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        self.close()
        state = self.__dict__.copy()
        del state['rFp']
        del state['wFp']
        del state['Fp']
        del state['rDatasets']
        del state['wdset']
        return state

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)
        self.rFp = {}
        self.wFp = {}
        self.Fp = ChainMap(self.rFp, self.wFp)
        self.rDatasets = {}
        self.wdset = None
        self.open(mode=self.mode)

    @property
    def backend_opts(self):
        return self._dflt_backend_opts

    @writer_checkout_only
    def _backend_opts_set(self, val):
        """Nonstandard descriptor method. See notes in ``backend_opts.setter``.
        """
        self._dflt_backend_opts = val
        return

    @backend_opts.setter
    def backend_opts(self, value):
        """
        Using seperate setter method (with ``@writer_checkout_only`` decorator
        applied) due to bug in python <3.8.

        From: https://bugs.python.org/issue19072
            > The classmethod decorator when applied to a function of a class,
            > does not honour the descriptor binding protocol for whatever it
            > wraps. This means it will fail when applied around a function which
            > has a decorator already applied to it and where that decorator
            > expects that the descriptor binding protocol is executed in order
            > to properly bind the function to the class.
        """
        return self._backend_opts_set(value)

    def open(self, mode: str, *, remote_operation: bool = False):
        """Open an hdf5 file handle in the Handler Singleton

        Parameters
        ----------
        mode : str
            one of `r` or `a` for read only / read-write.
        remote_operation : optional, kwarg only, bool
            if this hdf5 data is being created from a remote fetch operation, then
            we don't open any files for reading, and only open files for writing
            which exist in the remote data dir. (default is false, which means that
            write operations use the stage data dir and read operations use data store
            dir)
        """
        self.mode = mode
        if self.mode == 'a':
            process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
            process_dir.mkdir(exist_ok=True)
            for uidpth in process_dir.iterdir():
                if uidpth.suffix == '.hdf5':
                    file_pth = self.DATADIR.joinpath(uidpth.name)
                    self.rFp[uidpth.stem] = partial(
                        h5py.File, file_pth, 'r', swmr=True, libver='latest')

        if not remote_operation:
            if not self.STOREDIR.is_dir():
                return
            for uidpth in self.STOREDIR.iterdir():
                if uidpth.suffix == '.hdf5':
                    file_pth = self.DATADIR.joinpath(uidpth.name)
                    self.rFp[uidpth.stem] = partial(
                        h5py.File, file_pth, 'r', swmr=True, libver='latest')

    def close(self):
        """Close a file handle after writes have been completed

        behavior changes depending on write-enable or read-only file

        Returns
        -------
        bool
            True if success, otherwise False.
        """
        if self.mode == 'a':
            if self.w_uid in self.wFp:
                self.wFp[self.w_uid]['/'].attrs.modify('next_location', (self.hNextPath, self.hIdx))
                self.wFp[self.w_uid]['/'].attrs.modify('collections_remaining', self.hColsRemain)
                self.wFp[self.w_uid].flush()
            for uid in list(self.wFp.keys()):
                with suppress(AttributeError):
                    self.wFp[uid].close()
                del self.wFp[uid]
            self.wdset = None
            self.hMaxSize = None
            self.hNextPath = None
            self.hIdx = None
            self.hColsRemain = None
            self.w_uid = None

        for uid in list(self.rFp.keys()):
            with suppress(AttributeError):
                self.rFp[uid].close()
            del self.rFp[uid]
        self.rDatasets = {}

    @staticmethod
    def delete_in_process_data(repo_path: Path, *, remote_operation=False) -> None:
        """Removes some set of files entirely from the stage/remote directory.

        DANGER ZONE. This should essentially only be used to perform hard resets
        of the repository state.

        Parameters
        ----------
        repo_path : Path
            path to the repository on disk
        remote_operation : optional, kwarg only, bool
            If true, modify contents of the remote_dir, if false (default) modify
            contents of the staging directory.
        """
        data_dir = Path(repo_path, DIR_DATA, _FmtCode)
        PDIR = DIR_DATA_STAGE if not remote_operation else DIR_DATA_REMOTE
        process_dir = Path(repo_path, PDIR, _FmtCode)
        if not process_dir.is_dir():
            return

        for uidpth in process_dir.iterdir():
            if uidpth.suffix == '.hdf5':
                os.remove(process_dir.joinpath(uidpth.name))
                os.remove(data_dir.joinpath(uidpth.name))
        os.rmdir(process_dir)

    @staticmethod
    def _dataset_opts(complib: str, complevel: int, shuffle: Union[bool, str]) -> dict:
        """specify compression options for the hdf5 dataset.

        .. seealso:: :function:`_blosc_opts`

        to enable blosc compression, use the conda-forge `blosc-hdf5-plugin` package.

        .. seealso::

        * https://github.com/conda-forge/staged-recipes/pull/7650
        * https://github.com/h5py/h5py/issues/611

        Parameters
        ----------
        complib : str
            the compression lib to use, one of ['lzf', 'gzip', 'blosc:blosclz',
            'blosc:lz4', 'blosc:lz4hc', 'blosc:zlib', 'blosc:zstd']
        complevel : int
            compression level to specify (accepts values [0, 9] for all except 'lzf'
            where no complevel is accepted)
        shuffle : bool
            if True or `byte`, enable byte shuffle filter, if blosc
            compression, pass through 'bits' is accepted as well. False, or
            None indicates no shuffle should be applied.
        """
        # ---- blosc hdf5 plugin filters ----
        _blosc_compression = {
            'blosc:blosclz': 0,
            'blosc:lz4': 1,
            'blosc:lz4hc': 2,
            # Not built 'snappy': 3,
            'blosc:zlib': 4,
            'blosc:zstd': 5}
        _blosc_shuffle = {None: 0, 'none': 0, 'byte': 1, 'bit': 2}
        _blosc_complevel = {**{i: i for i in range(10)}, None: 9, 'none': 9}

        # ---- h5py built in filters ----
        _lzf_gzip_shuffle = {None: False, False: False, 'none': False, True: True, 'byte': True}
        _lzf_complevel = {False: None, None: None, 'none': None}
        _gzip_complevel = {**{i: i for i in range(10)}, None: 4, 'none': 4}

        if complib.startswith('blosc'):
            args = {
                'compression': 32001,
                'compression_opts': (
                    0, 0, 0, 0,
                    _blosc_complevel[complevel],
                    _blosc_shuffle[shuffle],
                    _blosc_compression[complib]),
                'shuffle': False}
        elif complib == 'lzf':
            args = {
                'shuffle': _lzf_gzip_shuffle[shuffle],
                'compression': complib,
                'compression_opts': _lzf_complevel[complevel]}
        elif complib == 'gzip':
            args = {
                'shuffle': _lzf_gzip_shuffle[shuffle],
                'compression': complib,
                'compression_opts': _gzip_complevel[complevel]}
        elif complib in (None, False, 'none'):
            args = {
                'shuffle': False,
                'compression': None,
                'compression_opts': None}
        else:
            raise ValueError(f'unknown value for opt arg `complib`: {complib}')
        return args

    def _create_schema(self, *, remote_operation: bool = False):
        """stores the shape and dtype as the schema of a column.

        Parameters
        ----------
        remote_operation : optional, kwarg only, bool
            if this schema is being created from a remote fetch operation, then do not
            place the file symlink in the staging directory. Instead symlink it
            to a special remote staging directory. (default is False, which places the
            symlink in the stage data directory.)

        Notes
        -----

        Parameters set for raw-data-chunk-cache (rdcc) values:

        * rdcc_nbytes: sets the total size (measured in bytes) of the raw data chunk
          cache for each dataset. This should be set to the size of each chunk times
          the number of chunks that are likely to be needed in cache.
        * rdcc_w0: sets the policy for chunks to be removed from the cache when more
          space is needed. If set to 0, always evict the least recently used chunk in
          cache. If set to 1, always evict the least recently used chunk which has
          been fully read or written. If the value is between 0 and 1, the behavior
          will be a blend of the two.
        * rdcc_nslots: The number of chunk slots in the cache for this entire file.
          In order for quick lookup, a hash map is used for each chunk value. For
          maximum performance, this value should be set approximately 100 times that
          number of chunks.

        .. seealso::

            http://docs.h5py.org/en/stable/high/file.html#chunk-cache

        """

        # -------------------- Chunk & RDCC Vals ------------------------------
        schema_shape = self.schema_shape
        itemsize = np.dtype(self.schema_dtype).itemsize
        expectedrows = COLLECTION_SIZE * COLLECTION_COUNT
        maindim = 0

        chunk_shape = calc_chunkshape(schema_shape, expectedrows, itemsize, maindim)
        if chunk_shape == (1,) and schema_shape == ():
            schema_shape = (1,)
        req_chunks_per_dim = [math.ceil(i / j) for i, j in zip(schema_shape, chunk_shape)]
        req_shape = [i * j for i, j in zip(req_chunks_per_dim, chunk_shape)]
        chunk_nbytes = np.prod(chunk_shape) * itemsize
        nchunks = np.prod(req_chunks_per_dim)

        rdcc_nbytes_val = chunk_nbytes * nchunks * COLLECTION_SIZE
        if rdcc_nbytes_val >= CHUNK_MAX_RDCC_NBYTES:
            rdcc_nbytes_val = CHUNK_MAX_RDCC_NBYTES
        rdcc_nslots_guess = nchunks * expectedrows * 100
        rdcc_nslots_prime_val = find_next_prime(rdcc_nslots_guess)

        # ---------------------------- File Creation --------------------------

        uid = random_string()
        file_path = Path(self.DATADIR, f'{uid}.hdf5')
        self.wFp[uid] = h5py.File(file_path,
                                  mode='w',
                                  libver='latest',
                                  rdcc_nbytes=rdcc_nbytes_val,
                                  rdcc_w0=CHUNK_RDCC_W0,
                                  rdcc_nslots=rdcc_nslots_prime_val)
        self.w_uid = uid
        self.wdset = None
        self.hNextPath = 0
        self.hIdx = 0
        self.hColsRemain = COLLECTION_COUNT
        self.hMaxSize = COLLECTION_SIZE

        process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
        Path(process_dir, f'{uid}.hdf5').touch()

        # ----------------------- Dataset Creation ----------------------------

        optKwargs = self._dataset_opts(**self._dflt_backend_opts)
        for dset_num in range(COLLECTION_COUNT):
            self.wFp[uid].create_dataset(
                f'/{dset_num}',
                shape=(COLLECTION_SIZE, *req_shape),
                dtype=self.schema_dtype,
                chunks=(1, *chunk_shape),
                **optKwargs)

        # ---------------------- Attribute Config Vals ------------------------

        self.wFp[self.w_uid]['/'].attrs['HANGAR_VERSION'] = __version__
        self.wFp[self.w_uid]['/'].attrs['schema_shape'] = self.schema_shape
        self.wFp[self.w_uid]['/'].attrs['schema_dtype_num'] = np.dtype(self.schema_dtype).num
        self.wFp[self.w_uid]['/'].attrs['next_location'] = (0, 0)
        self.wFp[self.w_uid]['/'].attrs['collection_max_size'] = COLLECTION_SIZE
        self.wFp[self.w_uid]['/'].attrs['collection_total'] = COLLECTION_COUNT
        self.wFp[self.w_uid]['/'].attrs['collections_remaining'] = COLLECTION_COUNT
        self.wFp[self.w_uid]['/'].attrs['rdcc_nbytes'] = rdcc_nbytes_val
        self.wFp[self.w_uid]['/'].attrs['rdcc_w0'] = CHUNK_RDCC_W0
        self.wFp[self.w_uid]['/'].attrs['rdcc_nslots'] = rdcc_nslots_prime_val
        self.wFp[self.w_uid]['/'].attrs['chunk_shape'] = chunk_shape
        if optKwargs['compression_opts'] is not None:
            self.wFp[self.w_uid]['/'].attrs['compression_opts'] = optKwargs['compression_opts']
        else:
            self.wFp[self.w_uid]['/'].attrs['compression_opts'] = False

        self.wFp[self.w_uid].flush()
        try:
            self.wFp[self.w_uid].swmr_mode = True
        except ValueError:
            assert self.wFp[self.w_uid].swmr_mode is True
        self.wdset = self.wFp[self.w_uid][f'/{self.hNextPath}']

    def read_data(self, hashVal: HDF5_01_DataHashSpec) -> np.ndarray:
        """Read data from an hdf5 file handle at the specified locations

        Parameters
        ----------
        hashVal : HDF5_01_DataHashSpec
            record specification parsed from its serialized store val in lmdb.

        Returns
        -------
        np.array
            requested data
        """
        dsetCol = f'/{hashVal.dataset}'
        srcSlc = (hashVal.dataset_idx, *[slice(0, dim) for dim in hashVal.shape])
        rdictkey = f'{hashVal.uid}{dsetCol}'

        if self.schema_dtype:  # if is not None
            destArr = np.empty(hashVal.shape, self.schema_dtype)
            if rdictkey in self.rDatasets:
                self.rDatasets[rdictkey].read_direct(destArr, srcSlc, None)
            else:
                try:
                    self.Fp[hashVal.uid][dsetCol].read_direct(destArr, srcSlc, None)
                    self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                except TypeError:
                    self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
                    self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                    self.rDatasets[rdictkey].read_direct(destArr, srcSlc, None)
                except KeyError:
                    process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
                    if Path(process_dir, f'{hashVal.uid}.hdf5').is_file():
                        file_pth = self.DATADIR.joinpath(f'{hashVal.uid}.hdf5')
                        self.rFp[hashVal.uid] = h5py.File(file_pth, 'r', swmr=True, libver='latest')
                        self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                        self.rDatasets[rdictkey].read_direct(destArr, srcSlc, None)
                    else:
                        raise
        else:
            if rdictkey in self.rDatasets:
                destArr = self.rDatasets[rdictkey][srcSlc]
            else:
                try:
                    destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
                    self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                except TypeError:
                    self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
                    destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
                    self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                except KeyError:
                    process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
                    if Path(process_dir, f'{hashVal.uid}.hdf5').is_file():
                        file_pth = self.DATADIR.joinpath(f'{hashVal.uid}.hdf5')
                        self.rFp[hashVal.uid] = h5py.File(file_pth, 'r', swmr=True, libver='latest')
                        destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
                        self.rDatasets[rdictkey] = self.Fp[hashVal.uid][dsetCol]
                    else:
                        raise

        if xxh64_hexdigest(destArr) != hashVal.checksum:
            # try casting to check if dtype does not match for all zeros case
            destArr = destArr.astype(np.typeDict[self.Fp[hashVal.uid]['/'].attrs['schema_dtype_num']])
            if xxh64_hexdigest(destArr) != hashVal.checksum:
                raise RuntimeError(
                    f'DATA CORRUPTION Checksum {xxh64_hexdigest(destArr)} != recorded {hashVal}')
        return destArr

    def write_data(self, array: np.ndarray, *, remote_operation: bool = False) -> bytes:
        """verifies correctness of array data and performs write operation.

        Parameters
        ----------
        array : np.ndarray
            tensor to write to group.
        remote_operation : optional, kwarg only, bool
            If this is a remote process which is adding data, any necessary
            hdf5 dataset files will be created in the remote data dir instead
            of the stage directory. (default is False, which is for a regular
            access process)

        Returns
        -------
        bytes
            string identifying the collection dataset and collection dim-0 index
            which the array can be accessed at.
        """
        checksum = xxh64_hexdigest(array)
        if self.w_uid in self.wFp:
            self.hIdx += 1
            if self.hIdx >= self.hMaxSize:
                self.wdset.flush()
                self.hIdx = 0
                self.hNextPath += 1
                self.hColsRemain -= 1
                self.wdset = self.wFp[self.w_uid][f'/{self.hNextPath}']
                if self.hColsRemain <= 1:
                    self.wFp[self.w_uid]['/'].attrs.modify('next_location', (self.hNextPath, self.hIdx))
                    self.wFp[self.w_uid]['/'].attrs.modify('collections_remaining', self.hColsRemain)
                    self.wFp[self.w_uid].flush()
                    self._create_schema(remote_operation=remote_operation)
        else:
            self._create_schema(remote_operation=remote_operation)

        destSlc = (self.hIdx, *[slice(0, dim) for dim in array.shape])
        self.wdset.write_direct(array, None, destSlc)
        self.wdset.flush()
        res = hdf5_01_encode(self.w_uid, checksum, self.hNextPath, self.hIdx, array.shape)
        return res
