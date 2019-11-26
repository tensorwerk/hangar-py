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

Separators used
---------------

*  ``SEP_KEY: ":"``
*  ``SEP_HSH: "$"``
*  ``SEP_LST: " "``
*  ``SEP_SLC: "*"``

Examples
--------

1)  Adding the first piece of data to a file:

    *  Array shape (Subarray Shape): (10, 10)
    *  File UID: "rlUK3C"
    *  xxhash64_hexdigest: 8067007c0f05c359
    *  Dataset Number: 16
    *  Collection Index: 105

    ``Record Data => "01:rlUK3C$8067007c0f05c359$16 105*10 10"``

1)  Adding to a piece of data to a the middle of a file:

    *  Array shape (Subarray Shape): (20, 2, 3)
    *  File UID: "rlUK3C"
    *  xxhash64_hexdigest: b89f873d3d153a9c
    *  Dataset Number: "3"
    *  Collection Index: 199

    ``Record Data => "01:rlUK3C$b89f873d3d153a9c$8 199*20 2 3"``


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
import os
import re
import time
import logging
from collections import ChainMap
from os.path import join as pjoin
from os.path import splitext as psplitext
from functools import partial
from typing import (
    MutableMapping, NamedTuple, Tuple, Optional, Union, Callable, Pattern)

import numpy as np
import h5py
try:
    # hdf5plugin warns if a filter is already loaded. we temporarily surpress
    # that here, then reset the logger level to it's initial version.
    _logger = logging.getLogger('hdf5plugin')
    _initialLevel = _logger.getEffectiveLevel()
    _logger.setLevel(logging.ERROR)
    import hdf5plugin
    assert 'blosc' in hdf5plugin.FILTERS
except (ImportError, ModuleNotFoundError):  # pragma: no cover
    pass
finally:
    _logger.setLevel(_initialLevel)
from xxhash import xxh64_hexdigest

from .. import __version__
from .. import constants as c
from ..utils import find_next_prime, symlink_rel, random_string, set_blosc_nthreads

set_blosc_nthreads()


# ----------------------------- Configuration ---------------------------------


# contents of a single hdf5 file
COLLECTION_SIZE = 100
COLLECTION_COUNT = 100
CHUNK_MAX_RDCC_NBYTES = 500_000_000
CHUNK_RDCC_W0 = 0.75


# -------------------------------- Parser Implementation ----------------------


_FmtCode = '01'
# match and remove the following characters: '['   ']'   '('   ')'   ','
_ShapeFmtRE: Pattern = re.compile('[,\(\)\[\]]')
# split up a formated parsed string into unique fields
_patern = fr'\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}'
_SplitDecoderRE: Pattern = re.compile(fr'[{_patern}]')


HDF5_01_DataHashSpec = NamedTuple('HDF5_01_DataHashSpec', [
    ('backend', str),
    ('uid', str),
    ('checksum', str),
    ('dataset', str),
    ('dataset_idx', int),
    ('shape', Tuple[int])])


def hdf5_01_encode(uid: str, checksum: str, dataset: str, dataset_idx: int,
                   shape: Tuple[int]) -> bytes:
    """converts the hdf5 data has spec to an appropriate db value

    Parameters
    ----------
    uid : str
        the file name prefix which the data is written to.
    checksum : int
        xxhash_64.hex_digest checksum of the data bytes in numpy array form.
    dataset : str
        collection (ie. hdf5 dataset) name to find this data piece.
    dataset_idx : int
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
    out_str = f'{_FmtCode}{c.SEP_KEY}'\
              f'{uid}{c.SEP_HSH}{checksum}{c.SEP_HSH}'\
              f'{dataset}{c.SEP_LST}{dataset_idx}{c.SEP_SLC}'\
              f'{_ShapeFmtRE.sub("", str(shape))}'
    return out_str.encode()


def hdf5_01_decode(db_val: bytes) -> HDF5_01_DataHashSpec:
    """converts an hdf5 data hash db val into an hdf5 data python spec.

    Parameters
    ----------
    db_val : bytestring
        data hash db value

    Returns
    -------
    HDF5_01_DataHashSpec
        hdf5 data hash specification containing `backend`, `schema`,
        `instance`, `dataset`, `dataset_idx`, `shape`
    """
    db_str = db_val.decode()
    _, uid, checksum, dataset_vs, shape_vs = _SplitDecoderRE.split(db_str)
    dataset, dataset_idx = dataset_vs.split(c.SEP_LST)
    # if the data is of empty shape -> shape_vs = '' str.split() default value
    # of none means split according to any whitespace, and discard empty strings
    # from the result. So long as c.SEP_LST = ' ' this will work
    shape = tuple(int(x) for x in shape_vs.split())
    raw_val = HDF5_01_DataHashSpec(backend=_FmtCode,
                                   uid=uid,
                                   checksum=checksum,
                                   dataset=dataset,
                                   dataset_idx=int(dataset_idx),
                                   shape=shape)
    return raw_val


# -------------------------- Filter Heuristics --------------------------------


def hdf5_01_heuristic_filter_opts(prototype: np.ndarray) -> dict:
    """generate default filter options from a prototype array

    Parameters
    ----------
    prototype : :class:`numpy.ndarray`
        sample array of expected shape and datatype

    Returns
    -------
    dict
        mapping containing default filter opts that the hdf5_00 storage manager
        will accept.

    TODO
    ----

    Do something with the prototype arrays, or get rid of the argument, it's
    just taking up space at this point.
    """
    opts = {
        'default': {
            'shuffle': 'byte',
            'complib': 'blosc:lz4hc',
            'complevel': 5,
        },
        'backup': {
            'shuffle': 'byte',
            'complib': 'lzf',
            'complevel': None,
        },
    }
    hdf5BloscAvail = h5py.h5z.filter_avail(32001)
    opts = opts['default'] if hdf5BloscAvail else opts['backup']
    return opts


# ------------------------- Accessor Object -----------------------------------


HDF5_01_MapTypes = MutableMapping[str, Union[h5py.File, Callable[[], h5py.File]]]


class HDF5_01_FileHandles(object):
    """Manage HDF5 file handles.

    When in SWMR-write mode, no more than a single file handle can be in the
    "writeable" state. This is an issue where multiple arraysets may need to
    write to the same arrayset schema.
    """

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.path: os.PathLike = repo_path
        self.schema_shape: tuple = schema_shape
        self.schema_dtype: np.dtype = schema_dtype
        self._dflt_backend_opts: Optional[dict] = None

        self.rFp: HDF5_01_MapTypes = {}
        self.wFp: HDF5_01_MapTypes = {}
        self.Fp: HDF5_01_MapTypes = ChainMap(self.rFp, self.wFp)

        self.mode: Optional[str] = None
        self.hIdx: Optional[int] = None
        self.w_uid: Optional[str] = None
        self.hMaxSize: Optional[int] = None
        self.hNextPath: Optional[int] = None
        self.hColsRemain: Optional[int] = None

        self.slcExpr = np.s_
        self.slcExpr.maketuple = False

        self.STAGEDIR: os.PathLike = pjoin(self.path, c.DIR_DATA_STAGE, _FmtCode)
        self.REMOTEDIR: os.PathLike = pjoin(self.path, c.DIR_DATA_REMOTE, _FmtCode)
        self.DATADIR: os.PathLike = pjoin(self.path, c.DIR_DATA, _FmtCode)
        self.STOREDIR: os.PathLike = pjoin(self.path, c.DIR_DATA_STORE, _FmtCode)
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.w_uid in self.wFp:
            self.wFp[self.w_uid]['/'].attrs.modify('next_location', (self.hNextPath, self.hIdx))
            self.wFp[self.w_uid]['/'].attrs.modify('collections_remaining', self.hColsRemain)
            self.wFp[self.w_uid].flush()

    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        self.close()
        time.sleep(0.1)  # buffer time
        state = self.__dict__.copy()
        del state['rFp']
        del state['wFp']
        del state['Fp']
        return state

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)
        self.rFp = {}
        self.wFp = {}
        self.Fp = ChainMap(self.rFp, self.wFp)
        self.open(mode=self.mode)

    @property
    def backend_opts(self):
        return self._dflt_backend_opts

    @backend_opts.setter
    def backend_opts(self, val):
        if self.mode == 'a':
            self._dflt_backend_opts = val
            return
        else:
            raise AttributeError(f"can't set property in read only mode")

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
            if not os.path.isdir(process_dir):
                os.makedirs(process_dir)

            process_uids = [psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.hdf5')]
            for uid in process_uids:
                file_pth = pjoin(process_dir, f'{uid}.hdf5')
                self.rFp[uid] = partial(h5py.File, file_pth, 'r', swmr=True, libver='latest')

        if not remote_operation:
            if not os.path.isdir(self.STOREDIR):
                return
            store_uids = [psplitext(x)[0] for x in os.listdir(self.STOREDIR) if x.endswith('.hdf5')]
            for uid in store_uids:
                file_pth = pjoin(self.STOREDIR, f'{uid}.hdf5')
                self.rFp[uid] = partial(h5py.File, file_pth, 'r', swmr=True, libver='latest')

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
                self.hMaxSize = None
                self.hNextPath = None
                self.hIdx = None
                self.hColsRemain = None
                self.w_uid = None
            for uid in list(self.wFp.keys()):
                try:
                    self.wFp[uid].close()
                except AttributeError:
                    pass
                del self.wFp[uid]

        for uid in list(self.rFp.keys()):
            try:
                self.rFp[uid].close()
            except AttributeError:
                pass
            del self.rFp[uid]

    @staticmethod
    def delete_in_process_data(repo_path: os.PathLike, *, remote_operation=False) -> None:
        """Removes some set of files entirely from the stage/remote directory.

        DANGER ZONE. This should essentially only be used to perform hard resets
        of the repository state.

        Parameters
        ----------
        repo_path : os.PathLike
            path to the repository on disk
        remote_operation : optional, kwarg only, bool
            If true, modify contents of the remote_dir, if false (default) modify
            contents of the staging directory.
        """
        data_dir = pjoin(repo_path, c.DIR_DATA, _FmtCode)
        PDIR = c.DIR_DATA_STAGE if not remote_operation else c.DIR_DATA_REMOTE
        process_dir = pjoin(repo_path, PDIR, _FmtCode)
        if not os.path.isdir(process_dir):
            return

        process_uids = (psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.hdf5'))
        for process_uid in process_uids:
            remove_link_pth = pjoin(process_dir, f'{process_uid}.hdf5')
            remove_data_pth = pjoin(data_dir, f'{process_uid}.hdf5')
            os.remove(remove_link_pth)
            os.remove(remove_data_pth)
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
        _blosc_shuffle = {
            None: 0,
            'none': 0,
            'byte': 1,
            'bit': 2}
        _blosc_compression = {
            'blosc:blosclz': 0,
            'blosc:lz4': 1,
            'blosc:lz4hc': 2,
            # Not built 'snappy': 3,
            'blosc:zlib': 4,
            'blosc:zstd': 5}
        _blosc_complevel = {
            **{i: i for i in range(10)},
            None: 9,
            'none': 9}

        # ---- h5py built in filters ----
        _lzf_gzip_shuffle = {
            None: False,
            False: False,
            'none': False,
            True: True,
            'byte': True}
        _lzf_complevel = {
            False: None,
            None: None,
            'none': None}
        _gzip_complevel = {
            **{i: i for i in range(10)},
            None: 4,
            'none': 4}

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
        """stores the shape and dtype as the schema of a arrayset.

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

        chunk_shape = self.schema_shape
        chunk_nbytes = np.dtype(self.schema_dtype).itemsize * np.prod(chunk_shape)
        rdcc_nbytes_val = chunk_nbytes * COLLECTION_SIZE
        if rdcc_nbytes_val >= CHUNK_MAX_RDCC_NBYTES:
            rdcc_nbytes_val = CHUNK_MAX_RDCC_NBYTES
        rdcc_nslots_guess = np.math.ceil(rdcc_nbytes_val / chunk_nbytes) * 100
        rdcc_nslots_prime_val = find_next_prime(rdcc_nslots_guess)

        # ---------------------------- File Creation --------------------------

        uid = random_string()
        file_path = pjoin(self.DATADIR, f'{uid}.hdf5')
        self.wFp[uid] = h5py.File(file_path,
                                  mode='w',
                                  libver='latest',
                                  rdcc_nbytes=rdcc_nbytes_val,
                                  rdcc_w0=CHUNK_RDCC_W0,
                                  rdcc_nslots=rdcc_nslots_prime_val)
        self.w_uid = uid
        self.hNextPath = 0
        self.hIdx = 0
        self.hColsRemain = COLLECTION_COUNT
        self.hMaxSize = COLLECTION_SIZE

        if remote_operation:
            symlink_file_path = pjoin(self.REMOTEDIR, f'{uid}.hdf5')
        else:
            symlink_file_path = pjoin(self.STAGEDIR, f'{uid}.hdf5')
        symlink_rel(file_path, symlink_file_path)

        # ----------------------- Dataset Creation ----------------------------

        optKwargs = self._dataset_opts(**self._dflt_backend_opts)
        for dset_num in range(COLLECTION_COUNT):
            self.wFp[uid].create_dataset(
                f'/{dset_num}',
                shape=(COLLECTION_SIZE, *chunk_shape),
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

    def read_data(self, hashVal: HDF5_01_DataHashSpec) -> np.ndarray:
        """Read data from an hdf5 file handle at the specified locations

        Parameters
        ----------
        hashVal : HDF5_01_DataHashSpec
            record specification parsed from its serialized store val in lmdb.

        Returns
        -------
        np.array
            requested data.
        """
        dsetCol = f'/{hashVal.dataset}'
        # if len(hashVal.shape) > 0:
        srcSlc = (self.slcExpr[int(hashVal.dataset_idx)], ...)
        destSlc = None

        if self.schema_dtype is not None:
            destArr = np.empty(hashVal.shape, self.schema_dtype)
            try:
                self.Fp[hashVal.uid][dsetCol].read_direct(destArr, srcSlc, destSlc)
            except TypeError:
                self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
                self.Fp[hashVal.uid][dsetCol].read_direct(destArr, srcSlc, destSlc)
            except KeyError:
                process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
                file_pth = pjoin(process_dir, f'{hashVal.uid}.hdf5')
                if os.path.islink(file_pth):
                    self.rFp[hashVal.uid] = h5py.File(file_pth, 'r', swmr=True, libver='latest')
                    self.Fp[hashVal.uid][dsetCol].read_direct(destArr, srcSlc, destSlc)
                else:
                    raise
        else:
            try:
                destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
            except TypeError:
                self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
                destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
            except KeyError:
                process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
                file_pth = pjoin(process_dir, f'{hashVal.uid}.hdf5')
                if os.path.islink(file_pth):
                    self.rFp[hashVal.uid] = h5py.File(file_pth, 'r', swmr=True, libver='latest')
                    destArr = self.Fp[hashVal.uid][dsetCol][srcSlc]
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
                self.hIdx = 0
                self.hNextPath += 1
                self.hColsRemain -= 1
                if self.hColsRemain <= 1:
                    self.wFp[self.w_uid]['/'].attrs.modify('next_location', (self.hNextPath, self.hIdx))
                    self.wFp[self.w_uid]['/'].attrs.modify('collections_remaining', self.hColsRemain)
                    self.wFp[self.w_uid].flush()
                    self._create_schema(remote_operation=remote_operation)
        else:
            self._create_schema(remote_operation=remote_operation)

        self.wFp[self.w_uid][f'/{self.hNextPath}'][self.hIdx] = array
        hashVal = hdf5_01_encode(uid=self.w_uid,
                                 checksum=checksum,
                                 dataset=self.hNextPath,
                                 dataset_idx=self.hIdx,
                                 shape=array.shape)
        return hashVal
