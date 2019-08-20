"""Local HDF5 Backend Implementation, Identifier: ``HDF5_00``

Backend Identifiers
===================

*  Backend: ``0``
*  Version: ``0``
*  Format Code: ``00``
*  Canonical Name: ``HDF5_00``

Storage Method
==============

*  Data is written to specific subarray indexes inside an HDF5 "dataset" in a
   single HDF5 File.

*  In each HDF5 File there are ``COLLECTION_COUNT`` "datasets" (named ``["0" :
   "{COLLECTION_COUNT}"]``). These are referred to as ``"dataset number"``

*  Each dataset is a zero-initialized array of:

   *  ``dtype: {schema_dtype}``; ie ``np.float32`` or ``np.uint8``

   *  ``shape: (COLLECTION_SIZE, *{schema_shape})``; ie ``(500, 10)`` or ``(500,
      4, 3)``. The first index in the dataset is referred to as a ``collection
      index``.

*  Compression Filters, Chunking Configuration/Options are applied globally for
   all ``datasets`` in a file at dataset creation time.

*  For subarrays with rank < rank of schema (in ``variable_sized`` arraysets),
   the data is written to the ``zeroth`` index of a larger hyperslab selection
   normally filled by a full rank sample. While this allows lower rank samples
   to be added, is complicates subarray slicing because we need to
   differentiate between user defined sample shapes where an array is provided
   with a zero-length dimension (``unsqueeze(0)``) and those which hangar
   specified for the subarray slice. A special shape value ``x`` is used to
   indicate this in the ``shape`` record field. When hangar encounters this,
   it will not create a ranged slice object, but a fixed dimension field
   (``value = 0``) upon reading/writing. Outisde of reading/writing, this
   ``x`` value will remain in the ``shape`` record.

Record Format
=============

Fields Recorded for Each Array
------------------------------

*  Format Code
*  File UID
*  Dataset Number (``0:COLLECTION_COUNT`` dataset selection)
*  Collection Index (``0:COLLECTION_SIZE`` dataset subarray selection)
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

    *  Array shape (Subarray Shape): (10)
    *  File UID: "2HvGf9"
    *  Dataset Number: "0"
    *  Collection Index: 0

   ``Record Data => "00:2HvGf9$0 0*10"``

2)  Adding to a piece of data to a the middle of a file:

    *  Array shape (Subarray Shape): (20, 2, 3)
    *  File UID: "WzUtdu"
    *  Dataset Number: "3"
    *  Collection Index: 199

    ``Record Data => "00:WzUtdu$3 199*20 2 3"``

3)  Adding to a piece of data with rank dimension < schema:

    *  Array shape (Subarray Shape): (5)  # schema rank = 3
    *  File UID: "O2EBcV"
    *  Dataset Number: "0"
    *  Collection Index: 2

    ``Record Data => b'00:O2EBcV$0 2*x x 5'``


Technical Notes
===============

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
"""
import math
import os
import re
import time
from collections import ChainMap
from os.path import join as pjoin
from os.path import splitext as psplitext
from functools import partial
from typing import MutableMapping, NamedTuple, Tuple, Optional, Union, Callable, Pattern

import h5py
import numpy as np

from .. import __version__
from .. import constants as c
from ..utils import find_next_prime, symlink_rel, random_string, set_blosc_nthreads

set_blosc_nthreads()


# ----------------------------- Configuration ---------------------------------


# contents of a single hdf5 file
COLLECTION_SIZE = 250
COLLECTION_COUNT = 100

# chunking options for compression schemes
CHUNK_MAX_NBYTES = 200_000  # < 256 KB to fit in L2 CPU Cache
CHUNK_MAX_RDCC_NBYTES = 100_000_000
CHUNK_RDCC_W0 = 0.75

# filter definition and backup selection if not available.
filter_opts = {
    'default': {
        'shuffle': 'bit',
        'complib': 'blosc:lz4',
        'complevel': 7,
        'fletcher32': True},
    'backup': {
        'shuffle': True,
        'complib': 'lzf',
        'complevel': None,
        'fletcher32': True},
}
hdf5BloscAvail = h5py.h5z.filter_avail(32001)
HDF5_FILTER = filter_opts['default'] if hdf5BloscAvail else filter_opts['backup']


# -------------------------------- Parser Implementation ----------------------


_FmtCode = '00'
# match and remove the following characters: '['   ']'   '('   ')'   ','
_ShapeFmtRE: Pattern = re.compile("[',\(\)\[\]]")
# split up a formated parsed string into unique fields
_SplitDecoderRE: Pattern = re.compile(fr'[\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}]')


HDF5_00_DataHashSpec = NamedTuple('HDF5_00_DataHashSpec',
                                  [('backend', str), ('uid', str),
                                   ('dataset', str), ('dataset_idx', int),
                                   ('shape', Tuple[int])])


def hdf5_00_encode(uid: str, dataset: str, dataset_idx: int, shape: Tuple[int]) -> bytes:
    """converts the hdf5 data has spec to an appropriate db value

    Parameters
    ----------
    uid : str
        the file name prefix which the data is written to.
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
    out_str = f'{_FmtCode}{c.SEP_KEY}{uid}'\
              f'{c.SEP_HSH}'\
              f'{dataset}{c.SEP_LST}{dataset_idx}'\
              f'{c.SEP_SLC}'\
              f'{_ShapeFmtRE.sub("", str(shape))}'
    return out_str.encode()


def hdf5_00_decode(db_val: bytes) -> HDF5_00_DataHashSpec:
    """converts an hdf5 data hash db val into an hdf5 data python spec.

    Parameters
    ----------
    db_val : bytestring
        data hash db value

    Returns
    -------
    HDF5_00_DataHashSpec
        hdf5 data hash specification containing `backend`, `schema`,
        `instance`, `dataset`, `dataset_idx`, `shape`
    """
    db_str = db_val.decode()
    _, uid, dataset_vs, shape_vs = _SplitDecoderRE.split(db_str)
    dataset, dataset_idx = dataset_vs.split(c.SEP_LST)
    # if the data is of empty shape -> shape_vs = '' str.split() default value
    # of none means split according to any whitespace, and discard empty strings
    # from the result. So long as c.SEP_LST = ' ' this will work
    shape = tuple([int(x) if x != 'x' else 'x' for x in shape_vs.split()])
    raw_val = HDF5_00_DataHashSpec(backend=_FmtCode,
                                   uid=uid,
                                   dataset=dataset,
                                   dataset_idx=int(dataset_idx),
                                   shape=shape)
    return raw_val


# ------------------------- Accessor Object -----------------------------------


HDF5_00_MapTypes = MutableMapping[str, Union[h5py.File, Callable[[], h5py.File]]]


class HDF5_00_FileHandles(object):
    """Manage HDF5 file handles.

    When in SWMR-write mode, no more than a single file handle can be in the
    "writeable" state. This is an issue where multiple arraysets may need to
    write to the same arrayset schema.
    """

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.path: os.PathLike = repo_path
        self.schema_shape: tuple = schema_shape
        self.schema_dtype: np.dtype = schema_dtype

        self.rFp: HDF5_00_MapTypes = {}
        self.wFp: HDF5_00_MapTypes = {}
        self.Fp: HDF5_00_MapTypes = ChainMap(self.rFp, self.wFp)

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

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)
        self.rFp = {}
        self.wFp = {}
        self.Fp = ChainMap(self.rFp, self.wFp)
        self.open(self.mode)

    def open(self, mode: str, *, remote_operation: bool = False):
        """Open an hdf5 file handle in the Handler Singleton

        Parameters
        ----------
        mode : str
            one of `r` or `a` for read only / read-write.
        repote_operation : optional, kwarg only, bool
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
    def _dataset_opts(complib: str, complevel: int, shuffle: Union[bool, str], fletcher32: bool) -> dict:
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
            'blosc:lz4', 'blosc:lz4hc', 'blosc:snappy', 'blosc:zlib', 'blosc:zstd']
        complevel : int
            compression level to specify (accepts values [0, 9] for all except 'lzf'
            where no complevel is accepted)
        shuffle : bool
            if True, enable byte shuffle filter, if blosc compression, pass through
            'bits' is accepted as well.
        fletcher32 : bool
            enable fletcher32 checksum validation of data integrity, (defaults to
            True, which is enabled)
        """
        if complib.startswith('blosc'):
            shuffle = 2 if shuffle == 'bit' else 1 if shuffle else 0
            compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
            complib = ['blosc:' + c for c in compressors].index(complib)
            args = {
                'compression': 32001,
                'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib),
                'fletcher32': fletcher32,
            }
            if shuffle:
                args['shuffle'] = False
        else:
            args = {
                'shuffle': shuffle,
                'compression': complib,
                'compression_opts': None if complib == 'lzf' else complevel,
                'fletcher32': fletcher32,
            }
        return args

    @staticmethod
    def _chunk_opts(sample_array: np.ndarray, max_chunk_nbytes: int) -> Tuple[list, int]:
        """Determine the chunk shape so each array chunk fits into configured nbytes.

        Currently the chunk nbytes are not user configurable. Instead the constant
        `HDF5_MAX_CHUNK_NBYTES` is sued to determine when to split.

        Parameters
        ----------
        sample_array : `np.array`
            Sample array whose shape and dtype should be used as the basis of the
            chunk shape determination
        max_chunk_nbytes : int
            how many bytes the array chunks should be limited to.

        Returns
        -------
        list
            list of ints of length == rank of `sample_array` specifying chunk sizes
            to split `sample_array` into nbytes
        int
            nbytes which the chunk will fit in. Will be <= `HDF5_MAX_CHUNK_NBYTES`
        """
        chunk_nbytes = sample_array.nbytes
        chunk_shape = list(sample_array.shape)
        shape_rank = len(chunk_shape)
        chunk_idx = 0

        while chunk_nbytes > max_chunk_nbytes:
            if chunk_idx >= shape_rank:
                chunk_idx = 0
            rank_dim = chunk_shape[chunk_idx]
            if rank_dim <= 2:
                chunk_idx += 1
                continue
            chunk_shape[chunk_idx] = math.floor(rank_dim / 2)
            chunk_nbytes = np.zeros(shape=chunk_shape, dtype=sample_array.dtype).nbytes
            chunk_idx += 1

        return (chunk_shape, chunk_nbytes)

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

        sample_array = np.zeros(self.schema_shape, dtype=self.schema_dtype)
        chunk_shape, chunk_nbytes = __class__._chunk_opts(
            sample_array=sample_array,
            max_chunk_nbytes=CHUNK_MAX_NBYTES)

        rdcc_nbytes_val = sample_array.nbytes * COLLECTION_SIZE
        if rdcc_nbytes_val < CHUNK_MAX_NBYTES:
            rdcc_nbytes_val = CHUNK_MAX_NBYTES
        elif rdcc_nbytes_val > CHUNK_MAX_RDCC_NBYTES:
            rdcc_nbytes_val = CHUNK_MAX_RDCC_NBYTES

        rdcc_nslots_guess = math.ceil(rdcc_nbytes_val / chunk_nbytes) * 100
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

        optKwargs = self._dataset_opts(**HDF5_FILTER)
        for dset_num in range(COLLECTION_COUNT):
            self.wFp[uid].create_dataset(
                f'/{dset_num}',
                shape=(COLLECTION_SIZE, *sample_array.shape),
                dtype=sample_array.dtype,
                maxshape=(COLLECTION_SIZE, *sample_array.shape),
                chunks=(1, *chunk_shape),
                **optKwargs)

        # ---------------------- Attribute Config Vals ------------------------

        self.wFp[self.w_uid]['/'].attrs['HANGAR_VERSION'] = __version__
        self.wFp[self.w_uid]['/'].attrs['schema_shape'] = sample_array.shape
        self.wFp[self.w_uid]['/'].attrs['schema_dtype_num'] = sample_array.dtype.num
        self.wFp[self.w_uid]['/'].attrs['next_location'] = (0, 0)
        self.wFp[self.w_uid]['/'].attrs['collection_max_size'] = COLLECTION_SIZE
        self.wFp[self.w_uid]['/'].attrs['collection_total'] = COLLECTION_COUNT
        self.wFp[self.w_uid]['/'].attrs['collections_remaining'] = COLLECTION_COUNT
        self.wFp[self.w_uid]['/'].attrs['rdcc_nbytes'] = rdcc_nbytes_val
        self.wFp[self.w_uid]['/'].attrs['rdcc_w0'] = CHUNK_RDCC_W0
        self.wFp[self.w_uid]['/'].attrs['rdcc_nslots'] = rdcc_nslots_prime_val
        self.wFp[self.w_uid]['/'].attrs['shuffle'] = HDF5_FILTER['shuffle']
        self.wFp[self.w_uid]['/'].attrs['complib'] = HDF5_FILTER['complib']
        self.wFp[self.w_uid]['/'].attrs['fletcher32'] = HDF5_FILTER['fletcher32']
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

    def read_data(self, hashVal: HDF5_00_DataHashSpec) -> np.ndarray:
        """Read data from an hdf5 file handle at the specified locations

        Parameters
        ----------
        hashVal : HDF5_00_DataHashSpec
            record specification parsed from its serialized store val in lmdb.

        Returns
        -------
        np.array
            requested data.
        """
        dsetIdx = int(hashVal.dataset_idx)
        dsetCol = f'/{hashVal.dataset}'

        srcSlc = (self.slcExpr[dsetIdx],
                  *(self.slcExpr[0:x] if x != 'x' else 0 for x in hashVal.shape))
        destSlc = None

        if self.schema_dtype is not None:
            outShape = hashVal.shape[hashVal.shape.count('x'):]
            destArr = np.empty((outShape), self.schema_dtype)
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

        nDimPad = len(self.schema_shape) - array.ndim
        padShape = tuple((*(nDimPad * ['x']), *array.shape))

        srcSlc = None
        destSlc = (self.slcExpr[self.hIdx],
                   *(self.slcExpr[0:x] if x != 'x' else 0 for x in padShape))
        self.wFp[self.w_uid][f'/{self.hNextPath}'].write_direct(array, srcSlc, destSlc)

        hashVal = hdf5_00_encode(uid=self.w_uid,
                                 dataset=self.hNextPath,
                                 dataset_idx=self.hIdx,
                                 shape=padShape)
        return hashVal
