"""Local Numpy memmap Backend Implementation, Identifier: ``NUMPY_10``

Backend Identifiers
===================

*  Backend: ``1``
*  Version: ``0``
*  Format Code: ``10``
*  Canonical Name: ``NUMPY_10``

Storage Method
==============

* Data is written to specific subarray indexes inside a numpy memmapped array on disk.

* Each file is a zero-initialized array of

  *  ``dtype: {schema_dtype}``; ie ``np.float32`` or ``np.uint8``

  *  ``shape: (COLLECTION_SIZE, *{schema_shape})``; ie ``(500, 10)`` or ``(500,
     4, 3)``. The first index in the array is referred to as a "collection
     index".

Compression Options
===================

Does not accept any compression options. No compression is applied.

Record Format
=============

Fields Recorded for Each Array
------------------------------

*  Format Code
*  File UID
*  xxhash64_hexdigest
*  Collection Index (0:COLLECTION_SIZE subarray selection)
*  Subarray Shape

Separators used
---------------

*  ``SEP_KEY: ":"``
*  ``SEP_HSH: "$"``
*  ``SEP_SLC: "*"``

Examples
--------

1)  Adding the first piece of data to a file:

    *  Array shape (Subarray Shape): (10, 10)
    *  File UID: "K3ktxv"
    *  xxhash64_hexdigest: 94701dd9f32626e2
    *  Collection Index: 488

    ``Record Data =>  "10:K3ktxv$94701dd9f32626e2$488*10 10"``

2)  Adding to a piece of data to a the middle of a file:

    *  Array shape (Subarray Shape): (20, 2, 3)
    *  File UID: "Mk23nl"
    *  xxhash64_hexdigest: 1363344b6c051b29
    *  Collection Index: 199

    ``Record Data => "10:Mk23nl$1363344b6c051b29$199*20 2 3"``


Technical Notes
===============

*  A typical numpy memmap file persisted to disk does not retain information
   about its datatype or shape, and as such must be provided when re-opened
   after close. In order to persist a memmap in ``.npy`` format, we use the a
   special function ``open_memmap`` imported from ``np.lib.format`` which can
   open a memmap file and persist necessary header info to disk in ``.npy``
   format.

*  On each write, an ``xxhash64_hexdigest`` checksum is calculated. This is not
   for use as the primary hash algorithm, but rather stored in the local record
   format itself to serve as a quick way to verify no disk corruption occurred.
   This is required since numpy has no built in data integrity validation
   methods when reading from disk.
"""
import os
import re
from collections import ChainMap
from functools import partial
from os.path import join as pjoin
from os.path import splitext as psplitext
from typing import MutableMapping, NamedTuple, Tuple, Optional
from xxhash import xxh64_hexdigest

import numpy as np
from numpy.lib.format import open_memmap

from .. import constants as c
from ..utils import random_string, symlink_rel


# ----------------------------- Configuration ---------------------------------


# number of subarray contents of a single numpy memmap file
COLLECTION_SIZE = 1000


# -------------------------------- Parser Implementation ----------------------

_FmtCode = '10'
# match and remove the following characters: '['   ']'   '('   ')'   ','
_ShapeFmtRE = re.compile('[,\(\)\[\]]')
# split up a formated parsed string into unique fields
_patern = fr'\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}'
_SplitDecoderRE = re.compile(fr'[{_patern}]')


NUMPY_10_DataHashSpec = NamedTuple('NUMPY_10_DataHashSpec',
                                   [('backend', str), ('uid', str),
                                    ('checksum', str), ('collection_idx', int),
                                    ('shape', Tuple[int])])


def numpy_10_encode(uid: str, checksum: str, collection_idx: int, shape: tuple) -> bytes:
    """converts the numpy data spect to an appropriate db value

    Parameters
    ----------
    uid : str
        file name (schema uid) of the np file to find this data piece in.
    checksum : int
        xxhash64_hexdigest checksum of the data as computed on that local machine.
    collection_idx : int
        collection first axis index in which this data piece resides.
    shape : tuple
        shape of the data sample written to the collection idx. ie: what
        subslices of the array should be read to retrieve the sample as
        recorded.

    Returns
    -------
    bytes
        hash data db value recording all input specifications
    """
    out_str = f'{_FmtCode}{c.SEP_KEY}'\
              f'{uid}{c.SEP_HSH}{checksum}{c.SEP_HSH}'\
              f'{collection_idx}{c.SEP_SLC}'\
              f'{_ShapeFmtRE.sub("", str(shape))}'
    return out_str.encode()


def numpy_10_decode(db_val: bytes) -> NUMPY_10_DataHashSpec:
    """converts a numpy data hash db val into a numpy data python spec

    Parameters
    ----------
    db_val : bytes
        data hash db val

    Returns
    -------
    DataHashSpec
        numpy data hash specification containing `backend`, `schema`, and
        `uid`, `collection_idx` and `shape` fields.
    """
    db_str = db_val.decode()
    _, uid, checksum, collection_idx, shape_vs = _SplitDecoderRE.split(db_str)
    # if the data is of empty shape -> shape_vs = '' str.split() default value
    # of none means split according to any whitespace, and discard empty strings
    # from the result. So long as c.SEP_LST = ' ' this will work
    shape = tuple(int(x) for x in shape_vs.split())
    raw_val = NUMPY_10_DataHashSpec(backend=_FmtCode,
                                    uid=uid,
                                    checksum=checksum,
                                    collection_idx=int(collection_idx),
                                    shape=shape)
    return raw_val


# ------------------------- Accessor Object -----------------------------------


class NUMPY_10_FileHandles(object):

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.repo_path = repo_path
        self.schema_shape = schema_shape
        self.schema_dtype = schema_dtype
        self._dflt_backend_opts: Optional[dict] = None

        self.rFp: MutableMapping[str, np.memmap] = {}
        self.wFp: MutableMapping[str, np.memmap] = {}
        self.Fp = ChainMap(self.rFp, self.wFp)

        self.mode: str = None
        self.w_uid: str = None
        self.hIdx: int = None

        self.slcExpr = np.s_
        self.slcExpr.maketuple = False

        self.STAGEDIR = pjoin(self.repo_path, c.DIR_DATA_STAGE, _FmtCode)
        self.REMOTEDIR = pjoin(self.repo_path, c.DIR_DATA_REMOTE, _FmtCode)
        self.DATADIR = pjoin(self.repo_path, c.DIR_DATA, _FmtCode)
        self.STOREDIR = pjoin(self.repo_path, c.DIR_DATA_STORE, _FmtCode)
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.w_uid in self.wFp:
            self.wFp[self.w_uid].flush()

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
        """open numpy file handle coded directories

        Parameters
        ----------
        mode : str
            one of `a` for `write-enabled` mode or `r` for read-only
        remote_operation : bool, optional, kwarg only
            True if remote operations call this method. Changes the symlink
            directories used while writing., by default False
        """
        self.mode = mode
        if self.mode == 'a':
            process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
            if not os.path.isdir(process_dir):
                os.makedirs(process_dir)

            process_uids = [psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.npy')]
            for uid in process_uids:
                file_pth = pjoin(process_dir, f'{uid}.npy')
                self.rFp[uid] = partial(open_memmap, file_pth, 'r')

        if not remote_operation:
            if not os.path.isdir(self.STOREDIR):
                return
            store_uids = [psplitext(x)[0] for x in os.listdir(self.STOREDIR) if x.endswith('.npy')]
            for uid in store_uids:
                file_pth = pjoin(self.STOREDIR, f'{uid}.npy')
                self.rFp[uid] = partial(open_memmap, file_pth, 'r')

    def close(self, *args, **kwargs):
        """Close any open file handles.
        """
        if self.mode == 'a':
            if self.w_uid in self.wFp:
                self.wFp[self.w_uid].flush()
                self.w_uid = None
                self.hIdx = None
            for k in list(self.wFp.keys()):
                del self.wFp[k]

        for k in list(self.rFp.keys()):
            del self.rFp[k]

    @staticmethod
    def delete_in_process_data(repo_path, *, remote_operation=False):
        """Removes some set of files entirely from the stage/remote directory.

        DANGER ZONE. This should essentially only be used to perform hard resets
        of the repository state.

        Parameters
        ----------
        repo_path : str
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

        process_uids = (psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.npy'))
        for process_uid in process_uids:
            remove_link_pth = pjoin(process_dir, f'{process_uid}.npy')
            remove_data_pth = pjoin(data_dir, f'{process_uid}.npy')
            os.remove(remove_link_pth)
            os.remove(remove_data_pth)
        os.rmdir(process_dir)

    def _create_schema(self, *, remote_operation: bool = False):
        """stores the shape and dtype as the schema of a arrayset.

        Parameters
        ----------
        remote_operation : optional, kwarg only, bool
            if this schema is being created from a remote fetch operation, then do not
            place the file symlink in the staging directory. Instead symlink it
            to a special remote staging directory. (default is False, which places the
            symlink in the stage data directory.)
        """
        uid = random_string()
        file_path = pjoin(self.DATADIR, f'{uid}.npy')
        m = open_memmap(file_path,
                        mode='w+',
                        dtype=self.schema_dtype,
                        shape=(COLLECTION_SIZE, *self.schema_shape))
        self.wFp[uid] = m
        self.w_uid = uid
        self.hIdx = 0

        if remote_operation:
            symlink_file_path = pjoin(self.REMOTEDIR, f'{uid}.npy')
        else:
            symlink_file_path = pjoin(self.STAGEDIR, f'{uid}.npy')
        symlink_rel(file_path, symlink_file_path)

    def read_data(self, hashVal: NUMPY_10_DataHashSpec) -> np.ndarray:
        """Read data from disk written in the numpy_00 fmtBackend

        Parameters
        ----------
        hashVal : NUMPY_10_DataHashSpec
            record specification stored in the db

        Returns
        -------
        np.ndarray
            tensor data stored at the provided hashVal specification.

        Raises
        ------
        RuntimeError
            If the recorded checksum does not match the received checksum.

        Notes
        -----

        TO AVOID DATA LOSS / CORRUPTION:

        * On a read operation, we copy memmap subarray tensor data to a new
          `np.ndarray` instance so as to prevent writes on a raw memmap result
          slice (a `np.memmap` instance) from propogating to data on disk.

        * This is an issue for reads from a write-enabled checkout where data
          was just written, since the np flag "WRITEABLE" and "OWNDATA" will be
          true, and writes to the returned array would be overwrite that data
          slice on disk.

        * For read-only checkouts, modifications to the resultant array would
          perform a "copy on write"-like operation which would be propogated to
          all future reads of the subarray from that process, but which would
          not be persisted to disk.
        """
        srcSlc = (self.slcExpr[hashVal.collection_idx],
                  *(self.slcExpr[0:x] for x in hashVal.shape))
        try:
            res = self.Fp[hashVal.uid][srcSlc]
        except TypeError:
            self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
            res = self.Fp[hashVal.uid][srcSlc]
        except KeyError:
            process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
            file_pth = pjoin(process_dir, f'{hashVal.uid}.npy')
            if os.path.islink(file_pth):
                self.rFp[hashVal.uid] = open_memmap(file_pth, 'r')
                res = self.Fp[hashVal.uid][srcSlc]
            else:
                raise

        out = np.array(res, dtype=res.dtype, order='C')
        if xxh64_hexdigest(out) != hashVal.checksum:
            raise RuntimeError(
                f'DATA CORRUPTION Checksum {xxh64_hexdigest(out)} != recorded {hashVal}')
        return out

    def write_data(self, array: np.ndarray, *, remote_operation: bool = False) -> bytes:
        """writes array data to disk in the numpy_00 fmtBackend

        Parameters
        ----------
        array : np.ndarray
            tensor to write to disk
        remote_operation : bool, optional, kwarg only
            True if writing in a remote operation, otherwise False. Default is
            False

        Returns
        -------
        bytes
            db hash record value specifying location information
        """
        checksum = xxh64_hexdigest(array)
        if self.w_uid in self.wFp:
            self.hIdx += 1
            if self.hIdx >= COLLECTION_SIZE:
                self.wFp[self.w_uid].flush()
                self._create_schema(remote_operation=remote_operation)
        else:
            self._create_schema(remote_operation=remote_operation)

        destSlc = (self.slcExpr[self.hIdx], *(self.slcExpr[0:x] for x in array.shape))
        self.wFp[self.w_uid][destSlc] = array
        hashVal = numpy_10_encode(uid=self.w_uid,
                                  checksum=checksum,
                                  collection_idx=self.hIdx,
                                  shape=array.shape)
        return hashVal
