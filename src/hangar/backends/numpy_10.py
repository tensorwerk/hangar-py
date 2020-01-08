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

Examples
--------

1)  Adding the first piece of data to a file:

    *  Array shape (Subarray Shape): (10, 10)
    *  File UID: "K3ktxv"
    *  xxhash64_hexdigest: 94701dd9f32626e2
    *  Collection Index: 488

    ``Record Data =>  "10:K3ktxv:94701dd9f32626e2:488:10 10"``

2)  Adding to a piece of data to a the middle of a file:

    *  Array shape (Subarray Shape): (20, 2, 3)
    *  File UID: "Mk23nl"
    *  xxhash64_hexdigest: 1363344b6c051b29
    *  Collection Index: 199

    ``Record Data => "10:Mk23nl:1363344b6c051b29:199:20 2 3"``


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
from pathlib import Path
from typing import MutableMapping, Optional

import numpy as np
from numpy.lib.format import open_memmap
from xxhash import xxh64_hexdigest

from ..constants import DIR_DATA_REMOTE, DIR_DATA_STAGE, DIR_DATA_STORE, DIR_DATA
from ..utils import random_string
from .specs import NUMPY_10_DataHashSpec

# ----------------------------- Configuration ---------------------------------

# number of subarray contents of a single numpy memmap file
COLLECTION_SIZE = 1000

# -------------------------------- Parser Implementation ----------------------

_FmtCode = '10'
# # match and remove the following characters: '['   ']'   '('   ')'   ','
_SRe = re.compile('[,\(\)\[\]]')


def numpy_10_encode(uid: str, cksum: str, collection_idx: int, shape: tuple) -> bytes:
    """converts the numpy data spect to an appropriate db value

    Parameters
    ----------
    uid : str
        file name (schema uid) of the np file to find this data piece in.
    cksum : int
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
    return f'10:{uid}:{cksum}:{collection_idx}:{_SRe.sub("", str(shape))}'.encode()


# ------------------------- Accessor Object -----------------------------------


class NUMPY_10_FileHandles(object):

    def __init__(self, repo_path: Path, schema_shape: tuple, schema_dtype: np.dtype):
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

        self.STAGEDIR: Path = Path(self.repo_path, DIR_DATA_STAGE, _FmtCode)
        self.REMOTEDIR: Path = Path(self.repo_path, DIR_DATA_REMOTE, _FmtCode)
        self.DATADIR: Path = Path(self.repo_path, DIR_DATA, _FmtCode)
        self.STOREDIR: Path = Path(self.repo_path, DIR_DATA_STORE, _FmtCode)
        self.DATADIR.mkdir(exist_ok=True)

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
            process_dir.mkdir(exist_ok=True)
            for uidpth in process_dir.iterdir():
                if uidpth.suffix == '.npy':
                    file_pth = self.DATADIR.joinpath(uidpth.name)
                    self.rFp[uidpth.stem] = partial(open_memmap, file_pth, 'r')

        if not remote_operation:
            if not self.STOREDIR.is_dir():
                return
            for uidpth in self.STOREDIR.iterdir():
                if uidpth.suffix == '.npy':
                    file_pth = self.DATADIR.joinpath(uidpth.name)
                    self.rFp[uidpth.stem] = partial(open_memmap, file_pth, 'r')

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
    def delete_in_process_data(repo_path: Path, *, remote_operation: bool = False):
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
        pdir = DIR_DATA_STAGE if not remote_operation else DIR_DATA_REMOTE
        process_dir = Path(repo_path, pdir, _FmtCode)
        if not process_dir.is_dir():
            return

        for uidpth in process_dir.iterdir():
            if uidpth.suffix == '.npy':
                os.remove(process_dir.joinpath(uidpth.name))
                os.remove(data_dir.joinpath(uidpth.name))
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
        file_path = self.DATADIR.joinpath(f'{uid}.npy')
        m = open_memmap(file_path,
                        mode='w+',
                        dtype=self.schema_dtype,
                        shape=(COLLECTION_SIZE, *self.schema_shape))
        self.wFp[uid] = m
        self.w_uid = uid
        self.hIdx = 0

        process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
        Path(process_dir, f'{uid}.npy').touch()

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
            if Path(process_dir, f'{hashVal.uid}.npy').is_file():
                file_pth = self.DATADIR.joinpath(f'{hashVal.uid}.npy')
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
                                  cksum=checksum,
                                  collection_idx=self.hIdx,
                                  shape=array.shape)
        return hashVal
