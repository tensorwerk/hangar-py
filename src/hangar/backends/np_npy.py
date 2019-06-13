'''Local Numpy NPY Backend Implementation, Identifier: NUMPY_01

Backend Identifiers
===================

* Format Code: 03
* Canonical Name: NUMPY_01

Storage Method
==============

* Data is written to independent files in numpy binary files on disk.

* Each file is exactally the data's shape and dtype.

Record Format
=============

Fields Recorded for Each Array
------------------------------

* Format Code
* Schema Hash
* Alder32 Checksum
* Filename (digest with `.npy` appended to end)

Seperators used
---------------

* SEP_KEY
* SEP_HSH

Examples
--------

Note: all examples use SEP_KEY: ":", SEP_HSH: "$"

1) Adding a piece of data

   * Schema Hash: "051a8e928804"
   * Alder32 Checksum: 900338819
   * Filename: "dcdcfec4a3986e7bd7bac96c4b30fd84d7363403.npy"

   Record Data => '03:051a8e928804$900338819$dcdcfec4a3986e7bd7bac96c4b30fd84d7363403.npy'

Technical Notes
===============

* This storage backend is primarily intended for server side usage.

* On each write, an alder32 checksum is calculated. This is not for use as the
  primary hash algorithm, but rather stored in the local record format itself to
  serve as a quick way to verify no disk corruption occured. This is required
  since numpy has no built in data integrity validation methods when reading
  from disk.
'''

import logging
import os
import re
import hashlib
from collections import namedtuple
from os.path import join as pjoin
from os.path import isdir as pisdir
from typing import MutableMapping
from zlib import adler32

import numpy as np

from .. import constants as c
from ..utils import random_string, symlink_rel

logger = logging.getLogger(__name__)

# -------------------------------- Parser Implementation ----------------------


class NUMPY_01_Parser(object):

    __slots__ = ['FmtCode', 'SplitDecoderRE', 'DataHashSpec']

    def __init__(self):

        self.FmtCode = '03'
        self.DataHashSpec = namedtuple(
            typename='DataHashSpec',
            field_names=['backend', 'schema_hash', 'checksum', 'file_name'])

        # split up a formated parsed string into unique fields
        self.SplitDecoderRE = re.compile(fr'[\{c.SEP_KEY}\{c.SEP_HSH}]')

    def encode(self, schema_hash: str, checksum: int, filename: os.PathLike) -> bytes:
        '''converts the numpy data spect to an appropriate db value

        Parameters
        ----------
        schema_hash: str
            schema_hash of the tensor data.
        checksum : int
            adler32 checksum of the data as computed on that local machine.
        filename : os.PathLike
            filename (the data hash digest with .npy appended) to locate the file in.

        Returns
        -------
        bytes
            hash data db value recording all input specifications
        '''
        out_str = f'{self.FmtCode}{c.SEP_KEY}'\
                  f'{schema_hash}{c.SEP_HSH}{checksum}{c.SEP_HSH}{filename}'
        return out_str.encode()

    def decode(self, db_val: bytes) -> namedtuple:
        '''converts a numpy data hash db val into a numpy data python spec

        Parameters
        ----------
        db_val : bytes
            data hash db val

        Returns
        -------
        namedtuple
            numpy data hash specification containing `backend`, `schema_hash`, and
            `checksum`, and `filename` fields.
        '''
        db_str = db_val.decode()
        _, schema_hash, checksum, filename = self.SplitDecoderRE.split(db_str)
        raw_val = self.DataHashSpec(backend=self.FmtCode,
                                    schema_hash=schema_hash,
                                    checksum=checksum,
                                    filename=filename)
        return raw_val


# ------------------------- Accessor Object -----------------------------------


class NUMPY_01_FileHandles(object):

    def __init__(self, repo_path: os.PathLike, *args, **kwargs):
        self.repo_path = repo_path

        # self.rFp: MutableMapping[str, np.memmap] = {}
        # self.wFp: MutableMapping[str, np.memmap] = {}
        # self.Fp = ChainMap(self.rFp, self.wFp)

        self.mode: str = None

        self.fmtParser = NUMPY_01_Parser()

        self.STAGEDIR = pjoin(self.repo_path, c.DIR_DATA_STAGE, self.fmtParser.FmtCode)
        self.REMOTEDIR = pjoin(self.repo_path, c.DIR_DATA_REMOTE, self.fmtParser.FmtCode)
        self.DATADIR = pjoin(self.repo_path, c.DIR_DATA, self.fmtParser.FmtCode)
        self.STOREDIR = pjoin(self.repo_path, c.DIR_DATA_STORE, self.fmtParser.FmtCode)
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def open(self, mode: str, *, remote_operation: bool = False):
        '''open numpy file handle coded directories

        Parameters
        ----------
        mode : str
            one of `a` for `write-enabled` mode or `r` for read-only
        remote_operation : bool, optional, kwarg only
            True if remote operations call this method. Changes the symlink
            directories used while writing., by default False
        '''
        self.mode = mode
        if self.mode == 'a':
            process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
            if not os.path.isdir(process_dir):
                os.makedirs(process_dir)

            # process_uids = [psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.npy')]
            # for uid in process_uids:
            #     file_pth = pjoin(process_dir, f'{uid}.npy')
            #     self.rFp[uid] = partial(open_memmap, file_pth, 'r')

        if not remote_operation:
            if not os.path.isdir(self.STOREDIR):
                return
            # store_uids = [psplitext(x)[0] for x in os.listdir(self.STOREDIR) if x.endswith('.npy')]
            # for uid in store_uids:
            #     file_pth = pjoin(self.STOREDIR, f'{uid}.npy')
            #     self.rFp[uid] = partial(open_memmap, file_pth, 'r')

    def close(self, *args, **kwargs):
        '''Close any open file handles.
        '''
        # if self.mode == 'a':
        #     if self.w_uid in self.wFp:
        #         self.wFp[self.w_uid].flush()
        #         self.w_uid = None
        #         self.hIdx = None
        #     for k in list(self.wFp.keys()):
        #         del self.wFp[k]

        # for k in list(self.rFp.keys()):
        #     del self.rFp[k]
        pass

    @staticmethod
    def delete_in_process_data(repo_path, *, remote_operation=False):
        '''Removes some set of files entirely from the stage/remote directory.

        DANGER ZONE. This should essentially only be used to perform hard resets
        of the repository state.

        Parameters
        ----------
        repo_path : str
            path to the repository on disk
        remote_operation : optional, kwarg only, bool
            If true, modify contents of the remote_dir, if false (default) modify
            contents of the staging directory.
        '''
        FmtCode = NUMPY_01_Parser().FmtCode
        data_dir = pjoin(repo_path, c.DIR_DATA, FmtCode)
        PDIR = c.DIR_DATA_STAGE if not remote_operation else c.DIR_DATA_REMOTE
        prcs_dir = pjoin(repo_path, PDIR, FmtCode)
        if not os.path.isdir(prcs_dir):
            return

        sub_dirs = (x for x in os.listdir(prcs_dir) if pisdir(pjoin(prcs_dir, x)))
        for sub_dir in sub_dirs:
            prcs_fnames = (x for x in os.listdir(pjoin(prcs_dir, sub_dir)) if x.endswith('.npy'))
            for process_fname in prcs_fnames:
                remove_link_pth = pjoin(prcs_dir, sub_dir, process_fname)
                remove_data_pth = pjoin(data_dir, sub_dir, process_fname)
                os.remove(remove_link_pth)
                os.remove(remove_data_pth)
            os.rmdir(prcs_dir, sub_dir)
        os.rmdir(prcs_dir)

    # def _create_schema(self, *, remote_operation: bool = False):
    #     '''stores the shape and dtype as the schema of a dataset.

    #     Parameters
    #     ----------
    #     remote_operation : optional, kwarg only, bool
    #         if this schema is being created from a remote fetch operation, then do not
    #         place the file symlink in the staging directory. Instead symlink it
    #         to a special remote staging directory. (default is False, which places the
    #         symlink in the stage data directory.)
    #     '''
    #     uid = random_string()
    #     file_path = pjoin(self.DATADIR, f'{uid}.npy')
    #     m = open_memmap(file_path,
    #                     mode='w+',
    #                     dtype=self.schema_dtype,
    #                     shape=(COLLECTION_SIZE, *self.schema_shape))
    #     self.wFp[uid] = m
    #     self.w_uid = uid
    #     self.hIdx = 0

    #     if remote_operation:
    #         symlink_file_path = pjoin(self.REMOTEDIR, f'{uid}.npy')
    #     else:
    #         symlink_file_path = pjoin(self.STAGEDIR, f'{uid}.npy')
    #     symlink_rel(file_path, symlink_file_path)

    def read_data(self, hashVal: namedtuple) -> np.ndarray:
        '''Read data from disk written in the numpy_00 fmtBackend

        Parameters
        ----------
        hashVal : namedtuple
            record specification stored in the db

        Returns
        -------
        np.ndarray
            tensor data stored at the provided hashVal specification.

        Raises
        ------
        RuntimeError
            If the recorded checksum does not match the recieved checksum.

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
        '''
        srcSlc = (self.slcExpr[int(hashVal.dataset_idx)], *(self.slcExpr[0:x] for x in hashVal.shape))

        try:
            res = self.Fp[hashVal.uid][srcSlc]
        except TypeError:
            self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
            res = self.Fp[hashVal.uid][srcSlc]
        except KeyError:
            file_pth = pjoin(self.STAGEDIR, f'{hashVal.uid}.npy')
            if (self.mode == 'a') and os.path.islink(file_pth):
                self.rFp[hashVal.uid] = open_memmap(file_pth, 'r')
                res = self.Fp[hashVal.uid][srcSlc]
            else:
                raise

        out = np.array(res, dtype=res.dtype, order='C')
        cksum = adler32(out)
        if cksum != int(hashVal.checksum):
            raise RuntimeError(f'DATA CORRUPTION ERROR: Checksum {cksum} != recorded for {hashVal}')
        return out

    def write_data(self, array: np.ndarray, *, remote_operation: bool = False) -> bytes:
        '''writes array data to disk in the numpy_00 fmtBackend

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
        '''
        checksum = adler32(array)
        full_hash = hashlib.blake2b(array.tobytes(), digest_size=20).hexdigest()
        hashdir = pjoin(self.DATADIR, full_hash[:2])
        fname = pjoin(hashdir, f'{full_hash}.npy')
        if not os.path.isdir(hashdir):
            os.makedirs(hashdir)

        with open(fname, 'xb') as fh:
            np.save(fh, array)

        hashVal = self.fmtParser.encode(schema_hash=self.w_uid,
                                        checksum=checksum,
                                        dataset_idx=self.hIdx,
                                        shape=array.shape)
        return hashVal
