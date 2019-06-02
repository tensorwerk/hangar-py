import logging
import os
import re
from collections import ChainMap, namedtuple
from functools import partial
from os.path import join as pjoin
from os.path import splitext as psplitext
from typing import MutableMapping
from zlib import adler32

import numpy as np

from .. import constants as c
from ..utils import random_string, symlink_rel

logger = logging.getLogger(__name__)


class NUMPY_00_Parser(object):

    __slots__ = ['FmtCode', 'SplitDecoderRE', 'ShapeFmtRE', 'DataHashSpec']

    def __init__(self):

        self.FmtCode = '01'
        self.DataHashSpec = namedtuple(
            typename='DataHashSpec',
            field_names=['backend', 'uid', 'checksum', 'dataset_idx', 'shape'])

        # match and remove the following characters: '['   ']'   '('   ')'   ','
        self.ShapeFmtRE = re.compile('[,\(\)\[\]]')
        # split up a formated parsed string into unique fields
        self.SplitDecoderRE = re.compile(fr'[\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}]')

    def encode(self, uid: str, checksum: int, dataset_idx: int, shape: tuple) -> bytes:
        '''converts the numpy data spect to an appropriate db value

        Parameters
        ----------
        uid : str
            file name (schema uid) of the np file to find this data piece in.
        checksum : int
            adler32 checksum of the data as computed on that local machine.
        dataset_idx : int
            collection first axis index in which this data piece resides.
        shape : tuple
            shape of the data sample written to the collection idx. ie:
            what subslices of the hdf5 dataset should be read to retrieve
            the sample as recorded.

        Returns
        -------
        bytes
            hash data db value recording all input specifications
        '''
        out_str = f'{self.FmtCode}{c.SEP_KEY}'\
                  f'{uid}{c.SEP_HSH}{checksum}'\
                  f'{c.SEP_HSH}'\
                  f'{dataset_idx}'\
                  f'{c.SEP_SLC}'\
                  f'{self.ShapeFmtRE.sub("", str(shape))}'
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
            numpy data hash specification containing `backend`, `schema`, and
            `uid`, `dataset_idx` and `shape` fields.
        '''
        db_str = db_val.decode()
        _, uid, checksum, dataset_idx, shape_vs = self.SplitDecoderRE.split(db_str)
        # if the data is of empty shape -> shape_vs = '' str.split() default
        # value of none means split according to any whitespace, and discard
        # empty strings from the result. So long as c.SEP_LST = ' ' this will
        # work
        shape = tuple(int(x) for x in shape_vs.split())
        raw_val = self.DataHashSpec(backend=self.FmtCode,
                                    uid=uid,
                                    checksum=checksum,
                                    dataset_idx=dataset_idx,
                                    shape=shape)
        return raw_val


class NUMPY_00_FileHandles(object):

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.repo_path = repo_path
        self.schema_shape = schema_shape
        self.schema_dtype = schema_dtype

        self.rFp: MutableMapping[str, np.memmap] = {}
        self.wFp: MutableMapping[str, np.memmap] = {}
        self.Fp = ChainMap(self.rFp, self.wFp)

        self.mode: str = None
        self.w_uid: str = None
        self.hIdx: int = None
        self.hMaxSize: int = None

        self.slcExpr = np.s_
        self.slcExpr.maketuple = False
        self.fmtParser = NUMPY_00_Parser()

        self.STAGEDIR = pjoin(self.repo_path, c.DIR_DATA_STAGE, self.fmtParser.FmtCode)
        self.REMOTEDIR = pjoin(self.repo_path, c.DIR_DATA_REMOTE, self.fmtParser.FmtCode)
        self.DATADIR = pjoin(self.repo_path, c.DIR_DATA, self.fmtParser.FmtCode)
        self.STOREDIR = pjoin(self.repo_path, c.DIR_DATA_STORE, self.fmtParser.FmtCode)
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.w_uid in self.wFp:
            self.wFp[self.w_uid].flush()

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

            process_uids = [psplitext(x)[0] for x in os.listdir(process_dir) if x.endswith('.npy')]
            for uid in process_uids:
                file_pth = pjoin(process_dir, f'{uid}.npy')
                self.rFp[uid] = partial(np.lib.format.open_memmap, file_pth, 'r')

        if not remote_operation:
            if not os.path.isdir(self.STOREDIR):
                return
            store_uids = [psplitext(x)[0] for x in os.listdir(self.STOREDIR) if x.endswith('.npy')]
            for uid in store_uids:
                file_pth = pjoin(self.STOREDIR, f'{uid}.npy')
                self.rFp[uid] = partial(np.lib.format.open_memmap, file_pth, 'r')

    def close(self, *args, **kwargs):
        '''Close any open file handles.
        '''
        if self.mode == 'a':
            if self.w_uid in self.wFp:
                self.wFp[self.w_uid].flush()
                self.w_uid = None
                self.hIdx = None
                self.hMaxSize = None
            for k in list(self.wFp.keys()):
                del self.wFp[k]

        for k in list(self.rFp.keys()):
            del self.rFp[k]

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
        FmtCode = NUMPY_00_Parser().FmtCode
        data_dir = pjoin(repo_path, c.DIR_DATA, FmtCode)
        PDIR = c.DIR_DATA_STAGE if not remote_operation else c.DIR_DATA_REMOTE
        process_dir = pjoin(repo_path, PDIR, FmtCode)
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
        '''stores the shape and dtype as the schema of a dataset.

        Parameters
        ----------
        remote_operation : optional, kwarg only, bool
            if this schema is being created from a remote fetch operation, then do not
            place the file symlink in the staging directory. Instead symlink it
            to a special remote staging directory. (default is False, which places the
            symlink in the stage data directory.)
        '''
        uid = random_string()
        file_path = pjoin(self.DATADIR, f'{uid}.npy')

        m = np.lib.format.open_memmap(file_path,
                                      mode='w+',
                                      dtype=self.schema_dtype,
                                      shape=(500, *self.schema_shape))
        self.wFp[uid] = m
        self.w_uid = uid
        self.hIdx = 0
        self.hMaxSize = 500  # TODO; CONFIGURE

        if remote_operation:
            symlink_file_path = pjoin(self.REMOTEDIR, f'{uid}.npy')
        else:
            symlink_file_path = pjoin(self.STAGEDIR, f'{uid}.npy')
        symlink_rel(file_path, symlink_file_path)

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
            If the recorded checksum does not match the recieved checksum value.
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
                self.rFp[hashVal.uid] = np.lib.format.open_memmap(file_pth, 'r')
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
        if self.w_uid in self.wFp:
            self.hIdx += 1
            if self.hIdx >= self.hMaxSize:
                self.wFp[self.w_uid].flush()
                self._create_schema(remote_operation=remote_operation)
        else:
            self._create_schema(remote_operation=remote_operation)

        destSlc = (self.slcExpr[self.hIdx], *(self.slcExpr[0:x] for x in array.shape))
        self.wFp[self.w_uid][destSlc] = array
        hashVal = self.fmtParser.encode(uid=self.w_uid,
                                        checksum=checksum,
                                        dataset_idx=self.hIdx,
                                        shape=array.shape)
        return hashVal
