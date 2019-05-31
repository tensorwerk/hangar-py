import logging
import os
import re
from functools import partial
from collections import namedtuple, ChainMap
from typing import MutableMapping
from os.path import join as pjoin
from os.path import splitext as psplitext

import numpy as np

from .. import config
from ..utils import symlink_rel, random_string

logger = logging.getLogger(__name__)

SEP = config.get('hangar.seps.key')
LISTSEP = config.get('hangar.seps.list')
SLICESEP = config.get('hangar.seps.slice')
HASHSEP = config.get('hangar.seps.hash')

STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
REMOTE_DATA_DIR = config.get('hangar.repository.remote_data_dir')
DATA_DIR = config.get('hangar.repository.data_dir')
STORE_DATA_DIR = config.get('hangar.repository.store_data_dir')


class NUMPY_00_Parser(object):

    __slots__ = ['FmtBackend', 'FmtCode', 'FmtCodeIdx', 'ShapeFmtRE', 'DataHashSpec']

    def __init__(self):

        self.FmtBackend = f'numpy_00'
        self.FmtCode = '01'
        self.FmtCodeIdx = 3

        # match and remove the following characters: '['   ']'   '('   ')'   ','
        self.ShapeFmtRE = re.compile('[,\(\)\[\]]')
        self.DataHashSpec = namedtuple(
            typename='DataHashSpec', field_names=['backend', 'uid', 'dataset_idx', 'shape'])

    def encode(self, uid: str, dataset_idx: int, shape: tuple) -> bytes:
        '''converts the numpy data spect to an appropriate db value

        Parameters
        ----------
        uid : str
            file name (schema uid) of the np file to find this data piece in.
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
        out_str = f'{self.FmtCode}{SEP}{uid}'\
                  f'{HASHSEP}'\
                  f'{dataset_idx}'\
                  f'{SLICESEP}'\
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
        db_str = db_val.decode()[self.FmtCodeIdx:]
        uid, _, dset_vals = db_str.partition(HASHSEP)
        dataset_idx, _, shape_vs = dset_vals.rpartition(SLICESEP)
        # if the data is of empty shape -> ()
        shape = () if shape_vs == '' else tuple([int(x) for x in shape_vs.split(LISTSEP)])
        raw_val = self.DataHashSpec(backend=self.FmtBackend,
                                    uid=uid,
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

        self.STAGEDIR = pjoin(self.repo_path, STAGE_DATA_DIR, self.fmtParser.FmtCode)
        self.REMOTEDIR = pjoin(self.repo_path, REMOTE_DATA_DIR, self.fmtParser.FmtCode)
        self.DATADIR = pjoin(self.repo_path, DATA_DIR, self.fmtParser.FmtCode)
        self.STOREDIR = pjoin(self.repo_path, STORE_DATA_DIR, self.fmtParser.FmtCode)
        if not os.path.isdir(self.DATADIR):
            os.makedirs(self.DATADIR)

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
    def remove_unused(repo_path, stagehashenv):
        '''If no changes made to staged hdf files, remove and unlik them from stagedir

        This searchs the stagehashenv file for all schemas & instances, and if any
        files are present in the stagedir without references in stagehashenv, the
        symlinks in stagedir and backing data files in datadir are removed.

        Parameters
        ----------
        repo_path : str
            path to the repository on disk
        stagehashenv : `lmdb.Environment`
            db where all stage hash additions are recorded

        '''
        from ..records.hashs import HashQuery

        FmtCode = NUMPY_00_Parser().FmtCode
        FmtBackend = NUMPY_00_Parser().FmtBackend
        dat_dir = pjoin(repo_path, DATA_DIR, FmtCode)
        stg_dir = pjoin(repo_path, STAGE_DATA_DIR, FmtCode)
        if not os.path.isdir(stg_dir):
            return

        stgHashs = HashQuery(stagehashenv).list_all_hash_values()
        stg_files = set(v.uid for v in stgHashs if v.backend == FmtBackend)
        stg_uids = set(psplitext(x)[0] for x in os.listdir(stg_dir) if x.endswith('.npy'))
        unused_uids = stg_uids.difference(stg_files)

        for unused_uid in unused_uids:
            remove_link_pth = pjoin(stg_dir, f'{unused_uid}.npy')
            remove_data_pth = pjoin(dat_dir, f'{unused_uid}.npy')
            os.remove(remove_link_pth)
            os.remove(remove_data_pth)

    def create_schema(self, *, remote_operation: bool = False):
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
        '''
        srcSlc = (self.slcExpr[int(hashVal.dataset_idx)], *(self.slcExpr[0:x] for x in hashVal.shape))
        try:
            res = self.Fp[hashVal.uid][srcSlc]
        except TypeError:
            self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
            res = self.Fp[hashVal.uid][srcSlc]
        return res

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

        Raises
        ------
        FileExistsError
            If the provided data hash exists on disk
        '''
        if self.w_uid in self.wFp:
            self.hIdx += 1
            if self.hIdx >= self.hMaxSize:
                self.wFp[self.w_uid].flush()
                self.create_schema(remote_operation=remote_operation)
        else:
            self.create_schema(remote_operation=remote_operation)

        destSlc = (self.slcExpr[self.hIdx], *(self.slcExpr[0:x] for x in array.shape))
        self.wFp[self.w_uid][destSlc] = array
        self.wFp[self.w_uid].flush()

        hashVal = self.fmtParser.encode(uid=self.w_uid,
                                        dataset_idx=self.hIdx,
                                        shape=array.shape)
        return hashVal