import logging
import os
import re
from collections import namedtuple
from typing import MutableMapping
from os.path import join as pjoin

import numpy as np

from .. import config
from ..utils import symlink_rel, random_string

logger = logging.getLogger(__name__)

SEP = config.get('hangar.seps.key')
LISTSEP = config.get('hangar.seps.list')
SLICESEP = config.get('hangar.seps.slice')
HASHSEP = config.get('hangar.seps.hash')


class NUMPY_00_Parser(object):

    __slots__ = ['FmtBackend', 'FmtCode', 'FmtCodeIdx', 'ShapeFmtRE', 'DataHashSpec']

    def __init__(self):

        self.FmtBackend = f'numpy_00'
        self.FmtCode = '01'
        self.FmtCodeIdx = 3

        # match and remove the following characters: '['   ']'   '('   ')'   ','
        self.ShapeFmtRE = re.compile('[,\(\)\[\]]')
        self.DataHashSpec = namedtuple(
            typename='DataHashSpec', field_names=['backend', 'schema', 'uid', 'dataset_idx', 'shape'])

    def encode(self, schema: str, uid: str, dataset_idx: int, shape: tuple) -> bytes:
        '''converts the numpy data spect to an appropriate db value

        Parameters
        ----------
        schema : str
            schema hash of this data piece
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
        out_str = f'{self.FmtCode}{SEP}'\
                  f'{schema}{LISTSEP}{uid}'\
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

        schema_vals, _, dset_vals = db_str.partition(HASHSEP)
        schema, uid = schema_vals.split(LISTSEP)

        dataset_idx, _, shape_vs = dset_vals.rpartition(SLICESEP)
        # dataset, dataset_idx = dataset_vs.split(LISTSEP)
        # if the data is of empty shape -> ()
        shape = () if shape_vs == '' else tuple([int(x) for x in shape_vs.split(LISTSEP)])

        raw_val = self.DataHashSpec(backend=self.FmtBackend,
                                    schema=schema,
                                    uid=uid,
                                    dataset_idx=dataset_idx,
                                    shape=shape)
        return raw_val

        # val = db_val.decode()[self.FmtCodeIdx:]
        # schema, fname = val.split(LISTSEP, 1)
        # raw_val = self.DataHashSpec(backend=self.FmtBackend, schema=schema, fname=fname)
        # return raw_val


class NUMPY_00_FileHandles(object):

    def __init__(self, repo_path):

        self.repo_path = repo_path
        self.fmtParser = NUMPY_00_Parser()

        self.STOREDIR: os.PathLike = None
        self.STAGEDIR: os.PathLike = None
        self.REMOTEDIR: os.PathLike = None
        self.DATADIR: os.PathLike = None

        self.repo_path = repo_path
        self.rFp: MutableMapping[str, np.memmap] = {}
        self.wFp: MutableMapping[str, np.memmap] = {}
        self.slcExpr = np.s_
        self.slcExpr.maketuple = False

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
        CONF_STORE_DIR = config.get('hangar.repository.store_data_dir')
        CONF_STAGE_DIR = config.get('hangar.repository.stage_data_dir')
        CONF_REMOTE_DIR = config.get('hangar.repository.remote_data_dir')
        CONF_DATA_DIR = config.get('hangar.repository.data_dir')

        self.STOREDIR = pjoin(self.repo_path, CONF_STORE_DIR, self.fmtParser.FmtCode)
        self.STAGEDIR = pjoin(self.repo_path, CONF_STAGE_DIR, self.fmtParser.FmtCode)
        self.REMOTEDIR = pjoin(self.repo_path, CONF_REMOTE_DIR, self.fmtParser.FmtCode)
        self.DATADIR = pjoin(self.repo_path, CONF_DATA_DIR, self.fmtParser.FmtCode)

        if mode == 'a':
            if not remote_operation and not os.path.isdir(self.DATADIR):
                os.makedirs(self.DATADIR)
            if not remote_operation and not os.path.isdir(self.STOREDIR):
                os.makedirs(self.STOREDIR)
            if not remote_operation and not os.path.isdir(self.STAGEDIR):
                os.makedirs(self.STAGEDIR)
            elif remote_operation and not os.path.isdir(self.REMOTEDIR):
                os.makedirs(self.REMOTEDIR)

    def close(self, *args, **kwargs):
        '''Close any open file handles.
        '''
        for k in list(self.wFp.keys()):
            self.wFp.flush()
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

        DATA_DIR = config.get('hangar.repository.data_dir')
        STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
        FmtCode = NUMPY_00_Parser().FmtCode
        FmtBackend = NUMPY_00_Parser().FmtBackend

        data_pth = pjoin(repo_path, DATA_DIR, FmtCode)
        stage_dir = pjoin(repo_path, STAGE_DATA_DIR, FmtCode)
        if os.path.isdir(stage_dir) is False:
            return

        stgHashs = HashQuery(stagehashenv).list_all_hash_values()
        stg_data_fs = set(v.fname for v in stgHashs if v.backend == FmtBackend)

        for stgDigestDir in os.listdir(stage_dir):
            subDigestPth = pjoin(stage_dir, stgDigestDir)
            if os.path.isdir(subDigestPth):
                stage_files = [x for x in os.listdir(subDigestPth) if x.endswith('.npy')]
                for instance in stage_files:
                    if instance in stg_data_fs:
                        continue
                    else:
                        remove_link_pth = pjoin(subDigestPth, instance)
                        remove_data_pth = pjoin(data_pth, stgDigestDir, instance)
                        os.remove(remove_link_pth)
                        os.remove(remove_data_pth)

    def read_data(self, hashVal: namedtuple, *args, **kwargs) -> np.ndarray:
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
        key = f'{hashVal.schema}_{hashVal.uid}'
        srcSlc = (self.slcExpr[hashVal.dataset_idx], *(self.slcExpr[0:x] for x in hashVal.shape))

        if key in self.rFp:
            out = self.rFp[key][srcSlc]
        elif key in self.wFp:
            out = self.wFp[key][srcSlc]
        else:
            stagePth = pjoin(self.STAGEDIR, hashVal.schema, f'{hashVal.uid}.npy')
            storePth = pjoin(self.STOREDIR, hashVal.schema, f'{hashVal.uid}.npy')
            if os.path.islink(stagePth):
                self.rFp[key] = np.lib.format.open_memmap(stagePth, mode='r')
                out = self.rFp[key][srcSlc]
            elif os.path.islink(storePth):
                self.rFp[key] = np.lib.format.open_memmap(storePth, mode='r')
                out = self.rFp[key][srcSlc]

        return out

        # fname = hashVal.fname
        # digest_prefix = fname[:2]

        # try:
        #     data_fp = pjoin(self.STOREDIR, digest_prefix, fname)
        #     data = np.load(data_fp)
        # except FileNotFoundError:
        #     data_fp = pjoin(self.STAGEDIR, digest_prefix, fname)
        #     data = np.load(data_fp)
        # return data

    def write_data(self, array: np.ndarray, dhash: str, shash: str,
                   *, remote_operation: bool = False) -> bytes:
        '''writes array data to disk in the numpy_00 fmtBackend

        Parameters
        ----------
        array : np.ndarray
            tensor to write to disk
        dhash : str
            hash of the tensor data
        shash : str
            hash of the sample's dataset schema
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
        if shash in self.wFp:
            mmapArr = self.wFp[shash]
            idx = int(mmapArr.item(0))
            uid = self.wFpUID[shash]
            mmapArr.itemset(0, idx + 1)
        else:
            uid = random_string()
            data_dir = pjoin(self.DATADIR, shash)
            data_fp = pjoin(data_dir, f'{uid}.npy')

        try:
            with open(data_fp, 'xb') as fh:
                np.save(fh, array)
        except FileNotFoundError:
            os.makedirs(data_dir)
            with open(data_fp, 'xb') as fh:
                np.save(fh, array)
        except FileExistsError:
            logger.error(f'dhash: {dhash}')
            raise

        if not remote_operation:
            symlink_dest = pjoin(self.STAGEDIR, dhash[:2], f'{dhash}.npy')
        else:
            symlink_dest = pjoin(self.REMOTEDIR, dhash[:2], f'{dhash}.npy')

        try:
            symlink_rel(data_fp, symlink_dest)
        except FileNotFoundError:
            symlink_dir = os.path.dirname(symlink_dest)
            os.makedirs(symlink_dir)
            symlink_rel(data_fp, symlink_dest)

        hashVal = self.fmtParser.encode(schema=shash, data_hash=dhash)
        return hashVal
