import math
import os
import subprocess
import re
from collections import namedtuple
from os.path import join as pjoin
import logging

import h5py
import numpy as np

from .. import __version__
from .. import config
from ..utils import find_next_prime

logger = logging.getLogger(__name__)


SEP = config.get('hangar.seps.key')
LISTSEP = config.get('hangar.seps.list')
SLICESEP = config.get('hangar.seps.slice')
HASHSEP = config.get('hangar.seps.hash')


class HDF5_00_Parser(object):

    __slots__ = ['FmtCode', 'FmtCodeIdx', 'DataShapeReplacementRE', 'DataHashSpec']

    def __init__(self):

        self.FmtCode = f'00{SEP}'
        self.FmtCodeIdx = len(self.FmtCode)

        # match and remove the following characters: '['   ']'   '('   ')'   ','
        self.DataShapeReplacementRE = re.compile('[,\(\)\[\]]')

        self.DataHashSpec = namedtuple(
            typename='DataHashVal',
            field_names=['schema', 'instance', 'dataset', 'dataset_idx', 'shape'])

    def encode(self, schema, instance, dataset, dataset_idx, shape) -> bytes:
        '''converts the hdf5 data has spec to an appropriate db value

        Parameters
        ----------
        schema : str
            hdf5 schema hash to find this data piece in.
        instance : str
            file name (schema instance) of the hdf5 file to find this data piece in.
        dataset : str
            collection (ie. hdf5 dataset) name to find find this data piece.
        dataset_idx : int or str
            collection first axis index in which this data piece resides.
        shape : tuple
            shape of the data sample written to the collection idx. ie:
            what subslices of the hdf5 dataset should be read to retrieve
            the sample as recorded.

        Returns
        -------
        bytes
            hash data db value recording all input specifications.
        '''
        out_str = f'{self.FmtCode}'\
                  f'{schema}{LISTSEP}{instance}'\
                  f'{HASHSEP}'\
                  f'{dataset}{LISTSEP}{dataset_idx}'\
                  f'{SLICESEP}'\
                  f'{self.DataShapeReplacementRE.sub("", str(shape))}'
        return out_str.encode()

    def decode(self, db_val: bytes) -> namedtuple:
        '''converts an hdf5 data hash db val into an hdf5 data python spec.

        Parameters
        ----------
        db_val : bytestring
            data hash db value

        Returns
        -------
        namedtuple
            hdf5 data hash specification in DataHashVal named tuple format
        '''
        db_str = db_val.decode()[self.FmtCodeIdx:]

        schema_vals, _, dset_vals = db_str.partition(HASHSEP)
        schema, instance = schema_vals.split(LISTSEP)

        dataset_vs, _, shape_vs = dset_vals.rpartition(SLICESEP)
        dataset, dataset_idx = dataset_vs.split(LISTSEP)
        # if the data is of empty shape -> ()
        shape = () if shape_vs == '' else tuple([int(x) for x in shape_vs.split(LISTSEP)])

        raw_val = self.DataHashSpec(schema=schema,
                                    instance=instance,
                                    dataset=dataset,
                                    dataset_idx=dataset_idx,
                                    shape=shape)
        return raw_val


class HDF5_00_FileHandlesSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        repo_pth = kwargs['repo_path']
        if repo_pth not in cls._instances:
            cls._instances[repo_pth] = super(HDF5_00_FileHandlesSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[repo_pth]


'''
Dense Array Methods
-------------------
'''


class HDF5_00_FileHandles(metaclass=HDF5_00_FileHandlesSingleton):
    '''Singleton to manage HDF5 file handles.

    When in SWMR-write mode, no more than a single file handle can be in the
    "writeable" state. This is an issue where multiple datasets may need to
    write to the same dataset schema.
    '''

    def __init__(self, repo_path):
        self.repo_path = repo_path
        self.rFp = {}
        self.wFp = {}
        self.hMaxSize = {}
        self.hNextPath = {}
        self.hIdx = {}
        self.hColsRemain = {}
        self.hNextInstance = {}
        self.slcExpr = np.s_
        self.slcExpr.maketuple = False
        self.fmtParser = HDF5_00_Parser()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for schema in list(self.wFp.keys()):
            for instance in list(self.wFp[schema].keys()):
                self.wFp[schema][instance]['/'].attrs.modify(
                    'next_dset', (self.hNextPath[schema][instance], self.hIdx[schema][instance]))
                self.wFp[schema][instance]['/'].attrs.modify(
                    'num_collections_remaining', self.hColsRemain[schema][instance])
                self.wFp[schema][instance].flush()

    @staticmethod
    def _dataset_opts(complib, complevel, shuffle, fletcher32):
        '''specify compression options for the hdf5 dataset.

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
        '''
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
    def _chunk_opts(sample_array, max_chunk_nbytes):
        '''Determine the chunk shape so each array chunk fits into configured nbytes.

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
        '''
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

    @staticmethod
    def create_schema(repo_path: str, schema_hash: str, sample_array: np.ndarray,
                      *, remote_operation: bool = False) -> h5py.File:
        '''stores the shape and dtype as the schema of a dataset.

        Parameters
        ----------
        repo_path : str
            path where the repository files can be accessed on the local disk.
        schema_hash : str
            hash of the dataset schema
        sample_array : np.ndarray
            sample input tensor (representitative of all data which will fill the dset) to
            extract the shape and dtype from.
        remote_operation : optional, kwarg only, bool
            if this schema is being created from a remote fetch operation, then do not
            place the file symlink in the staging directory. Instead symlink it
            to a special remote staging directory. (default is False, which places the
            symlink in the stage data directory.)

        Returns
        -------
        h5py.File
            File handle to the created h5py file

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

        '''
        HDF_DSET_FILTERS = config.get('hangar.hdf5.dataset.filters.default')
        if HDF_DSET_FILTERS['complib'].startswith('blosc'):
            bloscFilterAvail = h5py.h5z.filter_avail(32001)
            if not bloscFilterAvail:
                HDF_DSET_FILTERS = config.get('hangar.hdf5.dataset.filters.backup')

        HDF_CHUNK_OPTS = config.get('hangar.hdf5.dataset.chunking')
        HDF_DSET_CONTENTS = config.get('hangar.hdf5.contents')
        HDF_ATTRS = config.get('hangar.hdf5.attributes.keys')

        # ---------------------- Directory Access & Creation  ------------------

        STORE_DATA_DIR = config.get('hangar.repository.store_data_dir')
        STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
        REMOTE_DATA_DIR = config.get('hangar.repository.remote_data_dir')
        DATA_DIR = config.get('hangar.repository.data_dir')

        store_dir = pjoin(repo_path, STORE_DATA_DIR, f'hdf_{schema_hash}')
        stage_dir = pjoin(repo_path, STAGE_DATA_DIR, f'hdf_{schema_hash}')
        remote_dir = pjoin(repo_path, REMOTE_DATA_DIR, f'hdf_{schema_hash}')
        data_dir = pjoin(repo_path, DATA_DIR)

        num_schemas = 0
        if os.path.isdir(store_dir):
            num_schemas += len([x for x in os.listdir(store_dir) if x.endswith('.hdf5')])
        if os.path.isdir(stage_dir):
            num_schemas += len([x for x in os.listdir(stage_dir) if x.endswith('.hdf5')])
        if os.path.isdir(remote_dir):
            num_schemas += len([x for x in os.listdir(remote_dir) if x.endswith('.hdf5')])

        if not remote_operation and not os.path.isdir(stage_dir):
            os.makedirs(stage_dir)
        elif remote_operation and not os.path.isdir(remote_dir):
            os.makedirs(remote_dir)

        schema_just = str(num_schemas).rjust(3, '0')
        file_path = pjoin(data_dir, f'hdf_{schema_hash}_{schema_just}.hdf5')

        # -------------------- Chunk & RDCC Vals ------------------------------

        chunk_shape, chunk_nbytes = __class__._chunk_opts(
            sample_array=sample_array,
            max_chunk_nbytes=HDF_CHUNK_OPTS['max_nbytes'])

        rdcc_nbytes_val = math.ceil((sample_array.nbytes / chunk_nbytes) * chunk_nbytes * 10)
        if rdcc_nbytes_val < HDF_CHUNK_OPTS['max_nbytes']:
            rdcc_nbytes_val = HDF_CHUNK_OPTS['max_nbytes']
        elif rdcc_nbytes_val > HDF_CHUNK_OPTS['max_rdcc_nbytes']:
            rdcc_nbytes_val = HDF_CHUNK_OPTS['max_rdcc_nbytes']

        rdcc_nslots_guess = math.ceil(rdcc_nbytes_val / chunk_nbytes) * 100
        rdcc_nslots_prime_val = find_next_prime(rdcc_nslots_guess)

        # ---------------------------- File Creation --------------------------

        logger.debug(f'creating: {file_path}')
        fh = h5py.File(
            file_path,
            mode='w',
            libver='latest',
            rdcc_nbytes=rdcc_nbytes_val,
            rdcc_w0=HDF_CHUNK_OPTS['rdcc_w0'],
            rdcc_nslots=rdcc_nslots_prime_val)

        if remote_operation:
            symlink_file_path = pjoin(remote_dir, f'{schema_just}.hdf5')
            relpath = os.path.relpath(file_path, start=remote_dir)
        else:
            symlink_file_path = pjoin(stage_dir, f'{schema_just}.hdf5')
            relpath = os.path.relpath(file_path, start=stage_dir)

        os.symlink(relpath, symlink_file_path)

        # ----------------------- Dataset Creation ----------------------------

        optKwargs = __class__._dataset_opts(**HDF_DSET_FILTERS)

        for dset_num in range(HDF_DSET_CONTENTS['num_collections']):
            fh.create_dataset(
                f'/{dset_num}',
                shape=(HDF_DSET_CONTENTS['collection_size'], *sample_array.shape),
                dtype=sample_array.dtype,
                maxshape=(HDF_DSET_CONTENTS['collection_size'], *sample_array.shape),
                chunks=(1, *chunk_shape),
                **optKwargs)

        # ---------------------- Attribute Config Vals ------------------------

        fh['/'].attrs[HDF_ATTRS['hangar_version']] = __version__
        fh['/'].attrs[HDF_ATTRS['schema_hash']] = schema_hash
        fh['/'].attrs[HDF_ATTRS['schema_instance']] = num_schemas
        fh['/'].attrs[HDF_ATTRS['schema_shape']] = sample_array.shape
        fh['/'].attrs[HDF_ATTRS['schema_dtype']] = sample_array.dtype.num
        fh['/'].attrs[HDF_ATTRS['next_location']] = (0, 0)
        fh['/'].attrs[HDF_ATTRS['next_instance']] = schema_just
        fh['/'].attrs[HDF_ATTRS['collection_max_size']] = HDF_DSET_CONTENTS['collection_size']
        fh['/'].attrs[HDF_ATTRS['collection_total']] = HDF_DSET_CONTENTS['num_collections']
        fh['/'].attrs[HDF_ATTRS['collections_remaining']] = HDF_DSET_CONTENTS['num_collections']
        fh['/'].attrs[HDF_ATTRS['rdcc_nbytes']] = rdcc_nbytes_val
        fh['/'].attrs[HDF_ATTRS['rdcc_w0']] = HDF_CHUNK_OPTS['rdcc_w0']
        fh['/'].attrs[HDF_ATTRS['rdcc_nslots']] = rdcc_nslots_prime_val
        fh['/'].attrs[HDF_ATTRS['shuffle']] = optKwargs['shuffle']
        fh['/'].attrs[HDF_ATTRS['complib']] = HDF_DSET_FILTERS['complib']
        fh['/'].attrs[HDF_ATTRS['fletcher32']] = optKwargs['fletcher32']
        fh['/'].attrs[HDF_ATTRS['chunk_shape']] = chunk_shape
        if optKwargs['compression_opts'] is not None:
            fh['/'].attrs[HDF_ATTRS['comp_opts']] = optKwargs['compression_opts']
        else:
            fh['/'].attrs[HDF_ATTRS['comp_opts']] = False

        fh.flush()
        try:
            fh.swmr_mode = True
        except ValueError:
            assert fh.swmr_mode is True
        return fh

    @staticmethod
    def clear_consistency_flag(file_path: str) -> bool:
        '''Creates a subprocess to clear the hdf5 swmr consistency flag

        This is needed on non-standard file close operations. generally not
        an issue here because we use context managers to pretty much handle all
        access operations.

        Parameters
        ----------
        file_path : str
            path to the hdf5 file which needs the consistency flag cleared

        Returns
        -------
        bool
            True on success, False on error
        '''
        ret = subprocess.call(['h5clear', '-s', f'{file_path}'])
        if ret == 0:
            return True
        else:
            return False

    def open(self, repo_path: str, mode: str, *, remote_operation: bool = False):
        '''Open an hdf5 file handle in the Handler Singleton

        Parameters
        ----------
        repo_path : str
            directory path to the repository on disk
        mode : str
            one of `r` or `a` for read only / read-write.
        repote_operation : optional, kwarg only, bool
            if this hdf5 data is being created from a remote fetch operation, then
            we don't open any files for reading, and only open files for writing
            which exist in the remote data dir. (default is false, which means that
            write operations use the stage data dir and read operations use data store
            dir)
        '''

        if remote_operation:
            REMOTE_DATA_DIR = config.get('hangar.repository.remote_data_dir')
            process_dir = pjoin(repo_path, REMOTE_DATA_DIR)
        else:
            STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
            process_dir = pjoin(repo_path, STAGE_DATA_DIR)

        process_schemas = [x for x in os.listdir(process_dir) if x.startswith('hdf_')]

        if mode == 'a':
            for p_schema in process_schemas:
                schema_pth = pjoin(process_dir, p_schema)
                schHash = p_schema.replace('hdf_', '', 1)
                p_files = [x for x in os.listdir(schema_pth) if x.endswith('.hdf5')]
                latest_p = sorted(p_files, reverse=True)

                for idx, p_file in enumerate(latest_p):
                    file_pth = pjoin(schema_pth, p_file)
                    pInstance = p_file.replace('.hdf5', '')

                    if schHash not in self.wFp:
                        self.wFp[schHash] = {}
                        self.hNextPath[schHash] = {}
                        self.hIdx[schHash] = {}
                        self.hMaxSize[schHash] = {}
                        self.hColsRemain[schHash] = {}

                    if pInstance not in self.wFp[schHash]:
                        handle = h5py.File(file_pth, 'a', libver='latest')
                        handle.swmr_mode = True
                        next_dset = handle['/'].attrs['next_dset']
                        colsRemaining = int(handle['/'].attrs['num_collections_remaining'])

                        self.wFp[schHash][pInstance] = handle
                        self.hNextPath[schHash][pInstance] = next_dset[0]
                        self.hIdx[schHash][pInstance] = next_dset[1]
                        self.hMaxSize[schHash][pInstance] = handle['/'].attrs['max_size']
                        self.hColsRemain[schHash][pInstance] = colsRemaining
                        if idx == 0:
                            self.hNextInstance[schHash] = pInstance

        if not remote_operation:
            STORE_DATA_DIR = config.get('hangar.repository.store_data_dir')
            store_dir = pjoin(repo_path, STORE_DATA_DIR)
            store_schemas = [x for x in os.listdir(store_dir) if x.startswith('hdf_')]

            for store_schema in store_schemas:
                schema_pth = pjoin(store_dir, store_schema)
                schHash = store_schema.replace('hdf_', '', 1)
                store_files = [x for x in os.listdir(schema_pth) if x.endswith('.hdf5')]

                for store_file in store_files:
                    schInstance = store_file.replace('.hdf5', '')
                    file_pth = pjoin(schema_pth, store_file)
                    if schHash not in self.rFp:
                        self.rFp[schHash] = {}
                    if schInstance not in self.rFp[schHash]:
                        handle = h5py.File(file_pth, 'r', swmr=True, libver='latest')
                        self.rFp[schHash][schInstance] = handle

    def close(self, mode):
        '''Close a file handle after writes have been completed

        behavior changes depending on write-enable or read-only file

        Parameters
        ----------
        mode : str
            one of `r` or `a` specifying the mode which the file was opened in

        Returns
        -------
        bool
            True if success, otherwise False.
        '''
        if mode == 'a':
            for schema in list(self.wFp.keys()):
                for instance in list(self.wFp[schema].keys()):
                    self.wFp[schema][instance]['/'].attrs.modify(
                        'next_dset', (self.hNextPath[schema][instance], self.hIdx[schema][instance]))
                    self.wFp[schema][instance]['/'].attrs.modify(
                        'num_collections_remaining', self.hColsRemain[schema][instance])

                    self.wFp[schema][instance].flush()
                    self.wFp[schema][instance].close()

                self.hNextInstance.__delitem__(schema)
                self.wFp.__delitem__(schema)
                self.hMaxSize.__delitem__(schema)
                self.hNextPath.__delitem__(schema)
                self.hIdx.__delitem__(schema)
                self.hColsRemain.__delitem__(schema)
        else:
            for schema in list(self.rFp.keys()):
                for instance in list(self.rFp[schema].keys()):
                    self.rFp[schema][instance].close()
                    self.rFp[schema].__delitem__(instance)
                self.rFp.__delitem__(schema)

    def add_tensor_data(self, array, dhash, shash, *, remote_operation=False):
        '''verifies correctness of array data and performs write operation.

        Parameters
        ----------
        array : np.ndarray
            tensor to write to group.
        dhash : str
            hash of the tensor data to add.
        shash : str
            hash value of the schema dset file to place the array in.
        remote_operation : optional, kwarg only, bool
            If this is a remote process which is adding data, any necessary
            hdf5 dataset files will be created in the remote data dir instead
            of the stage directory. (default is False, which is for a regular
            access process)

        Returns
        -------
        tuple
            string identifying the collection dataset and collection dim-0 index
            which the array can be accessed at.
        '''
        nxt_instance = self.hNextInstance[shash]
        idx = self.hIdx[shash][nxt_instance] + 1
        nxt_dspth = self.hNextPath[shash][nxt_instance]

        if idx >= self.hMaxSize[shash][nxt_instance]:
            idx = 0
            nxt_dspth += 1
            self.hNextPath[shash][nxt_instance] += 1
            self.hColsRemain[shash][nxt_instance] -= 1

            if self.hColsRemain[shash][nxt_instance] <= 1:
                sample_array = self.wFp[shash][nxt_instance]['/0'][(0)]
                h = self.create_schema(repo_path=self.repo_path,
                                       schema_hash=shash,
                                       sample_array=sample_array,
                                       remote_operation=remote_operation)
                h.close()
                self.open(self.repo_path,
                          mode='a',
                          remote_operation=remote_operation)

        srcSlc = None
        destSlc = (self.slcExpr[idx], *(self.slcExpr[0:x] for x in array.shape))

        self.wFp[shash][nxt_instance][f'/{nxt_dspth}'].write_direct(array, srcSlc, destSlc)
        self.hIdx[shash][nxt_instance] = idx
        hashVal = self.fmtParser.encode(schema=shash,
                                        instance=nxt_instance,
                                        dataset=nxt_dspth,
                                        dataset_idx=idx,
                                        shape=array.shape)
        return hashVal

    def read_data(self, dbHashVal, mode, dtype):
        '''Read data from an hdf5 file handle at the specified locations

        Parameters
        ----------
        dbHashVal : bytes
            Hash specification stored in the DB.
        mode : str
            one of 'r' or 'a', indicating which mode this process is opening the file in.
        dtype : int
            numeric type code of the output data.

        Returns
        -------
        np.array
            requested data.
        '''
        hashVal = self.fmtParser.decode(dbHashVal)
        dsetIdx = int(hashVal.dataset_idx)
        dsetCol = f'/{hashVal.dataset}'
        fSchema = hashVal.schema
        fInstance = hashVal.instance

        srcSlc = (self.slcExpr[dsetIdx], *(self.slcExpr[0:x] for x in hashVal.shape))
        destSlc = None
        destArr = np.empty((hashVal.shape), np.typeDict[dtype])

        if mode == 'r':
            self.rFp[fSchema][fInstance][dsetCol].read_direct(destArr, srcSlc, destSlc)
        else:
            try:
                self.rFp[fSchema][fInstance][dsetCol].read_direct(destArr, srcSlc, destSlc)
            except KeyError:
                self.wFp[fSchema][fInstance][dsetCol].read_direct(destArr, srcSlc, destSlc)

        return destArr