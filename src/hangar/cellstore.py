import hashlib
import logging
import os
import warnings
from multiprocessing import cpu_count, get_context
from typing import (
    Iterator, Iterable, List, Mapping, Optional, Tuple, Union)

import lmdb
import numpy as np

from .backends import BACKEND_ACCESSOR_MAP
from .backends import backend_decoder, backend_from_heuristics
from .context import TxnRegister
from .records import parsing
from .records.queries import RecordQuery
from .utils import cm_weakref_obj_proxy, is_suitable_user_key

logger = logging.getLogger(__name__)


class CellstoreDataReader(object):
    '''Class implementing get access to data in a cellstore.

    The methods implemented here are common to the :class:`CellstoreDataWriter`
    accessor class as well as to this ``"read-only"`` method. Though minimal,
    the behavior of read and write checkouts is slightly unique, with the main
    difference being that ``"read-only"`` checkouts implement both thread and
    process safe access methods. This is not possible for ``"write-enabled"``
    checkouts, and attempts at multiprocess/threaded writes will generally
    fail with cryptic error messages.
    '''

    def __init__(self,
                 repo_pth: os.PathLike,
                 dset_name: str,
                 default_schema_hash: str,
                 samplesAreNamed: bool,
                 isVar: bool,
                 varMaxShape: list,
                 varDtypeNum: int,
                 dataenv: lmdb.Environment,
                 hashenv: lmdb.Environment,
                 mode: str,
                 *args, **kwargs):
        '''Developer documentation for init method

        The location of the data references can be transparently specified by
        feeding in a different dataenv argument. For staged reads -> ``dataenv =
        lmdb.Environment(STAGING_DB)``. For commit read -> ``dataenv =
        lmdb.Environment(COMMIT_DB)``.

        Parameters
        ----------
        repo_pth : os.PathLike
            path to the repository on disk.
        dset_name : str
            name of the cellstore
        schema_hashes : list of str
            list of all schemas referenced in the cellstore
        samplesAreNamed : bool
            do samples have names or not.
        isVar : bool
            is the cellstore schema variable shape or not
        varMaxShape : list or tuple of int
            schema size (max) of the cellstore data
        varDtypeNum : int
            datatype numeric code of the cellstore data
        dataenv : lmdb.Environment
            environment of the cellstore references to read
        hashenv : lmdb.Environment
            environment of the repository hash records
        mode : str, optional
            mode to open the file handles in. 'r' for read only, 'a' for read/write, defaults
            to 'r'
        '''
        self._mode = mode
        self._path = repo_pth
        self._dsetn = dset_name
        self._schema_variable = isVar
        self._schema_dtype_num = varDtypeNum
        self._samples_are_named = samplesAreNamed
        self._schema_max_shape = tuple(varMaxShape)
        self._default_schema_hash = default_schema_hash

        self._is_conman: bool = False
        self._index_expr_factory = np.s_
        self._index_expr_factory.maketuple = False
        self._contains_partial_remote_data: bool = False

        # ------------------------ backend setup ------------------------------

        self._fs = {}
        for backend, accessor in BACKEND_ACCESSOR_MAP.items():
            if accessor is not None:
                self._fs[backend] = accessor(
                    repo_path=self._path,
                    schema_shape=self._schema_max_shape,
                    schema_dtype=np.typeDict[self._schema_dtype_num])
                self._fs[backend].open(self._mode)

        # -------------- Sample backend specification parsing -----------------

        self._sspecs = {}
        _TxnRegister = TxnRegister()
        hashTxn = _TxnRegister.begin_reader_txn(hashenv)
        try:
            dsetNamesSpec = RecordQuery(dataenv).cellstore_data_records(self._dsetn)
            for dsetNames, dataSpec in dsetNamesSpec:
                hashKey = parsing.hash_data_db_key_from_raw_key(dataSpec.data_hash)
                hash_ref = hashTxn.get(hashKey)
                be_loc = backend_decoder(hash_ref)
                self._sspecs[dsetNames.data_name] = be_loc
                if (be_loc.backend == '50') and (not self._contains_partial_remote_data):
                    warnings.warn(
                        f'Cellstore: {self._dsetn} contains `reference-only` samples, with '
                        f'actual data residing on a remote server. A `fetch-data` '
                        f'operation is required to access these samples.', UserWarning)
                    self._contains_partial_remote_data = True
        finally:
            _TxnRegister.abort_reader_txn(hashenv)

    def __enter__(self):
        self._is_conman = True
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        return

    def __getitem__(self, key):
        '''Retrieve a sample with a given key. Convenience method for dict style access.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : string
            sample key to retrieve from the cellstore

        Returns
        -------
        np.array
            sample array data corresponding to the provided key
        '''
        return self.get(key)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        '''Check how many samples are present in a given cellstore

        Returns
        -------
        int
            number of samples the cellstore contains
        '''
        return len(self._sspecs)

    def __contains__(self, key):
        '''Determine if a key is a valid sample name in the cellstore

        Parameters
        ----------
        key : string
            name to check if it is a sample in the cellstore

        Returns
        -------
        bool
            True if key exists, else False
        '''
        exists = key in self._sspecs
        return exists

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Cellstore Name             : {self._dsetn}\
                \n    Schema Hash              : {self._default_schema_hash}\
                \n    Variable Shape           : {bool(int(self._schema_variable))}\
                \n    (max) Shape              : {self._schema_max_shape}\
                \n    Datatype                 : {np.typeDict[self._schema_dtype_num]}\
                \n    Named Samples            : {bool(self._samples_are_named)}\
                \n    Access Mode              : {self._mode}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self._contains_partial_remote_data)}\n'
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._path}, '\
              f'dset_name={self._dsetn}, '\
              f'default_schema_hash={self._default_schema_hash}, '\
              f'isVar={self._schema_variable}, '\
              f'varMaxShape={self._schema_max_shape}, '\
              f'varDtypeNum={self._schema_dtype_num}, '\
              f'mode={self._mode})'
        return res

    def _open(self):
        for val in self._fs.values():
            val.open(mode=self._mode)

    def _close(self):
        for val in self._fs.values():
            val.close()

    @property
    def name(self):
        '''Name of the cellstore. Read-Only attribute.
        '''
        return self._dsetn

    @property
    def dtype(self):
        '''Datatype of the cellstore schema. Read-only attribute.
        '''
        return np.typeDict[self._schema_dtype_num]

    @property
    def shape(self):
        '''Shape (or `max_shape`) of the cellstore sample tensors. Read-only attribute.
        '''
        return self._schema_max_shape

    @property
    def variable_shape(self):
        '''Bool indicating if cellstore schema is variable sized. Read-only attribute.
        '''
        return self._schema_variable

    @property
    def named_samples(self):
        '''Bool indicating if samples are named. Read-only attribute.
        '''
        return self._samples_are_named

    @property
    def iswriteable(self):
        '''Bool indicating if this cellstore object is write-enabled. Read-only attribute.
        '''
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> bool:
        '''Bool indicating if all samples exist locally or if some reference remote sources.
        '''
        return bool(self._contains_partial_remote_data)

    @property
    def remote_reference_sample_keys(self) -> List[str]:
        '''Returns sample names whose data is stored in a remote server reference.

        Returns
        -------
        List[str]
            list of sample keys in the cellstore.
        '''
        remote_keys = []
        if self.contains_remote_references is True:
            for sampleName, beLoc in self._sspecs.items():
                if beLoc.backend == '50':
                    remote_keys.append(sampleName)
        return remote_keys

    def keys(self) -> Iterator[Union[str, int]]:
        '''generator which yields the names of every sample in the cellstore

        For write enabled checkouts, is technically possible to iterate over the
        cellstore object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[Union[str, int]]
            keys of one sample at a time inside the cellstore
        '''
        for name in tuple(self._sspecs.keys()):
            yield name

    def values(self) -> Iterator[np.ndarray]:
        '''generator which yields the tensor data for every sample in the cellstore

        For write enabled checkouts, is technically possible to iterate over the
        cellstore object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[np.ndarray]
            values of one sample at a time inside the cellstore
        '''
        for name in tuple(self._sspecs.keys()):
            yield self.get(name)

    def items(self) -> Iterator[Tuple[Union[str, int], np.ndarray]]:
        '''generator yielding two-tuple of (name, tensor), for every sample in the cellstore.

        For write enabled checkouts, is technically possible to iterate over the
        cellstore object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[Tuple[Union[str, int], np.ndarray]]
            sample name and stored value for every sample inside the cellstore
        '''
        for name in tuple(self._sspecs.keys()):
            yield (name, self.get(name))

    def get(self, name: str) -> np.ndarray:
        '''Retrieve a sample in the cellstore with a specific name.

        The method is thread/process safe IF used in a read only checkout. Use
        this if the calling application wants to manually manage multiprocess
        logic for data retrieval. Otherwise, see the :py:meth:`get_batch` method
        to retrieve multiple data samples simultaneously. This method uses
        multiprocess pool of workers (managed by hangar) to drastically increase
        access speed and simplify application developer workflows.

        .. note::

            in most situations, we have observed little to no performance
            improvements when using multithreading. However, access time can be
            nearly linearly decreased with the number of CPU cores / workers if
            multiprocessing is used.

        Parameters
        ----------
        name : str
            Name of the sample to retrieve data for.

        Returns
        -------
        np.array
            Tensor data stored in the cellstore archived with provided name(s).

        Raises
        ------
        KeyError
            if the cellstore does not contain data with the provided name
        '''
        try:
            spec = self._sspecs[name]
            data = self._fs[spec.backend].read_data(spec)
            return data
        except KeyError:
            raise KeyError(f'HANGAR KEY ERROR:: data: {name} not in dset: {self._dsetn}')

    def get_batch(self, names: list,
                  *, n_cpus: int = None, start_method: str = 'spawn') -> list:
        '''Retrieve a batch of sample data with the provided names.

        This method is (technically) thread & process safe, though it should not
        be called in parallel via multithread/process application code; This
        method has been seen to drastically decrease retrieval time of sample
        batches (as compared to looping over single sample names sequentially).
        Internally it implements a multiprocess pool of workers (managed by
        hangar) to simplify application developer workflows.

        Parameters
        ----------
        name : list, tuple
            list/tuple of sample names to retrieve data for.
        n_cpus : int, kwarg-only
            if not None, uses num_cpus / 2 of the system for retrieval. Setting
            this value to ``1`` will not use a multiprocess pool to perform the
            work. Default is None
        start_method : str, kwarg-only
            One of 'spawn', 'fork', 'forkserver' specifying the process pool
            start method. Not all options are available on all platforms. see
            python multiprocess docs for details. Default is 'spawn'.

        Returns
        -------
        list(np.ndarray)
            Tensor data stored in the cellstore archived with provided name(s).

            If a single sample name is passed in as the, the corresponding
            np.array data will be returned.

            If a list/tuple of sample names are pass in the ``names`` argument,
            a tuple of size ``len(names)`` will be returned where each element
            is an np.array containing data at the position it's name listed in
            the ``names`` parameter.

        Raises
        ------
        KeyError
            if the cellstore does not contain data with the provided name
        '''
        n_jobs = n_cpus if isinstance(n_cpus, int) else int(cpu_count() / 2)
        with get_context(start_method).Pool(n_jobs) as p:
            data = p.map(self.get, names)
        return data


class CellstoreDataWriter(CellstoreDataReader):
    '''Class implementing methods to write data to a cellstore.

    Writer specific methods are contained here, and while read functionality is
    shared with the methods common to :class:`CellstoreDataReader`. Write-enabled
    checkouts are not thread/process safe for either ``writes`` OR ``reads``,
    a restriction we impose for ``write-enabled`` checkouts in order to ensure
    data integrity above all else.

    .. seealso:: :class:`CellstoreDataReader`

    '''

    def __init__(self,
                 stagehashenv: lmdb.Environment,
                 default_schema_backend: str,
                 *args, **kwargs):
        '''Developer documentation for init method.

        Extends the functionality of the CellstoreDataReader class. The __init__
        method requires quite a number of ``**kwargs`` to be passed along to the
        :class:`CellstoreDataReader` class.

        Parameters
        ----------
            stagehashenv : lmdb.Environment
                db where the newly added staged hash data records are stored
            default_schema_backend : str
                backend code to act as default where new data samples are added.
            **kwargs:
                See args of :class:`CellstoreDataReader`
        '''

        super().__init__(*args, **kwargs)

        self._stagehashenv = stagehashenv
        self._dflt_backend: str = default_schema_backend
        self._dataenv: lmdb.Environment = kwargs['dataenv']
        self._hashenv: lmdb.Environment = kwargs['hashenv']

        self._TxnRegister = TxnRegister()
        self._hashTxn: Optional[lmdb.Transaction] = None
        self._dataTxn: Optional[lmdb.Transaction] = None

    def __enter__(self):
        self._is_conman = True
        self._hashTxn = self._TxnRegister.begin_writer_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)
        self._stageHashTxn = self._TxnRegister.begin_writer_txn(self._stagehashenv)
        for k in self._fs.keys():
            self._fs[k].__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._hashTxn = self._TxnRegister.commit_writer_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.commit_writer_txn(self._dataenv)
        self._stageHashTxn = self._TxnRegister.commit_writer_txn(self._stagehashenv)
        for k in self._fs.keys():
            self._fs[k].__exit__(*exc)

    def __setitem__(self, key: Union[str, int], value: np.ndarray) -> Union[str, int]:
        '''Store a piece of data in a cellstore. Convenience method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : Union[str, int]
            name of the sample to add to the cellstore
        value : np.array
            tensor data to add as the sample

        Returns
        -------
        Union[str, int]
            sample name of the stored data (assuming operation was successful)
        '''
        self.add(value, key)
        return key

    def __delitem__(self, key: Union[str, int]) -> Union[str, int]:
        '''Remove a sample from the cellstore. Convenience method to :meth:`remove`.

        .. seealso:: :meth:`remove`

        Parameters
        ----------
        key : Union[str, int]
            Name of the sample to remove from the cellstore

        Returns
        -------
        Union[str, int]
            Name of the sample removed from the cellstore (assuming operation successful)
        '''
        return self.remove(key)

    @property
    def _backend(self) -> str:
        '''The default backend for the cellstore which can be written to

        Returns
        -------
        str
            numeric format code of the default backend.
        '''
        return self._dflt_backend

    def add(self, data: np.ndarray, name: Union[str, int] = None,
            **kwargs) -> Union[str, int]:
        '''Store a piece of data in a cellstore

        Parameters
        ----------
        data : np.ndarray
            data to store as a sample in the cellstore.
        name : Union[str, int], optional
            name to assign to the same (assuming the cellstore accepts named
            samples), If str, can only contain alpha-numeric ascii characters
            (in addition to '-', '.', '_'). Integer key must be >= 0. by default
            None

        Returns
        -------
        Union[str, int]
            sample name of the stored data (assuming the operation was successful)

        Raises
        ------
        ValueError
            If no `name` arg was provided for cellstore requiring named samples.
        ValueError
            If input data tensor rank exceeds specified rank of cellstore samples.
        ValueError
            For variable shape cellstores, if a dimension size of the input data
            tensor exceeds specified max dimension size of the cellstore samples.
        ValueError
            For fixed shape cellstores, if input data dimensions do not exactly match
            specified cellstore dimensions.
        ValueError
            If type of `data` argument is not an instance of np.ndarray.
        ValueError
            If `data` is not "C" contiguous array layout.
        ValueError
            If the datatype of the input data does not match the specified data type of
            the cellstore
        LookupError
            If a data sample with the same name and hash value already exists in the
            cellstore.
        '''

        # ------------------------ argument type checking ---------------------

        try:
            if self._samples_are_named and not is_suitable_user_key(name):
                raise ValueError(
                    f'Name provided: `{name}` type: {type(name)} is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace) or int >= 0')
            elif not self._samples_are_named:
                name = kwargs['bulkn'] if 'bulkn' in kwargs else parsing.generate_sample_name()

            if not isinstance(data, np.ndarray):
                raise ValueError(f'`data` argument type: {type(data)} != `np.ndarray`')
            elif data.dtype.num != self._schema_dtype_num:
                raise ValueError(
                    f'dtype: {data.dtype} != dset: {np.typeDict[self._schema_dtype_num]}.')
            elif not data.flags.c_contiguous:
                raise ValueError(f'`data` must be "C" contiguous array.')

            if self._schema_variable is True:
                if data.ndim != len(self._schema_max_shape):
                    raise ValueError(
                        f'`data` rank: {data.ndim} != dset rank: {len(self._schema_max_shape)}')
                for dDimSize, schDimSize in zip(data.shape, self._schema_max_shape):
                    if dDimSize > schDimSize:
                        raise ValueError(
                            f'dimensions of `data`: {data.shape} exceed variable max '
                            f'dims of dset: {self._dsetn} specified max dimensions: '
                            f'{self._schema_max_shape}: SIZE: {dDimSize} > {schDimSize}')
            elif data.shape != self._schema_max_shape:
                raise ValueError(
                    f'`data` shape: {data.shape} != fixed dset shape: {self._schema_max_shape}')

        except ValueError as e:
            logger.error(e, exc_info=False)
            raise e from None

        # --------------------- add data to storage backend -------------------

        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            full_hash = hashlib.blake2b(data.tobytes(), digest_size=20).hexdigest()
            hashKey = parsing.hash_data_db_key_from_raw_key(full_hash)

            # check if data record already exists
            dataRecKey = parsing.data_record_db_key_from_raw_key(self._dsetn, name)
            existingDataRecVal = self._dataTxn.get(dataRecKey, default=False)
            if existingDataRecVal:
                existingDataRec = parsing.data_record_raw_val_from_db_val(existingDataRecVal)
                if full_hash == existingDataRec.data_hash:
                    raise LookupError(
                        f'Cellstore: {self._dsetn} already contains identical object named:'
                        f'{name} with same hash value: {full_hash}')

            # write new data if data hash does not exist
            existingHashVal = self._hashTxn.get(hashKey, default=False)
            if existingHashVal is False:
                hashVal = self._fs[self._dflt_backend].write_data(data)
                self._hashTxn.put(hashKey, hashVal)
                self._stageHashTxn.put(hashKey, hashVal)
                self._sspecs[name] = backend_decoder(hashVal)
            else:
                self._sspecs[name] = backend_decoder(existingHashVal)

            # add the record to the db
            dataRecVal = parsing.data_record_db_val_from_raw_val(full_hash)
            self._dataTxn.put(dataRecKey, dataRecVal)

            if existingDataRecVal is False:
                dsetCountKey = parsing.cellstore_record_count_db_key_from_raw_key(self._dsetn)
                dsetCountVal = self._dataTxn.get(dsetCountKey, default='0'.encode())
                newDsetCount = parsing.cellstore_record_count_raw_val_from_db_val(dsetCountVal) + 1
                newDsetCountVal = parsing.cellstore_record_count_db_val_from_raw_val(newDsetCount)
                self._dataTxn.put(dsetCountKey, newDsetCountVal)

        except LookupError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if tmpconman:
                self.__exit__()

        return name

    def remove(self, name: Union[str, int]) -> Union[str, int]:
        '''Remove a sample with the provided name from the cellstore.

        .. Note::

            This operation will NEVER actually remove any data from disk. If
            you commit a tensor at any point in time, **it will always remain
            accessible by checking out a previous commit** when the tensor was
            present. This is just a way to tell Hangar that you don't want some
            piece of data to clutter up the current version of the repository.

        .. Warning::

            Though this may change in a future release, in the current version of
            Hangar, we cannot recover references to data which was added to the
            staging area, written to disk, but then removed **before** a commit
            operation was run. This would be a similar sequence of events as:
            checking out a `git` branch, changing a bunch of text in the file, and
            immediately performing a hard reset. If it was never committed, git
            doesn't know about it, and (at the moment) neither does Hangar.

        Parameters
        ----------
        name : Union[str, int]
            name of the sample to remove.

        Returns
        -------
        Union[str, int]
            If the operation was successful, name of the data sample deleted.

        Raises
        ------
        KeyError
            If a sample with the provided name does not exist in the cellstore.
        '''
        if not self._is_conman:
            self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)

        dataKey = parsing.data_record_db_key_from_raw_key(self._dsetn, name)
        try:
            isRecordDeleted = self._dataTxn.delete(dataKey)
            if isRecordDeleted is False:
                raise KeyError(f'No sample: {name} type: {type(name)} exists in: {self._dsetn}')
            del self._sspecs[name]

            dsetDataCountKey = parsing.cellstore_record_count_db_key_from_raw_key(self._dsetn)
            dsetDataCountVal = self._dataTxn.get(dsetDataCountKey)
            newDsetDataCount = parsing.cellstore_record_count_raw_val_from_db_val(dsetDataCountVal) - 1

            # if this is the last data piece existing in a cellstore, remove the cellstore
            if newDsetDataCount == 0:
                dsetSchemaKey = parsing.cellstore_record_schema_db_key_from_raw_key(self._dsetn)
                self._dataTxn.delete(dsetDataCountKey)
                self._dataTxn.delete(dsetSchemaKey)
                totalNumDsetsKey = parsing.cellstore_total_count_db_key()
                totalNumDsetsVal = self._dataTxn.get(totalNumDsetsKey)
                newTotalNumDsets = parsing.cellstore_total_count_raw_val_from_db_val(totalNumDsetsVal) - 1
                # if no more cellstores exist, delete the indexing key
                if newTotalNumDsets == 0:
                    self._dataTxn.delete(totalNumDsetsKey)
                # otherwise just decrement the count of dsets
                else:
                    newTotalNumDsetsVal = parsing.cellstore_total_count_db_val_from_raw_val(newTotalNumDsets)
                    self._dataTxn.put(newTotalNumDsetsVal)
            # otherwise just decrement the cellstore record count
            else:
                newDsetDataCountVal = parsing.cellstore_record_count_db_val_from_raw_val(newDsetDataCount)
                self._dataTxn.put(dsetDataCountKey, newDsetDataCountVal)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if not self._is_conman:
                self._TxnRegister.commit_writer_txn(self._dataenv)

        return name


'''
Constructor and Interaction Class for Cellstores
--------------------------------------------------
'''


class Cellstores(object):
    '''Common access patterns and initialization/removal of cellstores in a checkout.

    This object is the entry point to all tensor data stored in their individual
    cellstores. Each cellstore contains a common schema which dictates the general
    shape, dtype, and access patters which the backends optimize access for. The
    methods contained within allow us to create, remove, query, and access these
    collections of common tensors.
    '''

    def __init__(self,
                 mode: str,
                 repo_pth: os.PathLike,
                 cellstores: Mapping[str, Union[CellstoreDataReader, CellstoreDataWriter]],
                 hashenv: Optional[lmdb.Environment] = None,
                 dataenv: Optional[lmdb.Environment] = None,
                 stagehashenv: Optional[lmdb.Environment] = None):
        '''Developer documentation for init method.

        .. warning::

            This class should not be instantiated directly. Instead use the factory
            functions :py:meth:`_from_commit` or :py:meth:`_from_staging` to return
            a pre-initialized class instance appropriately constructed for either a
            read-only or write-enabled checkout.

        Parameters
        ----------
        mode : str
            one of 'r' or 'a' to indicate read or write mode
        repo_pth : os.PathLike
            path to the repository on disk
        cellstores : Mapping[str, Union[CellstoreDataReader, CellstoreDataWriter]]
            dictionary of CellstoreData objects
        hashenv : Optional[lmdb.Environment]
            environment handle for hash records
        dataenv : Optional[lmdb.Environment]
            environment handle for the unpacked records. `data` is means to refer to
            the fact that the stageenv is passed in for for write-enabled, and a
            cmtrefenv for read-only checkouts.
        stagehashenv : Optional[lmdb.Environment]
            environment handle for newly added staged data hash records.
        '''
        self._mode = mode
        self._repo_pth = repo_pth
        self._cellstores = cellstores
        self._is_conman = False
        self._contains_partial_remote_data: bool = False

        if (mode == 'a'):
            self._hashenv = hashenv
            self._dataenv = dataenv
            self._stagehashenv = stagehashenv

        self.__setup()

    def __setup(self):
        '''Do not allow users to use internal functions
        '''
        self._from_commit = None  # should never be able to access
        self._from_staging_area = None  # should never be able to access
        if self._mode == 'r':
            self.init_cellstore = None
            self.remove_dset = None
            self.multi_add = None
            self.__delitem__ = None
            self.__setitem__ = None

    def _open(self):
        for v in self._cellstores.values():
            v._open()

    def _close(self):
        for v in self._cellstores.values():
            v._close()

# ------------- Methods Available To Both Read & Write Checkouts ------------------

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__}\
                \n    Writeable: {bool(0 if self._mode == "r" else 1)}\
                \n    Cellstore Names / Partial Remote References:\
                \n      - ' + '\n      - '.join(
            f'{dsetn} / {dset.contains_remote_references}'
            for dsetn, dset in self._cellstores.items())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'cellstores={self._cellstores}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        '''Let ipython know that any key based access can use the cellstore keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for the
        sanity of developers, this is the best way to ensure users can autocomplete
        keys.

        Returns
        -------
        list
            list of strings, each being one of the cellstore keys for access.
        '''
        return self.keys()

    def __getitem__(self, key):
        '''Dict style access to return the cellstore object with specified key/name.

        Parameters
        ----------
        key : string
            name of the cellstore object to get.

        Returns
        -------
        :class:`CellstoreDataReader` or :class:`CellstoreDataWriter`
            The object which is returned depends on the mode of checkout specified.
            If the cellstore was checked out with write-enabled, return writer object,
            otherwise return read only object.
        '''
        return self.get(key)

    def __setitem__(self, key, value):
        '''Specifically prevent use dict style setting for cellstore objects.

        Cellstores must be created using the factory function :py:meth:`init_cellstore`.

        Raises
        ------
        PermissionError
            This operation is not allowed under any circumstance

        '''
        msg = f'HANGAR NOT ALLOWED:: To add a cellstore use `init_cellstore` method.'
        raise PermissionError(msg)

    def __contains__(self, key: str) -> bool:
        '''Determine if a cellstore with a particular name is stored in the checkout

        Parameters
        ----------
        key : str
            name of the cellstore to check for

        Returns
        -------
        bool
            True if a cellstore with the provided name exists in the checkout,
            otherwise False.
        '''
        return True if key in self._cellstores else False

    def __len__(self):
        return len(self._cellstores)

    def __iter__(self):
        return iter(self._cellstores)

    @property
    def iswriteable(self):
        '''Bool indicating if this cellstore object is write-enabled. Read-only attribute.
        '''
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> Mapping[str, bool]:
        '''Dict of bool indicating data reference locality in each cellstore.

        Returns
        -------
        Mapping[str, bool]
            For each cellstore name key, boolean value where False indicates all
            samples in cellstore exist locally, True if some reference remote
            sources.
        '''
        res: Mapping[str, bool] = {}
        for dsetn, dset in self._cellstores.items():
            res[dsetn] = dset.contains_remote_references
        return res

    @property
    def remote_sample_keys(self) -> Mapping[str, Iterable[Union[int, str]]]:
        '''Determine cellstores samples names which reference remote sources.

        Returns
        -------
        Mapping[str, Iterable[Union[int, str]]]
            dict where keys are cellstore names and values are iterables of
            samples in the cellstore containing remote references
        '''
        res: Mapping[str, Iterable[Union[int, str]]] = {}
        for dsetn, dset in self._cellstores.items():
            res[dsetn] = dset.remote_reference_sample_keys
        return res

    def keys(self) -> List[str]:
        '''list all cellstore keys (names) in the checkout

        Returns
        -------
        List[str]
            list of cellstore names
        '''
        return list(self._cellstores.keys())

    def values(self) -> Iterable[Union[CellstoreDataReader, CellstoreDataWriter]]:
        '''yield all cellstore object instances in the checkout.

        Yields
        -------
        Iterable[Union[CellstoreDataReader, CellstoreDataWriter]]
            Generator of CellstoreData accessor objects (set to read or write mode
            as appropriate)
        '''
        for dsetObj in self._cellstores.values():
            wr = cm_weakref_obj_proxy(dsetObj)
            yield wr

    def items(self) -> Iterable[Tuple[str, Union[CellstoreDataReader, CellstoreDataWriter]]]:
        '''generator providing access to cellstore_name, :class:`Cellstores`

        Yields
        ------
        Iterable[Tuple[str, Union[CellstoreDataReader, CellstoreDataWriter]]]
            returns two tuple of all all cellstore names/object pairs in the checkout.
        '''
        for dsetN, dsetObj in self._cellstores.items():
            wr = cm_weakref_obj_proxy(dsetObj)
            yield (dsetN, wr)

    def get(self, name: str) -> Union[CellstoreDataReader, CellstoreDataWriter]:
        '''Returns a cellstore access object.

        This can be used in lieu of the dictionary style access.

        Parameters
        ----------
        name : str
            name of the cellstore to return

        Returns
        -------
        Union[CellstoreDataReader, CellstoreDataWriter]
            CellstoreData accessor (set to read or write mode as appropriate) which
            governs interaction with the data

        Raises
        ------
        KeyError
            If no cellstore with the given name exists in the checkout
        '''
        try:
            wr = cm_weakref_obj_proxy(self._cellstores[name])
            return wr
        except KeyError:
            e = KeyError(f'No cellstore exists with name: {name}')
            logger.error(e, exc_info=False)
            raise e

# ------------------------ Writer-Enabled Methods Only ------------------------------

    def __delitem__(self, key: str) -> str:
        '''remove a cellstore and all data records if write-enabled process.

        Parameters
        ----------
        key : str
            Name of the cellstore to remove from the repository. This will remove
            all records from the staging area (though the actual data and all
            records are still accessible) if they were previously committed

        Returns
        -------
        str
            If successful, the name of the removed cellstore.

        Raises
        ------
        PermissionError
            If this is a read-only checkout, no operation is permitted.
        '''
        return self.remove_dset(key)

    def __enter__(self):
        self._is_conman = True
        for dskey in list(self._cellstores):
            self._cellstores[dskey].__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        for dskey in list(self._cellstores):
            self._cellstores[dskey].__exit__(*exc)

    def multi_add(self, mapping: dict) -> str:
        '''Add related samples to un-named cellstores with the same generated key.

        If you have multiple cellstores in a checkout whose samples are related to
        each other in some manner, there are two ways of associating samples
        together:

        1) using named cellstores and setting each tensor in each cellstore to the
           same sample "name" using un-named cellstores.
        2) using this "add" method. which accepts a dictionary of "cellstore
           names" as keys, and "tensors" (ie. individual samples) as values.

        When method (2) - this method - is used, the internally generated sample
        ids will be set to the same value for the samples in each cellstore. That
        way a user can iterate over the cellstore key's in one sample, and use
        those same keys to get the other related tensor samples in another
        cellstore.

        Parameters
        ----------
        mapping: dict
            Dict mapping (any number of) cellstore names to tensor data (samples)
            which to add. The cellstores must exist, and must be set to accept
            samples which are not named by the user

        Returns
        -------
        str
            generated id (key) which each sample is stored under in their
            corresponding cellstore. This is the same for all samples specified in
            the input dictionary.


        Raises
        ------
        KeyError
            If no cellstore with the given name exists in the checkout
        '''
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            if not all([k in self._cellstores for k in mapping.keys()]):
                raise KeyError(
                    f'some key(s): {mapping.keys()} not in dset(s): {self._cellstores.keys()}')
            data_name = parsing.generate_sample_name()
            for k, v in mapping.items():
                self._cellstores[k].add(v, bulkn=data_name)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise e from None

        finally:
            if tmpconman:
                self.__exit__()

        return data_name

    def init_cellstore(self,
                       name: str,
                       shape: Union[int, Tuple[int]] = None,
                       dtype: np.dtype = None,
                       prototype: np.ndarray = None,
                       named_samples: bool = True,
                       variable_shape: bool = False,
                       *,
                       backend: str = None):
        '''Initializes a cellstore in the repository.

        Cellstores are groups of related data pieces (samples). All samples within
        a cellstore have the same data type, and number of dimensions. The size of
        each dimension can be either fixed (the default behavior) or variable
        per sample.

        For fixed dimension sizes, all samples written to the cellstore must have
        the same size that was initially specified upon cellstore initialization.
        Variable size cellstores on the other hand, can write samples with
        dimensions of any size less than a maximum which is required to be set
        upon cellstore creation.

        Parameters
        ----------
        name : str
            The name assigned to this cellstore.
        shape : Union[int, Tuple[int]]
            The shape of the data samples which will be written in this cellstore.
            This argument and the `dtype` argument are required if a `prototype`
            is not provided, defaults to None.
        dtype : np.dtype
            The datatype of this cellstore. This argument and the `shape` argument
            are required if a `prototype` is not provided., defaults to None.
        prototype : np.ndarray
            A sample array of correct datatype and shape which will be used to
            initialize the cellstore storage mechanisms. If this is provided, the
            `shape` and `dtype` arguments must not be set, defaults to None.
        named_samples : bool, optional
            If the samples in the cellstore have names associated with them. If set,
            all samples must be provided names, if not, no name will be assigned.
            defaults to True, which means all samples should have names.
        variable_shape : bool, optional
            If this is a variable sized cellstore. If true, a the maximum shape is
            set from the provided `shape` or `prototype` argument. Any sample
            added to the cellstore can then have dimension sizes <= to this
            initial specification (so long as they have the same rank as what
            was specified) defaults to False.
        backend : DEVELOPER USE ONLY. str, optional, kwarg only
            Backend which should be used to write the cellstore files on disk.

        Returns
        -------
        :class:`CellstoreDataWriter`
            instance object of the initialized cellstore.

        Raises
        ------
        ValueError
            If provided name contains any non ascii, non alpha-numeric characters.
        ValueError
            If required `shape` and `dtype` arguments are not provided in absence of
            `prototype` argument.
        ValueError
            If `prototype` argument is not a C contiguous ndarray.
        LookupError
            If a cellstore already exists with the provided name.
        ValueError
            If rank of maximum tensor shape > 31.
        ValueError
            If zero sized dimension in `shape` argument
        ValueError
            If the specified backend is not valid.
        '''

        # ------------- Checks for argument validity --------------------------

        try:

            if not is_suitable_user_key(name):
                raise ValueError(
                    f'Cellstore name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace).')
            if name in self._cellstores:
                raise LookupError(f'KEY EXISTS: cellstore already exists with name: {name}.')

            if prototype is not None:
                if not isinstance(prototype, np.ndarray):
                    raise ValueError(
                        f'If specified (not None) `prototype` argument be `np.ndarray`-like.'
                        f'Invalid value: {prototype} of type: {type(prototype)}')
                elif not prototype.flags.c_contiguous:
                    raise ValueError(f'`prototype` must be "C" contiguous array.')
            elif isinstance(shape, (tuple, list, int)) and (dtype is not None):
                prototype = np.zeros(shape, dtype=dtype)
            else:
                raise ValueError(f'`shape` & `dtype` args required if no `prototype` set.')

            if (0 in prototype.shape) or (prototype.ndim > 31):
                raise ValueError(
                    f'Invalid shape specification with ndim: {prototype.ndim} and shape: '
                    f'{prototype.shape}. Array rank > 31 dimensions not allowed AND '
                    'all dimension sizes must be > 0.')

            if backend is not None:
                if backend not in BACKEND_ACCESSOR_MAP:
                    raise ValueError(f'Backend specifier: {backend} not known')
            else:
                backend = backend_from_heuristics(prototype)

        except (ValueError, LookupError) as e:
            logger.error(e, exc_info=False)
            raise e

        # ----------- Determine schema format details -------------------------

        schema_format = np.array(
            (*prototype.shape, prototype.size, prototype.dtype.num), dtype=np.uint64)
        schema_hash = hashlib.blake2b(schema_format.tobytes(), digest_size=6).hexdigest()

        dsetCountKey = parsing.cellstore_record_count_db_key_from_raw_key(name)
        dsetCountVal = parsing.cellstore_record_count_db_val_from_raw_val(0)
        dsetSchemaKey = parsing.cellstore_record_schema_db_key_from_raw_key(name)
        dsetSchemaVal = parsing.cellstore_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=variable_shape,
            schema_max_shape=prototype.shape,
            schema_dtype=prototype.dtype.num,
            schema_is_named=named_samples,
            schema_default_backend=backend)

        # -------- set vals in lmdb only after schema is sure to exist --------

        dataTxn = TxnRegister().begin_writer_txn(self._dataenv)
        hashTxn = TxnRegister().begin_writer_txn(self._hashenv)
        numDsetsCountKey = parsing.cellstore_total_count_db_key()
        numDsetsCountVal = dataTxn.get(numDsetsCountKey, default=('0'.encode()))
        numDsets_count = parsing.cellstore_total_count_raw_val_from_db_val(numDsetsCountVal)
        numDsetsCountVal = parsing.cellstore_record_count_db_val_from_raw_val(numDsets_count + 1)
        hashSchemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashSchemaVal = dsetSchemaVal

        dataTxn.put(dsetCountKey, dsetCountVal)
        dataTxn.put(numDsetsCountKey, numDsetsCountVal)
        dataTxn.put(dsetSchemaKey, dsetSchemaVal)
        hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)
        TxnRegister().commit_writer_txn(self._dataenv)
        TxnRegister().commit_writer_txn(self._hashenv)

        self._cellstores[name] = CellstoreDataWriter(
            stagehashenv=self._stagehashenv,
            repo_pth=self._repo_pth,
            dset_name=name,
            default_schema_hash=schema_hash,
            samplesAreNamed=named_samples,
            isVar=variable_shape,
            varMaxShape=prototype.shape,
            varDtypeNum=prototype.dtype.num,
            hashenv=self._hashenv,
            dataenv=self._dataenv,
            mode='a',
            default_schema_backend=backend)

        return self.get(name)

    def remove_dset(self, dset_name: str) -> str:
        '''remove the cellstore and all data contained within it from the repository.

        Parameters
        ----------
        dset_name : str
            name of the cellstore to remove

        Returns
        -------
        str
            name of the removed cellstore

        Raises
        ------
        KeyError
            If a cellstore does not exist with the provided name
        '''
        datatxn = TxnRegister().begin_writer_txn(self._dataenv)
        try:
            if dset_name not in self._cellstores:
                e = KeyError(f'HANGAR KEY ERROR:: Cannot remove: {dset_name}. Key does not exist.')
                logger.error(e, exc_info=False)
                raise e
            self._cellstores[dset_name]._close()
            self._cellstores.__delitem__(dset_name)

            dsetCountKey = parsing.cellstore_record_count_db_key_from_raw_key(dset_name)
            numDsetsKey = parsing.cellstore_total_count_db_key()
            arraysInDset = datatxn.get(dsetCountKey)
            recordsToDelete = parsing.cellstore_total_count_raw_val_from_db_val(arraysInDset)
            recordsToDelete = recordsToDelete + 1  # depends on num subkeys per array recy
            with datatxn.cursor() as cursor:
                cursor.set_key(dsetCountKey)
                for i in range(recordsToDelete):
                    cursor.delete()
            cursor.close()

            dsetSchemaKey = parsing.cellstore_record_schema_db_key_from_raw_key(dset_name)
            datatxn.delete(dsetSchemaKey)
            numDsetsVal = datatxn.get(numDsetsKey)
            numDsets = parsing.cellstore_total_count_raw_val_from_db_val(numDsetsVal) - 1
            if numDsets == 0:
                datatxn.delete(numDsetsKey)
            else:
                numDsetsVal = parsing.cellstore_total_count_db_val_from_raw_val(numDsets)
                datatxn.put(numDsetsKey, numDsetsVal)
        finally:
            TxnRegister().commit_writer_txn(self._dataenv)

        return dset_name

# ------------------------ Class Factory Functions ------------------------------

    @classmethod
    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        '''Class method factory to checkout :class:`Cellstores` in write-enabled mode

        This is not a user facing operation, and should never be manually called
        in normal operation. Once you get here, we currently assume that
        verification of the write lock has passed, and that write operations are
        safe.

        Parameters
        ----------
        repo_pth : string
            directory path to the hangar repository on disk
        hashenv : lmdb.Environment
            environment where tensor data hash records are open in write mode.
        stageenv : lmdb.Environment
            environment where staging records (dataenv) are opened in write mode.
        stagehashenv: lmdb.Environment
            environment where the staged hash records are stored in write mode

        Returns
        -------
        :class:`Cellstores`
            Interface class with write-enabled attributes activated and any
            cellstores existing initialized in write mode via
            :class:`.cellstore.CellstoreDataWriter`.
        '''

        cellstores = {}
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()
        for dsetName, schemaSpec in stagedSchemaSpecs.items():
            cellstores[dsetName] = CellstoreDataWriter(
                stagehashenv=stagehashenv,
                repo_pth=repo_pth,
                dset_name=dsetName,
                default_schema_hash=schemaSpec.schema_hash,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                hashenv=hashenv,
                dataenv=stageenv,
                mode='a',
                default_schema_backend=schemaSpec.schema_default_backend)

        return cls('a', repo_pth, cellstores, hashenv, stageenv, stagehashenv)

    @classmethod
    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        '''Class method factory to checkout :class:`.cellstore.Cellstores` in read-only mode

        This is not a user facing operation, and should never be manually called
        in normal operation. For read mode, no locks need to be verified, but
        construction should occur through the interface to the
        :class:`Cellstores` class.

        Parameters
        ----------
        repo_pth : string
            directory path to the hangar repository on disk
        hashenv : lmdb.Environment
            environment where tensor data hash records are open in read-only mode.
        cmtrefenv : lmdb.Environment
            environment where staging checkout records are opened in read-only mode.

        Returns
        -------
        :class:`Cellstores`
            Interface class with all write-enabled attributes deactivated
            cellstores initialized in read mode via :class:`.cellstore.CellstoreDataReader`.
        '''
        cellstores = {}
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()
        for dsetName, schemaSpec in cmtSchemaSpecs.items():
            cellstores[dsetName] = CellstoreDataReader(
                repo_pth=repo_pth,
                dset_name=dsetName,
                default_schema_hash=schemaSpec.schema_hash,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                dataenv=cmtrefenv,
                hashenv=hashenv,
                mode='r')

        return cls('r', repo_pth, cellstores, None, None, None)
