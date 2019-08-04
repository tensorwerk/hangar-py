import os
import hashlib
import logging
import warnings
from typing import Optional, MutableMapping, List, Union, Mapping, Iterator, Tuple
from multiprocessing import get_context, cpu_count

import lmdb
import numpy as np

from .context import TxnRegister
from .backends import backend_decoder, BACKEND_ACCESSOR_MAP, backend_from_heuristics
from .records import parsing
from .records.queries import RecordQuery
from .utils import is_suitable_user_key, cm_weakref_obj_proxy

logger = logging.getLogger(__name__)


class DatasetDataReader(object):
    '''Class implementing get access to data in a dataset

    The location of the data references can be transparently specified by
    feeding in a different dataenv argument. For staged reads -> ``dataenv =
    lmdb.Environment(STAGING_DB)``. For commit read -> ``dataenv =
    lmdb.Environment(COMMIT_DB)``.

    Parameters
    ----------
    repo_pth : str
        path to the repository on disk.
    dset_name : str
        name of the dataset
    schema_hashes : list of str
        list of all schemas referenced in the dataset
    samplesAreNamed : bool
        do samples have names or not.
    isVar : bool
        is the dataset schema variable shape or not
    varMaxShape : list or tuple of int
        schema size (max) of the dataset data
    varDtypeNum : int
        datatype numeric code of the dataset data
    dataenv : lmdb.Environment
        environment of the dataset references to read
    hashenv : lmdb.Environment
        environment of the repository hash records
    mode : str, optional
        mode to open the file handles in. 'r' for read only, 'a' for read/write, defaults
        to 'r'
    '''

    def __init__(self,
                 repo_pth: str,
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
            dsetNamesSpec = RecordQuery(dataenv).dataset_data_records(self._dsetn)
            for dsetNames, dataSpec in dsetNamesSpec:
                hashKey = parsing.hash_data_db_key_from_raw_key(dataSpec.data_hash)
                hash_ref = hashTxn.get(hashKey)
                be_loc = backend_decoder(hash_ref)
                self._sspecs[dsetNames.data_name] = be_loc
                if (be_loc.backend == '50') and (not self._contains_partial_remote_data):
                    warnings.warn(
                        f'Dataset: {self._dsetn} contains `reference-only` samples, with '
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
            sample key to retrieve from the dataset

        Returns
        -------
        np.array
            sample array data corresponding to the provided key
        '''
        return self.get(key)

    def __iter__(self):
        return self.keys()

    def __len__(self):
        '''Check how many samples are present in a given dataset

        Returns
        -------
        int
            number of samples the dataset contains
        '''
        return len(self._sspecs)

    def __contains__(self, key):
        '''Determine if a key is a valid sample name in the dataset

        Parameters
        ----------
        key : string
            name to check if it is a sample in the dataset

        Returns
        -------
        bool
            True if key exists, else False
        '''
        exists = key in self._sspecs
        return exists

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Dataset Name             : {self._dsetn}\
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
        '''Name of the dataset. Read-Only attribute.
        '''
        return self._dsetn

    @property
    def dtype(self):
        '''Datatype of the dataset schema. Read-only attribute.
        '''
        return np.typeDict[self._schema_dtype_num]

    @property
    def shape(self):
        '''Shape (or `max_shape`) of the dataset sample tensors. Read-only attribute.
        '''
        return self._schema_max_shape

    @property
    def variable_shape(self):
        '''Bool indicating if dataset schema is variable sized. Read-only attribute.
        '''
        return self._schema_variable

    @property
    def named_samples(self):
        '''Bool indicating if samples are named. Read-only attribute.
        '''
        return self._samples_are_named

    @property
    def iswriteable(self):
        '''Bool indicating if this dataset object is write-enabled. Read-only attribute.
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
            list of sample keys in the dataset.
        '''
        remote_keys = []
        if self.contains_remote_references is True:
            for sampleName, beLoc in self._sspecs.items():
                if beLoc.backend == '50':
                    remote_keys.append(sampleName)
        return remote_keys

    def keys(self) -> Iterator[str]:
        '''generator which yields the names of every sample in the dataset

        unlike write-enabled checkouts, reader checkouts do not make a copy
        of the key list while iterating as the checkout has no way to modify
        the contents of the list.

        Yields
        ------
        Iterator[str]
            keys of one sample at a time inside the dataset
        '''
        for name in self._sspecs:
            yield name

    def values(self) -> Iterator[np.ndarray]:
        '''generator which yields the tensor data for every sample in the dataset

        unlike write-enabled checkouts, reader checkouts do not make a copy
        of the key/value list while iterating as the checkout has no way to modify
        the contents of the list.

        Yields
        ------
        Iterator[np.ndarray]
            values of one sample at a time inside the dataset
        '''
        for name in self._sspecs:
            yield self.get(name)

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        '''generator yielding two-tuple of (name, tensor), for every sample in the dataset.

        unlike write-enabled checkouts, reader checkouts do not make a copy
        of the key/values list while iterating as the checkout has no way to modify
        the contents of the list.

        Yields
        ------
        Iterator[Tuple[str, np.ndarray]]
            sample name and stored value for every sample inside the dataset
        '''
        for name in self._sspecs:
            yield (name, self.get(name))

    def get(self, name: str) -> np.ndarray:
        '''Retrieve a sample in the dataset with a specific name.

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
            Tensor data stored in the dataset archived with provided name(s).

        Raises
        ------
        KeyError
            if the dataset does not contain data with the provided name
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
            Tensor data stored in the dataset archived with provided name(s).

            If a single sample name is passed in as the, the corresponding
            np.array data will be returned.

            If a list/tuple of sample names are pass in the ``names`` argument,
            a tuple of size ``len(names)`` will be returned where each element
            is an np.array containing data at the position it's name listed in
            the ``names`` parameter.

        Raises
        ------
        KeyError
            if the dataset does not contain data with the provided name
        '''
        n_jobs = n_cpus if isinstance(n_cpus, int) else int(cpu_count() / 2)
        with get_context(start_method).Pool(n_jobs) as p:
            data = p.map(self.get, names)
        return data


class DatasetDataWriter(DatasetDataReader):
    '''Class implementing methods to write data to a dataset.

    Extends the functionality of the DatasetDataReader class. The __init__
    method requires quite a number of ``**kwargs`` to be passed along to the
    :class:`DatasetDataReader` class.

    .. seealso:: :class:`DatasetDataReader`

    Parameters
    ----------
        stagehashenv : lmdb.Environment
            db where the newly added staged hash data records are stored
        default_schema_backend : str
            backend code to act as default where new data samples are added.
        **kwargs:
            See args of :class:`DatasetDataReader`
    '''

    def __init__(self, stagehashenv, default_schema_backend, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self._stagehashenv = stagehashenv
        self._dflt_backend: str = default_schema_backend
        self._dataenv: lmdb.Environment = kwargs['dataenv']
        self._hashenv: lmdb.Environment = kwargs['hashenv']

        self._TxnRegister = TxnRegister()
        self._Query = RecordQuery(self._dataenv)
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
        '''Store a piece of data in a dataset. Convenience method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : Union[str, int]
            name of the sample to add to the dataset
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
        '''Remove a sample from the dataset. Convenience method to :meth:`remove`.

        .. seealso:: :meth:`remove`

        Parameters
        ----------
        key : Union[str, int]
            Name of the sample to remove from the dataset

        Returns
        -------
        Union[str, int]
            Name of the sample removed from the dataset (assuming operation successful)
        '''
        return self.remove(key)

    @property
    def _backend(self) -> str:
        '''The default backend for the dataset which can be written to

        Returns
        -------
        str
            numeric format code of the default backend.
        '''
        return self._dflt_backend

    def keys(self) -> Iterator[str]:
        '''generator which yields the names of every sample in the dataset

        For write enabled checkouts, is technically possible to iterate over the
        dataset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginging the loop.)
        However, in order to avoid differences between read-write checkouts we
        still just return the results as a generator to the user.

        Yields
        ------
        Iterator[str]
            name of keys inside the dataset
        '''
        for name in list(self._sspecs):
            yield name

    def values(self) -> Iterator[np.ndarray]:
        '''generator which yields the tensor data for every sample in the dataset

        For write enabled checkouts, is technically possible to iterate over the
        dataset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginging the loop.)
        However, in order to avoid differences between read-write checkouts we
        still just return the results as a generator to the user.

        Yields
        ------
        Iterator[np.ndarray]
            values of one sample at a time inside the dataset
        '''
        for name in list(self._sspecs):
            yield self.get(name)

    def items(self) -> Iterator[Tuple[str, np.ndarray]]:
        '''generator yielding two-tuple of (name, tensor), for every sample in the dataset.

        For write enabled checkouts, is technically possible to iterate over the
        dataset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginging the loop.)
        However, in order to avoid differences between read-write checkouts we
        still just return the results as a generator to the user.

        Yields
        ------
        Iterator[Tuple[str, np.ndarray]]
            sample name and stored value for every sample inside the dataset
        '''
        for name in list(self._sspecs):
            yield (name, self.get(name))

    def add(self, data: np.ndarray, name: Union[str, int] = None, **kwargs) -> Union[str, int]:
        '''Store a piece of data in a dataset

        Parameters
        ----------
        data : np.ndarray
            data to store as a sample in the dataset.
        name : Union[str, int], optional
            name to assign to the same (assuming the dataset accepts named
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
            If no `name` arg was provided for dataset requiring named samples.
        ValueError
            If input data tensor rank exceeds specified rank of dataset samples.
        ValueError
            For variable shape datasets, if a dimension size of the input data
            tensor exceeds specified max dimension size of the dataset samples.
        ValueError
            For fixed shape datasets, if input data dimensions do not exactly match
            specified dataset dimensions.
        ValueError
            If type of `data` argument is not an instance of np.ndarray.
        ValueError
            If `data` is not "C" contiguous array layout.
        ValueError
            If the datatype of the input data does not match the specified data type of
            the dataset
        LookupError
            If a data sample with the same name and hash value already exists in the
            dataset.
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
                        f'Dataset: {self._dsetn} already contains identical object named:'
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
                dsetCountKey = parsing.dataset_record_count_db_key_from_raw_key(self._dsetn)
                dsetCountVal = self._dataTxn.get(dsetCountKey, default='0'.encode())
                newDsetCount = parsing.dataset_record_count_raw_val_from_db_val(dsetCountVal) + 1
                newDsetCountVal = parsing.dataset_record_count_db_val_from_raw_val(newDsetCount)
                self._dataTxn.put(dsetCountKey, newDsetCountVal)

        except LookupError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if tmpconman:
                self.__exit__()

        return name

    def remove(self, name: Union[str, int]) -> Union[str, int]:
        '''Remove a sample with the provided name from the dataset.

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
            If a sample with the provided name does not exist in the dataset.
        '''
        if not self._is_conman:
            self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)

        dataKey = parsing.data_record_db_key_from_raw_key(self._dsetn, name)
        try:
            isRecordDeleted = self._dataTxn.delete(dataKey)
            if isRecordDeleted is False:
                raise KeyError(f'No sample: {name} type: {type(name)} exists in: {self._dsetn}')
            del self._sspecs[name]

            dsetDataCountKey = parsing.dataset_record_count_db_key_from_raw_key(self._dsetn)
            dsetDataCountVal = self._dataTxn.get(dsetDataCountKey)
            newDsetDataCount = parsing.dataset_record_count_raw_val_from_db_val(dsetDataCountVal) - 1

            # if this is the last data piece existing in a dataset, remove the dataset
            if newDsetDataCount == 0:
                dsetSchemaKey = parsing.dataset_record_schema_db_key_from_raw_key(self._dsetn)
                self._dataTxn.delete(dsetDataCountKey)
                self._dataTxn.delete(dsetSchemaKey)
                totalNumDsetsKey = parsing.dataset_total_count_db_key()
                totalNumDsetsVal = self._dataTxn.get(totalNumDsetsKey)
                newTotalNumDsets = parsing.dataset_total_count_raw_val_from_db_val(totalNumDsetsVal) - 1
                # if no more datasets exist, delete the indexing key
                if newTotalNumDsets == 0:
                    self._dataTxn.delete(totalNumDsetsKey)
                # otherwise just decrement the count of dsets
                else:
                    newTotalNumDsetsVal = parsing.dataset_total_count_db_val_from_raw_val(newTotalNumDsets)
                    self._dataTxn.put(newTotalNumDsetsVal)
            # otherwise just decrement the dataset record count
            else:
                newDsetDataCountVal = parsing.dataset_record_count_db_val_from_raw_val(newDsetDataCount)
                self._dataTxn.put(dsetDataCountKey, newDsetDataCountVal)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if not self._is_conman:
                self._TxnRegister.commit_writer_txn(self._dataenv)

        return name


'''
Constructor and Interaction Class for Datasets
--------------------------------------------------
'''


class Datasets(object):
    '''Common access patterns and initialization/removal of datasets in a checkout.

    .. warning::

        This class should not be instantiated directly. Instead use the factory
        functions :py:meth:`_from_commit` or :py:meth:`_from_staging` to
        return a pre-initialized class instance appropriately constructed for
        either a read-only or write-enabled checkout.

    Parameters
    ----------
    mode : str
        one of 'r' or 'a' to indicate read or write mode
    repo_pth : os.PathLike
        path to the repository on disk
    datasets : dict of obj
        dictionary of DatasetData objects
    hashenv : lmdb.Environment
        environment handle for hash records
    dataenv : lmdb.Environment
        environment handle for the unpacked records. `data` is means to refer to
        the fact that the stageenv is passed in for for write-enabled, and a
        cmtrefenv for read-only checkouts.
    stagehashenv : lmdb.Environment
        environment handle for newly added staged data hash records.
    '''

    def __init__(self,
                 mode: str,
                 repo_pth: os.PathLike,
                 datasets: Mapping[str, Union[DatasetDataReader, DatasetDataWriter]],
                 hashenv: Optional[lmdb.Environment] = None,
                 dataenv: Optional[lmdb.Environment] = None,
                 stagehashenv: Optional[lmdb.Environment] = None):

        self._mode = mode
        self._repo_pth = repo_pth
        self._datasets = datasets
        self._is_conman = False
        self._contains_partial_remote_data: bool = False

        if (mode == 'a'):
            self._hashenv = hashenv
            self._dataenv = dataenv
            self._stagehashenv = stagehashenv
            self._Query = RecordQuery(self._dataenv)

        self.__setup()

    def __setup(self):
        '''Do not allow users to use internal functions
        '''
        self._from_commit = None  # should never be able to access
        self._from_staging_area = None  # should never be able to access
        if self._mode == 'r':
            self.init_dataset = None
            self.remove_dset = None
            self.multi_add = None
            self.__delitem__ = None
            self.__setitem__ = None

    def _open(self):
        for v in self._datasets.values():
            v._open()

    def _close(self):
        for v in self._datasets.values():
            v._close()

# ------------- Methods Available To Both Read & Write Checkouts ------------------

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__}\
                \n    Writeable: {bool(0 if self._mode == "r" else 1)}\
                \n    Dataset Names / Partial Remote References:\
                \n      - '                            + '\n      - '.join(
            f'{dsetn} / {dset.contains_remote_references}'
            for dsetn, dset in self._datasets.items())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'datasets={self._datasets}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        '''Let ipython know that any key based access can use the dataset keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for the
        sanity of developers, this is the best way to ensure users can autocomplete
        keys.

        Returns
        -------
        list
            list of strings, each being one of the dataset keys for access.
        '''
        return self.keys()

    def __getitem__(self, key):
        '''Dict style access to return the dataset object with specified key/name.

        Parameters
        ----------
        key : string
            name of the dataset object to get.

        Returns
        -------
        :class:`DatasetDataReader` or :class:`DatasetDataWriter`
            The object which is returned depends on the mode of checkout specified.
            If the dataset was checked out with write-enabled, return writer object,
            otherwise return read only object.
        '''
        return self.get(key)

    def __setitem__(self, key, value):
        '''Specifically prevent use dict style setting for dataset objects.

        Datasets must be created using the factory function :py:meth:`init_dataset`.

        Raises
        ------
        PermissionError
            This operation is not allowed under any circumstance

        '''
        msg = f'HANGAR NOT ALLOWED:: To add a dataset use `init_dataset` method.'
        raise PermissionError(msg)

    def __contains__(self, key):
        return True if key in self._datasets else False

    def __len__(self):
        return len(self._datasets)

    def __iter__(self):
        return iter(self._datasets)

    @property
    def iswriteable(self):
        '''Bool indicating if this dataset object is write-enabled. Read-only attribute.
        '''
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> MutableMapping[str, bool]:
        '''Dict of bool indicating data reference locality in each dataset.

        False is all samples in dataset exist locally, True if some reference remote sources.
        '''
        res = {}
        for dsetn, dset in self._datasets.items():
            res[dsetn] = dset.contains_remote_references
        return res

    @property
    def remote_reference_dset_sample_keys(self) -> MutableMapping[str, List[str]]:
        '''Return list of sample keys containing data stored in a remote reference for each dataset

        Returns
        -------
        MutableMapping[str, List[str]]
            dict where keys are dataset names and values are samples in the dataset containing
            remote references
        '''
        res: MutableMapping[str, List[str]] = {}
        for dsetn, dset in self._datasets.items():
            res[dsetn] = dset.remote_reference_sample_keys
        return res

    def keys(self):
        '''list all dataset keys (names) in the checkout

        Returns
        -------
        list of str
            list of dataset names
        '''
        return self._datasets.keys()

    def values(self):
        '''list all dataset object instances in the checkout

        Yields
        -------
        object
            Generator of DatasetData accessor objects (set to read or write mode
            as appropriate)
        '''
        for dsetObj in self._datasets.values():
            wr = cm_weakref_obj_proxy(dsetObj)
            yield wr

    def items(self):
        '''generator providing access to dataset_name, :class:`Datasets`

        Yields
        ------
        tuple
            containing (name, :class:`Datasets`) for all datasets.

        '''
        for dsetN, dsetObj in self._datasets.items():
            wr = cm_weakref_obj_proxy(dsetObj)
            yield (dsetN, wr)

    def get(self, name):
        '''Returns a dataset access object.

        This can be used in lieu of the dictionary style access.

        Parameters
        ----------
        name : str
            name of the dataset to return

        Returns
        -------
        ObjectProxy
            DatasetData accessor (set to read or write mode as appropriate) proxy

        Raises
        ------
        KeyError
            If no dataset with the given name exists in the checkout
        '''
        try:
            wr = cm_weakref_obj_proxy(self._datasets[name])
            return wr
        except KeyError:
            e = KeyError(f'No dataset exists with name: {name}')
            logger.error(e, exc_info=False)
            raise e

# ------------------------ Writer-Enabled Methods Only ------------------------------

    def __delitem__(self, key):
        '''remove a dataset and all data records if write-enabled process.

        Parameters
        ----------
        key : string
            name of the dataset to remove from the repository. This will remove
            all records from the staging area (though the actual data and all
            records are still accessible) if they were previously committed

        Raises
        ------
        PermissionError
            If this is a read-only checkout, no operation is permitted.
        '''
        self.remove_dset(key)

    def __enter__(self):
        self._is_conman = True
        for dskey in list(self._datasets):
            self._datasets[dskey].__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        for dskey in list(self._datasets):
            self._datasets[dskey].__exit__(*exc)

    def multi_add(self, mapping: dict) -> str:
        '''Add related samples to un-named datasets with the same generated key.

        If you have multiple datasets in a checkout whose samples are related to
        each other in some manner, there are two ways of associating samples
        together:

        1) using named datasets and setting each tensor in each dataset to the
           same sample "name" using un-named datasets.
        2) using this "add" method. which accepts a dictionary of "dataset
           names" as keys, and "tensors" (ie. individual samples) as values.

        When method (2) - this method - is used, the internally generated sample
        ids will be set to the same value for the samples in each dataset. That
        way a user can iterate over the dataset key's in one sample, and use
        those same keys to get the other related tensor samples in another
        dataset.

        Parameters
        ----------
        mapping: dict
            Dict mapping (any number of) dataset names to tensor data (samples)
            which to add. The datasets must exist, and must be set to accept
            samples which are not named by the user

        Returns
        -------
        str
            generated id (key) which each sample is stored under in their
            corresponding dataset. This is the same for all samples specified in
            the input dictionary.


        Raises
        ------
        KeyError
            If no dataset with the given name exists in the checkout
        '''
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            if not all([k in self._datasets for k in mapping.keys()]):
                raise KeyError(
                    f'some key(s): {mapping.keys()} not in dset(s): {self._datasets.keys()}')
            data_name = parsing.generate_sample_name()
            for k, v in mapping.items():
                self._datasets[k].add(v, bulkn=data_name)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise e from None

        finally:
            if tmpconman:
                self.__exit__()

        return data_name

    def init_dataset(self, name: str, shape=None, dtype=None, prototype=None,
                     named_samples=True, variable_shape=False, *, backend: str = None):
        '''Initializes a dataset in the repository.

        Datasets are groups of related data pieces (samples). All samples within
        a dataset have the same data type, and number of dimensions. The size of
        each dimension can be either fixed (the default behavior) or variable
        per sample.

        For fixed dimension sizes, all samples written to the dataset must have
        the same size that was initially specified upon dataset initialization.
        Variable size datasets on the other hand, can write samples with
        dimensions of any size less than a maximum which is required to be set
        upon dataset creation.

        Parameters
        ----------
        name : str
            The name assigned to this dataset.
        shape : tuple, optional
            The shape of the data samples which will be written in this dataset.
            This argument and the `dtype` argument are required if a `prototype`
            is not provided, defaults to None.
        dtype : np.dtype, optional
            The datatype of this dataset. This argument and the `shape` argument
            are required if a `prototype` is not provided., defaults to None.
        prototype : np.array, optional
            A sample array of correct datatype and shape which will be used to
            initialize the dataset storage mechanisms. If this is provided, the
            `shape` and `dtype` arguments must not be set, defaults to None.
        named_samples : bool, optional
            If the samples in the dataset have names associated with them. If set,
            all samples must be provided names, if not, no name will be assigned.
            defaults to True, which means all samples should have names.
        variable_shape : bool, optional
            If this is a variable sized dataset. If true, a the maximum shape is
            set from the provided `shape` or `prototype` argument. Any sample
            added to the dataset can then have dimension sizes <= to this
            initial specification (so long as they have the same rank as what
            was specified) defaults to False.
        backend : DEVELOPER USE ONLY. str, optional, kwarg only
            Backend which should be used to write the dataset files on disk.

        Returns
        -------
        :class:`DatasetDataWriter`
            instance object of the initialized dataset.

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
            If a dataset already exists with the provided name.
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
                    f'Dataset name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace).')
            if name in self._datasets:
                raise LookupError(f'KEY EXISTS: dataset already exists with name: {name}.')

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

        dsetCountKey = parsing.dataset_record_count_db_key_from_raw_key(name)
        dsetCountVal = parsing.dataset_record_count_db_val_from_raw_val(0)
        dsetSchemaKey = parsing.dataset_record_schema_db_key_from_raw_key(name)
        dsetSchemaVal = parsing.dataset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=variable_shape,
            schema_max_shape=prototype.shape,
            schema_dtype=prototype.dtype.num,
            schema_is_named=named_samples,
            schema_default_backend=backend)

        # -------- set vals in lmdb only after schema is sure to exist --------

        dataTxn = TxnRegister().begin_writer_txn(self._dataenv)
        hashTxn = TxnRegister().begin_writer_txn(self._hashenv)
        numDsetsCountKey = parsing.dataset_total_count_db_key()
        numDsetsCountVal = dataTxn.get(numDsetsCountKey, default=('0'.encode()))
        numDsets_count = parsing.dataset_total_count_raw_val_from_db_val(numDsetsCountVal)
        numDsetsCountVal = parsing.dataset_record_count_db_val_from_raw_val(numDsets_count + 1)
        hashSchemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashSchemaVal = dsetSchemaVal

        dataTxn.put(dsetCountKey, dsetCountVal)
        dataTxn.put(numDsetsCountKey, numDsetsCountVal)
        dataTxn.put(dsetSchemaKey, dsetSchemaVal)
        hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)
        TxnRegister().commit_writer_txn(self._dataenv)
        TxnRegister().commit_writer_txn(self._hashenv)

        self._datasets[name] = DatasetDataWriter(
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

    def remove_dset(self, dset_name):
        '''remove the dataset and all data contained within it from the repository.

        Parameters
        ----------
        dset_name : str
            name of the dataset to remove

        Returns
        -------
        str
            name of the removed dataset

        Raises
        ------
        KeyError
            If a dataset does not exist with the provided name
        '''
        datatxn = TxnRegister().begin_writer_txn(self._dataenv)
        try:
            if dset_name not in self._datasets:
                e = KeyError(f'HANGAR KEY ERROR:: Cannot remove: {dset_name}. Key does not exist.')
                logger.error(e, exc_info=False)
                raise e
            self._datasets[dset_name]._close()
            self._datasets.__delitem__(dset_name)

            dsetCountKey = parsing.dataset_record_count_db_key_from_raw_key(dset_name)
            numDsetsKey = parsing.dataset_total_count_db_key()
            arraysInDset = datatxn.get(dsetCountKey)
            recordsToDelete = parsing.dataset_total_count_raw_val_from_db_val(arraysInDset)
            recordsToDelete = recordsToDelete + 1  # depends on num subkeys per array recy
            with datatxn.cursor() as cursor:
                cursor.set_key(dsetCountKey)
                for i in range(recordsToDelete):
                    cursor.delete()
            cursor.close()

            dsetSchemaKey = parsing.dataset_record_schema_db_key_from_raw_key(dset_name)
            datatxn.delete(dsetSchemaKey)
            numDsetsVal = datatxn.get(numDsetsKey)
            numDsets = parsing.dataset_total_count_raw_val_from_db_val(numDsetsVal) - 1
            if numDsets == 0:
                datatxn.delete(numDsetsKey)
            else:
                numDsetsVal = parsing.dataset_total_count_db_val_from_raw_val(numDsets)
                datatxn.put(numDsetsKey, numDsetsVal)
        finally:
            TxnRegister().commit_writer_txn(self._dataenv)

        return dset_name

# ------------------------ Class Factory Functions ------------------------------

    @classmethod
    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        '''Class method factory to checkout :class:`Datasets` in write-enabled mode

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
        :class:`Datasets`
            Interface class with write-enabled attributes activated and any
            datasets existing initialized in write mode via
            :class:`DatasetDataWriter`.
        '''

        datasets = {}
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()
        for dsetName, schemaSpec in stagedSchemaSpecs.items():
            datasets[dsetName] = DatasetDataWriter(
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

        return cls('a', repo_pth, datasets, hashenv, stageenv, stagehashenv)

    @classmethod
    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        '''Class method factory to checkout :class:`Datasets` in read-only mode

        This is not a user facing operation, and should never be manually called
        in normal operation. For read mode, no locks need to be verified, but
        construction should occur through the interface to the
        :class:`Datasets` class.

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
        :class:`Datasets`
            Interface class with all write-enabled attributes deactivated
            datasets initialized in read mode via :class:`DatasetDataReader`.
        '''
        datasets = {}
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()
        for dsetName, schemaSpec in cmtSchemaSpecs.items():
            datasets[dsetName] = DatasetDataReader(
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

        return cls('r', repo_pth, datasets, None, None, None)
