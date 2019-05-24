import hashlib
import os
from typing import Optional
from uuid import uuid1
import logging

import numpy as np
import lmdb

from . import config
from .context import TxnRegister
from .hdf5_store import FileHandles
from .records import parsing
from .records.queries import RecordQuery
from .utils import is_ascii_alnum, cm_weakref_obj_proxy

logger = logging.getLogger(__name__)


class DatasetDataReader(object):
    '''Class implementing get access to data in a dataset

    The location of the data references can be transparently specified by feeding in a
    different dataenv argument. For staged reads -> ``dataenv = lmdb.Environment(STAGING_DB)``.
    For commit read -> ``dataenv = lmdb.Environment(COMMIT_DB)``.

    Parameters
    ----------
    repo_pth : str
        path to the repository on disk.
    dset_name : str
        name of the dataset
    schema_hashes : list of str
        list of all schemas referenced in the dataset
    schema_uuid : str
        uuid of the dataset instance
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
                 schema_uuid: str,
                 samplesAreNamed: bool,
                 isVar: bool,
                 varMaxShape: list,
                 varDtypeNum: int,
                 dataenv: lmdb.Environment,
                 hashenv: lmdb.Environment,
                 mode: str,
                 **kwargs):

        self._path = repo_pth
        self._dsetn = dset_name
        self._dataenv = dataenv
        self._hashenv = hashenv
        self._default_schema_hash = default_schema_hash
        self._schema_uuid = schema_uuid
        self._samples_are_named = samplesAreNamed
        self._schema_variable = isVar
        self._schema_max_shape = tuple(varMaxShape)
        self._index_expr_factory = np.s_
        self._index_expr_factory.maketuple = False
        self._schema_dtype_num = varDtypeNum
        self._mode = mode
        self._fs = FileHandles(repo_path=repo_pth)
        self._Query = RecordQuery(self._dataenv)
        self._TxnRegister = TxnRegister()

        self._fs.open(self._path, self._mode)

        self._hashTxn: Optional[lmdb.Transaction] = None
        self._dataTxn: Optional[lmdb.Transaction] = None
        self._is_conman = False

    def __enter__(self):
        self._is_conman = True
        self._hashTxn = self._TxnRegister.begin_reader_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.begin_reader_txn(self._dataenv)
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._hashTxn = self._TxnRegister.abort_reader_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.abort_reader_txn(self._dataenv)

    def __getitem__(self, key):
        '''Retrieve a sample with a given key. Convenience method for dict style access.

        .. seealso:: :meth:`get()`

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
        return len(self._Query.dataset_data_names(self._dsetn))

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
        if not self._is_conman:
            self._dataTxn = self._TxnRegister.begin_reader_txn(self._dataenv)

        try:
            ref_key = parsing.data_record_db_key_from_raw_key(self._dsetn, key)
            data_ref = self._dataTxn.get(ref_key, default=False)
            keyExists = bool(data_ref)
        finally:
            if not self._is_conman:
                self._dataTxn = self._TxnRegister.abort_reader_txn(self._dataenv)

        return keyExists

    def _repr_pretty_(self, p, cycle):
        res = f'\n Hangar {self.__class__.__name__} \
                \n    Dataset Name     : {self._dsetn}\
                \n    Schema UUID      : {self._schema_uuid}\
                \n    Schema Hash      : {self._default_schema_hash}\
                \n    Variable Shape   : {bool(int(self._schema_variable))}\
                \n    (max) Shape      : {self._schema_max_shape}\
                \n    Datatype         : {np.typeDict[self._schema_dtype_num]}\
                \n    Named Samples    : {bool(self._samples_are_named)}\
                \n    Access Mode      : {self._mode}\
                \n    Num Samples      : {self.__len__()}\n'
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._path}, '\
              f'dset_name={self._dsetn}, '\
              f'default_schema_hash={self._default_schema_hash}, '\
              f'schema_uuid={self._schema_uuid}, '\
              f'isVar={self._schema_variable}, '\
              f'varMaxShape={self._schema_max_shape}, '\
              f'varDtypeNum={self._schema_dtype_num}, '\
              f'dataenv={self._dataenv}, '\
              f'hashenv={self._hashenv}, '\
              f'mode={self._mode})'
        return res

    def _close(self):
        self._fs.close(mode=self._mode)

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

    def keys(self):
        '''generator which yields the names of every sample in the dataset
        '''
        data_names = self._Query.dataset_data_names(self._dsetn)
        for name in data_names:
            yield name

    def values(self):
        '''generator which yields the tensor data for every sample in the dataset
        '''
        data_names = self._Query.dataset_data_names(self._dsetn)
        for name in data_names:
            yield self.get(name)

    def items(self):
        '''generator yielding two-tuple of (name, tensor), for every sample in the dataset.
        '''
        data_names = self._Query.dataset_data_names(self._dsetn)
        for name in data_names:
            yield (name, self.get(name))

    def get(self, name):
        '''Retrieve a dataset data sample with the provided sample name

        Parameters
        ----------
        name : str
            name of the sample to retrieve

        Returns
        -------
        np.array
            tensor data stored in the dataset archived with provided name

        Raises
        ------
        KeyError
            if the dataset does not contain data with the provided name
        '''
        try:
            tmpconman = False if self._is_conman else True
            if tmpconman:
                self.__enter__()

            ref_key = parsing.data_record_db_key_from_raw_key(self._dsetn, name)
            data_ref = self._dataTxn.get(ref_key, default=False)
            if data_ref is False:
                raise KeyError(f'HANGAR KEY ERROR:: data: {name} not in dset: {self._dsetn}')

            dataSpec = parsing.data_record_raw_val_from_db_val(data_ref)
            hashKey = parsing.hash_data_db_key_from_raw_key(dataSpec.data_hash)
            hash_ref = self._hashTxn.get(hashKey)
            hashVal = parsing.hash_data_raw_val_from_db_val(hash_ref)
            # This is a tight enough loop where keyword args can impact performance
            data = self._fs.read_data(hashVal, self._mode, self._schema_dtype_num)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if tmpconman:
                self.__exit__()

        return data


class DatasetDataWriter(DatasetDataReader):
    '''Class implementing methods to write data to a dataset.

    Extends the functionality of the DatasetDataReader class. The __init__ method requires
    quite a number of ``**kwargs`` to be passed along to the :class:`DatasetDataReader`
    class.

    .. seealso:: :class:`DatasetDataReader`

    Parameters
    ----------
        **kwargs:
            See args of :class:`DatasetDataReader`
    '''

    def __init__(self, stagehashenv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stagehashenv = stagehashenv
        self._setup_file_access()

    def _setup_file_access(self):
        '''Internal method used to open handles to the backing store.
        '''
        if self._default_schema_hash not in self._fs.wHands:
            sample_array = np.zeros(self._schema_max_shape, dtype=np.typeDict[self._schema_dtype_num])
            self._fs.create_schema(self._path, self._default_schema_hash, sample_array)
        self._fs.open(self._path, self._mode)

    def __enter__(self):
        self._is_conman = True
        self._hashTxn = self._TxnRegister.begin_writer_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)
        self._stageHashTxn = self._TxnRegister.begin_writer_txn(self._stagehashenv)
        self._fs.__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._hashTxn = self._TxnRegister.commit_writer_txn(self._hashenv)
        self._dataTxn = self._TxnRegister.commit_writer_txn(self._dataenv)
        self._stageHashTxn = self._TxnRegister.commit_writer_txn(self._stagehashenv)
        self._fs.__exit__(*exc)

    def __setitem__(self, key, value):
        '''Store a piece of data in a dataset. Convenince method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : str
            name of the sample to add to the dataset
        value : np.array
            tensor data to add as the sample

        Returns
        -------
        str
            sample name of the stored data (assuming operation was successful)
        '''
        self.add(value, key)
        return key

    def __delitem__(self, key):
        '''Remove a sample from the dataset. Convenence method to :meth:`remove`.

        .. seealso:: :meth:`remove`

        Parameters
        ----------
        key : str
            Name of the sample to remove from the dataset

        Returns
        -------
        str
            Name of the sample removed from the dataset (assuming operation successful)
        '''
        return self.remove(key)

    def add(self, data, name=None, **kwargs):
        '''Store a piece of data in a dataset

        Parameters
        ----------
        data : np.array
            data to store as a sample in the dataset
        name : str, optional
            name to assign to the same (assuming the dataset accepts named
            samples), by default None

        Returns
        -------
        str
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
            For fixed shape datasets, if input data dimensions do not exactally match
            specified dataset dimensions.
        ValueError
            If type of `data` argument is not an instance of np.ndarray.
        ValueError
            If the datatype of the input data does not match the specifed data type of
            the dataset
        LookupError
            If a data sample with the same name and hash value already exists in the
            dataset.
        '''

        # ------------------------ argument type checking ---------------------

        try:
            if self._samples_are_named:
                if not (isinstance(name, str) and is_ascii_alnum(name)):
                    msg = f'HANGAR VALUE ERROR:: name: {name} invalid. Must be of type str & '\
                          f'only contain alpha-numeric ascii characters (no whitespace).'
                    raise ValueError(msg)
            else:
                name = kwargs['bulkn'] if 'bulkn' in kwargs else parsing.generate_sample_name()

            if not isinstance(data, np.ndarray):
                msg = f'HANGAR VALUE ERROR:: `data` argument type: {type(data)} != `np.ndarray`'
                raise ValueError(msg)
            elif data.dtype.num != self._schema_dtype_num:
                msg = f'HANGAR VALUE ERROR:: data type of input data: {data.dtype} != type of '\
                      f'specified type: {np.typeDict[self._schema_dtype_num]}.'
                raise ValueError(msg)

            if self._schema_variable is True:
                if data.ndim != len(self._schema_max_shape):
                    msg = f'HANGAR VALUE ERROR:: rank of input tensor: {data.ndim} exceeds '\
                          f'rank specified max rank: {len(self._schema_max_shape)}'
                    raise ValueError(msg)
                for dataDimSize, schemaDimSize in zip(data.shape, self._schema_max_shape):
                    if dataDimSize > schemaDimSize:
                        msg = f'HANGAR VALUE ERROR:: dimensions of input data: {data.shape} '\
                              f'exceed variable max dims of dset: {self._dsetn} specified '\
                              f'max dimensions: {self._schema_max_shape}. DIM SIZE: '\
                              f'{dataDimSize} > {schemaDimSize}'
                        raise ValueError(msg)
            elif data.shape != self._schema_max_shape:
                msg = f'HANGAR VALUE ERROR:: shape of input data: {data.shape} != fixed '\
                      f'dims of dset: {self._dsetn} specified dimss: {self._schema_max_shape}'
                raise ValueError(msg)

        except ValueError as e:
            logger.error(e, exc_info=False)
            raise e from None

        # --------------------- add data to storage backend -------------------

        try:
            tmpconman = False if self._is_conman else True
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
                    msg = f'HANGAR KEY EXISTS ERROR:: dataset: {self._dsetn} already contains '\
                          f'identical object named: {name} with same hash value: {full_hash}'
                    raise LookupError(msg)

            # write new data if data hash does not exist
            existingHashVal = self._hashTxn.get(hashKey, default=False)
            if existingHashVal is False:
                hdf_instance, hdf_dset, hdf_idx = self._fs.add_tensor_data(
                    data, self._default_schema_hash)
                hashVal = parsing.hash_data_db_val_from_raw_val(
                    self._default_schema_hash, hdf_instance, hdf_dset, hdf_idx, data.shape)
                self._hashTxn.put(hashKey, hashVal)
                self._stageHashTxn.put(hashKey, hashVal)

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

    def remove(self, name):
        '''Remove a sample with the provided name from the dataset.

        .. Note::

            This operation will NEVER actually remove any data from disk. If
            you commit a tensor at any point in time, **it will always remain
            accessable by checking out a previous commit** when the tensor was
            present. This is just a way to tell Hangar that you don't want some
            piece of data to clutter up the current version of the repository.

        .. Warning::

            Though this may change in a future release, in the current version of
            Hangar, we cannot recover references to data which was added to the
            staging area, written to disk, but then removed **before** a commit
            operation was run. This would be a similar sequence of events as:
            checking out a `git` branch, changing a bunch of text in the file, and
            immediatly performing a hard reset. If it was never committed, git
            doesn't know about it, and (at the moment) neither does Hangar.

        Parameters
        ----------
        name : str
            name of the sample to remove.

        Returns
        -------
        str
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
                msg = f'HANGAR KEY ERROR:: no sample: {name} exists in dset: {self._dsetn}'
                raise KeyError(msg)

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
    '''Common access patterns and initilization/removal of datasets in a checkout.

    .. warning::

        This class should not be instantiated directly. Instead use the factory
        functions :py:meth:`_from_commit` or :py:meth:`_from_staging` to
        return a pre-initialized class instance appropriatly constructed for
        either a read-only or write-enabled checkout.

    Parameters
    ----------
    repo_pth : str
        path to the repository on disk
    datasets : dict of obj
        dictionary of DatasetData objects
    hashenv : lmdb.Environment
        environment handle for hash records
    dataenv : lmdb.Environment
        environment handle for the unpacked records (stage for
        write-enabled, cmtrefenv for read-only)
    mode : str
        one of 'r' or 'a' to indicate read or write mode
    '''

    def __init__(self, repo_pth, datasets, hashenv, dataenv, stagehashenv, mode):
        self._mode = mode
        self._hashenv = hashenv
        self._dataenv = dataenv
        self._stagehashenv = stagehashenv
        self._repo_pth = repo_pth
        self._datasets = datasets
        self._is_conman = False
        self._Query = RecordQuery(self._dataenv)

        self.__setup()

    def __setup(self):
        '''Do not allow users to use internal functions
        '''
        self._from_commit = None
        self._from_staging_area = None
        if self._mode == 'r':
            self.init_dataset = None
            self.remove_dset = None

# ------------- Methods Available To Both Read & Write Checkouts ------------------

    def _repr_pretty_(self, p, cycle):
        res = f'\n Hangar {self.__class__.__name__}\
                \n     Writeable: {bool(0 if self._mode == "r" else 1)}\
                \n     Dataset Names:\
                \n       - ' + '\n       - '.join(self._datasets.keys())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'datasets={self._datasets}, '\
              f'hashenv={self._hashenv}, '\
              f'dataenv={self._dataenv}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        '''Let ipython know that any key based access can use the dataset keys

        Since we don't want to inheret from dict, nor mess with `__dir__` for the
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
        msg = f'HANGAR NOT ALLOWED ERROR:: To add a dataset use `init_dataset` method.'
        e = PermissionError(msg)
        logger.error(e, exc_info=False)
        raise e

    def __contains__(self, key):
        return True if key in self._datasets else False

    def __delitem__(self, key):
        '''remove a dataset and all data records if write-enabled process.

        Parameters
        ----------
        key : string
            name of the dataset to remove from the repository. This will
            remove all records from the staging area (though the actual data
            and all records are still accessable) if they were previously commited

        Raises
        ------
        PermissionError
            If this is a read-only checkout, no operation is permitted.
        '''
        if self._mode == 'r':
            msg = 'HANGAR NOT ALLOWED ERROR:: Cannot remove a dataset in read-only checkout'
            e = PermissionError(msg)
            logger.error(e, exc_info=False)
            raise e
        else:
            self.remove_dset(key)

    def __len__(self):
        return len(self._datasets)

    def __iter__(self):
        return iter(self._datasets)

    @property
    def iswriteable(self):
        '''Bool indicating if this dataset object is write-enabled. Read-only attribute.
        '''
        return False if self._mode == 'r' else True

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
            msg = f'HANGAR KEY ERROR:: No dataset exists with name: {name}'
            e = KeyError(msg)
            logger.error(e, exc_info=False)
            raise e

# ------------------------ Writer-Enabled Methods Only ------------------------------

    def __enter__(self):
        self._is_conman = True
        for dskey in list(self._datasets):
            self._datasets[dskey].__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        for dskey in list(self._datasets):
            self._datasets[dskey].__exit__(*exc)

    def add(self, mapping: dict) -> str:
        '''Add related samples to un-named datasets with the same generated key.

        If you have multiple datasets in a checkout whose samples are related to
        eachother in some manner, there are two ways of associating samples
        together:

        1) using named datasets and setting each tensor in each dataset to the
           same sample "name" using un-named datasets.
        2) using this "add" method. which accepts a dictionary of "dataset
           names" as keys, and "tensors" (ie. individual samples) as values.

        When method (2) - this method - is used, the internally generated sample
        ids will be set to the same value for the samples in each dataset. That
        way a user can iterate over the dataset key's in one sample, and use
        those same keys to get the other releated tensor samples in another
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
            tmpconman = False if self._is_conman else True
            if tmpconman:
                self.__enter__()

            assert all([k in self._datasets for k in mapping.keys()])
            data_name = parsing.generate_sample_name()
            for k, v in mapping.items():
                self._datasets[k].add(v, bulkn=data_name)

        except AssertionError:
            msg = f'HANGAR KEY ERROR:: one of keys: {mapping.keys()} not in '\
                  f'datasets: {self._datasets.keys()}'
            logger.error(msg)
            raise KeyError(msg) from None
        finally:
            if tmpconman:
                self.__exit__()

        return data_name

    def init_dataset(self, name: str, shape=None, dtype=None, prototype=None, *,
                     samples_are_named=True, variable_shape=False, max_shape=None):
        '''Initializes a dataset in the repository.

        Datasets are groups of related data pieces (samples). All samples within
        a dataset have the same data type, and number of dimensions. The size of
        each dimension can be either fixed (the default behavior) or variable
        per sample.

        For fixed dimension sizes, all samples written to the dataset must have
        the same size that was initially specified upon dataset initialization.
        Variable size datasets on the other hand, can write samples with
        dimensions of any size less than a maximum which is required to be set
        upon datset creation.

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
        samples_are_named : bool, optional
            If the samples in the dataset have names associated with them. If set,
            all samples must be provided names, if not, no name will be assigned.
            defaults to True, which means all samples should have names.
        variable_shape : bool, optional
            If this is a variable sized dataset. If true, a `max_shape` argument
            must be specified, defaults to False.
        max_shape : tuple of int, optional
            The maximum size for each dimension which a data sample can be set
            with. The number of dimensions must match that specified in the
            `shape` or `prototype` argument, and each dimension size must be >=
            the equivalent dimension size specified. defaults to None.

        Returns
        -------
        :class:`DatasetDataWriter`
            instance object of the initialized dataset.

        Raises
        ------
        ValueError
            If provided name contains any non ascii, non alpha-numeric characters.
        SyntaxError
            If required `shape` and `dtype` arguments are not provided in absense of
            `prototype` argument.
        LookupError
            If a dataset already exists with the provided name.
        ValueError
            If provided prototype shape (or `shape` argument) not <= `max_shape`
            value if `variable_shape=True`.
        ValueError
            If rank of maximum tensor shape > 31.
        ValueError
            If zero sized dimension in `shape` argument
        ValueError
            If zero sized dimension in `max_shape` argument
        '''

        # ------------- Checks for argument validity --------------------------

        if not is_ascii_alnum(name):
            msg = f'Dataset name provided: `{name}` is invalid. Must only contain '\
                  f'alpha-numeric ascii with no whitespace characters.'
            e = ValueError(msg)
            logger.error(e, exc_info=False)
            raise e

        if name in self._datasets:
            e = LookupError(f'KEY EXISTS ERROR: dataset already exists with name: {name}.')
            logger.error(e)
            raise e

        if prototype is not None:
            style = 'prototype'
            tenShape = prototype.shape
        elif (shape is not None) and (dtype is not None):
            tenShape = tuple(shape) if isinstance(shape, list) else shape
            prototype = np.zeros(tenShape, dtype=dtype)
            style = 'provided'
        else:
            e = SyntaxError(f'Both `shape` & `dtype` required if `prototype` not specified.')
            logger.error(e, exc_info=False)
            raise e

        if variable_shape is True:
            maxShape = tuple(max_shape) if isinstance(max_shape, list) else max_shape
            if not np.all(np.less_equal(prototype.shape, maxShape)):
                msg = f'Variable shape `max_shape` value: {maxShape} not <= specified '\
                      f'prototype shape: {tenShape}.'
                e = ValueError(msg)
                logger.error(e)
                raise e
            prototype = np.zeros(maxShape, dtype=prototype.dtype)
            style = f'{style} + variable_shape'
        else:
            maxShape = tenShape

        if 0 in tenShape:
            raise ValueError(f'Invalid `shape`: {tenShape}. Dimension sizes must be > 0')
        elif 0 in maxShape:
            raise ValueError(f'Invalid `max_shape`: {maxShape}. Dimension sizes must be > 0')

        if len(maxShape) > 31:
            e = ValueError(f'Maximum tensor rank must be <= 31. specified: {len(maxShape)}')
            logger.error(e, exc_info=False)
            raise e

        msg = f'Dataset Specification:: '\
              f'Name: `{name}`, '\
              f'Initialization style: `{style}`, '\
              f'Shape: `{prototype.shape}`, '\
              f'DType: `{prototype.dtype}`, '\
              f'Samples Named: `{samples_are_named}`, '\
              f'Variable Shape: `{variable_shape}`, '\
              f'Max Shape: `{maxShape}`'
        logger.info(msg)

        # ----------- Determine schema format details -------------------------

        dset_uuid = uuid1().hex
        schema_format = np.array(
            (*prototype.shape, prototype.size, prototype.dtype.num), dtype=np.uint64)
        schema_hash = hashlib.blake2b(schema_format.tobytes(), digest_size=6).hexdigest()

        dsetCountKey = parsing.dataset_record_count_db_key_from_raw_key(name)
        dsetCountVal = parsing.dataset_record_count_db_val_from_raw_val(0)
        dsetSchemaKey = parsing.dataset_record_schema_db_key_from_raw_key(name)
        dsetSchemaVal = parsing.dataset_record_schema_db_val_from_raw_val(
            schema_uuid=dset_uuid,
            schema_hash=schema_hash,
            schema_is_var=variable_shape,
            schema_max_shape=prototype.shape,
            schema_dtype=prototype.dtype.num,
            schema_is_named=samples_are_named)

        # -------- Create staged schema storage backend -----------------------
        #
        # If this is a dataset with a different name from an existing one, but
        # which has the same schema hash, don't create a second one. Without
        # this method an operation like:
        #
        #     testProto = trainProto = np.array(shape=Foo, dtype=bar)
        #     init_dataset(name='test', prototype=testProto)
        #     init_dataset(name='train', prototype=trainProto)
        #
        # would create two backends and the first would not be used since the
        # last backend takes priority.
        #

        STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
        stage_dir = os.path.join(self._repo_pth, STAGE_DATA_DIR, f'hdf_{schema_hash}')
        if not os.path.isdir(stage_dir):
            f_handle = FileHandles(repo_path=self._repo_pth).create_schema(
                self._repo_pth, schema_hash, prototype)
            f_handle.close()

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
            schema_uuid=dset_uuid,
            samplesAreNamed=samples_are_named,
            isVar=variable_shape,
            varMaxShape=prototype.shape,
            varDtypeNum=prototype.dtype.num,
            hashenv=self._hashenv,
            dataenv=self._dataenv,
            mode='a')

        logger.info(f'Dataset Initialized: `{name}`')
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
                msg = f'HANGAR KEY ERROR:: Cannot remove: {dset_name}. No dset exists with that name.'
                raise KeyError(msg)
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
        except KeyError as e:
            logger.error(e)
            raise
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
            environment where the staged hash records are sored in write mode

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
                schema_uuid=schemaSpec.schema_uuid,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                hashenv=hashenv,
                dataenv=stageenv,
                mode='a')

        return cls(repo_pth, datasets, hashenv, stageenv, stagehashenv, 'a')

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
                schema_uuid=schemaSpec.schema_uuid,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                dataenv=cmtrefenv,
                hashenv=hashenv,
                mode='r')

        return cls(repo_pth, datasets, hashenv, cmtrefenv, None, 'r')
