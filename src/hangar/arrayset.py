import os
import warnings
from multiprocessing import cpu_count, get_context
from typing import Iterator, Iterable, List, Mapping, Optional, Tuple, Union, NamedTuple

import lmdb
import numpy as np

from .backends import BACKEND_ACCESSOR_MAP
from .backends import backend_decoder
from .backends import is_local_backend
from .backends import parse_user_backend_opts
from .context import TxnRegister
from .utils import cm_weakref_obj_proxy, is_suitable_user_key, is_ascii
from .records.hashmachine import schema_hash_digest
from .records.hashmachine import array_hash_digest
from .records.queries import RecordQuery
from .records.parsing import hash_data_db_key_from_raw_key
from .records.parsing import generate_sample_name
from .records.parsing import hash_schema_db_key_from_raw_key
from .records.parsing import data_record_db_key_from_raw_key
from .records.parsing import data_record_raw_val_from_db_val
from .records.parsing import data_record_db_val_from_raw_val
from .records.parsing import arrayset_record_count_range_key
from .records.parsing import arrayset_record_schema_db_key_from_raw_key
from .records.parsing import arrayset_record_schema_db_val_from_raw_val


CompatibleArray = NamedTuple(
    'CompatibleArray', [('compatible', bool), ('reason', str)])


class ArraysetDataReader(object):
    """Class implementing get access to data in a arrayset.

    The methods implemented here are common to the :class:`.ArraysetDataWriter`
    accessor class as well as to this ``"read-only"`` method. Though minimal,
    the behavior of read and write checkouts is slightly unique, with the main
    difference being that ``"read-only"`` checkouts implement both thread and
    process safe access methods. This is not possible for ``"write-enabled"``
    checkouts, and attempts at multiprocess/threaded writes will generally
    fail with cryptic error messages.
    """

    def __init__(self,
                 repo_pth: os.PathLike,
                 aset_name: str,
                 default_schema_hash: str,
                 samplesAreNamed: bool,
                 isVar: bool,
                 varMaxShape: list,
                 varDtypeNum: int,
                 dataenv: lmdb.Environment,
                 hashenv: lmdb.Environment,
                 mode: str,
                 default_schema_backend: str,
                 default_backend_opts: str,
                 *args, **kwargs):
        """Developer documentation for init method

        The location of the data references can be transparently specified by
        feeding in a different dataenv argument. For staged reads -> ``dataenv =
        lmdb.Environment(STAGING_DB)``. For commit read -> ``dataenv =
        lmdb.Environment(COMMIT_DB)``.

        Parameters
        ----------
        repo_pth : os.PathLike
            path to the repository on disk.
        aset_name : str
            name of the arrayset
        schema_hashes : list of str
            list of all schemas referenced in the arrayset
        samplesAreNamed : bool
            do samples have names or not.
        isVar : bool
            is the arrayset schema variable shape or not
        varMaxShape : list or tuple of int
            schema size (max) of the arrayset data
        varDtypeNum : int
            datatype numeric code of the arrayset data
        dataenv : lmdb.Environment
            environment of the arrayset references to read
        hashenv : lmdb.Environment
            environment of the repository hash records
        mode : str, optional
            mode to open the file handles in. 'r' for read only, 'a' for read/write, defaults
            to 'r'
        """
        self._mode = mode
        self._path = repo_pth
        self._asetn = aset_name
        self._schema_variable = isVar
        self._schema_dtype_num = varDtypeNum
        self._samples_are_named = samplesAreNamed
        self._schema_max_shape = tuple(varMaxShape)
        self._dflt_schema_hash = default_schema_hash
        self._dflt_backend = default_schema_backend
        self._dflt_backend_opts = default_backend_opts

        self._is_conman: bool = False
        self._index_expr_factory = np.s_
        self._index_expr_factory.maketuple = False
        self._contains_partial_remote_data: bool = False

        # -------------- Sample backend specification parsing -----------------

        self._sspecs = {}
        _TxnRegister = TxnRegister()
        hashTxn = _TxnRegister.begin_reader_txn(hashenv)
        try:
            used_bes = set()
            asetNamesSpec = RecordQuery(dataenv).arrayset_data_records(self._asetn)
            for asetNames, dataSpec in asetNamesSpec:
                hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
                hash_ref = hashTxn.get(hashKey)
                be_loc = backend_decoder(hash_ref)
                self._sspecs[asetNames.data_name] = be_loc
                used_bes.add(be_loc.backend)

            if not all([is_local_backend(be) for be in used_bes]):
                self._contains_partial_remote_data = True
                warnings.warn(
                    f'Arrayset: {self._asetn} contains `reference-only` samples, with '
                    f'actual data residing on a remote server. A `fetch-data` '
                    f'operation is required to access these samples.', UserWarning)
        finally:
            _TxnRegister.abort_reader_txn(hashenv)

        # ------------------------ backend setup ------------------------------

        self._fs = {}
        for be, accessor in BACKEND_ACCESSOR_MAP.items():
            if (self._mode == 'a') or (be in used_bes):
                if accessor is None:
                    continue
                self._fs[be] = accessor(
                    repo_path=self._path,
                    schema_shape=self._schema_max_shape,
                    schema_dtype=np.typeDict[self._schema_dtype_num])
                self._fs[be].open(mode=self._mode)

    def __enter__(self):
        self._is_conman = True
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        return

    def __getitem__(self, key: Union[str, int]) -> np.ndarray:
        """Retrieve a sample with a given key, convenience method for dict style access.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : Union[str, int]
            sample key to retrieve from the arrayset

        Returns
        -------
        :class:`numpy.ndarray`
            sample array data corresponding to the provided key
        """
        return self.get(key)

    def __iter__(self) -> Iterator[Union[str, int]]:
        return self.keys()

    def __len__(self) -> int:
        """Check how many samples are present in a given arrayset

        Returns
        -------
        int
            number of samples the arrayset contains
        """
        return len(self._sspecs)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Determine if a key is a valid sample name in the arrayset

        Parameters
        ----------
        key : Union[str, int]
            name to check if it is a sample in the arrayset

        Returns
        -------
        bool
            True if key exists, else False
        """
        exists = key in self._sspecs
        return exists

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Arrayset Name             : {self._asetn}\
                \n    Schema Hash              : {self._dflt_schema_hash}\
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
              f'aset_name={self._asetn}, '\
              f'default_schema_hash={self._dflt_schema_hash}, '\
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
    def name(self) -> str:
        """Name of the arrayset. Read-Only attribute.
        """
        return self._asetn

    @property
    def dtype(self) -> np.dtype:
        """Datatype of the arrayset schema. Read-only attribute.
        """
        return np.typeDict[self._schema_dtype_num]

    @property
    def shape(self) -> Tuple[int]:
        """Shape (or `max_shape`) of the arrayset sample tensors. Read-only attribute.
        """
        return self._schema_max_shape

    @property
    def variable_shape(self) -> bool:
        """Bool indicating if arrayset schema is variable sized. Read-only attribute.
        """
        return self._schema_variable

    @property
    def named_samples(self) -> bool:
        """Bool indicating if samples are named. Read-only attribute.
        """
        return self._samples_are_named

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this arrayset object is write-enabled. Read-only attribute.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating if all samples exist locally or if some reference remote sources.
        """
        return bool(self._contains_partial_remote_data)

    @property
    def remote_reference_keys(self) -> List[Union[str, int]]:
        """Returns sample names whose data is stored in a remote server reference.

        Returns
        -------
        List[Union[str, int]]
            list of sample keys in the arrayset.
        """
        keys = []
        if self.contains_remote_references is True:
            keys = [name for name, be in self._sspecs.items() if not is_local_backend(be)]
        return keys

    @property
    def backend(self) -> str:
        """The default backend for the arrayset.

        Returns
        -------
        str
            numeric format code of the default backend.
        """
        return self._dflt_backend

    @property
    def backend_opts(self):
        """The opts applied to the default backend.

        Returns
        -------
        dict
            config settings used to set up filters
        """
        return self._dflt_backend_opts

    def keys(self, local: bool = False) -> Iterator[Union[str, int]]:
        """generator which yields the names of every sample in the arrayset

        Parameters
        ----------
        local : bool, optional
            if True, returned keys will only correspond to data which is
            available for reading on the local disk, by default False

        Yields
        ------
        Iterator[Union[str, int]]
            keys of one sample at a time inside the arrayset

        Notes
        -----
        For write enabled checkouts, is technically possible to iterate over the
        arrayset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.
        """
        if not local:
            for name in tuple(self._sspecs.keys()):
                yield name
        else:
            for name, be in tuple(self._sspecs.items()):
                if is_local_backend(be):
                    yield name

    def values(self, local=False) -> Iterator[np.ndarray]:
        """generator which yields the tensor data for every sample in the arrayset

        Parameters
        ----------
        local : bool, optional
            if True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False

        Yields
        ------
        Iterator[:class:`numpy.ndarray`]
            values of one sample at a time inside the arrayset

        Notes
        -----
        For write enabled checkouts, is technically possible to iterate over the
        arrayset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.
        """
        if not local:
            for name in tuple(self._sspecs.keys()):
                yield self.get(name)
        else:
            for name, be in tuple(self._sspecs.items()):
                if is_local_backend(be):
                    yield self.get(name)

    def items(self, local=False) -> Iterator[Tuple[Union[str, int], np.ndarray]]:
        """generator yielding two-tuple of (name, tensor), for every sample in the arrayset.

        Parameters
        ----------
        local : bool, optional
            if True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False

        Yields
        ------
        Iterator[Tuple[Union[str, int], :class:`numpy.ndarray`]]
            sample name and stored value for every sample inside the arrayset

        Notes
        -----
        For write enabled checkouts, is technically possible to iterate over the
        arrayset object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.
        """
        if not local:
            for name in tuple(self._sspecs.keys()):
                yield (name, self.get(name))
        else:
            for name, be in tuple(self._sspecs.items()):
                if is_local_backend(be):
                    yield (name, self.get(name))

    def get(self, name: Union[str, int]) -> np.ndarray:
        """Retrieve a sample in the arrayset with a specific name.

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
        name : Union[str, int]
            Name of the sample to retrieve data for.

        Returns
        -------
        :class:`numpy.ndarray`
            Tensor data stored in the arrayset archived with provided name(s).

        Raises
        ------
        KeyError
            if the arrayset does not contain data with the provided name
        """
        try:
            spec = self._sspecs[name]
            data = self._fs[spec.backend].read_data(spec)
            return data
        except KeyError:
            raise KeyError(f'HANGAR KEY ERROR:: data: {name} not in aset: {self._asetn}')

    def get_batch(self,
                  names: Iterable[Union[str, int]],
                  *,
                  n_cpus: int = None,
                  start_method: str = 'spawn') -> List[np.ndarray]:
        """Retrieve a batch of sample data with the provided names.

        This method is (technically) thread & process safe, though it should not
        be called in parallel via multithread/process application code; This
        method has been seen to drastically decrease retrieval time of sample
        batches (as compared to looping over single sample names sequentially).
        Internally it implements a multiprocess pool of workers (managed by
        hangar) to simplify application developer workflows.

        Parameters
        ----------
        name : Iterable[Union[str, int]]
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
        List[:class:`numpy.ndarray`]
            Tensor data stored in the arrayset archived with provided name(s).

            If a single sample name is passed in as the, the corresponding
            np.array data will be returned.

            If a list/tuple of sample names are pass in the ``names`` argument,
            a tuple of size ``len(names)`` will be returned where each element
            is an np.array containing data at the position it's name listed in
            the ``names`` parameter.

        Raises
        ------
        KeyError
            if the arrayset does not contain data with the provided name
        """
        n_jobs = n_cpus if isinstance(n_cpus, int) else int(cpu_count() / 2)
        with get_context(start_method).Pool(n_jobs) as p:
            data = p.map(self.get, names)
        return data


class ArraysetDataWriter(ArraysetDataReader):
    """Class implementing methods to write data to a arrayset.

    Writer specific methods are contained here, and while read functionality is
    shared with the methods common to :class:`.ArraysetDataReader`. Write-enabled
    checkouts are not thread/process safe for either ``writes`` OR ``reads``,
    a restriction we impose for ``write-enabled`` checkouts in order to ensure
    data integrity above all else.

    .. seealso:: :class:`.ArraysetDataReader`

    """

    def __init__(self,
                 stagehashenv: lmdb.Environment,
                 *args, **kwargs):
        """Developer documentation for init method.

        Extends the functionality of the ArraysetDataReader class. The __init__
        method requires quite a number of ``**kwargs`` to be passed along to the
        :class:`ArraysetDataReader` class.

        Parameters
        ----------
            stagehashenv : lmdb.Environment
                db where the newly added staged hash data records are stored
            default_schema_backend : str
                backend code to act as default where new data samples are added.
            **kwargs:
                See args of :class:`ArraysetDataReader`
        """
        super().__init__(*args, **kwargs)

        self._fs[self._dflt_backend].backend_opts = self._dflt_backend_opts

        self._stagehashenv = stagehashenv
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
        """Store a piece of data in a arrayset. Convenience method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : Union[str, int]
            name of the sample to add to the arrayset
        value : :class:`numpy.ndarray`
            tensor data to add as the sample

        Returns
        -------
        Union[str, int]
            sample name of the stored data (assuming operation was successful)
        """
        self.add(value, key)
        return key

    def __delitem__(self, key: Union[str, int]) -> Union[str, int]:
        """Remove a sample from the arrayset. Convenience method to :meth:`remove`.

        .. seealso:: :meth:`remove`

        Parameters
        ----------
        key : Union[str, int]
            Name of the sample to remove from the arrayset

        Returns
        -------
        Union[str, int]
            Name of the sample removed from the arrayset (assuming operation successful)
        """
        return self.remove(key)

    def _verify_array_compatible(self, data: np.ndarray) -> CompatibleArray:
        """Determine if an array is compatible with the arraysets schema

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            array to check compatibility for

        Returns
        -------
        CompatibleArray
            compatible and reason field
        """
        reason = ''
        if not isinstance(data, np.ndarray):
            reason = f'`data` argument type: {type(data)} != `np.ndarray`'
        elif data.dtype.num != self._schema_dtype_num:
            reason = f'dtype: {data.dtype} != aset: {np.typeDict[self._schema_dtype_num]}.'
        elif not data.flags.c_contiguous:
            reason = f'`data` must be "C" contiguous array.'

        if reason == '':
            if self._schema_variable is True:
                if data.ndim != len(self._schema_max_shape):
                    reason = f'`data` rank: {data.ndim} != aset rank: {len(self._schema_max_shape)}'
                for dDimSize, schDimSize in zip(data.shape, self._schema_max_shape):
                    if dDimSize > schDimSize:
                        reason = f'dimensions of `data`: {data.shape} exceed variable max '\
                                 f'dims of aset: {self._asetn} specified max dimensions: '\
                                 f'{self._schema_max_shape}: SIZE: {dDimSize} > {schDimSize}'
            elif data.shape != self._schema_max_shape:
                reason = f'data shape: {data.shape} != fixed aset shape: {self._schema_max_shape}'

        compatible = True if reason == '' else False
        res = CompatibleArray(compatible=compatible, reason=reason)
        return res

    def change_backend(self, backend_opts: Union[str, dict]):
        """Change the default backend and filters applied to future data writes.

        .. warning::

           This method is meant for advanced users only. Please refer to the
           hangar backend codebase for information on accepted parameters and
           options.

        Parameters
        ----------
        backend_opts : Optional[Union[str, dict]]
            If str, backend format code to specify, opts are automatically
            inffered. If dict, key ``backend`` must have a valid backend format code
            value, and the rest of the items are assumed to be valid specs for that
            particular backend. If none, both backend and opts are inffered from
            the array prototype

        Raises
        ------
        RuntimeError
            If this method was called while this arrayset is invoked in a
            context manager
        ValueError
            If the backend format code is not valid.
        """
        if self._is_conman:
            raise RuntimeError('Cannot call method inside arrayset context manager.')

        proto = np.zeros(self.shape, dtype=self.dtype)
        beopts = parse_user_backend_opts(backend_opts=backend_opts,
                                         prototype=proto,
                                         named_samples=self.named_samples,
                                         variable_shape=self.variable_shape)

        # ----------- Determine schema format details -------------------------

        schema_hash = schema_hash_digest(shape=proto.shape,
                                         size=proto.size,
                                         dtype_num=proto.dtype.num,
                                         named_samples=self.named_samples,
                                         variable_shape=self.variable_shape,
                                         backend_code=beopts.backend,
                                         backend_opts=beopts.opts)
        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(self.name)
        asetSchemaVal = arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=self.variable_shape,
            schema_max_shape=proto.shape,
            schema_dtype=proto.dtype.num,
            schema_is_named=self.named_samples,
            schema_default_backend=beopts.backend,
            schema_default_backend_opts=beopts.opts)

        hashSchemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        dataTxn = TxnRegister().begin_writer_txn(self._dataenv)
        hashTxn = TxnRegister().begin_writer_txn(self._hashenv)
        try:
            dataTxn.put(asetSchemaKey, asetSchemaVal)
            hashTxn.put(hashSchemaKey, asetSchemaVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self._dataenv)
            TxnRegister().commit_writer_txn(self._hashenv)

        if beopts.backend not in self._fs:
            self._fs[beopts.backend] = BACKEND_ACCESSOR_MAP[beopts.backend](
                repo_path=self._path,
                schema_shape=self._schema_max_shape,
                schema_dtype=np.typeDict[self._schema_dtype_num])
        else:
            self._fs[beopts.backend].close()
        self._fs[beopts.backend].open(mode=self._mode)
        self._fs[beopts.backend].backend_opts = beopts.opts
        self._dflt_backend = beopts.backend
        self._dflt_backend_opts = beopts.opts
        self._dflt_schema_hash = schema_hash
        return

    def add(self, data: np.ndarray, name: Union[str, int] = None,
            **kwargs) -> Union[str, int]:
        """Store a piece of data in a arrayset

        Parameters
        ----------
        data : :class:`numpy.ndarray`
            data to store as a sample in the arrayset.
        name : Union[str, int], optional
            name to assign to the same (assuming the arrayset accepts named
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
            If no `name` arg was provided for arrayset requiring named samples.
        ValueError
            If input data tensor rank exceeds specified rank of arrayset samples.
        ValueError
            For variable shape arraysets, if a dimension size of the input data
            tensor exceeds specified max dimension size of the arrayset samples.
        ValueError
            For fixed shape arraysets, if input data dimensions do not exactly match
            specified arrayset dimensions.
        ValueError
            If type of `data` argument is not an instance of np.ndarray.
        ValueError
            If `data` is not "C" contiguous array layout.
        ValueError
            If the datatype of the input data does not match the specified data type of
            the arrayset
        """

        # ------------------------ argument type checking ---------------------

        try:
            if self._samples_are_named and not is_suitable_user_key(name):
                raise ValueError(
                    f'Name provided: `{name}` type: {type(name)} is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace) or int >= 0. '
                    f'Must be <= 64 characters long.')
            elif not self._samples_are_named:
                name = kwargs['bulkn'] if 'bulkn' in kwargs else generate_sample_name()

            isCompat = self._verify_array_compatible(data)
            if not isCompat.compatible:
                raise ValueError(isCompat.reason)

        except ValueError as e:
            raise e from None

        # --------------------- add data to storage backend -------------------

        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            full_hash = array_hash_digest(data)
            hashKey = hash_data_db_key_from_raw_key(full_hash)
            # check if data record already exists with given key
            dataRecKey = data_record_db_key_from_raw_key(self._asetn, name)
            existingDataRecVal = self._dataTxn.get(dataRecKey, default=False)
            if existingDataRecVal:
                # check if data record already with same key & hash value
                existingDataRec = data_record_raw_val_from_db_val(existingDataRecVal)
                if full_hash == existingDataRec.data_hash:
                    return name

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
            dataRecVal = data_record_db_val_from_raw_val(full_hash)
            self._dataTxn.put(dataRecKey, dataRecVal)

        finally:
            if tmpconman:
                self.__exit__()

        return name

    def remove(self, name: Union[str, int]) -> Union[str, int]:
        """Remove a sample with the provided name from the arrayset.

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
            If a sample with the provided name does not exist in the arrayset.
        """
        tmpconman = not self._is_conman
        if tmpconman:
            self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)

        dataKey = data_record_db_key_from_raw_key(self._asetn, name)
        try:
            isRecordDeleted = self._dataTxn.delete(dataKey)
            if isRecordDeleted is False:
                raise KeyError(f'No sample {name} in {self._asetn}')
            del self._sspecs[name]
        except KeyError as e:
            raise e
        finally:
            if tmpconman:
                self._TxnRegister.commit_writer_txn(self._dataenv)
        return name


"""
Constructor and Interaction Class for Arraysets
--------------------------------------------------
"""


class Arraysets(object):
    """Common access patterns and initialization/removal of arraysets in a checkout.

    This object is the entry point to all tensor data stored in their individual
    arraysets. Each arrayset contains a common schema which dictates the general
    shape, dtype, and access patters which the backends optimize access for. The
    methods contained within allow us to create, remove, query, and access these
    collections of common tensors.
    """

    def __init__(self,
                 mode: str,
                 repo_pth: os.PathLike,
                 arraysets: Mapping[str, Union[ArraysetDataReader, ArraysetDataWriter]],
                 hashenv: Optional[lmdb.Environment] = None,
                 dataenv: Optional[lmdb.Environment] = None,
                 stagehashenv: Optional[lmdb.Environment] = None):
        """Developer documentation for init method.

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
        arraysets : Mapping[str, Union[ArraysetDataReader, ArraysetDataWriter]]
            dictionary of ArraysetData objects
        hashenv : Optional[lmdb.Environment]
            environment handle for hash records
        dataenv : Optional[lmdb.Environment]
            environment handle for the unpacked records. `data` is means to refer to
            the fact that the stageenv is passed in for for write-enabled, and a
            cmtrefenv for read-only checkouts.
        stagehashenv : Optional[lmdb.Environment]
            environment handle for newly added staged data hash records.
        """
        self._mode = mode
        self._repo_pth = repo_pth
        self._arraysets = arraysets
        self._is_conman = False
        self._contains_partial_remote_data: bool = False

        if (mode == 'a'):
            self._hashenv = hashenv
            self._dataenv = dataenv
            self._stagehashenv = stagehashenv

        self.__setup()

    def __setup(self):
        """Do not allow users to use internal functions
        """
        self._from_commit = None  # should never be able to access
        self._from_staging_area = None  # should never be able to access
        if self._mode == 'r':
            self.init_arrayset = None
            self.remove_aset = None
            self.multi_add = None
            self.__delitem__ = None
            self.__setitem__ = None

    def _open(self):
        for v in self._arraysets.values():
            v._open()

    def _close(self):
        for v in self._arraysets.values():
            v._close()

# ------------- Methods Available To Both Read & Write Checkouts ------------------

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__}\
                \n    Writeable: {bool(0 if self._mode == "r" else 1)}\
                \n    Arrayset Names / Partial Remote References:\
                \n      - ' + '\n      - '.join(
            f'{asetn} / {aset.contains_remote_references}'
            for asetn, aset in self._arraysets.items())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'arraysets={self._arraysets}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        """Let ipython know that any key based access can use the arrayset keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the arrayset keys for access.
        """
        return self.keys()

    def __getitem__(self, key: str) -> Union[ArraysetDataReader, ArraysetDataWriter]:
        """Dict style access to return the arrayset object with specified key/name.

        Parameters
        ----------
        key : string
            name of the arrayset object to get.

        Returns
        -------
        Union[:class:`.ArraysetDataReader`, :class:`.ArraysetDataWriter`]
            The object which is returned depends on the mode of checkout
            specified. If the arrayset was checked out with write-enabled,
            return writer object, otherwise return read only object.
        """
        return self.get(key)

    def __setitem__(self, key, value):
        """Specifically prevent use dict style setting for arrayset objects.

        Arraysets must be created using the method :meth:`init_arrayset`.

        Raises
        ------
        PermissionError
            This operation is not allowed under any circumstance

        """
        msg = f'Not allowed! To add a arrayset use `init_arrayset` method.'
        raise PermissionError(msg)

    def __contains__(self, key: str) -> bool:
        """Determine if a arrayset with a particular name is stored in the checkout

        Parameters
        ----------
        key : str
            name of the arrayset to check for

        Returns
        -------
        bool
            True if a arrayset with the provided name exists in the checkout,
            otherwise False.
        """
        return True if key in self._arraysets else False

    def __len__(self) -> int:
        return len(self._arraysets)

    def __iter__(self) -> Iterable[str]:
        return iter(self._arraysets)

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this arrayset object is write-enabled. Read-only attribute.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> Mapping[str, bool]:
        """Dict of bool indicating data reference locality in each arrayset.

        Returns
        -------
        Mapping[str, bool]
            For each arrayset name key, boolean value where False indicates all
            samples in arrayset exist locally, True if some reference remote
            sources.
        """
        res: Mapping[str, bool] = {}
        for asetn, aset in self._arraysets.items():
            res[asetn] = aset.contains_remote_references
        return res

    @property
    def remote_sample_keys(self) -> Mapping[str, Iterable[Union[int, str]]]:
        """Determine arraysets samples names which reference remote sources.

        Returns
        -------
        Mapping[str, Iterable[Union[int, str]]]
            dict where keys are arrayset names and values are iterables of
            samples in the arrayset containing remote references
        """
        res: Mapping[str, Iterable[Union[int, str]]] = {}
        for asetn, aset in self._arraysets.items():
            res[asetn] = aset.remote_reference_keys
        return res

    def keys(self) -> List[str]:
        """list all arrayset keys (names) in the checkout

        Returns
        -------
        List[str]
            list of arrayset names
        """
        return list(self._arraysets.keys())

    def values(self) -> Iterable[Union[ArraysetDataReader, ArraysetDataWriter]]:
        """yield all arrayset object instances in the checkout.

        Yields
        -------
        Iterable[Union[:class:`.ArraysetDataReader`, :class:`.ArraysetDataWriter`]]
            Generator of ArraysetData accessor objects (set to read or write mode
            as appropriate)
        """
        for asetN in list(self._arraysets.keys()):
            asetObj = self._arraysets[asetN]
            wr = cm_weakref_obj_proxy(asetObj)
            yield wr

    def items(self) -> Iterable[Tuple[str, Union[ArraysetDataReader, ArraysetDataWriter]]]:
        """generator providing access to arrayset_name, :class:`Arraysets`

        Yields
        ------
        Iterable[Tuple[str, Union[:class:`.ArraysetDataReader`, :class:`.ArraysetDataWriter`]]]
            returns two tuple of all all arrayset names/object pairs in the checkout.
        """
        for asetN in list(self._arraysets.keys()):
            asetObj = self._arraysets[asetN]
            wr = cm_weakref_obj_proxy(asetObj)
            yield (asetN, wr)

    def get(self, name: str) -> Union[ArraysetDataReader, ArraysetDataWriter]:
        """Returns a arrayset access object.

        This can be used in lieu of the dictionary style access.

        Parameters
        ----------
        name : str
            name of the arrayset to return

        Returns
        -------
        Union[:class:`.ArraysetDataReader`, :class:`.ArraysetDataWriter`]
            ArraysetData accessor (set to read or write mode as appropriate) which
            governs interaction with the data

        Raises
        ------
        KeyError
            If no arrayset with the given name exists in the checkout
        """
        try:
            wr = cm_weakref_obj_proxy(self._arraysets[name])
            return wr
        except KeyError:
            e = KeyError(f'No arrayset exists with name: {name}')
            raise e from None

# ------------------------ Writer-Enabled Methods Only ------------------------------

    def _any_is_conman(self) -> bool:
        """Determine if self or any contains arrayset class is conman.

        Returns
        -------
        bool
            [description]
        """
        res = any([self._is_conman, *[x._is_conman for x in self._arraysets.values()]])
        return res

    def __delitem__(self, key: str) -> str:
        """remove a arrayset and all data records if write-enabled process.

        Parameters
        ----------
        key : str
            Name of the arrayset to remove from the repository. This will remove
            all records from the staging area (though the actual data and all
            records are still accessible) if they were previously committed

        Returns
        -------
        str
            If successful, the name of the removed arrayset.

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opned in a connection manager.
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any arraysets class is opened in a context manager')
        return self.remove_aset(key)

    def __enter__(self):
        self._is_conman = True
        for asetN in list(self._arraysets.keys()):
            self._arraysets[asetN].__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        for asetN in list(self._arraysets.keys()):
            self._arraysets[asetN].__exit__(*exc)

    def multi_add(self, mapping: Mapping[str, np.ndarray]) -> str:
        """Add related samples to un-named arraysets with the same generated key.

        If you have multiple arraysets in a checkout whose samples are related to
        each other in some manner, there are two ways of associating samples
        together:

        1) using named arraysets and setting each tensor in each arrayset to the
           same sample "name" using un-named arraysets.
        2) using this "add" method. which accepts a dictionary of "arrayset
           names" as keys, and "tensors" (ie. individual samples) as values.

        When method (2) - this method - is used, the internally generated sample
        ids will be set to the same value for the samples in each arrayset. That
        way a user can iterate over the arrayset key's in one sample, and use
        those same keys to get the other related tensor samples in another
        arrayset.

        Parameters
        ----------
        mapping: Mapping[str, :class:`numpy.ndarray`]
            Dict mapping (any number of) arrayset names to tensor data (samples)
            which to add. The arraysets must exist, and must be set to accept
            samples which are not named by the user

        Returns
        -------
        str
            generated id (key) which each sample is stored under in their
            corresponding arrayset. This is the same for all samples specified in
            the input dictionary.


        Raises
        ------
        KeyError
            If no arrayset with the given name exists in the checkout
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            if not all([k in self._arraysets for k in mapping.keys()]):
                raise KeyError(
                    f'some key(s): {mapping.keys()} not in aset(s): {self._arraysets.keys()}')
            data_name = generate_sample_name()
            for k, v in mapping.items():
                self._arraysets[k].add(v, bulkn=data_name)
        except KeyError as e:
            raise e from None
        finally:
            if tmpconman:
                self.__exit__()

        return data_name

    def init_arrayset(self,
                      name: str,
                      shape: Union[int, Tuple[int]] = None,
                      dtype: np.dtype = None,
                      prototype: np.ndarray = None,
                      named_samples: bool = True,
                      variable_shape: bool = False,
                      *,
                      backend_opts: Optional[Union[str, dict]] = None) -> ArraysetDataWriter:
        """Initializes a arrayset in the repository.

        Arraysets are groups of related data pieces (samples). All samples within
        a arrayset have the same data type, and number of dimensions. The size of
        each dimension can be either fixed (the default behavior) or variable
        per sample.

        For fixed dimension sizes, all samples written to the arrayset must have
        the same size that was initially specified upon arrayset initialization.
        Variable size arraysets on the other hand, can write samples with
        dimensions of any size less than a maximum which is required to be set
        upon arrayset creation.

        Parameters
        ----------
        name : str
            The name assigned to this arrayset.
        shape : Union[int, Tuple[int]]
            The shape of the data samples which will be written in this arrayset.
            This argument and the `dtype` argument are required if a `prototype`
            is not provided, defaults to None.
        dtype : :class:`numpy.dtype`
            The datatype of this arrayset. This argument and the `shape` argument
            are required if a `prototype` is not provided., defaults to None.
        prototype : :class:`numpy.ndarray`
            A sample array of correct datatype and shape which will be used to
            initialize the arrayset storage mechanisms. If this is provided, the
            `shape` and `dtype` arguments must not be set, defaults to None.
        named_samples : bool, optional
            If the samples in the arrayset have names associated with them. If set,
            all samples must be provided names, if not, no name will be assigned.
            defaults to True, which means all samples should have names.
        variable_shape : bool, optional
            If this is a variable sized arrayset. If true, a the maximum shape is
            set from the provided ``shape`` or ``prototype`` argument. Any sample
            added to the arrayset can then have dimension sizes <= to this
            initial specification (so long as they have the same rank as what
            was specified) defaults to False.
        backend_opts : Optional[Union[str, dict]], optional
            ADVANCED USERS ONLY, backend format code and filter opts to apply
            to arrayset data. If None, automatically infered and set based on
            data shape and type. by default None

        Returns
        -------
        :class:`.ArraysetDataWriter`
            instance object of the initialized arrayset.

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opened in a connection manager.
        ValueError
            If provided name contains any non ascii letter characters
            characters, or if the string is longer than 64 characters long.
        ValueError
            If required `shape` and `dtype` arguments are not provided in absence of
            `prototype` argument.
        ValueError
            If `prototype` argument is not a C contiguous ndarray.
        LookupError
            If a arrayset already exists with the provided name.
        ValueError
            If rank of maximum tensor shape > 31.
        ValueError
            If zero sized dimension in `shape` argument
        ValueError
            If the specified backend is not valid.
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any arraysets class is opened in a context manager')

        # ------------- Checks for argument validity --------------------------

        try:
            if (not is_suitable_user_key(name)) or (not is_ascii(name)):
                raise ValueError(
                    f'Arrayset name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace). '
                    f'Must be <= 64 characters long')
            if name in self._arraysets:
                raise LookupError(f'Arrayset already exists with name: {name}.')

            if prototype is not None:
                if not isinstance(prototype, np.ndarray):
                    raise ValueError(
                        f'If not `None`, `prototype` argument be `np.ndarray`-like.'
                        f'Invalid value: {prototype} of type: {type(prototype)}')
                elif not prototype.flags.c_contiguous:
                    raise ValueError(f'`prototype` must be "C" contiguous array.')
            elif isinstance(shape, (tuple, list, int)) and (dtype is not None):
                prototype = np.zeros(shape, dtype=dtype)
            else:
                raise ValueError(f'`shape` & `dtype` required if no `prototype` set.')

            if (0 in prototype.shape) or (prototype.ndim > 31):
                raise ValueError(
                    f'Invalid shape specification with ndim: {prototype.ndim} and '
                    f'shape: {prototype.shape}. Array rank > 31 dimensions not '
                    f'allowed AND all dimension sizes must be > 0.')

            beopts = parse_user_backend_opts(backend_opts=backend_opts,
                                             prototype=prototype,
                                             named_samples=named_samples,
                                             variable_shape=variable_shape)
        except (ValueError, LookupError) as e:
            raise e from None

        # ----------- Determine schema format details -------------------------

        schema_hash = schema_hash_digest(shape=prototype.shape,
                                         size=prototype.size,
                                         dtype_num=prototype.dtype.num,
                                         named_samples=named_samples,
                                         variable_shape=variable_shape,
                                         backend_code=beopts.backend,
                                         backend_opts=beopts.opts)

        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(name)
        asetSchemaVal = arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=variable_shape,
            schema_max_shape=prototype.shape,
            schema_dtype=prototype.dtype.num,
            schema_is_named=named_samples,
            schema_default_backend=beopts.backend,
            schema_default_backend_opts=beopts.opts)

        # -------- set vals in lmdb only after schema is sure to exist --------

        dataTxn = TxnRegister().begin_writer_txn(self._dataenv)
        hashTxn = TxnRegister().begin_writer_txn(self._hashenv)
        hashSchemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        hashSchemaVal = asetSchemaVal
        dataTxn.put(asetSchemaKey, asetSchemaVal)
        hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)
        TxnRegister().commit_writer_txn(self._dataenv)
        TxnRegister().commit_writer_txn(self._hashenv)

        self._arraysets[name] = ArraysetDataWriter(
            stagehashenv=self._stagehashenv,
            repo_pth=self._repo_pth,
            aset_name=name,
            default_schema_hash=schema_hash,
            samplesAreNamed=named_samples,
            isVar=variable_shape,
            varMaxShape=prototype.shape,
            varDtypeNum=prototype.dtype.num,
            hashenv=self._hashenv,
            dataenv=self._dataenv,
            mode='a',
            default_schema_backend=beopts.backend,
            default_backend_opts=beopts.opts)

        return self.get(name)

    def remove_aset(self, aset_name: str) -> str:
        """remove the arrayset and all data contained within it.

        Parameters
        ----------
        aset_name : str
            name of the arrayset to remove

        Returns
        -------
        str
            name of the removed arrayset

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opned in a connection manager.
        KeyError
            If a arrayset does not exist with the provided name
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any arraysets class is opened in a context manager')

        datatxn = TxnRegister().begin_writer_txn(self._dataenv)
        try:
            if aset_name not in self._arraysets:
                e = KeyError(f'Cannot remove: {aset_name}. Key does not exist.')
                raise e from None

            self._arraysets[aset_name]._close()
            self._arraysets.__delitem__(aset_name)

            with datatxn.cursor() as cursor:
                cursor.first()
                asetRangeKey = arrayset_record_count_range_key(aset_name)
                recordsExist = cursor.set_range(asetRangeKey)
                while recordsExist:
                    k = cursor.key()
                    if k.startswith(asetRangeKey):
                        recordsExist = cursor.delete()
                    else:
                        recordsExist = False

            asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(aset_name)
            datatxn.delete(asetSchemaKey)
        finally:
            TxnRegister().commit_writer_txn(self._dataenv)

        return aset_name

# ------------------------ Class Factory Functions ------------------------------

    @classmethod
    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        """Class method factory to checkout :class:`Arraysets` in write-enabled mode

        This is not a user facing operation, and should never be manually
        called in normal operation. Once you get here, we currently assume that
        verification of the write lock has passed, and that write operations
        are safe.

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
        :class:`.Arraysets`
            Interface class with write-enabled attributes activated and any
            arraysets existing initialized in write mode via
            :class:`.arrayset.ArraysetDataWriter`.
        """

        arraysets = {}
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()
        for asetName, schemaSpec in stagedSchemaSpecs.items():
            arraysets[asetName] = ArraysetDataWriter(
                stagehashenv=stagehashenv,
                repo_pth=repo_pth,
                aset_name=asetName,
                default_schema_hash=schemaSpec.schema_hash,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                hashenv=hashenv,
                dataenv=stageenv,
                mode='a',
                default_schema_backend=schemaSpec.schema_default_backend,
                default_backend_opts=schemaSpec.schema_default_backend_opts)

        return cls('a', repo_pth, arraysets, hashenv, stageenv, stagehashenv)

    @classmethod
    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        """Class method factory to checkout :class:`.arrayset.Arraysets` in read-only mode

        This is not a user facing operation, and should never be manually called
        in normal operation. For read mode, no locks need to be verified, but
        construction should occur through the interface to the
        :class:`Arraysets` class.

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
        :class:`.Arraysets`
            Interface class with all write-enabled attributes deactivated
            arraysets initialized in read mode via :class:`.arrayset.ArraysetDataReader`.
        """
        arraysets = {}
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()
        for asetName, schemaSpec in cmtSchemaSpecs.items():
            arraysets[asetName] = ArraysetDataReader(
                repo_pth=repo_pth,
                aset_name=asetName,
                default_schema_hash=schemaSpec.schema_hash,
                samplesAreNamed=schemaSpec.schema_is_named,
                isVar=schemaSpec.schema_is_var,
                varMaxShape=schemaSpec.schema_max_shape,
                varDtypeNum=schemaSpec.schema_dtype,
                dataenv=cmtrefenv,
                hashenv=hashenv,
                mode='r',
                default_schema_backend=schemaSpec.schema_default_backend,
                default_backend_opts=schemaSpec.schema_default_backend_opts)

        return cls('r', repo_pth, arraysets, None, None, None)
