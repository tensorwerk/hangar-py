from contextlib import contextmanager
import os
from typing import Tuple, List, Union, NamedTuple, Sequence, Dict, Iterable

import numpy as np

from ..utils import is_suitable_user_key
from ..backends import (
    backend_decoder,
    is_local_backend,
    parse_user_backend_opts,
    BACKEND_ACCESSOR_MAP,
    AccessorMapType,
    DataHashSpecsType,
)
from ..records.hashmachine import array_hash_digest, schema_hash_digest
from ..records.parsing import (
    arrayset_record_schema_db_key_from_raw_key,
    arrayset_record_schema_db_val_from_raw_val,
    arrayset_record_schema_raw_val_from_db_val,
    data_record_db_key_from_raw_key,
    data_record_db_val_from_raw_val,
    data_record_raw_val_from_db_val,
    hash_data_db_key_from_raw_key,
    hash_schema_db_key_from_raw_key,
    RawArraysetSchemaVal,
    generate_sample_name,
)


KeyType = Union[str, int]
KeyArrMap = Dict[KeyType, np.ndarray]
KeyArrType = Union[Tuple[KeyType, np.ndarray], List[Union[KeyType, np.ndarray]]]
GetKeysType = Union[KeyType, Sequence[KeyType]]


class CompatibleArray(NamedTuple):
    compatible: bool
    reason: str


class SampleReaderModifier(object):
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
                 aset_name: str,
                 samples: Dict[KeyType, DataHashSpecsType],
                 backend_handles: AccessorMapType,
                 schema_spec: RawArraysetSchemaVal,
                 repo_path: os.PathLike,
                 *args, **kwargs):

        self._mode = 'r'
        self._asetn = aset_name
        self._samples = samples
        self._be_fs = backend_handles
        self._path = repo_path

        self._schema_spec = schema_spec
        self._schema_variable = schema_spec.schema_is_var
        self._schema_dtype_num = schema_spec.schema_dtype
        self._samples_are_named = schema_spec.schema_is_named
        self._schema_max_shape = tuple(schema_spec.schema_max_shape)
        self._dflt_schema_hash = schema_spec.schema_hash
        self._dflt_backend = schema_spec.schema_default_backend
        self._dflt_backend_opts = schema_spec.schema_default_backend_opts
        self._contains_subsamples = schema_spec.schema_contains_subsamples

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

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar Sample Arrayset \
                \n    Arrayset Name             : {self._asetn}\
                \n    Schema Hash              : {self._dflt_schema_hash}\
                \n    Variable Shape           : {bool(int(self._schema_variable))}\
                \n    (max) Shape              : {self._schema_max_shape}\
                \n    Datatype                 : {np.typeDict[self._schema_dtype_num]}\
                \n    Named Samples            : {bool(self._samples_are_named)}\
                \n    Access Mode              : {self._mode}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self.contains_remote_references)}\
                \n    Contains Subsamples      : False\n'
        p.text(res)

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
        return list(self.keys())

    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        if self._mode == 'a':
            raise TypeError(f'cannot pickle write enabled checkout samples')
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    def __getitem__(self, key: KeyType) -> np.ndarray:
        """Retrieve data for some sample key via dict style access conventions.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : KeyTyle
            Sample key to retrieve from the arrayset.

        Returns
        -------
        :class:`numpy.ndarray`
            Sample array data corresponding to the provided key.
        """
        return self.get(key)

    def __iter__(self) -> Iterable[KeyType]:
        """Create iterator yielding an arrayset sample key for every call to ``next``.

        Yields
        -------
        Iterable[KeyType]
            Sample key contained in the arrayset.
        """
        yield from self.keys()

    def __len__(self) -> int:
        """Check how many samples are present in a given arrayset.

        Returns
        -------
        int
            number of samples the arrayset contains.
        """
        return len(self._samples)

    def __contains__(self, key: KeyType) -> bool:
        """Determine if a key is a valid sample name in the arrayset.

        Parameters
        ----------
        key : KeyType
            name to check if it is a sample in the arrayset

        Returns
        -------
        bool
            True if key exists, else False
        """
        return key in self._samples

    def _open(self):
        for val in self._be_fs.values():
            val.open(mode=self._mode)

    def _close(self):
        for val in self._be_fs.values():
            val.close()

    @property
    def arrayset(self) -> str:
        """Name of the arrayset.
        """
        return self._asetn

    @property
    def dtype(self) -> np.dtype:
        """Datatype of the arrayset schema.
        """
        return np.typeDict[self._schema_dtype_num]

    @property
    def shape(self) -> Tuple[int]:
        """Shape (or `max_shape`) of the arrayset sample tensors.
        """
        return self._schema_max_shape

    @property
    def variable_shape(self) -> bool:
        """Bool indicating if arrayset schema is variable sized.
        """
        return self._schema_variable

    @property
    def named_samples(self) -> bool:
        """Bool indicating if samples are named.
        """
        return self._samples_are_named

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this arrayset object is write-enabled.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating all samples in arrayset exist on local disk.

        The data associated with samples referencing some remore server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if atleast one sample in the arrayset references data stored
            on some remote server. True if all sample data is available on the
            machine's local disk.
        """
        for sample_spec in self._samples.values():
            if not is_local_backend(sample_spec):
                return True
        return False

    @property
    def remote_reference_keys(self) -> List[KeyType]:
        """Compute sample names whose data is stored in a remote server reference.

        Returns
        -------
        List[KeyType]
            list of sample keys in the arrayset whose data references indicate
            they are stored on a remote server.
        """
        keys = []
        for sample_name, sample_spec in self._samples.items():
            if not is_local_backend(sample_spec):
                keys.append(sample_name)
        return keys

    @property
    def backend(self) -> str:
        """Default backend new data is written to in this arrayset.

        Returns
        -------
        str
            numeric format code of the default backend.
        """
        return self._dflt_backend

    @property
    def backend_opts(self):
        """Storage backend options used when writing new data in this arrayset.

        Returns
        -------
        dict
            Config settings used to set up filters / other behaviors in the
            default storage backend.
        """
        return self._dflt_backend_opts

    @property
    def contains_subsamples(self) -> bool:
        """Bool indicating if sub-samples are contained in this arrayset container.

        Returns
        -------
        bool
            True if subsamples are included, False otherwise. For this arrayset
            class, no subsamples are stored; the result will always be False.
        """
        return False

    def keys(self, local: bool = False) -> Iterable[KeyType]:
        """Generator yielding the name (key) of every sample in the arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys will only correspond to data which is
            available for reading on the local disk, by default False.

        Yields
        ------
        Iterable[KeyType]
            Keys of one sample at a time inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield sample_name
        else:
            for sample_name, sample_spec in tuple(self._samples.items()):
                if is_local_backend(sample_spec):
                    yield sample_name

    def values(self, local: bool = False) -> Iterable[np.ndarray]:
        """Generator yielding the tensor data for every sample in arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[:class:`numpy.ndarray`]
            Values of one sample at a time inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield self.get(sample_name)
        else:
            for sample_name, sample_spec in tuple(self._samples.items()):
                if is_local_backend(sample_spec):
                    yield self.get(sample_name)

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, np.ndarray]]:
        """Generator yielding (name, tensor) tuple for every sample in arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, np.ndarray]]
            Sample name and stored value for every sample inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield (sample_name, self.get(sample_name))
        else:
            for sample_name, sample_spec in tuple(self._samples.items()):
                if is_local_backend(sample_spec):
                    yield (sample_name, self.get(sample_name))

    def get(self, keys: GetKeysType) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve tensor data for some sample key(s) in the arrayset.

        Parameters
        ----------
        keys : GetKeysType
            Single key, or sequence (list or tuple) of sample keys to retrieve
            data for.

        Returns
        -------
        Union[:class:`np.ndarray`, KeyArrMap]:
            Tensor data stored in the arrayset archived with provided name(s).

            If single item is passed to the ``key`` parameter, then an instance
            of a single :class:`np.ndarray` will be returned.

            If a sequence (list or tuple) of keys are provided, then a dictionary
            will be returned containing a mapping of each element in ``keys`` to
            it's corresponding :class:`np.ndarray` instance.

        Raises
        ------
        KeyError
            if the arrayset does not contain data with the provided name
        """
        try:
            if isinstance(keys, (str, int)):
                sample = keys
                spec = self._samples[keys]
                return self._be_fs[spec.backend].read_data(spec)
            elif isinstance(keys, (list, tuple)):
                res = {}
                for sample in keys:
                    spec = self._samples[sample]
                    res[sample] = self._be_fs[spec.backend].read_data(spec)
                return res
            else:
                raise ValueError(f'sample keys argument: {keys} not valid format')
        except KeyError:
            raise KeyError(f'No sample key {sample} exists in arrayset.')


class SampleWriterModifier(SampleReaderModifier):

    def __init__(self, aset_txn_ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'a'
        self._txnctx = aset_txn_ctx

    def __enter__(self):
        self._txnctx.open_write()
        for k in self._be_fs.keys():
            self._be_fs[k].__enter__()
        return self

    def __exit__(self, *exc):
        self._txnctx.close_write()
        for k in self._be_fs.keys():
            self._be_fs[k].__exit__(*exc)

    def __setitem__(self, key: KeyType, value: np.ndarray) -> KeyType:
        """Store a piece of data in a arrayset. Convenience method to :meth:`add`.

        .. seealso::

            :meth:`add` for the actual method called.

            :meth:`update` for an implementation analogous to python's built in
            :meth:`dict.update` method which accepts a dict or iterable of key/value
            pairs to add in the same operation.

        Parameters
        ----------
        key : KeyType
            Key (name) of the sample to add to the arrayset.
        value : :class:`numpy.ndarray`
            Tensor data to add as the sample.

        Returns
        -------
        KeyType
            Sample key (name) of the stored data (if operation was successful)
        """
        return self.add(key, value)

    def __delitem__(self, key: KeyType) -> KeyType:
        """Remove a sample from the arrayset. Convenience method to :meth:`delete`.

        .. seealso::

            :meth:`delete` (the analogous named operation for this method)

            :meth:`pop` to return a records value and then delete it in the same
            operation

        Parameters
        ----------
        key : KeyType
            Name of the sample to remove from the arrayset.

        Returns
        -------
        KeyType
            Name of the sample removed from the arrayset (assuming operation successful)
        """
        return self.delete(key)

    @property
    def _is_conman(self):
        return self._txnctx.is_conman

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
        SCHEMA_DTYPE = self._schema_spec.schema_dtype
        MAX_SHAPE = self._schema_spec.schema_max_shape

        reason = ''
        if not isinstance(data, np.ndarray):
            reason = f'`data` argument type: {type(data)} != `np.ndarray`'
        elif data.dtype.num != SCHEMA_DTYPE:
            reason = f'dtype: {data.dtype} != aset: {np.typeDict[SCHEMA_DTYPE]}.'
        elif not data.flags.c_contiguous:
            reason = f'`data` must be "C" contiguous array.'

        if reason == '':
            if self._schema_spec.schema_is_var is True:
                if data.ndim != len(MAX_SHAPE):
                    reason = f'data rank {data.ndim} != aset rank {len(MAX_SHAPE)}'
                for dDimSize, schDimSize in zip(data.shape, MAX_SHAPE):
                    if dDimSize > schDimSize:
                        reason = f'shape {data.shape} exceeds schema max {MAX_SHAPE}'
            elif data.shape != MAX_SHAPE:
                reason = f'data shape {data.shape} != fixed schema {MAX_SHAPE}'

        compatible = True if reason == '' else False
        res = CompatibleArray(compatible=compatible, reason=reason)
        return res

    @contextmanager
    def _txn_backend_context(self):
        tmpconman = False
        try:
            if self._txnctx.is_conman:
                yield self._txnctx
            else:
                tmpconman = True
                self._txnctx = self._txnctx.open_write()
                for k in self._be_fs.keys():
                    self._be_fs[k].__enter__()
                yield self._txnctx
        finally:
            if tmpconman:
                for k in self._be_fs.keys():
                    self._be_fs[k].__exit__()
                self._txnctx.close_write()

    def add(self, key: KeyType, value: np.ndarray, **kwargs) -> KeyType:
        """Store a piece of data in a arrayset with a given name and value

        .. seealso::

            :meth:`update` which implements functionality similar to python's
            builtin :meth:`dict.update` method, accepting either a dictionary
            or other iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        key : KeyType, optional
            name to assign to the same (assuming the arrayset accepts named
            samples), If str, can only contain alpha-numeric ascii characters
            (in addition to '-', '.', '_'). Integer key must be >= 0. by default
            None
        value : :class:`numpy.ndarray`
            data to store as a sample in the arrayset.

        Returns
        -------
        KeyType
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
            if self._schema_spec.schema_is_named and not is_suitable_user_key(key):
                raise ValueError(f'Sample name `{key}` is not suitable.')
            elif not self._schema_spec.schema_is_named:
                key = kwargs['bulkn'] if 'bulkn' in kwargs else generate_sample_name()

            isCompat = self._verify_array_compatible(value)
            if not isCompat.compatible:
                raise ValueError(isCompat.reason)
        except ValueError as e:
            raise e from None

        # --------------------- add data to storage backend -------------------

        with self._txn_backend_context() as ctx:

            full_hash = array_hash_digest(value)
            hashKey = hash_data_db_key_from_raw_key(full_hash)
            # check if data record already exists with given key
            dataRecKey = data_record_db_key_from_raw_key(self._asetn, key)
            existingDataRecVal = ctx.dataTxn.get(dataRecKey, default=False)
            if existingDataRecVal:
                # check if data record already with same key & hash value
                existingDataRec = data_record_raw_val_from_db_val(existingDataRecVal)
                if full_hash == existingDataRec.data_hash:
                    return key

            # write new data if data hash does not exist
            existingHashVal = ctx.hashTxn.get(hashKey, default=False)
            if existingHashVal is False:
                hashVal = self._be_fs[self._dflt_backend].write_data(value)
                ctx.hashTxn.put(hashKey, hashVal)
                ctx.stageHashTxn.put(hashKey, hashVal)
                hash_spec = backend_decoder(hashVal)
            else:
                hash_spec = backend_decoder(existingHashVal)

            # add the record to the db
            dataRecVal = data_record_db_val_from_raw_val(full_hash)
            ctx.dataTxn.put(dataRecKey, dataRecVal)
            self._samples[key] = hash_spec

        return key

    def update(self, other: Union[KeyArrMap, KeyArrType, Sequence[KeyArrType]]) -> List[KeyType]:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[KeyArrMap, KeyArrType, Sequence[KeyArrType]]
            Dictionary mapping sample names to :class:`np.ndarray` instances,
            length two sequence (list or tuple) of sample name and
            :class:`np.ndarray` instance, or sequence of multiple length two
            sequences (list or tuple) of sample names / :class:`np.ndarray`
            instances to store. If sample name is string type, can only contain
            alpha-numeric ascii characters (in addition to '-', '.', '_').
            Integer key must be >= 0.

        Returns
        -------
        List[KeyType]
            sample name(s) of the stored data (assuming the operation was successful)

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
        res = []
        with self._txn_backend_context():
            if isinstance(other, dict):
                for key, value in other.items():
                    res.append(self.add(key, value))
            elif isinstance(other, (list, tuple)):
                if len(other) == 2:
                    if is_suitable_user_key(other[0]) and isinstance(other[1], np.ndarray):
                        other = [other]  # contain in other list for iteration.
                for kvpair in other:
                    if not isinstance(kvpair, (list, tuple)) or (len(kvpair) != 2):
                        raise ValueError(f'length of other element {kvpair} != 2. No-Op.')
                for key, value in other:
                    res.append(self.add(key, value))
        return res

    def delete(self, keys: Union[KeyType, Sequence[KeyType]]) -> Union[KeyType, Sequence[KeyType]]:
        """Remove some key / value pair(s) from the arrayset, returning the deleted keys.

        Parameters
        ----------
        keys : Union[KeyType, Sequence[KeyType]]
            Either a single sample key to remove, or a sequence (list or tuple)
            of keys to remove from the arrayset

        Returns
        -------
        Union[KeyType, Sequence[KeyType]]
            Upon success, the key(s) removed from the arrayset

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        res = keys  # return same type of key as passed in
        with self._txn_backend_context() as ctx:
            if not isinstance(keys, (list, tuple)):
                keys = [keys]
            for key in keys:
                try:
                    dataKey = data_record_db_key_from_raw_key(self._asetn, key)
                    del self._samples[key]
                    isRecordDeleted = ctx.dataTxn.delete(dataKey)
                    if isRecordDeleted is False:
                        raise KeyError
                except KeyError:
                    raise KeyError(f'No sample {key} in {self._asetn}')
            return res

    def pop(self, keys: Union[KeyType, Sequence[KeyType]]) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        keys : Union[KeyType, Sequence[KeyType]]
            Either a single sample key to remove, or a sequence (list or tuple)
            of keys to remove from the arrayset

        Returns
        -------
        Union[np.ndarray, KeyArrMap]
            Upon success, the value of the removed key (if a single element is
            passed), or a dictionary of key/value pairs of the removed samples.

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        values = self.get(keys)
        self.delete(keys)
        return values

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
            inferred. If dict, key ``backend`` must have a valid backend format code
            value, and the rest of the items are assumed to be valid specs for that
            particular backend. If none, both backend and opts are inferred from
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
        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(self.arrayset)
        asetSchemaVal = arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=self.variable_shape,
            schema_max_shape=proto.shape,
            schema_dtype=proto.dtype.num,
            schema_is_named=self.named_samples,
            schema_default_backend=beopts.backend,
            schema_default_backend_opts=beopts.opts,
            schema_contains_subsamples=self._contains_subsamples)

        hashSchemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        with self._txnctx.write() as ctx:
            ctx.dataTxn.put(asetSchemaKey, asetSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, asetSchemaVal, overwrite=False)
        rawAsetSchema = arrayset_record_schema_raw_val_from_db_val(asetSchemaVal)

        if beopts.backend not in self._be_fs:
            self._be_fs[beopts.backend] = BACKEND_ACCESSOR_MAP[beopts.backend](
                repo_path=self._path,
                schema_shape=self._schema_max_shape,
                schema_dtype=np.typeDict[self._schema_dtype_num])
        else:
            self._be_fs[beopts.backend].close()
        self._be_fs[beopts.backend].open(mode=self._mode)
        self._be_fs[beopts.backend].backend_opts = beopts.opts
        self._dflt_backend = beopts.backend
        self._dflt_backend_opts = beopts.opts
        self._dflt_schema_hash = schema_hash
        self._schema_spec = rawAsetSchema
        return
