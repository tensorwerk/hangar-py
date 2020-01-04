from contextlib import ExitStack
from pathlib import Path
from typing import Tuple, List, Union, NamedTuple, Sequence, Dict, Iterable, Type, Optional

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
# GetKeysType = Union[KeyType, Sequence[KeyType]]
AsetTxnType = Type['AsetTxn']


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
                 repo_path: Path,
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

    def _ipython_key_completions_(self):  # pragma: no cover
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

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

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

    def __getitem__(self, key: KeyType) -> np.ndarray:
        """Retrieve data for some sample key via dict style access conventions.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : KeyType
            Sample key to retrieve from the arrayset.

        Returns
        -------
        :class:`numpy.ndarray`
            Sample array data corresponding to the provided key.

        Raises
        ------
        KeyError
            if no sample with the requested key exists.
        """
        try:
            spec = self._samples[key]
            return self._be_fs[spec.backend].read_data(spec)
        except KeyError:
            raise KeyError(f'No sample key {key} exists in arrayset.')

    def get(self, *keys: KeyType) -> Union[KeyArrMap, np.ndarray]:
        """Retrieve tensor data for some sample key(s) in the arrayset.

        Parameters
        ----------
        *keys : KeyType
            Return the value for the key if a sample exists with that name.

        Returns
        -------
        Union[:class:`np.ndarray`, KeyArrMap]
            Tensor data stored in the arrayset archived with provided name(s).

            If multiple keys passed as a parameter, a dict mapping key names
            to tensor data.

        Raises
        ------
        KeyError
            if the arrayset does not contain data with the provided name
        """
        if len(keys) > 1:
            res = {}
            for key in keys:
                res[key] = self[key]
        else:
            res = self[keys[0]]
        return res

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
                yield self[sample_name]
        else:
            for sample_name, sample_spec in tuple(self._samples.items()):
                if is_local_backend(sample_spec):
                    yield self[sample_name]

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
                yield sample_name, self[sample_name]
        else:
            for sample_name, sample_spec in tuple(self._samples.items()):
                if is_local_backend(sample_spec):
                    yield sample_name, self[sample_name]


class SampleWriterModifier(SampleReaderModifier):

    def __init__(self, aset_txn_ctx: AsetTxnType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'a'
        self._txnctx = aset_txn_ctx
        self._stack: Optional[ExitStack] = None
        self._enter_count = 0

    def __enter__(self):
        with ExitStack() as stack:
            self._txnctx.open_write()
            stack.callback(self._txnctx.close_write)
            if self._enter_count == 0:
                for k in self._be_fs.keys():
                    stack.enter_context(self._be_fs[k])
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1

    @property
    def _is_conman(self) -> bool:
        return bool(self._enter_count)

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
        return CompatibleArray(compatible, reason)

    def _set_arg_validate(self, key: KeyType, value: np.ndarray) -> bool:

        if not is_suitable_user_key(key):
            raise ValueError(f'Sample name `{key}` is not suitable.')
        isCompat = self._verify_array_compatible(value)
        if not isCompat.compatible:
            raise ValueError(isCompat.reason)

    def _perform_set(self, key: KeyType, value: np.ndarray) -> KeyType:
        """Internal write method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key : KeyType
            sample key to store
        value : np.ndarray
            tensor data to store

        Returns
        -------
        KeyType
            name of saved data.
        """
        full_hash = array_hash_digest(value)
        hashKey = hash_data_db_key_from_raw_key(full_hash)
        # check if data record already exists with given key
        dataRecKey = data_record_db_key_from_raw_key(self._asetn, key)
        existingDataRecVal = self._txnctx.dataTxn.get(dataRecKey, default=False)
        if existingDataRecVal:
            # check if data record already with same key & hash value
            existingDataRec = data_record_raw_val_from_db_val(existingDataRecVal)
            if full_hash == existingDataRec.data_hash:
                return key

        # write new data if data hash does not exist
        existingHashVal = self._txnctx.hashTxn.get(hashKey, default=False)
        if existingHashVal is False:
            hashVal = self._be_fs[self._dflt_backend].write_data(value)
            self._txnctx.hashTxn.put(hashKey, hashVal)
            self._txnctx.stageHashTxn.put(hashKey, hashVal)
            hash_spec = backend_decoder(hashVal)
        else:
            hash_spec = backend_decoder(existingHashVal)

        # add the record to the db
        dataRecVal = data_record_db_val_from_raw_val(full_hash)
        self._txnctx.dataTxn.put(dataRecKey, dataRecVal)
        self._samples[key] = hash_spec
        return key

    def __setitem__(self, key: KeyType, value: np.ndarray) -> None:
        """Store a piece of data in a arrayset.

        .. seealso::

            :meth:`add` alternative way to set key / value where input is expressed
            as parameters in a method call.

            :meth:`update` for an implementation analogous to python's built in
            :meth:`dict.update` method which accepts a dict or iterable of
            key/value pairs to add in the same operation.

        Parameters
        ----------
        key : KeyType, optional
            name to assign to the same (assuming the arrayset accepts named
            samples), If str, can only contain alpha-numeric ascii characters
            (in addition to '-', '.', '_'). Integer key must be >= 0. by default
            None
        value : :class:`numpy.ndarray`
            data to store as a sample in the arrayset.

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
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(key, value)
            self._perform_set(key, value)

    def add(self, key: KeyType, value: np.ndarray) -> KeyType:
        """Store a piece of data in a arrayset with a given name and value

        .. seealso::

            :meth:`__setitem__` which implements identical functionality through a
            dict style interface. The only difference is that upon successful
            completion, this method returns the name of the key which was set,
            while :meth:`__setitem__` does not return anything.

            :meth:`update` which implements functionality similar to python's
            builtin :meth:`dict.update` method, accepting either a dictionary or
            other iterable (of length two) listing key / value pairs.

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
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(key, value)
            return self._perform_set(key, value)

    def append(self, value: np.ndarray) -> KeyType:
        """TODO: is this the right way we should be handling unnamed samples?

        Thesis:
            Rather than have seperate arraysets for each named and unnamed sample type,
            you should be able to append to any arrayset without a name, and one will be
            generated for you?
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            key = generate_sample_name()
            self._set_arg_validate(key, value)
            return self._perform_set(key, value)

    def _from_sequence(self, seq: Sequence[KeyArrType]) -> Sequence[KeyType]:
        # assume callers already set up context manager
        for idx, kv_double in enumerate(seq):
            if len(kv_double) != 2:
                raise ValueError(
                    f"dictionary update sequence element #{idx} ({kv_double}) has "
                    f"length {len(kv_double)}; 2 is required")
            self._set_arg_validate(kv_double[0], kv_double[1])
        saved_keys = []
        for key, val in seq:
            self._perform_set(key, val)
            saved_keys.append(key)
        return saved_keys

    def _merge(self, mapping: KeyArrMap) -> Sequence[KeyType]:
        # assume callers already set up context manager
        for key, val in mapping.items():
            self._set_arg_validate(key, val)
        saved_keys = []
        for key, val in mapping.items():
            self._perform_set(key, val)
            saved_keys.append(key)
        return saved_keys

    def update(self,
               other: Union[None, KeyArrMap, KeyArrType, Sequence[KeyArrType]] = None,
               **kwargs) -> List[KeyType]:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, KeyArrMap, KeyArrType, Sequence[KeyArrType]], optional
            Dictionary mapping sample names to :class:`np.ndarray` instances,
            length two sequence (list or tuple) of sample name and
            :class:`np.ndarray` instance, or sequence of multiple length two
            sequences (list or tuple) of sample names / :class:`np.ndarray`
            instances to store. If sample name is string type, can only contain
            alpha-numeric ascii characters (in addition to '-', '.', '_').
            Integer key must be >= 0. By default, None, in which care assignments
            will be made from keywork args
        **kwargs
            keyword arguments provided will be saved with keywords as sample keys
            (string type only) and values as np.array instances.


        Returns
        -------
        List[KeyType]
            sample name(s) of the stored data (assuming the operation was
            successful)

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
            For fixed shape arraysets, if input data dimensions do not exactly
            match specified arrayset dimensions.
        ValueError
            If type of `data` argument is not an instance of np.ndarray.
        ValueError
            If `data` is not "C" contiguous array layout.
        ValueError
            If the datatype of the input data does not match the specified data
            type of the arrayset
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            # Note: we have to merge kwargs dict (if it exists) with `other` before
            # operating on either one. This is so that the validation and write
            # methods occur in one operation; if any element in any one of the inputs,
            # is invalid, no data will be written from another.

            saved_keys = []
            if isinstance(other, dict):
                if kwargs:
                    other.update(kwargs)
                saved_keys.extend(self._merge(other))
            elif isinstance(other, (list, tuple)):
                if kwargs:
                    other = list(other)
                    other.extend(list(kwargs.items()))
                saved_keys.extend(self._from_sequence(other))
            elif other is None:
                saved_keys.extend(self._merge(kwargs))
            else:
                raise ValueError(f'Type of `other` {type(other)} must be mapping, list, tuple')

            return saved_keys

    def _perform_del(self, key: KeyType) -> KeyType:
        """Internal del method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key : KeyType
            sample key remove.

        Returns
        -------
        KeyType
            name of removed sample.
        """
        dataKey = data_record_db_key_from_raw_key(self._asetn, key)
        isRecordDeleted = self._txnctx.dataTxn.delete(dataKey)
        if isRecordDeleted is False:
            raise KeyError
        del self._samples[key]
        return key

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
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            if key not in self:
                raise KeyError(f'No sample {key} in {self._asetn}')
            self._perform_del(key)

    def delete(self, *keys: KeyType) -> Union[KeyType, Sequence[KeyType]]:
        """Remove some key / value pair(s) from the arrayset, returning the deleted keys.

        Parameters
        ----------
        *keys : KeyType
            name (or names) or samples to remove from the arrayset

        Returns
        -------
        Union[KeyType, Sequence[KeyType]]
            Upon success, the key(s) removed from the arrayset

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if len(keys) > 1:
                for key in keys:
                    if key not in self:
                        raise KeyError(f'No sample with name {key} exists.')
                removed = []
                for key in keys:
                    removed.append(self._perform_del(key))
                return removed
            else:
                if keys[0] not in self:
                    raise KeyError(f'No sample with name {keys[0]} exists.')
                return self._perform_del(keys[0])

    def pop(self, *keys: KeyType) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        *keys : KeyType
            name (or names) of samples to remove from the arrayset and get values
            values back for.

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
        values = self.get(*keys)
        self.delete(*keys)
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
