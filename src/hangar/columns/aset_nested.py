from contextlib import ExitStack
from pathlib import Path
from typing import Tuple, List, Union, NamedTuple, Sequence, Dict, Iterable, Any, Type, Optional
from weakref import proxy

import numpy as np

from ..utils import is_suitable_user_key
from ..backends import (
    backend_decoder,
    is_local_backend,
    parse_user_backend_opts,
    BACKEND_ACCESSOR_MAP,
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

AsetTxnType = Type['AsetTxn']
KeyType = Union[str, int]
KeysType = Union[KeyType, Sequence[KeyType]]
EllipsisType = type(Ellipsis)
GetKeysType = Union[KeysType, EllipsisType, slice]
KeyArrMap = Dict[KeyType, np.ndarray]
KeyArrType = Union[Tuple[KeyType, np.ndarray], List[Union[KeyType, np.ndarray]]]
MapKeyArrType = Union[KeyArrMap, Sequence[KeyArrType]]


class CompatibleArray(NamedTuple):
    compatible: bool
    reason: str


class SubsampleName(NamedTuple):
    sample: str
    subsample: str


class SubsampleReader(object):

    __slots__ = ('_asetn', '_samplen', '_be_fs', '_subsamples', '__weakref__')

    def __init__(self, asetn: str, samplen: str, be_handles: BACKEND_ACCESSOR_MAP,
                 specs: Dict[KeyType, DataHashSpecsType]):

        self._asetn = asetn
        self._samplen = samplen
        self._be_fs = be_handles
        self._subsamples = specs

    def __repr__(self):
        res = f'{self.__class__}('\
              f'aset_name={self._asetn}, '\
              f'sample_name={self._samplen})'
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Arrayset Name            : {self._asetn}\
                \n    Sample Name              : {self._samplen}\
                \n    Mode (read/write)        : "r"\
                \n    Number of Subsamples     : {self.__len__()}\n'
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

    def __enter__(self):
        self._enter_count += 1
        return self

    def __exit__(self, *exc):
        self._enter_count -= 1
        return

    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        if hasattr(self, '_mode'):
            if self._mode == 'a':
                raise TypeError(f'cannot pickle write enabled checkout samples')
        state = {}
        state['_asetn'] = self._asetn
        state['_samplen'] = self._samplen
        state['_be_fs'] = self._be_fs
        state['_subsamples'] = self._subsamples
        return state

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.
        """
        self._asetn = state['_asetn']
        self._samplen = state['_samplen']
        self._be_fs = state['_be_fs']
        self._subsamples = state['_subsamples']

    def __len__(self) -> int:
        return len(self._subsamples)

    def __contains__(self, key: KeyType) -> bool:
        return key in self._subsamples

    def __iter__(self) -> Iterable[KeyType]:
        yield from self.keys()

    def _perform_get(self, key: KeyType) -> np.ndarray:
        try:
            spec = self._subsamples[key]
            return self._be_fs[spec.backend].read_data(spec)
        except KeyError:
            raise KeyError(f'No subsample key {key} exists in sample {self.sample}.')

    def __getitem__(self, key: KeyType) -> np.ndarray:
        """Retrieve data for some subsample key via dict style access conventions.

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
        # select subsample(s) with regular keys
        if isinstance(key, (str, int)):
            return self._perform_get(key)
        # select all subsamples
        elif key is Ellipsis:
            res = {}
            for subsample in tuple(self._subsamples.keys()):
                res[subsample] = self._perform_get(subsample)
            return res
        # slice subsamples by sorted order of keys
        elif isinstance(key, slice):
            res = {}
            subsample_spec_slice = tuple(self._subsamples.items())[key]
            for subsample, spec in subsample_spec_slice:
                res[subsample] = self._perform_get(subsample)
            return res
        else:
            raise ValueError(f'subsample keys argument: {key} not valid format')

    @property
    def _enter_count(self):
        return self._be_fs['enter_count']

    @_enter_count.setter
    def _enter_count(self, value):
        self._be_fs['enter_count'] = value

    @property
    def _is_conman(self):
        return bool(self._enter_count)

    @property
    def sample(self) -> KeyType:
        """Return name (key) of this sample.
        """
        return self._samplen

    @property
    def arrayset(self) -> str:
        """Return name (key) of arrayset this sample is contained in.
        """
        return self._asetn

    @property
    def data(self) -> KeyArrMap:
        """Return dict mapping every subsample key / data value stored in the sample.

        Returns
        -------
        KeyArrMap
            Dictionary mapping subsample name(s) (keys) to their stored
            values as :class:`np.ndarray` instances.
        """
        return self.get(...)

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating all subsamples in sample arrayset exist on local disk.

        The data associated with subsamples referencing some remore server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if at least one subsample in the arrayset references data stored
            on some remote server. True if all sample data is available on the
            machine's local disk.
        """
        for subsample_spec in tuple(self._subsamples.values()):
            if not is_local_backend(subsample_spec):
                return True
        return False

    @property
    def remote_reference_keys(self) -> List[KeyType]:
        """Compute subsample names whose data is stored in a remote server reference.

        Returns
        -------
        List[KeyType]
            list of subsample keys in the arrayset whose data references indicate
            they are stored on a remote server.
        """
        keys = []
        for subsample_name, subsample_spec in tuple(self._subsamples.items()):
            if not is_local_backend(subsample_spec):
                keys.append(subsample_name)
        return keys

    def keys(self, local: bool = False) -> Iterable[KeyType]:
        """Generator yielding the name (key) of every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys will only correspond to data which is
            available for reading on the local disk, by default False.

        Yields
        ------
        Iterable[KeyType]
            Keys of one subsample at a time inside the sample.
        """
        if not local:
            for subsample_name in tuple(self._subsamples.keys()):
                yield subsample_name
        else:
            for subsample_name, subsample_spec in tuple(self._subsamples.items()):
                if is_local_backend(subsample_spec):
                    yield subsample_name

    def values(self, local: bool = False) -> Iterable[np.ndarray]:
        """Generator yielding the tensor data for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[:class:`numpy.ndarray`]
            Values of one subsample at a time inside the sample.
        """
        if not local:
            for subsample_name in tuple(self._subsamples.keys()):
                yield self.get(subsample_name)
        else:
            for subsample_name, subsample_spec in tuple(self._subsamples.items()):
                if is_local_backend(subsample_spec):
                    yield self.get(subsample_name)

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, np.ndarray]]:
        """Generator yielding (name, tensor) tuple for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, np.ndarray]]
            Name and stored value for every subsample inside the sample.
        """
        if not local:
            for subsample_name in tuple(self._subsamples.keys()):
                yield (subsample_name, self.get(subsample_name))
        else:
            for subsample_name, subsample_spec in tuple(self._subsamples.items()):
                if is_local_backend(subsample_spec):
                    yield (subsample_name, self.get(subsample_name))

    def get(self, *keys: Union[None, GetKeysType]) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve the data associated with some subsample key(s)

        Parameters
        ----------
        keys : GetKeysType
            The name of the subsample(s) to retrieve. Passing a single
            subsample key will return the stored :class:`numpy.ndarray`
            associated with that key. Passing a sequence of keys (list or
            tuple) will return a dict mapping each key to it's stored array
            data.

            Alternatively, ``slice`` syntax can be used to retrieve a
            selection of subsample keys/values. An empty slice (``:`` == ``slice(None)``)
            or ``Ellipsis`` (``...``) will return all subsample keys/values.
            Passing a non-empty slice (``[1:5] == slice(1, 5)``) will select
            keys to retrieve by enumerating all subsamples and retrieving
            the element (key) for each step across the range. Note: order of
            enumeration is not guaranteed; do not rely on any ordering observed
            when using this method.


        Returns
        -------
        Union[np.ndarray, KeyArrMap]
            single :class:`numpy.ndarray` if a single sample key was passed in, else
            a dictionary mapping subsample names to array data.
        """
        if len(keys) > 1:
            res = {}
            for key in keys:
                res[key] = self._perform_get(key)
        else:
            res = self[keys[0]]
        return res


class SubsampleWriter(SubsampleReader):

    __slots__ = ('_txnctx', '_stack')

    def __init__(self, aset_txn_ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._txnctx = aset_txn_ctx
        self._stack: Optional[ExitStack] = None

    def __repr__(self):
        res = f'{self.__class__}('\
              f'aset_name={self._asetn}, '\
              f'sample_name={self._samplen}, '\
              f'default_backend={self._schema_spec.schema_default_backend})'
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__} \
                \n    Arrayset Name            : {self._asetn}\
                \n    Sample Name              : {self._samplen}\
                \n    Mode (read/write)        : "a"\
                \n    Default Backend          : {self._schema_spec.schema_default_backend}\
                \n    Number of Subsamples     : {self.__len__()}\n'
        p.text(res)

    def __enter__(self):
        with ExitStack() as stack:
            self._txnctx.open_write()
            stack.callback(self._txnctx.close_write)
            if self._enter_count == 0:
                for k in self._be_fs.keys():
                    if k in ('enter_count', 'schema_spec'):
                        continue
                    stack.enter_context(self._be_fs[k])
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1
        if self._enter_count == 0:
            self._stack = None

    @property
    def _schema_spec(self) -> RawArraysetSchemaVal:
        return self._be_fs['schema_spec']

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
            subsample key to store
        value : np.ndarray
            tensor data to store

        Returns
        -------
        KeyType
            name of saved data subsample.
        """
        full_hash = array_hash_digest(value)
        hashKey = hash_data_db_key_from_raw_key(full_hash)
        # check if data record already exists with given key
        dataRecKey = data_record_db_key_from_raw_key(
            self._asetn, self._samplen, subsample=key)
        existingDataRecVal = self._txnctx.dataTxn.get(dataRecKey, default=False)
        if existingDataRecVal:
            # check if data record already with same key & hash value
            existingDataRec = data_record_raw_val_from_db_val(existingDataRecVal)
            if full_hash == existingDataRec.data_hash:
                return key

        # write new data if data hash does not exist
        existingHashVal = self._txnctx.hashTxn.get(hashKey, default=False)
        if existingHashVal is False:
            backendCode = self._schema_spec.schema_default_backend
            hashVal = self._be_fs[backendCode].write_data(value)
            self._txnctx.hashTxn.put(hashKey, hashVal)
            self._txnctx.stageHashTxn.put(hashKey, hashVal)
            hash_spec = backend_decoder(hashVal)
        else:
            hash_spec = backend_decoder(existingHashVal)

        # add the record to the db
        dataRecVal = data_record_db_val_from_raw_val(full_hash)
        self._txnctx.dataTxn.put(dataRecKey, dataRecVal)
        self._subsamples[key] = hash_spec
        return key

    def __setitem__(self, key: KeyType, value: np.ndarray) -> None:
        """Store a piece of data as a subsample. Convenience method to :meth:`add`.

        .. seealso::

            :meth:`add` for the actual method called.

            :meth:`update` for an implementation analogous to python's built in
            :meth:`dict.update` method which accepts a dict or iterable of key/value
            pairs to add in the same operation.

        Parameters
        ----------
        key : KeyType
            Key (name) of the subsample to add to the arrayset.
        value : :class:`numpy.ndarray`
            Tensor data to add as the sample.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(key, value)
            self._perform_set(key, value)

    def add(self, key: KeyType, value: np.ndarray) -> KeyType:
        """Store a piece of data in a subsample with a given name and value

        .. seealso::

            :meth:`update` which implements functionality similar to python's
            builtin :meth:`dict.update` method, accepting either a dictionary
            or other iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        key : KeyType
            Name to assign to the subsample, If str, can only contain
            alpha-numeric ascii characters (in addition to '-', '.', '_').
            Integer key must be >= 0.
        value : :class:`numpy.ndarray`
            data to store as a sample in the arrayset.

        Returns
        -------
        KeyType
            sample name of the stored data (assuming the operation was successful)
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

    def _from_sequence(self, seq: Sequence[KeyArrType]) -> List[KeyType]:
        # assume callers already set up context manager
        for idx, kv_double in enumerate(seq):
            if len(kv_double) != 2:
                raise ValueError(
                    f"dictionary update sequence element #{idx} ({kv_double}) has "
                    f"length {len(kv_double)}; 2 is required")
            self._set_arg_validate(kv_double[0], kv_double[1])
        saved_keys = []
        for key, val in seq:
            saved_keys.append(self._perform_set(key, val))
        return saved_keys

    def _merge(self, mapping: KeyArrMap) -> List[KeyType]:
        # assume callers already set up context manager
        for key, val in mapping.items():
            self._set_arg_validate(key, val)
        saved_keys = []
        for key, val in mapping.items():
            saved_keys.append(self._perform_set(key, val))
        return saved_keys

    def update(self, other: Union[None, MapKeyArrType] = None, **kwargs) -> List[KeyType]:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, MapKeyArrType]
            Accepts either another dictionary object or an iterable of key/value
            pairs (as tuples or other iterables of length two). mapping sample
            names to :class:`np.ndarray` instances, If sample name is string type,
            can only contain alpha-numeric ascii characters (in addition to '-',
            '.', '_'). Integer key must be >=
            0.
        **kwargs
            keyword arguments provided will be saved with keywords as subsample
            keys (string type only) and values as np.array instances.

        Returns
        -------
        List[KeyType]
            sample name(s) of the stored data (assuming the operation was successful)
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            # Note: we have to merge kwargs dict (if it exists) with `other` before
            # operating on either one. This is so that the validation and write
            # methods occur in one operation; if any element in any one of the inputs,
            # is invalid, no data will be written from another.

            if isinstance(other, dict):
                if kwargs:
                    other.update(kwargs)
                return self._merge(other)
            elif isinstance(other, (list, tuple)):
                if kwargs:
                    other = list(other)
                    other.extend(list(kwargs.items()))
                return self._from_sequence(other)
            elif other is None:
                return self._merge(kwargs)
            else:
                raise ValueError(f'Type of `other` {type(other)} must be mapping, list, tuple')

    def _perform_del(self, key: KeyType) -> KeyType:
        """Internal del method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key : KeyType
            subsample key remove.

        Returns
        -------
        KeyType
            name of removed subsample.
        """
        dbKey = data_record_db_key_from_raw_key(
            self._asetn, self._samplen, subsample=key)
        isRecordDeleted = self._txnctx.dataTxn.delete(dbKey)
        if isRecordDeleted is False:
            raise KeyError(f'No subsample {key} in sample {self._samplen}')
        del self._subsamples[key]
        return key

    def __delitem__(self, key: KeyType):
        """Remove a subsample from the arrayset. Convenience method to :meth:`delete`.

        .. seealso::

            :meth:`delete` (the analogous named operation for this method)

            :meth:`pop` to return a records value and then delete it in the same
            operation

        Parameters
        ----------
        key : KeyType
            Name of the sample to remove from the arrayset.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            if key not in self:
                raise KeyError(f'No subsample {key} in sample {self._samplen}')
            self._perform_del(key)

    def delete(self, *keys: KeyType) -> Union[KeyType, Sequence[KeyType]]:
        """Remove some key / value subsample pairs from a sample, returning the deleted keys.

        Parameters
        ----------
        *keys : KeyType
            name (or names) or subsample(s) to remove from the sample

        Returns
        -------
        Union[KeyType, Sequence[KeyType]]
            Upon success, the key(s) removed from the sample

        Raises
        ------
        KeyError
            If there is no subsample with the provided key(s)
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

    def pop(self, *keys: KeysType) -> Union[np.ndarray, KeyArrMap]:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        *keys : KeysType
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
        values = self.get(*keys)
        self.delete(*keys)
        return values


SubsampleTypes = Union[SubsampleReader, SubsampleWriter]


class SubsampleReaderModifier(object):

    def __init__(self,
                 aset_name: str,
                 samples: Dict[KeyType, SubsampleTypes],
                 backend_handles: Dict[str, Any],
                 schema_spec: RawArraysetSchemaVal,
                 repo_path: Path,
                 *args, **kwargs):

        self._mode = 'r'
        self._asetn = aset_name
        self._samples = samples
        self._be_fs = backend_handles
        self._path = repo_path
        self._stack: Optional[ExitStack] = None

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
        res = f'Hangar Subsample Arrayset \
                \n    Arrayset Name            : {self._asetn}\
                \n    Schema Hash              : {self._dflt_schema_hash}\
                \n    Variable Shape           : {bool(int(self._schema_variable))}\
                \n    (max) Shape              : {self._schema_max_shape}\
                \n    Datatype                 : {np.typeDict[self._schema_dtype_num]}\
                \n    Named Samples            : {bool(self._samples_are_named)}\
                \n    Access Mode              : {self._mode}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self.contains_remote_references)}\
                \n    Contains Subsamples      : True\
                \n    Number of Subsamples     : {self.num_subsamples}\n'
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

    def __enter__(self):
        self._enter_count += 1
        return self

    def __exit__(self, *exc):
        self._enter_count -= 1
        return

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

    def __getitem__(self, key: KeyType) -> SubsampleTypes:
        """Get the sample access class for some sample key.

        Parameters
        ----------
        key : KeyType
            Name of sample to retrieve

        Returns
        -------
        SubsampleTypes
            Sample accessor corresponding to the given key
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
        """Find number of samples in the arrayset
        """
        return len(self._samples)

    def __contains__(self, key: KeyType) -> bool:
        """Determine if some sample key exists in the arrayset.

        Parameters
        ----------
        key : KeyType
            Key to check for existence as sample in arrayset.

        Returns
        -------
        bool
            True if sample key exists, otherwise False
        """
        return key in self._samples

    def _open(self):
        for val in self._be_fs.values():
            try:
                # since we are storing non backend accessor information in the be_fs
                # weakref proxy for the purpose of memory savings, not all elements
                # have a `open` method
                val.open(mode=self._mode)
            except AttributeError:
                pass

    def _close(self):
        for val in self._be_fs.values():
            # since we are storing non backend accessor information in the be_fs
            # weakref proxy for the purpose of memory savings, not all elements
            # have a `close` method
            try:
                val.close()
            except AttributeError:
                pass

    @property
    def _enter_count(self):
        return self._be_fs['enter_count']

    @_enter_count.setter
    def _enter_count(self, value):
        self._be_fs['enter_count'] = value

    @property
    def _is_conman(self):
        return bool(self._enter_count)

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
        """Bool indicating if all samples exist locally or if some reference remote sources.
        """
        for subsample in tuple(self._samples.values()):
            if subsample.contains_remote_references:
                return True
        return False

    @property
    def remote_reference_keys(self) -> List[KeyType]:
        """Returns sample names who have subsamples whose data references remote sources.

        Returns
        -------
        List[KeyType]
            list of sample keys in the arrayset.
        """
        keys = []
        for sample_name, subsample in tuple(self._samples.items()):
            if subsample.contains_remote_references:
                keys.append(sample_name)
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

    @property
    def contains_subsamples(self) -> bool:
        """Bool indicating if sub-samples are contained in this arrayset container.

        Returns
        -------
        bool
            True if subsamples are included, False otherwise. For this arrayset
            class, subsamples are stored; the result will always be True
        """
        return True

    @property
    def num_subsamples(self) -> int:
        """Calculate total number of subsamples existing in all samples in arrayset
        """
        total = 0
        for sample in self._samples.values():
            total += len(sample)
        return total

    def keys(self, local: bool = False) -> Iterable[KeyType]:
        """Generator yielding the name (key) of every sample in the arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys will only correspond to samples in which
            every subsample has data available for reading on the local disk,
            by default False.

        Yields
        ------
        Iterable[KeyType]
            Keys of one sample at a time inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield sample_name
        else:
            for sample_name, subsample in tuple(self._samples.items()):
                if not subsample.contains_remote_references:
                    yield sample_name

    def values(self, local: bool = False) -> Iterable[SubsampleTypes]:
        """Generator yielding the subsample accessor class for every sample in arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to samples in which
            every subsample has data available for reading on the local disk,
            by default False.

        Yields
        ------
        Iterable[SubsampleTypes]
            Subsample accessor class of one sample at a time inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield self.get(sample_name)
        else:
            for sample_name, subsample in tuple(self._samples.items()):
                if not subsample.contains_remote_references:
                    yield self.get(sample_name)

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, SubsampleTypes]]:
        """Generator yielding (name, subsample accessor) tuple for every sample in arrayset.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to samples in
            which every subsample has data available for reading on the local
            disk, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, SubsampleTypes]]
            Sample name and subsample accessor class for every sample inside the arrayset.
        """
        if not local:
            for sample_name in tuple(self._samples.keys()):
                yield (sample_name, self.get(sample_name))
        else:
            for sample_name, subsample in tuple(self._samples.items()):
                if not subsample.contains_remote_references:
                    yield (sample_name, self.get(sample_name))

    def get(self, key: GetKeysType) -> SubsampleTypes:
        """Retrieve tensor data for some sample key(s) in the arrayset.

        Parameters
        ----------
        key : GetKeysType
            Sample key to retrieve data for.

        Returns
        -------
        SubsampleTypes:
            Sample accessor class given by name ``key`` which can be used to
            access subsample data.

        Raises
        ------
        KeyError
            if the arrayset does not contain a sample with the provided key.
        """
        try:
            return proxy(self._samples[key])
        except KeyError:
            raise KeyError(f'No sample key {key} exists in arrayset.')


class SubsampleWriterModifier(SubsampleReaderModifier):

    def __init__(self, aset_txn_ctx: AsetTxnType, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mode = 'a'
        self._txnctx = aset_txn_ctx
        self._stack: ExitStack = ExitStack()

    def __enter__(self):
        with ExitStack() as stack:
            self._txnctx.open_write()
            stack.callback(self._txnctx.close_write)
            if self._enter_count == 0:
                for k in tuple(self._be_fs.keys()):
                    if k in ('enter_count', 'schema_spec'):
                        continue
                    stack.enter_context(self._be_fs[k])
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1
        return

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

    def _set_arg_validate(self, sample_key: KeyType, subsample_map: MapKeyArrType):

        if not is_suitable_user_key(sample_key):
            raise ValueError(f'Sample name `{sample_key}` is not suitable.')

        if isinstance(subsample_map, (tuple, list)):
            subsample_map = dict(subsample_map)

        for subsample_key, subsample_val in subsample_map.items():
            if not is_suitable_user_key(subsample_key):
                raise ValueError(f'Sample name `{sample_key}` is not suitable.')
            isCompat = self._verify_array_compatible(subsample_val)
            if not isCompat.compatible:
                raise ValueError(isCompat.reason)

    def _perform_set(self, key: KeyType, value: MapKeyArrType) -> List[KeyType]:
        if key in self._samples:
            return self._samples[key].update(value)
        else:
            self._samples[key] = SubsampleWriter(
                aset_txn_ctx=proxy(self._txnctx),
                asetn=self._asetn,
                samplen=key,
                be_handles=proxy(self._be_fs),
                specs={})
            try:
                # TODO: class method to eliminate double validation check?
                return self._samples[key].update(value)
            except Exception as e:
                del self._samples[key]
                raise e

    def __setitem__(self, key: KeyType, value: MapKeyArrType) -> None:
        """Store some subsample key / subsample data map pairs, overwriting existing keys.

        .. seealso::

            :meth:`add` for the actual implementation of the method and docstring
            for this methods parameters
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(key, value)
            self._perform_set(key, value)

    def add(self, sample: KeyType, subsample_map: MapKeyArrType) -> Tuple[KeyType, Sequence[KeyType]]:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        sample : KeyType
            Sample key which the value in the subsample_map should be stored in.
        subsample_map : MapKeyArrType
            Dictionary mapping sample names to :class:`np.ndarray` instances,
            length two sequence (list or tuple) of subsample name(s) and
            :class:`np.ndarray` instance, or sequence of multiple length two
            sequences (list or tuple) of sample name(s) / :class:`np.ndarray`
            instance(s) to store. If sample name is string type, can only contain
            alpha-numeric ascii characters (in addition to '-', '.', '_').
            Integer key must be >= 0.

        Returns
        -------
        Tuple[KeyType, Sequence[KeyType]]
            two tuple of sample name saved, and sequence of subsample names saved.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(sample, subsample_map)
            subsample_keys = self._perform_set(sample, subsample_map)
            return (sample, subsample_keys)

    def _from_sequence(self, seq) -> Sequence[Tuple[KeyType, Sequence[KeyType]]]:
        """input [[sample_key, sample_data_map], ..., [sample_key, sample_data_map]]
        """
        # assume callers already set up context manager
        for idx, kv_double in enumerate(seq):
            if len(kv_double) != 2:
                raise ValueError(
                    f"dictionary update sequence element #{idx} ({kv_double}) has "
                    f"length {len(kv_double)}; 2 is required")
            self._set_arg_validate(kv_double[0], kv_double[1])
        saved_keys = []
        for key, val in seq:
            subsample_keys = self._perform_set(key, val)
            saved_keys.append((key, subsample_keys))
        return saved_keys

    def _merge(self, mapping) -> Sequence[Tuple[KeyType, Sequence[KeyType]]]:
        """input Dict[sample_key, [MapKeyArrType]]
        """
        # assume callers already set up context manager
        for key, val in mapping.items():
            self._set_arg_validate(key, val)
        saved_keys = []
        for key, val in mapping.items():
            subsample_keys = self._perform_set(key, val)
            saved_keys.append((key, subsample_keys))
        return saved_keys

    def update(
            self,
            other: Union[None, Dict[KeyType, MapKeyArrType],
                         Sequence[Sequence[Union[KeyType, MapKeyArrType]]]],
            **kwargs
    ) -> Sequence[Tuple[KeyType, Sequence[KeyType]]]:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, Dict[KeyType, MapKeyArrType],
                      Sequence[Sequence[Union[KeyType, MapKeyArrType]]]]

            Dictionary mapping sample names to subsample data maps. Or Squence
            (list or tuple) where element one is the sample name and element two is
            a subsample data map.

        **kwargs :
            keyword arguments provided will be saved with keywords as sample
            keys (string type only) and values as a mapping of subarray keys to
            :class:`np.ndarray` instances.

        Returns
        -------
        Sequence[Tuple[KeyType, Sequence[KeyType]]]
            Sequence of two tuple which contains fields listing `sample` and
            sequence of `subsample` keys which data were added into.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            # Note: we have to merge kwargs dict (if it exists) with `other` before
            # operating on either one. This is so that the validation and write
            # methods occur in one operation; if any element in any one of the inputs,
            # is invalid, no data will be written from another.

            if isinstance(other, dict):
                if kwargs:
                    other.update(kwargs)
                return self._merge(other)
            elif isinstance(other, (list, tuple)):
                if kwargs:
                    other = list(other)
                    other.extend(list(kwargs.items()))
                return self._from_sequence(other)
            elif other is None:
                return self._merge(kwargs)
            else:
                raise ValueError(f'Type of `other` {type(other)} must be mapping, list, tuple')

    def _perform_del(self, key: KeyType) -> KeyType:
        sample = self._samples[key]
        subsample_keys = list(sample.keys())
        sample.delete(*subsample_keys)
        del self._samples[key]
        return key

    def __delitem__(self, key: KeyType) -> None:
        """Remove a sample (including all contained subsamples) from the arrayset.

        .. seealso::

            :meth:`delete` for the actual implementation of the method and docstring
            for this methods parameters
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            if key not in self:
                raise KeyError(f'No sample {key} in {self._asetn}')
            self._perform_del(key)

    def delete(self, *keys: KeysType) -> KeysType:
        """Delete a sample (and all contained subsamples), returning removed sample key.

        Parameters
        ----------
        keys : KeysType
            Either a single sample key to remove, or a sequence (list or tuple)
            of keys to remove from the arrayset

        Returns
        -------
        KeysType
            Upon success, the sample key(s) removed from the arrayset
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

    def pop(self, *keys: KeysType) -> Dict[KeyType, KeyArrMap]:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        *keys : KeysType
            Either a single sample key to remove, or a sequence (list or tuple)
            of keys to remove from the arrayset

        Returns
        -------
        Dict[KeyType, KeyArrMap]
            Upon success, a nested dictionary mapping sample names to a dict of
            subsample names and subsample values for every sample key passed
            into this method.

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            for key in keys:
                if key not in self:
                    raise KeyError(f'sample name {key} does not exist.')

            res = {}
            for key in keys:
                sample = self._samples[key]
                res[key] = sample.data
                self.delete(key)
            return res

    def change_backend(self, backend_opts: Union[str, dict]):
        """Change the default backend and filters applied to future data writes.

        .. warning::

           This method is meant for advanced users only. Please refer to the
           hangar backend codebase for information on accepted parameters and
           options.

        Parameters
        ----------
        backend_opts : Union[str, dict]
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
        self._be_fs['schema_spec'] = rawAsetSchema
        self._dflt_backend = beopts.backend
        self._dflt_backend_opts = beopts.opts
        self._dflt_schema_hash = schema_hash
        self._schema_spec = rawAsetSchema
        return
