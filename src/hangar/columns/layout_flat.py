"""Accessor class for columns containing single-level key/value mappings

The FlatSample container is used to store data (in any backend) in a column
containing a single level key/value mapping from names/ids to data.

All backends are supported.
"""
from contextlib import ExitStack
from pathlib import Path
from typing import (
    Tuple, List, Union, NamedTuple, Sequence, Dict, Iterable, Type, Optional, Any
)

import numpy as np

from .validation import DataValidator
from .constructors import FlatSampleBuilder
from ..utils import is_suitable_user_key, valfilter, valfilterfalse
from ..op_state import reader_checkout_only, writer_checkout_only
from ..backends import (
    backend_decoder,
    parse_user_backend_opts,
    BACKEND_ACCESSOR_MAP,
    AccessorMapType,
    DataHashSpecsType,
)
from ..records.hashmachine import array_hash_digest, schema_hash_digest, metadata_hash_digest
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
MapKeyArrType = Union[KeyArrMap, Sequence[KeyArrType]]
AsetTxnType = Type['AsetTxn']


class FlatSample(metaclass=FlatSampleBuilder):
    """Class implementing get access to data in a arrayset.

    This class exposes the standard API to access data stored in a single level
    key / value mapping column. Usage is modeled after the python :py:`dict`
    style syntax -- with a few additional utility and inspection methods
    and properties added. Methods named after those of a python :py:`dict`
    have syntactically identical arguments and behavior to that of the standard
    library.

    If not opened in a ``write-enabled`` checkout, then attempts to add or delete
    data or container properties will raise an exception (in the form of a
    :py:`PermissionError`). No changes will be propogated unless a ``write-enabled``
    checkout is used.

    This object can be serialized -- pickled -- for parallel processing / reading
    if opened in a ``read-only`` checkout. Parallel operations are both
    thread and process safe, though performance may significantly differ between
    multithreaded vs multiprocessed code (depending on the backend data is stored
    in). Attempts to serialize objects opened in ``write-enabled`` checkouts are not
    supported and will raise a :py:`PermissionError` if attempted. This behavior
    is enforced in order to ensure data and record integrity while writing to
    the repository.
    """

    __slots__ = ('_mode', '_asetn', '_samples', '_be_fs', '_path',
                 '_schema_spec', '_schema_variable', '_schema_dtype_num',
                 '_schema_max_shape', '_dflt_schema_hash', '_dflt_backend',
                 '_dflt_backend_opts', '_contains_subsamples', '_stack',
                 '_txnctx', '_enter_count', '_datavalidator')
    _attrs = __slots__

    def __init__(self,
                 aset_name: str,
                 samples: Dict[KeyType, DataHashSpecsType],
                 backend_handles: AccessorMapType,
                 schema_spec: RawArraysetSchemaVal,
                 repo_path: Path,
                 mode: str,
                 aset_ctx: Optional[AsetTxnType] = None,
                 *args, **kwargs):

        self._stack: Optional[ExitStack] = None
        self._mode = mode
        self._asetn = aset_name
        self._samples = samples
        self._be_fs = backend_handles
        self._path = repo_path

        self._schema_spec = schema_spec
        # self._schema_variable = schema_spec.schema_is_var
        # self._schema_dtype_num = schema_spec.schema_dtype
        # self._schema_max_shape = tuple(schema_spec.schema_max_shape)
        self._dflt_schema_hash = schema_spec['schema_hash']
        self._dflt_backend = schema_spec['backend']
        self._dflt_backend_opts = schema_spec['backend_options']
        # self._contains_subsamples = schema_spec.schema_contains_subsamples

        self._txnctx = aset_ctx
        self._enter_count = 0

        self._datavalidator = DataValidator()
        self._datavalidator.schema = self._schema_spec

    @property
    def _debug_(self):  # pragma: no cover
        return {
            '__class__': self.__class__,
            '_mode': self._mode,
            '_asetn': self._asetn,
            '_be_fs': self._be_fs,
            '_path': self._path,
            '_schema_spec': self._schema_spec,
            '_schema_variable': self._schema_variable,
            '_schema_dtype_num': self._schema_dtype_num,
            '_schema_max_shape': self._schema_max_shape,
            '_dflt_schema_hash': self._dflt_schema_hash,
            '_dflt_backend': self._dflt_backend,
            '_dflt_backend_opts': self._dflt_backend_opts,
            '_contains_subsamples': self._contains_subsamples,
            '_txnctx': self._txnctx._debug_,
            '_stack': self._stack._exit_callbacks if self._stack else self._stack,
            '_enter_count': self._enter_count,
            '_datavalidator': self._datavalidator.__dict__,
        }

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
        res = f'Hangar {self.__class__.__qualname__} \
                \n    Arrayset Name            : {self._asetn}\
                \n    Schema Hash              : {self._dflt_schema_hash}\
                \n    Access Mode              : {self._mode}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self.contains_remote_references)}\
                \n    Contains Subsamples      : False\n'
        p.text(res)
                # \n    (max) Shape              : {self._schema_max_shape}\
                # \n    Datatype                 : {np.typeDict[self._schema_dtype_num]}\
                # \n    Variable Shape           : {bool(int(self._schema_variable))}\

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

    @reader_checkout_only
    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def __setstate__(self, state: dict) -> None:
        """ensure multiprocess operations can pickle relevant data.

        Technically should be decorated with @reader_checkout_only, but since
        at instance creation that is not an attribute, the decorator won't
        know. Since only readers can be pickled, This isn't much of an issue.
        """
        for slot, value in state.items():
            setattr(self, slot, value)

    def __enter__(self):
        if self._mode == 'r':
            return self
        else:
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
        if self._mode == 'r':
            return
        else:
            self._stack.close()
            self._enter_count -= 1

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        self._close()
        for attr in self._attrs:
            delattr(self, attr)

    def __getattr__(self, name):
        """Raise permission error after checkout is closed.

         Only runs after a call to :meth:`_destruct`, which is responsible for
         deleting all attributes from the object instance.
        """
        try:
            self.__getattribute__('_mode')  # once checkout is closed, this won't exist.
        except AttributeError:
            err = (f'Unable to operate on past checkout objects which have been '
                   f'closed. No operation occurred. Please use a new checkout.')
            raise PermissionError(err) from None
        return self.__getattribute__(name)


    @property
    def _is_conman(self) -> bool:
        return bool(self._enter_count)

    def __iter__(self) -> Iterable[KeyType]:
        """Create iterator yielding an arrayset sample keys.

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
        spec = self._samples[key]
        return self._be_fs[spec.backend].read_data(spec)

    def get(self, key: KeyType, default: Any = None) -> Union[np.ndarray, Any]:
        """Retrieve the data associated with some sample key

        Parameters
        ----------
        key : KeyType
            The name of the subsample(s) to retrieve. Passing a single
            subsample key will return the stored :class:`numpy.ndarray`
        default : Any
            if a `key` parameter is not found, then return this value instead.
            By default, None.

        Returns
        -------
        np.ndarray
            :class:`numpy.ndarray` array data stored under subsample key
            if key exists, else default value if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def arrayset(self) -> str:
        """Name of the arrayset.
        """
        return self._asetn

    # @property
    # def dtype(self) -> np.dtype:
    #     """Datatype of the arrayset schema.
    #     """
    #     return np.typeDict[self._schema_dtype_num]
    #
    # @property
    # def shape(self) -> Tuple[int]:
    #     """Shape (or `max_shape`) of the arrayset sample tensors.
    #     """
    #     return self._schema_max_shape
    #
    # @property
    # def variable_shape(self) -> bool:
    #     """Bool indicating if arrayset schema is variable sized.
    #     """
    #     return self._schema_variable

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this arrayset object is write-enabled.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating if all samples in arrayset exist on local disk.

        The data associated with samples referencing some remote server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if at least one sample in the arrayset references data stored
            on some remote server. True if all sample data is available on the
            machine's local disk.
        """
        return not all(map(lambda x: x.islocal, self._samples.values()))

    @property
    def remote_reference_keys(self) -> Tuple[KeyType]:
        """Compute sample names whose data is stored in a remote server reference.

        Returns
        -------
        Tuple[KeyType]
            list of sample keys in the arrayset whose data references indicate
            they are stored on a remote server.
        """
        return tuple(valfilterfalse(lambda x: x.islocal, self._samples).keys())

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
        """
        return False

    def _mode_local_aware_key_looper(self, local: bool) -> Iterable[KeyType]:
        """Generate keys for iteration with dict update safety ensured.

        Parameters
        ----------
        local : bool
            True if keys should be returned which only exist on the local machine.
            Fale if remote sample keys should be excluded.

        Returns
        -------
        Iterable[KeyType]
            Sample keys conforming to the `local` argument spec.
        """
        if local:
            if self._mode == 'r':
                yield from valfilter(lambda x: x.islocal, self._samples).keys()
            else:
                yield from tuple(valfilter(lambda x: x.islocal, self._samples).keys())
        else:
            if self._mode == 'r':
                yield from self._samples.keys()
            else:
                yield from tuple(self._samples.keys())

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
        yield from self._mode_local_aware_key_looper(local)

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
        for key in self._mode_local_aware_key_looper(local):
            yield self[key]

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
        for key in self._mode_local_aware_key_looper(local):
            yield (key, self[key])

    # ---------------- writer methods only after this point -------------------

    def _set_arg_validate(self, key: KeyType, value: np.ndarray) -> None:
        """Verify if key / value pair is valid to be written in this arrayset

        Parameters
        ----------
        key : KeyType
            name to associate with this data piece
        value : :class:`np.ndarray`
            piece of data to store in the arrayset

        Raises
        ------
        ValueError
            If key is not valid type/contents or if value is not correct object
            type / if it does not conform to arrayset schema
        """
        if not is_suitable_user_key(key):
            raise ValueError(f'Sample name `{key}` is not suitable.')

        isCompat = self._datavalidator.verify_data_compatible(value)
        if not isCompat.compatible:
            raise ValueError(isCompat.reason)

    @writer_checkout_only
    def _perform_set(self, key: KeyType, value: np.ndarray) -> None:
        """Internal write method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key : KeyType
            sample key to store
        value : np.ndarray
            tensor data to store
        """
        # full_hash = array_hash_digest(value)  # TODO TESTING
        full_hash = metadata_hash_digest(value)
        hashKey = hash_data_db_key_from_raw_key(full_hash)
        # check if data record already exists with given key
        dataRecKey = data_record_db_key_from_raw_key(self._asetn, key)
        existingDataRecVal = self._txnctx.dataTxn.get(dataRecKey, default=False)
        if existingDataRecVal:
            # check if data record already with same key & hash value
            existingDataRec = data_record_raw_val_from_db_val(existingDataRecVal)
            if full_hash == existingDataRec.data_hash:
                return

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
            name to assign to the sample (assuming the arrayset accepts named
            samples), If str, can only contain alpha-numeric ascii characters
            (in addition to '-', '.', '_'). Integer key must be >= 0. by
            default, None
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

    def append(self, value: np.ndarray) -> KeyType:
        """Store some data in a sample with an automatically generated key.

        This method should only be used if the context some piece of data is
        used in is independent from it's value (ie. when reading data back,
        there is no useful information which needs to be conveyed between the
        data source's name/id and the value of that piece of information.)
        Think carefully before going this route, as this posit does not apply
        to many common use cases.

        To store the data with a user defined key, use :meth:`update` or
        :meth:`__setitem__`

        Parameters
        ----------
        value: :class:`numpy.ndarray`
            Piece of data to store in the arrayset.

        Returns
        -------
        KeyType
            Name of the generated key this data is stored with.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            key = generate_sample_name()
            self._set_arg_validate(key, value)
            self._perform_set(key, value)
            return key

    def update(self, other: Union[None, MapKeyArrType] = None, **kwargs) -> None:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, MapKeyArrType], optional
            Accepts either another dictionary object or an iterable of
            key/value pairs (as tuples or other iterables of length two).
            mapping sample names to :class:`np.ndarray` instances, If sample
            name is string type, can only contain alpha-numeric ascii
            characters (in addition to '-', '.', '_'). Int key must be >= 0. By
            default, None.
        **kwargs
            keyword arguments provided will be saved with keywords as sample keys
            (string type only) and values as np.array instances.


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

            if other:
                if not isinstance(other, dict):
                    other = dict(other)
                else:
                    other = other.copy()
            elif other is None:
                other = {}
            if kwargs:
                # we have to merge kwargs dict with `other` before operating on
                # either so all validation and writing occur atomically
                other.update(kwargs)

            for key, val in other.items():
                self._set_arg_validate(key, val)
            for key, val in other.items():
                self._perform_set(key, val)

    @writer_checkout_only
    def __delitem__(self, key: KeyType) -> None:
        """Remove a sample from the arrayset. Convenience method to :meth:`delete`.

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

            if key not in self._samples:
                raise KeyError(key)

            dataKey = data_record_db_key_from_raw_key(self._asetn, key)
            isRecordDeleted = self._txnctx.dataTxn.delete(dataKey)
            if isRecordDeleted is False:
                raise RuntimeError(
                    f'Internal error. Not able to delete key {key} from staging '
                    f'db even though existance passed in memory verification. '
                    f'Please report this message in full to the hangar development team.',
                    f'Specified key: <{type(key)} {key}>', f'Calculated dataKey: <{dataKey}>',
                    f'isRecordDeleted: <{isRecordDeleted}>', f'DEBUG STRING: {self._debug_}')
            del self._samples[key]

    def pop(self, key: KeyType) -> np.ndarray:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        key : KeysType
            Sample key to remove

        Returns
        -------
        :class:`np.ndarray`
            Upon success, the value of the removed key.

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        value = self[key]
        del self[key]
        return value

    @writer_checkout_only
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
            inferred. If dict, key ``backend`` must have a valid backend format
            code value, and the rest of the items are assumed to be valid specs
            for that particular backend. If none, both backend and opts are
            inferred from the array prototype

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
                                         variable_shape=self.variable_shape)

        # ----------- Determine schema format details -------------------------

        schema_hash = schema_hash_digest(shape=proto.shape,
                                         size=proto.size,
                                         dtype_num=proto.dtype.num,
                                         variable_shape=self.variable_shape,
                                         contains_subsamples=self.contains_subsamples,
                                         backend_code=beopts.backend,
                                         backend_opts=beopts.opts)
        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(self.arrayset)
        asetSchemaVal = arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=self.variable_shape,
            schema_max_shape=proto.shape,
            schema_dtype=proto.dtype.num,
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
        self._datavalidator.schema = self._schema_spec
        return
