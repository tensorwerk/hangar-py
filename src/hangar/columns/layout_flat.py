"""Accessor class for columns containing single-level key/value mappings

The FlatSampleReader container is used to store data (in any backend) in a column
containing a single level key/value mapping from names/ids to data.

All backends are supported.
"""
from contextlib import ExitStack
from pathlib import Path
from typing import Tuple, Union, Iterable, Optional, Any

from .common import open_file_handles
from ..records import (
    data_record_db_val_from_digest,
    data_record_digest_val_from_db_val,
    flat_data_db_key_from_names,
    hash_data_db_key_from_raw_key,
    schema_db_key_from_column,
    schema_hash_db_key_from_digest,
    schema_hash_record_db_val_from_spec,
    schema_record_db_val_from_digest
)
from ..records.parsing import generate_sample_name
from ..backends import backend_decoder
from ..op_state import reader_checkout_only
from ..utils import is_suitable_user_key, valfilter, valfilterfalse


KeyType = Union[str, int]


class FlatSampleReader:
    """Class implementing get access to data in a column.

    This class exposes the standard API to access data stored in a single level
    key / value mapping column. Usage is modeled after the python :class:`dict`
    style syntax -- with a few additional utility and inspection methods and
    properties added. Methods named after those of a python :class:`dict` have
    syntactically identical arguments and behavior to that of the standard
    library.

    If not opened in a ``write-enabled`` checkout, then attempts to add or
    delete data or container properties will raise an exception (in the form of
    a :class:`PermissionError`). No changes will be propogated unless a
    ``write-enabled`` checkout is used.

    This object can be serialized -- pickled -- for parallel processing /
    reading if opened in a ``read-only`` checkout. Parallel operations are both
    thread and process safe, though performance may significantly differ
    between multithreaded vs multiprocessed code (depending on the backend data
    is stored in). Attempts to serialize objects opened in ``write-enabled``
    checkouts are not supported and will raise a :class:`PermissionError` if
    attempted. This behavior is enforced in order to ensure data and record
    integrity while writing to the repository.
    """

    __slots__ = ('_mode', '_column_name', '_samples', '_be_fs',
                 '_path', '_stack', '_enter_count', '_schema')
    _attrs = __slots__

    def __init__(self,
                 columnname: str,
                 samples,
                 backend_handles,
                 schema,
                 repo_path: Path,
                 mode: str,
                 *args, **kwargs):

        self._stack: Optional[ExitStack] = None
        self._mode = mode
        self._column_name = columnname
        self._samples = samples
        self._be_fs = backend_handles
        self._path = repo_path
        self._schema = schema
        self._enter_count = 0

    @property
    def _debug_(self):  # pragma: no cover
        return {
            '__class__': self.__class__,
            '_mode': self._mode,
            '_column_name': self._column_name,
            '_be_fs': self._be_fs,
            '_path': self._path,
            '_contains_subsamples': self.contains_subsamples,
            '_stack': self._stack._exit_callbacks if self._stack else self._stack,
            '_enter_count': self._enter_count,
        }

    def __repr__(self):
        res = (
            f'{self.__class__.__qualname__}('
            f'repo_pth={self._path}, '
            f'aset_name={self._column_name}, '
            f"{[f'{key}={val}, ' for key, val in self._schema.schema.items()]}, "
            f'mode={self._mode})')
        return res

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__qualname__} \
                \n    Column Name              : {self._column_name}\
                \n    Writeable                : {self.iswriteable}\
                \n    Column Type              : {self.column_type}\
                \n    Column Layout            : {self.column_layout}\
                \n    Schema Type              : {self.schema_type}\
                \n    DType                    : {self.dtype}\
                \n    Shape                    : {self.shape}\
                \n    Number of Samples        : {self.__len__()}\
                \n    Partial Remote Data Refs : {bool(self.contains_remote_references)}\n'
        p.text(res)

    def _ipython_key_completions_(self):  # pragma: no cover
        """Let ipython know that any key based access can use the column keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the column keys for access.
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
        return self

    def __exit__(self, *exc):
        return

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
        """Create iterator yielding an column sample keys.

        Yields
        -------
        Iterable[KeyType]
            Sample key contained in the column.
        """
        yield from self.keys()

    def __len__(self) -> int:
        """Check how many samples are present in a given column.
        """
        return len(self._samples)

    def __contains__(self, key: KeyType) -> bool:
        """Determine if a key is a valid sample name in the column.
        """
        return key in self._samples

    def _open(self):
        for val in self._be_fs.values():
            val.open(mode=self._mode)

    def _close(self):
        for val in self._be_fs.values():
            val.close()

    def __getitem__(self, key: KeyType):
        """Retrieve data for some sample key via dict style access conventions.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : KeyType
            Sample key to retrieve from the column.

        Returns
        -------
        value
            Data corresponding to the provided sample key.

        Raises
        ------
        KeyError
            if no sample with the requested key exists.
        """
        spec = self._samples[key]
        return self._be_fs[spec.backend].read_data(spec)

    def get(self, key: KeyType, default=None):
        """Retrieve the data associated with some sample key

        Parameters
        ----------
        key : KeyType
            The name of the subsample(s) to retrieve. Passing a single
            subsample key will return the stored data value.
        default : Any
            if a `key` parameter is not found, then return this value instead.
            By default, None.

        Returns
        -------
        value
            data data stored under subsample key if key exists, else
            default value if not found.
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    def column(self) -> str:
        """Name of the column.
        """
        return self._column_name

    @property
    def column_type(self):
        """Data container type of the column ('ndarray', 'str', etc).
        """
        return self._schema.column_type

    @property
    def column_layout(self):
        """Column layout type ('nested', 'flat', etc).
        """
        return self._schema.column_layout

    @property
    def schema_type(self):
        """Schema type of the contained data ('variable_shape', 'fixed_shape', etc).
        """
        return self._schema.schema_type

    @property
    def dtype(self):
        """Dtype of the columns data (np.float, str, etc).
        """
        return self._schema.dtype

    @property
    def shape(self):
        """(Max) shape of data that can (is) written in the column.
        """
        try:
            return self._schema.shape
        except AttributeError:
            return None

    @property
    def backend(self) -> str:
        """Code indicating which backing store is used when writing data.
        """
        return self._schema.backend

    @property
    def backend_options(self):
        """Filter / Compression options applied to backend when writing data.
        """
        return self._schema.backend_options

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this column object is write-enabled.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_subsamples(self) -> bool:
        """Bool indicating if sub-samples are contained in this column container.
        """
        return False

    @property
    def contains_remote_references(self) -> bool:
        """Bool indicating if all samples in column exist on local disk.

        The data associated with samples referencing some remote server will
        need to be downloaded (``fetched`` in the hangar vocabulary) before
        they can be read into memory.

        Returns
        -------
        bool
            False if at least one sample in the column references data stored
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
            list of sample keys in the column whose data references indicate
            they are stored on a remote server.
        """
        return tuple(valfilterfalse(lambda x: x.islocal, self._samples).keys())

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

    def values(self, local: bool = False) -> Iterable[Any]:
        """Generator yielding the data for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned values will only correspond to data which is
            available for reading on the local disk. No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Any]
            Values of one subsample at a time inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield self[key]

    def items(self, local: bool = False) -> Iterable[Tuple[KeyType, Any]]:
        """Generator yielding (name, data) tuple for every subsample.

        Parameters
        ----------
        local : bool, optional
            If True, returned keys/values will only correspond to data which is
            available for reading on the local disk, No attempt will be made to
            read data existing on a remote server, by default False.

        Yields
        ------
        Iterable[Tuple[KeyType, Any]]
            Name and stored value for every subsample inside the sample.
        """
        for key in self._mode_local_aware_key_looper(local):
            yield (key, self[key])

# ---------------- writer methods only after this point -------------------


class FlatSampleWriter(FlatSampleReader):

    __slots__ = ('_txnctx',)
    _attrs = __slots__ + FlatSampleReader.__slots__

    def __init__(self, aset_ctx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._txnctx = aset_ctx

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

    def _set_arg_validate(self, key, value):
        """Verify if key / value pair is valid to be written in this column

        Parameters
        ----------
        key
            name to associate with this data piece
        value
            piece of data to store in the column

        Raises
        ------
        ValueError
            If key is not valid type/contents or if value is not correct object
            type / if it does not conform to column schema
        """
        if not is_suitable_user_key(key):
            raise ValueError(f'Sample name `{key}` is not suitable.')

        isCompat = self._schema.verify_data_compatible(value)
        if not isCompat.compatible:
            raise ValueError(isCompat.reason)

    def _perform_set(self, key, value):
        """Internal write method. Assumes all arguments validated and context is open

        Parameters
        ----------
        key
            sample key to store
        value
            data to store
        """
        full_hash = self._schema.data_hash_digest(value)

        hashKey = hash_data_db_key_from_raw_key(full_hash)
        # check if data record already exists with given key
        dataRecKey = flat_data_db_key_from_names(self._column_name, key)
        existingDataRecVal = self._txnctx.dataTxn.get(dataRecKey, default=False)
        if existingDataRecVal:
            # check if data record already with same key & hash value
            existingDataRec = data_record_digest_val_from_db_val(existingDataRecVal)
            if full_hash == existingDataRec.digest:
                return

        # write new data if data hash does not exist
        existingHashVal = self._txnctx.hashTxn.get(hashKey, default=False)
        if existingHashVal is False:
            hashVal = self._be_fs[self._schema.backend].write_data(value)
            self._txnctx.hashTxn.put(hashKey, hashVal)
            self._txnctx.stageHashTxn.put(hashKey, hashVal)
            hash_spec = backend_decoder(hashVal)
        else:
            hash_spec = backend_decoder(existingHashVal)
            if hash_spec.backend not in self._be_fs:
                # when adding data which is already stored in the repository, the
                # backing store for the existing data location spec may not be the
                # same as the backend which the data piece would have been saved in here.
                #
                # As only the backends actually referenced by a columns samples are
                # initialized (accessible by the column), there is no guarantee that
                # an accessor exists for such a sample. In order to prevent internal
                # errors from occurring due to an uninitialized backend if a previously
                # existing data piece is "saved" here and subsequently read back from
                # the same writer checkout, we perform an existence check and backend
                # initialization, if appropriate.
                fh = open_file_handles(backends=(hash_spec.backend,),
                                       path=self._path,
                                       mode='a',
                                       schema=self._schema)
                self._be_fs[hash_spec.backend] = fh[hash_spec.backend]

        # add the record to the db
        dataRecVal = data_record_db_val_from_digest(full_hash)
        self._txnctx.dataTxn.put(dataRecKey, dataRecVal)
        self._samples[key] = hash_spec

    def __setitem__(self, key, value):
        """Store a piece of data in a column.

        .. seealso::

            :meth:`update` for an implementation analogous to python's built in
            :meth:`dict.update` method which accepts a dict or iterable of
            key/value pairs to add in the same operation.

        Parameters
        ----------
        key
            name to assign to the sample (assuming the column accepts named
            samples), If str, can only contain alpha-numeric ascii characters
            (in addition to '-', '.', '_'). Integer key must be >= 0. by
            default, None
        value
            data to store as a sample in the column.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self._set_arg_validate(key, value)
            self._perform_set(key, value)

    def append(self, value) -> KeyType:
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
        value
            Piece of data to store in the column.

        Returns
        -------
        KeyType
            Name of the generated key this data is stored with.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            key = generate_sample_name()
            while key in self._samples:
                key = generate_sample_name()
            self._set_arg_validate(key, value)
            self._perform_set(key, value)
            return key

    def update(self, other=None, **kwargs):
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other
            Accepts either another dictionary object or an iterable of
            key/value pairs (as tuples or other iterables of length two).
            mapping sample names to data value instances instances, If sample
            name is string type, can only contain alpha-numeric ascii
            characters (in addition to '-', '.', '_'). Int key must be >= 0. By
            default, None.
        **kwargs
            keyword arguments provided will be saved with keywords as sample keys
            (string type only) and values as np.array instances.
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

    def __delitem__(self, key: KeyType) -> None:
        """Remove a sample from the column. Convenience method to :meth:`delete`.

        .. seealso::

            :meth:`pop` to return a value and then delete it in the same operation

        Parameters
        ----------
        key : KeyType
            Name of the sample to remove from the column.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if key not in self._samples:
                raise KeyError(key)

            dataKey = flat_data_db_key_from_names(self._column_name, key)
            isRecordDeleted = self._txnctx.dataTxn.delete(dataKey)
            if isRecordDeleted is False:
                raise RuntimeError(
                    f'Internal error. Not able to delete key {key} from staging '
                    f'db even though existance passed in memory verification. '
                    f'Please report this message in full to the hangar development team.',
                    f'Specified key: <{type(key)} {key}>', f'Calculated dataKey: <{dataKey}>',
                    f'isRecordDeleted: <{isRecordDeleted}>', f'DEBUG STRING: {self._debug_}')
            del self._samples[key]

    def pop(self, key: KeyType):
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        key : KeysType
            Sample key to remove

        Returns
        -------
        value
            Upon success, the value of the removed key.

        Raises
        ------
        KeyError
            If there is no sample with some key in the column.
        """
        value = self[key]
        del self[key]
        return value

    def change_backend(self, backend: str, backend_options: Optional[dict] = None):
        """Change the default backend and filters applied to future data writes.

        .. warning::

           This method is meant for advanced users only. Please refer to the
           hangar backend codebase for information on accepted parameters and
           options.

        Parameters
        ----------
        backend : str
            Backend format code to swtich to.
        backend_options : Optional[dict]
            Backend option specification to use (if specified). If left to
            default value of None, then default options for backend are
            automatically used.

        Raises
        ------
        RuntimeError
            If this method was called while this column is invoked in a
            context manager
        ValueError
            If the backend format code is not valid.
        """
        if self._is_conman:
            raise RuntimeError('Cannot call method inside column context manager.')

        self._schema.change_backend(backend, backend_options=backend_options)

        new_schema_digest = self._schema.schema_hash_digest()
        columnSchemaKey = schema_db_key_from_column(self._column_name, layout=self.column_layout)
        columnSchemaVal = schema_record_db_val_from_digest(new_schema_digest)
        hashSchemaKey = schema_hash_db_key_from_digest(new_schema_digest)
        hashSchemaVal = schema_hash_record_db_val_from_spec(self._schema.schema)

        # -------- set vals in lmdb only after schema is sure to exist --------

        with self._txnctx.write() as ctx:
            ctx.dataTxn.put(columnSchemaKey, columnSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)

        new_backend = self._schema.backend
        if new_backend not in self._be_fs:
            fhands = open_file_handles(
                backends=[new_backend],
                path=self._path,
                mode='a',
                schema=self._schema)
            self._be_fs[new_backend] = fhands[new_backend]
        else:
            self._be_fs[new_backend].close()
        self._be_fs[new_backend].open(mode='a')
        self._be_fs[new_backend].backend_opts = self._schema.backend_options
        return
