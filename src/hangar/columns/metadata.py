from contextlib import ExitStack
from pathlib import Path
from typing import Optional, Union, Iterator, Tuple, Dict, Mapping, List, Sequence, Any, Iterable

import lmdb

from ..records.hashmachine import metadata_hash_digest
from ..records.parsing import (
    hash_meta_db_key_from_raw_key,
    hash_meta_db_val_from_raw_val,
    hash_meta_raw_val_from_db_val,
    metadata_record_db_key_from_raw_key,
    metadata_record_db_val_from_raw_val,
    metadata_record_raw_val_from_db_val,
    generate_sample_name,
)
from ..records.queries import RecordQuery
from ..txnctx import TxnRegister
from ..utils import is_suitable_user_key, is_ascii


KeyTypes = Union[str, int]
KeyValMap = Mapping[KeyTypes, str]
KeyValType = Union[Tuple[KeyTypes, str], List[Union[KeyTypes, str]]]
MapKeyValType = Union[KeyValMap, Sequence[KeyValType]]


class MetadataReader(object):
    """Class implementing get access to the metadata in a repository.

    Unlike the :class:`~.columns.arrayset.Arraysets` and the equivalent
    Metadata classes do not need a factory function or class to coordinate
    access through the checkout. This is primarily because the metadata is
    only stored at a single level, and because the long term storage is
    must simpler than for array data (just write to a lmdb database).

    .. note::

        It is important to realize that this is not intended to serve as a general
        store large amounts of textual data, and has no optimization to support
        such use cases at this time. This should only serve to attach helpful
        labels, or other quick information primarily intended for human
        book-keeping, to the main tensor data!

    .. note::

        Write-enabled metadata objects are not thread or process safe. Read-only
        checkouts can use multithreading safety to retrieve data via the standard
        :py:meth:`.MetadataReader.get` calls

    """

    def __init__(self,
                 mode: str,
                 repo_pth: Path,
                 dataenv: lmdb.Environment,
                 labelenv: lmdb.Environment,
                 *args, **kwargs):
        """Developer documentation for init method.

        Parameters
        ----------
        mode : str
            'r' for read-only, 'a' for write-enabled
        repo_pth : Path
            path to the repository on disk.
        dataenv : lmdb.Environment
            the lmdb environment in which the data records are stored. this is
            the same as the arrayset data record environments.
        labelenv : lmdb.Environment
            the lmdb environment in which the label hash key / values are stored
            permanently. When opened in by this reader instance, no write access
            is allowed.
        """
        self._mode = mode
        self._path = repo_pth
        self._labelenv: lmdb.Environment = labelenv
        self._labelTxn: Optional[lmdb.Transaction] = None
        self._TxnRegister = TxnRegister()

        self._stack: Optional[ExitStack] = None
        self._enter_count: int = 0

        self._mspecs: Dict[KeyTypes, bytes] = {}
        metaNamesSpec = RecordQuery(dataenv).metadata_records()
        for metaNames, metaSpec in metaNamesSpec:
            labelKey = hash_meta_db_key_from_raw_key(metaSpec.meta_hash)
            self._mspecs[metaNames.meta_name] = labelKey

    def __enter__(self):
        with ExitStack() as stack:
            self._enter_count += 1
            self._labelTxn = self._TxnRegister.begin_reader_txn(self._labelenv)
            stack.callback(self._TxnRegister.abort_reader_txn, self._labelenv)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._stack = None
        self._enter_count -= 1

    def __len__(self) -> int:
        """Determine how many metadata key/value pairs are in the checkout

        Returns
        -------
        int
            number of metadata key/value pairs.
        """
        return len(self._mspecs)

    def __contains__(self, key: KeyTypes) -> bool:
        """Determine if a key with the provided name is in the metadata

        Parameters
        ----------
        key : KeyTypes
            key to check for containment testing

        Returns
        -------
        bool
            True if key exists, False otherwise
        """
        return True if key in self._mspecs else False

    def __iter__(self) -> Iterator[KeyTypes]:
        yield from self.keys()

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar Metadata\
                \n    Writeable: {self.iswriteable}\
                \n    Number of Keys: {self.__len__()}\n'
        p.text(res)

    def __repr__(self):
        res = f'Hangar Metadata\
                \n    Writeable: {self.iswriteable}\
                \n    Number of Keys: {self.__len__()}\n'
        return res

    @property
    def _is_conman(self):
        return bool(self._enter_count)

    @property
    def iswriteable(self) -> bool:
        """Read-only attribute indicating if this metadata object is write-enabled.

        Returns
        -------
        bool
            True if write-enabled checkout, Otherwise False.
        """
        return False if self._mode == 'r' else True

    def __getitem__(self, key: KeyTypes) -> str:
        """Retrieve a metadata sample with a key. Convenience method for dict style access.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : KeyTypes
            metadata key to retrieve from the checkout

        Returns
        -------
        string
            value of the metadata key/value pair stored in the checkout.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            metaVal = self._labelTxn.get(self._mspecs[key])
            meta_val = hash_meta_raw_val_from_db_val(metaVal)
            return meta_val

    def get(self, key: KeyTypes, default: Optional[Any] = None) -> str:
        """retrieve a piece of metadata from the checkout.

        Parameters
        ----------
        key : KeyTypes
            The name of the metadata piece to retrieve.
        default : Optional[Any]
            If key is not found in metadata records, returns this value instead
            of raising an exception like :meth:`__getitem__` does.

        Returns
        -------
        str
            The stored metadata value associated with the key.
        """
        try:
            return self[key]
        except KeyError:
            return default

    def _mode_aware_key_looper(self) -> Iterable[KeyTypes]:
        """Generate keys for iteration with dict update safety ensured.

        Returns
        -------
        Iterable[KeyTypes]
            metadata keys
        """
        if self._mode == 'r':
            yield from self._mspecs.keys()
        else:
            yield from tuple(self._mspecs.keys())

    def keys(self) -> Iterable[KeyTypes]:
        """generator which yields the names of every metadata piece in the checkout.

        If accessed by a write-enabled checkout, runtime safety is guarrenteed
        even if items are added / removed / modified while iterating over
        results of this method. (Also true for read-only checkouts.)

        Yields
        ------
        Iterable[KeyTypes]
            keys of one metadata sample at a time
        """
        yield from self._mode_aware_key_looper()

    def values(self) -> Iterable[str]:
        """generator yielding all metadata values in the checkout

        If accessed by a write-enabled checkout, runtime safety is guarrenteed
        even if items are added / removed / modified while iterating over
        results of this method. (Also true for read-only checkouts.)

        Yields
        ------
        Iterable[str]
            values of one metadata piece at a time
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            for key in self._mode_aware_key_looper():
                yield self[key]

    def items(self) -> Iterable[Tuple[KeyTypes, str]]:
        """generator yielding key/value for all metadata recorded in checkout.

        If accessed by a write-enabled checkout, runtime safety is guarrenteed
        even if items are added / removed / modified while iterating over
        results of this method. (Also true for read-only checkouts.)

        Yields
        ------
        Iterable[Tuple[KeyTypes, np.ndarray]]
            metadata key and stored value for every piece in the checkout.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            for key in self._mode_aware_key_looper():
                yield (key, self[key])

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)

    def __getattr__(self, name):
        """Raise permission error after checkout is closed.

         Only runs after a call to :meth:`_destruct`, which is responsible
         for deleting all attributes from the object instance.
        """
        try:
            self.__getattribute__('_mode')  # once checkout is closed, this won't exist.
        except AttributeError:
            err = (f'Unable to operate on past checkout objects which have been '
                   f'closed. No operation occurred. Please use a new checkout.')
            raise PermissionError(err) from None
        return self.__getattribute__(name)


class MetadataWriter(MetadataReader):
    """Class implementing write access to repository metadata.

    Similar to the :class:`~.columns.arrayset.ArraysetDataWriter`, this class
    inherits the functionality of the :class:`~.metadata.MetadataReader` for reading. The
    only difference is that the reader will be initialized with data records
    pointing to the staging area, and not a commit which is checked out.

    .. note::

       Write-enabled metadata objects are not thread or process safe. Read-only
       checkouts can use multithreading safety to retrieve data via the
       standard :py:meth:`.MetadataReader.get` calls

    .. seealso::

        :class:`.MetadataReader` for the intended use of this functionality.
    """

    def __init__(self, *args, **kwargs):
        """Developer documentation of init method

        Parameters
        ----------
        *args
            Arguments passed to :class:`MetadataReader`
        **kwargs
            KeyWord arguments passed to :class:`MetadataReader`
        """

        super().__init__(*args, **kwargs)
        self._dataenv: lmdb.Environment = kwargs['dataenv']
        self._dataTxn: Optional[lmdb.Transaction] = None

    def __enter__(self):
        with ExitStack() as stack:
            self._enter_count += 1
            self._labelTxn = self._TxnRegister.begin_writer_txn(self._labelenv)
            stack.callback(self._TxnRegister.commit_writer_txn, self._labelenv)
            self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)
            stack.callback(self._TxnRegister.commit_writer_txn, self._dataenv)
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._stack = None
        self._enter_count -= 1

    def __set_arg_validate(self, key: KeyTypes, val: str) -> None:
        """Verify user provided key/value pair is valid, raising ValueError if problem.
        """
        if not is_suitable_user_key(key):
            raise ValueError(
                f'Metadata key {key} of type {type(key)} invalid. Must be int '
                f'ascii string with only alpha-numeric / "." "_" "-" characters. '
                f'Must be <= 64 characters long.')
        elif not (isinstance(val, str) and is_ascii(val)):
            raise ValueError(f'Metadata Value `{val}` not valid. Must be ascii str')

    def _perform_set(self, key: KeyTypes, val: str):
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            val_hash = metadata_hash_digest(value=val)
            hashKey = hash_meta_db_key_from_raw_key(val_hash)
            metaRecKey = metadata_record_db_key_from_raw_key(key)
            metaRecVal = metadata_record_db_val_from_raw_val(val_hash)
            # check if meta record already exists with same key
            existingMetaRecVal = self._dataTxn.get(metaRecKey, default=False)
            if existingMetaRecVal:
                existingMetaRec = metadata_record_raw_val_from_db_val(existingMetaRecVal)
                # check if meta record already exists with same key/val
                if val_hash == existingMetaRec.meta_hash:
                    return

            # write new data if label hash does not exist
            existingHashVal = self._labelTxn.get(hashKey, default=False)
            if existingHashVal is False:
                hashVal = hash_meta_db_val_from_raw_val(val)
                self._labelTxn.put(hashKey, hashVal)

            self._dataTxn.put(metaRecKey, metaRecVal)
            self._mspecs[key] = hashKey
            return

    def __setitem__(self, key: KeyTypes, value: str) -> None:
        """Store a key/value pair as metadata. Convenience method to :meth:`add`.

        .. seealso:: :meth:`update`

        Parameters
        ----------
        key : KeyTypes
            Name of the metadata piece, alphanumeric ascii characters only.
        value : string
            Metadata value to store in the repository, any length of valid
            ascii characters.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            self.__set_arg_validate(key, value)
            self._perform_set(key, value)

    def update(self, other: Union[None, MapKeyValType] = None, **kwargs) -> None:
        """Store some data with the key/value pairs from other, overwriting existing keys.

        :meth:`update` implements functionality similar to python's builtin
        :meth:`dict.update` method, accepting either a dictionary or other
        iterable (of length two) listing key / value pairs.

        Parameters
        ----------
        other : Union[None, MapKeyValType], optional
            Accepts either another dictionary object or an iterable of
            key/value pairs (as tuples or other iterables of length two).
            mapping sample names to :class:`str` instances, If sample
            name is string type, can only contain alpha-numeric ascii
            characters (in addition to '-', '.', '_'). Int key must be >= 0.
            By default, None.
        **kwargs
            keyword arguments provided will be saved with keywords as sample keys
            (string type only) and values as string instances.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if other and not isinstance(other, dict):
                other = dict(other)
            elif other is None:
                other = {}
            if kwargs:
                # we have to merge kwargs dict with `other` before operating on
                # either so all validation and writing occur atomically
                other.update(kwargs)

            for key, val in other.items():
                self.__set_arg_validate(key, val)
            for key, val in other.items():
                self._perform_set(key, val)

    def append(self, value: str) -> KeyTypes:
        """TODO: is this the right way we should be handling unnamed samples?
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)
            key = generate_sample_name()
            self.__set_arg_validate(key, value)
            self._perform_set(key, value)
            return key

    def __delitem__(self, key: KeyTypes):
        """Remove a key/value pair from metadata.

        .. seealso:: :meth:`pop` for an atomic get value and delete operation

        Parameters
        ----------
        key : KeyTypes
            Name of the metadata piece to remove.

        Raises
        ------
        KeyError
            If no item exists with the specified key
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if key not in self._mspecs:
                raise KeyError(key)
            metaRecKey = metadata_record_db_key_from_raw_key(key)
            delete_succeeded = self._dataTxn.delete(metaRecKey)
            if delete_succeeded is False:
                raise RuntimeError(
                    f'Internal error, could not delete metadata key `{key}` with '
                    f' metaRecKey `{metaRecKey}` from refs db even though in memory '
                    f'verification succeeded. Please report to hangar dev team.')
            del self._mspecs[key]

    def pop(self, key: KeyTypes) -> str:
        """Retrieve some value for some key(s) and delete it in the same operation.

        Parameters
        ----------
        key : KeysType
            Sample key to remove

        Returns
        -------
        str
            Upon success, the value of the removed key.

        Raises
        ------
        KeyError
            If there is no sample with some key in the arrayset.
        """
        value = self[key]
        del self[key]
        return value

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)
