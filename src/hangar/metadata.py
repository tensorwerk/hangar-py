import os
from typing import Optional, Union, Iterator, Tuple, Dict

import lmdb

from .context import TxnRegister
from .records import parsing
from .records.queries import RecordQuery
from .records.hashmachine import metadata_hash_digest
from .utils import is_suitable_user_key, is_ascii


class MetadataReader(object):
    """Class implementing get access to the metadata in a repository.

    Unlike the :class:`~.arrayset.ArraysetDataReader` and
    :class:`~.arrayset.ArraysetDataWriter`, the equivalent Metadata classes do
    not need a factory function or class to coordinate access through the
    checkout. This is primarily because the metadata is only stored at a single
    level, and because the long term storage is must simpler than for array
    data (just write to a lmdb database).

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
                 repo_pth: os.PathLike,
                 dataenv: lmdb.Environment,
                 labelenv: lmdb.Environment,
                 *args, **kwargs):
        """Developer documentation for init method.

        Parameters
        ----------
        mode : str
            'r' for read-only, 'a' for write-enabled
        repo_pth : os.PathLike
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
        self._is_conman: bool = False
        self._labelenv: lmdb.Environment = labelenv
        self._labelTxn: Optional[lmdb.Transaction] = None
        self._TxnRegister = TxnRegister()

        self._mspecs: Dict[Union[str, int], bytes] = {}
        metaNamesSpec = RecordQuery(dataenv).metadata_records()
        for metaNames, metaSpec in metaNamesSpec:
            labelKey = parsing.hash_meta_db_key_from_raw_key(metaSpec.meta_hash)
            self._mspecs[metaNames.meta_name] = labelKey

    def __enter__(self):
        self._is_conman = True
        self._labelTxn = self._TxnRegister.begin_reader_txn(self._labelenv)
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._labelTxn = self._TxnRegister.abort_reader_txn(self._labelenv)

    def __len__(self) -> int:
        """Determine how many metadata key/value pairs are in the checkout

        Returns
        -------
        int
            number of metadata key/value pairs.
        """
        return len(self._mspecs)

    def __getitem__(self, key: Union[str, int]) -> str:
        """Retrieve a metadata sample with a key. Convenience method for dict style access.

        .. seealso:: :meth:`get`

        Parameters
        ----------
        key : Union[str, int]
            metadata key to retrieve from the checkout

        Returns
        -------
        string
            value of the metadata key/value pair stored in the checkout.
        """
        return self.get(key)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Determine if a key with the provided name is in the metadata

        Parameters
        ----------
        key : Union[str, int]
            key to check for containment testing

        Returns
        -------
        bool
            True if key exists, False otherwise
        """
        if key in self._mspecs:
            return True
        else:
            return False

    def __iter__(self) -> Iterator[Union[int, str]]:
        return self.keys()

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
    def iswriteable(self) -> bool:
        """Read-only attribute indicating if this metadata object is write-enabled.

        Returns
        -------
        bool
            True if write-enabled checkout, Otherwise False.
        """
        return False if self._mode == 'r' else True

    def keys(self) -> Iterator[Union[str, int]]:
        """generator which yields the names of every metadata piece in the checkout.

        For write enabled checkouts, is technically possible to iterate over the
        metadata object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[Union[str, int]]
            keys of one metadata sample at a time
        """
        for name in tuple(self._mspecs.keys()):
            yield name

    def values(self) -> Iterator[str]:
        """generator yielding all metadata values in the checkout

        For write enabled checkouts, is technically possible to iterate over the
        metadata object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[str]
            values of one metadata piece at a time
        """
        for name in tuple(self._mspecs.keys()):
            yield self.get(name)

    def items(self) -> Iterator[Tuple[Union[str, int], str]]:
        """generator yielding key/value for all metadata recorded in checkout.

        For write enabled checkouts, is technically possible to iterate over the
        metadata object while adding/deleting data, in order to avoid internal
        python runtime errors (``dictionary changed size during iteration`` we
        have to make a copy of they key list before beginning the loop.) While
        not necessary for read checkouts, we perform the same operation for both
        read and write checkouts in order in order to avoid differences.

        Yields
        ------
        Iterator[Tuple[Union[str, int], np.ndarray]]
            metadata key and stored value for every piece in the checkout.
        """
        for name in tuple(self._mspecs.keys()):
            yield (name, self.get(name))

    def get(self, key: Union[str, int]) -> str:
        """retrieve a piece of metadata from the checkout.

        Parameters
        ----------
        key : Union[str, int]
            The name of the metadata piece to retrieve.

        Returns
        -------
        str
            The stored metadata value associated with the key.

        Raises
        ------
        ValueError
            If the `key` is not str type or contains whitespace or non
            alpha-numeric characters.
        KeyError
            If no metadata exists in the checkout with the provided key.
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()
            metaVal = self._labelTxn.get(self._mspecs[key])
            meta_val = parsing.hash_meta_raw_val_from_db_val(metaVal)
        except KeyError:
            raise KeyError(f'The checkout does not contain metadata with key: {key}')
        finally:
            if tmpconman:
                self.__exit__()
        return meta_val


class MetadataWriter(MetadataReader):
    """Class implementing write access to repository metadata.

    Similar to the :class:`~.arrayset.ArraysetDataWriter`, this class
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
        self._is_conman = True
        self._labelTxn = self._TxnRegister.begin_writer_txn(self._labelenv)
        self._dataTxn = self._TxnRegister.begin_writer_txn(self._dataenv)
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._labelTxn = self._TxnRegister.commit_writer_txn(self._labelenv)
        self._dataTxn = self._TxnRegister.commit_writer_txn(self._dataenv)

    def __setitem__(self, key: Union[str, int], value: str) -> Union[str, int]:
        """Store a key/value pair as metadata. Convenience method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : Union[str, int]
            name of the key to add as metadata
        value : string
            value to add as metadata

        Returns
        -------
        Union[str, int]
            key of the stored metadata sample (assuming operation was successful)
        """
        return self.add(key, value)

    def __delitem__(self, key: Union[str, int]) -> Union[str, int]:
        """Remove a key/value pair from metadata. Convenience method to :meth:`remove`.

        .. seealso:: :meth:`remove` for the function this calls into.

        Parameters
        ----------
        key : Union[str, int]
            Name of the metadata piece to remove.

        Returns
        -------
        Union[str, int]
            Metadata key removed from the checkout (assuming operation successful)
        """
        return self.remove(key)

    def add(self, key: Union[str, int], value: str) -> Union[str, int]:
        """Add a piece of metadata to the staging area of the next commit.

        Parameters
        ----------
        key : Union[str, int]
            Name of the metadata piece, alphanumeric ascii characters only.
        value : string
            Metadata value to store in the repository, any length of valid
            ascii characters.

        Returns
        -------
        Union[str, int]
            The name of the metadata key written to the database if the
            operation succeeded.

        Raises
        ------
        ValueError
            If the `key` contains any whitespace or non alpha-numeric
            characters or is longer than 64 characters.
        ValueError
            If the `value` contains any non ascii characters.
        """
        try:
            if not is_suitable_user_key(key):
                raise ValueError(
                    f'Metadata key: {key} of type: {type(key)} invalid. Must be int '
                    f'ascii string with only alpha-numeric / "." "_" "-" characters. '
                    f'Must be <= 64 characters long.')
            elif not (isinstance(value, str) and is_ascii(value)):
                raise ValueError(
                    f'Metadata Value: `{value}` not allowed. Must be ascii-only string')
        except ValueError as e:
            raise e from None

        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            val_hash = metadata_hash_digest(value=value)
            hashKey = parsing.hash_meta_db_key_from_raw_key(val_hash)
            metaRecKey = parsing.metadata_record_db_key_from_raw_key(key)
            metaRecVal = parsing.metadata_record_db_val_from_raw_val(val_hash)
            # check if meta record already exists with same key
            existingMetaRecVal = self._dataTxn.get(metaRecKey, default=False)
            if existingMetaRecVal:
                existingMetaRec = parsing.metadata_record_raw_val_from_db_val(existingMetaRecVal)
                # check if meta record already exists with same key/val
                if val_hash == existingMetaRec.meta_hash:
                    return key

            # write new data if label hash does not exist
            existingHashVal = self._labelTxn.get(hashKey, default=False)
            if existingHashVal is False:
                hashVal = parsing.hash_meta_db_val_from_raw_val(value)
                self._labelTxn.put(hashKey, hashVal)

            self._dataTxn.put(metaRecKey, metaRecVal)
            self._mspecs[key] = hashKey

        finally:
            if tmpconman:
                self.__exit__()
        return key

    def remove(self, key: Union[str, int]) -> Union[str, int]:
        """Remove a piece of metadata from the staging area of the next commit.

        Parameters
        ----------
        key : Union[str, int]
            Metadata name to remove.

        Returns
        -------
        Union[str, int]
            Name of the metadata key/value pair removed, if the operation was
            successful.

        Raises
        ------
        KeyError
            If the checkout does not contain metadata with the provided key.
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()
            if key not in self._mspecs:
                raise KeyError(f'No metadata exists with key: {key}')

            metaRecKey = parsing.metadata_record_db_key_from_raw_key(key)
            delete_succeeded = self._dataTxn.delete(metaRecKey)
            if delete_succeeded is False:
                raise KeyError(f'No metadata exists with key: {key}')
            del self._mspecs[key]

        except (KeyError, ValueError) as e:
            raise e from None
        finally:
            if tmpconman:
                self.__exit__()
        return key
