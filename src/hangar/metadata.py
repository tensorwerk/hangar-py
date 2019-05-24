import hashlib
from typing import Optional
import logging

import lmdb

from .context import TxnRegister
from .records import parsing
from .records.queries import RecordQuery
from .utils import is_ascii_alnum, is_ascii

logger = logging.getLogger(__name__)


class MetadataReader(object):
    '''Class implementing get access to the metadata in a repository.

    Unlike the :class:`hangar.dataset.DatasetDataReader` and
    :class:`hangar.dataset.DatasetDataWriter`, the equivalent Metadata classes
    do not need a factory function or class to coordinate access through the
    checkout. This is primarily because the metadata is only stored at a single
    level, and because the long term storage is must simpler than for array data
    (just write to a lmdb database).

    .. note::

        It is important to realize that this is not intended to serve as a general
        store large amounts of textual data, and has no optimization to support such
        use cases at this time. This should only serve to attach helpful labels, or
        other quick information primarily intdented for human book-keeping, to the
        main tensor data!

    Parameters
    ----------
    dataenv : lmdb.Environment
        the lmdb enviornment in which the data records are stored. this is
        the same as the dataset data record environments.
    labelenv : lmdb.Environment
        the lmdb envirionment in which the label hash key / values are stored
        permanently. When opened in by this reader instance, no write access
        is allowed.
    '''

    def __init__(self, dataenv: lmdb.Environment, labelenv: lmdb.Environment):

        self._dataenv: lmdb.Environment = dataenv
        self._labelenv: lmdb.Environment = labelenv
        self._labelTxn: Optional[lmdb.Transaction] = None
        self._dataTxn: Optional[lmdb.Transaction] = None

        self._Query = RecordQuery(self._dataenv)

        self._is_writeable: bool = False
        self.__is_conman: bool = False

    def __enter__(self):
        self.__is_conman = True
        self._labelTxn = TxnRegister().begin_reader_txn(self._labelenv)
        self._dataTxn = TxnRegister().begin_reader_txn(self._dataenv)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__is_conman = False
        self._labelTxn = TxnRegister().abort_reader_txn(self._labelenv)
        self._dataTxn = TxnRegister().abort_reader_txn(self._dataenv)

    def __len__(self):
        '''Determine how many metadata key/value pairs are in the checkout

        Returns
        -------
        int
            number of metadata key/value pairs.
        '''
        return len(self._Query.metadata_names())

    def __getitem__(self, key):
        '''Retrieve a metadata sample with a key. Convenience method for dict style access.

        .. seealso:: :meth:`get()`

        Parameters
        ----------
        key : string
            metadata key to retrieve from the dataset

        Returns
        -------
        string
            value of the metadata key/value pair stored in the checkout.
        '''
        return self.get(key)

    def __contains__(self, key):
        '''Determine if a key with the provided name is in the metadata

        Parameters
        ----------
        key : string
            key to check for containment testing

        Returns
        -------
        bool
            True if key exists, False otherwise
        '''
        names = self._Query.metadata_names()
        if key in names:
            return True
        else:
            return False

    def __iter__(self):
        return self.keys()

    def _repr_pretty_(self, p, cycle):
        res = f'\n Hangar Metadata\
                \n     Writeable: {self.iswriteable}\
                \n     Number of Keys: {len(self)}\n'
        p.text(res)

    def __repr__(self):
        res = f'\n Hangar Metadata\
                \n     Writeable: {self.iswriteable}\
                \n     Number of Keys: {len(self)}\n'
        return res

    @property
    def iswriteable(self):
        '''Bool indicating if this metadata object is write-enabled. Read-only attribute.
        '''
        return self._is_writeable

    def keys(self):
        '''generator returning all metadata key names in the checkout
        '''
        names = self._Query.metadata_names()
        for name in names:
            yield name

    def values(self):
        '''generator returning all metadata values in the checkout
        '''
        names = self._Query.metadata_names()
        for name in names:
            yield self.get(name)

    def items(self):
        '''generator returning all key/value pairs in the checkout.
        '''
        names = self._Query.metadata_names()
        for name in names:
            yield (name, self.get(name))

    def get(self, key):
        '''retrieve a piece of metadata from the checkout.

        Parameters
        ----------
        key : string
            The name of the metadata piece to retrieve.

        Returns
        -------
        string
            The stored metadata value associated with the key.

        Raises
        ------
        ValueError
            If the `key` is not str type or contains whitespace or non
            alpha-numeric characters.
        KeyError
            If no metadata exists in the checkout with the provided key.
        '''
        if not self.__is_conman:
            self._labelTxn = TxnRegister().begin_reader_txn(self._labelenv)
            self._dataTxn = TxnRegister().begin_reader_txn(self._dataenv)

        try:
            if not (isinstance(key, str) and is_ascii_alnum(key)):
                msg = f'HANGAR VALUE ERROR:: metadata key: `{key}` not allowed. '\
                      f'Must be str with only alpha-numeric ascii (no whitespace) chars.'
                raise ValueError(msg)

            refKey = parsing.metadata_record_db_key_from_raw_key(key)
            hashVal = self._dataTxn.get(refKey, default=False)
            if hashVal is False:
                msg = f'HANGAR KEY ERROR:: No metadata key: `{key}` exists in checkout'
                raise KeyError(msg)

            hash_spec = parsing.metadata_record_raw_val_from_db_val(hashVal)
            metaKey = parsing.hash_meta_db_key_from_raw_key(hash_spec)
            metaVal = self._labelTxn.get(metaKey)
            meta_value = parsing.hash_meta_raw_val_from_db_val(metaVal)

        except KeyError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if not self.__is_conman:
                self._labelTxn = TxnRegister().abort_reader_txn(self._labelenv)
                self._dataTxn = TxnRegister().abort_reader_txn(self._dataenv)

        return meta_value


class MetadataWriter(MetadataReader):
    '''Class implementing write access to repository metadata.

    Similar to the :class:`hangar.dataset.DatasetDataWriter`, this class
    inherets the functionality of the :class:`MetadataReader` for reading. The
    only difference is that the reader will be initialized with a data record
    lmdb environment pointing to the staging area, and not a commit which is
    checked out.

    .. seealso::

        :class:`MetadataReader` for the intended use of this functionality.

    Parameters
    ----------
    dataenv : lmdb.Environment
        lmdb environment pointing to the staging area db which is opened in
        write mode
    labelenv : lmdb.Environment
        lmdb environment pointing to the label hash/value store db.
    '''

    def __init__(self, dataenv, labelenv):

        super().__init__(dataenv, labelenv)
        self.__is_conman: bool = False
        self._is_writeable: bool = True

    def __enter__(self):
        self.__is_conman = True
        self._labelTxn = TxnRegister().begin_writer_txn(self._labelenv)
        self._dataTxn = TxnRegister().begin_writer_txn(self._dataenv)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__is_conman = False
        self._labelTxn = TxnRegister().commit_writer_txn(self._labelenv)
        self._dataTxn = TxnRegister().commit_writer_txn(self._dataenv)

    def __setitem__(self, key, value):
        '''Store a key/value pair as metadata. Convenince method to :meth:`add`.

        .. seealso:: :meth:`add`

        Parameters
        ----------
        key : string
            name of the key to add as metadata
        value : string
            value to add as metadata

        Returns
        -------
        str
            key of the stored metadata sample (assuming operation was successful)
        '''
        return self.add(key, value)

    def __delitem__(self, key):
        '''Remove a key/value pair from metadata. Convenence method to :meth:`remove`.

        .. seealso:: :meth:`remove`

        Parameters
        ----------
        key : str
            Name of the metadata piece to remove.

        Returns
        -------
        str
            Metadata key removed from the dataset (assuming operation successful)
        '''
        return self.remove(key)

    def add(self, key, value):
        '''Add a piece of metadata to the staging area of the next commit.

        Parameters
        ----------
        key : string
            Name of the metadata piece, alphanumeric ascii chracters only
        value : string
            Metadata value to store in the repository, any length of valid
            ascii characters.

        Returns
        -------
        str
            The name of the metadata key written to the database if the
            operation succeeded.

        Raises
        ------
        ValueError
            If the `key` contains any whitespace or non alpha-numeric characters.
        ValueError
            If the `value` contains any non ascii characters.
        LookupError
            If an identical key/value pair exists in the checkout
        '''
        try:
            if not (isinstance(key, str) and is_ascii_alnum(key)):
                msg = f'HANGAR VALUE ERROR:: metadata key: `{key}` not allowed. '\
                      f'Must be str with only alpha-numeric ascii (no whitespace) chars.'
                raise ValueError(msg)
            elif not (isinstance(value, str) and is_ascii(value)):
                msg = f'HANGAR VALUE ERROR:: metadata value: `{value}` not allowed. '\
                      f'Must be str with only ascii characters.'
                raise ValueError(msg)
        except ValueError as e:
            logger.error(e, exc_info=False)
            raise e from None

        if not self.__is_conman:
            self._labelTxn = TxnRegister().begin_writer_txn(self._labelenv)
            self._dataTxn = TxnRegister().begin_writer_txn(self._dataenv)

        try:
            val_hash = hashlib.blake2b(value.encode(), digest_size=20).hexdigest()
            metaRecKey = parsing.metadata_record_db_key_from_raw_key(key)
            metaRecVal = parsing.metadata_record_db_val_from_raw_val(val_hash)
            metaHashKey = parsing.hash_meta_db_key_from_raw_key(val_hash)
            metaHashVal = parsing.hash_meta_db_val_from_raw_val(value)

            existingMetaRecVal = self._dataTxn.get(metaRecKey, default=False)
            if existingMetaRecVal:
                if metaRecVal == existingMetaRecVal:
                    msg = f'HANGAR KEY EXISTS ERROR:: metadata already contains key: `{key}` '\
                          f'with value: `{value}` & hash: {val_hash}'
                    raise LookupError(msg)
            else:
                # increment metadata record count
                metaCountKey = parsing.metadata_count_db_key()
                metaCountVal = self._dataTxn.get(metaCountKey, default='0'.encode())
                meta_count = parsing.metadata_count_raw_val_from_db_val(metaCountVal) + 1
                newMetaCountVal = parsing.metadata_count_db_val_from_raw_val(meta_count)
                self._dataTxn.put(metaCountKey, newMetaCountVal)
            self._labelTxn.put(metaHashKey, metaHashVal, overwrite=False)
            self._dataTxn.put(metaRecKey, metaRecVal)

        except LookupError as e:
            logger.error(e, exc_info=False)
            raise

        finally:
            if not self.__is_conman:
                self._labelTxn = TxnRegister().commit_writer_txn(self._labelenv)
                self._dataTxn = TxnRegister().commit_writer_txn(self._dataenv)

        return key

    def remove(self, key):
        '''Remove a piece of metadata from the staging area of the next commit.

        Parameters
        ----------
        key : str
            Metadata name to remove.

        Returns
        -------
        str
            name of the metadata key/value pair removed, if the operation was successful.

        Raises
        ------
        ValueError
            If the key provided is not string type and containing only
            ascii-alphanumeric characters.
        KeyError
            If the checkout does not contain metadata with the provided key.
        '''
        if not self.__is_conman:
            self._dataTxn = TxnRegister().begin_writer_txn(self._dataenv)

        try:
            if not (isinstance(key, str) and is_ascii_alnum(key)):
                msg = f'HANGAR VALUE ERROR:: metadata key: `{key}` not allowed. '\
                      f'Must be str with only alpha-numeric ascii (no whitespace) chars.'
                raise ValueError(msg)

            metaRecKey = parsing.metadata_record_db_key_from_raw_key(key)
            delete_succeeded = self._dataTxn.delete(metaRecKey)
            if delete_succeeded is False:
                msg = f'HANGAR KEY ERROR:: No metadata exists with key: {key}'
                raise KeyError(msg)

            metaRecCountKey = parsing.metadata_count_db_key()
            metaRecCountVal = self._dataTxn.get(metaRecCountKey)
            meta_count = parsing.metadata_count_raw_val_from_db_val(metaRecCountVal)
            meta_count -= 1
            if meta_count == 0:
                self._dataTxn.delete(metaRecCountKey)
            else:
                newMetaRecCountVal = parsing.metadata_count_db_val_from_raw_val(meta_count)
                self._dataTxn.put(metaRecCountKey, newMetaRecCountVal)

        except (KeyError, ValueError) as e:
            logger.error(e, exc_info=False)
            raise e from None

        finally:
            if not self.__is_conman:
                self._dataTxn = TxnRegister().commit_writer_txn(self._dataenv)
        return key
