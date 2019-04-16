import hashlib

from .context import TxnRegister
from .records import parsing
from .records.queries import RecordQuery


class MetadataReader(object):
    '''Class implementing get access to the metadata in a repository.

    Unlike the :class:`DatasetDataReader` and :class:`DatasetDataWriter`, the
    equivalent Metadata classes do not need a factory function or class to
    coordinate access through the checkout. This is primarily because the
    metadata is only stored at a single level, and because the long term storage
    is must simpler than for array data (just write to a lmdb database).

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

    def __init__(self, dataenv, labelenv):

        self._dataenv = dataenv
        self._labelenv = labelenv
        self._labelTxn = None
        self._dataTxn = None
        self._Query = RecordQuery(self._dataenv)
        self.__is_conman = False

    def __enter__(self):
        self.__is_conman = True
        self._labelTxn = TxnRegister().begin_reader_txn(self._labelenv)
        self._dataTxn = TxnRegister().begin_reader_txn(self._dataenv)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__is_conman = False
        self._labelTxn = TxnRegister().abort_reader_txn(self._labelenv)
        self._dataTxn = TxnRegister().abort_reader_txn(self._dataenv)

    def __getitem__(self, key):
        return self.get(key)

    def __iter__(self):
        raise NotImplementedError()

    def __repr__(self):
        res = f'\n Hangar Metadata\
                \n     Number of Keys : {len(self._Query.metadata_names())}\
                \n     Access Mode    : r\n'
        return res

    def keys(self):
        names = self._Query.metadata_names()
        for name in names:
            yield name

    def values(self):
        names = self._Query.metadata_names()
        for name in names:
            yield self.get(name)

    def items(self):
        names = self._Query.metadata_names()
        for name in names:
            yield (name, self.get(name))

    def get(self, key):
        '''retrieve a piece of metadata from the checkout.

        Parameters
        ----------
        key : string, alphanumeric ascii characters only.
            The name of the metadata pice to retrieve.

        Returns
        -------
        string
            The stored metadata value associated with the key.

        Raises
        ------
        ValueError
            If key contains non-alphanumeric or non-ascii characters.
        '''
        if (not key.isascii()) or (not key.isalnum()):
            raise ValueError(f'key: "{key}" cannot contain non-alphanumeric non-ascii characters.')

        if not self.__is_conman:
            self._labelTxn = TxnRegister().begin_reader_txn(self._labelenv)
            self._dataTxn = TxnRegister().begin_reader_txn(self._dataenv)

        try:
            refKey = parsing.metadata_record_db_key_from_raw_key(key)
            hashVal = self._dataTxn.get(refKey, default=False)
            if not hashVal:
                print(f'No metadata with key: {key} exists')
                return False

            hash_spec = parsing.metadata_record_raw_val_from_db_val(hashVal)
            metaKey = parsing.hash_meta_db_key_from_raw_key(hash_spec)
            metaVal = self._labelTxn.get(metaKey)
            meta_value = parsing.hash_meta_raw_val_from_db_val(metaVal)
        finally:
            if not self.__is_conman:
                self._labelTxn = TxnRegister().abort_reader_txn(self._labelenv)
                self._dataTxn = TxnRegister().abort_reader_txn(self._dataenv)

        return meta_value


class MetadataWriter(MetadataReader):
    '''Class implementing write access to repository metadata.

    Similar to the :class:`DatasetDataWriter`, this class inherets the
    functionality of the :class:`MetadataReader` for reading. The only
    difference is that the reader will be initialized with a data record
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

        self._dataenv = None
        self._labelenv = None
        super().__init__(dataenv, labelenv)

        self._dataTxn = None
        self._labelTxn = None
        self.__is_conman = False

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
        return self.add(key, value)

    def __delitem__(self, key):
        return self.remove(key)

    def __missing__(self):
        raise NotImplementedError()

    def __contains__(self, key):
        raise NotImplementedError()

    def __repr__(self):
        res = f'\n Hangar Metadata\
                \n     Number of Keys : {len(self._Query.metadata_names())}\
                \n     Access Mode    : a\n'
        return res

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
        False
            If the operation was successful
        string
            The name of the metadata key written to the database if the
            operation succeeded.
        '''
        if (not key.isascii()) or (not key.isalnum()):
            err = f'key: "{key}" cannot contain non-alphanumeric non-ascii characters.'
            raise ValueError(err)
        elif not value.isascii():
            err = f'Value: "{value}" cannot contain non-ascii characters.'
            raise ValueError(err)

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
                    print('metadata record with same name and value exists. no-op.')
                    return False
            else:
                # increment metadata record count
                metaCountKey = parsing.metadata_count_db_key()
                metaCountVal = self._dataTxn.get(metaCountKey, default='0'.encode())
                meta_count = parsing.metadata_count_raw_val_from_db_val(metaCountVal) + 1
                newMetaCountVal = parsing.metadata_count_db_val_from_raw_val(meta_count)
                self._dataTxn.put(metaCountKey, newMetaCountVal)

            self._labelTxn.put(metaHashKey, metaHashVal, overwrite=False)
            self._dataTxn.put(metaRecKey, metaRecVal)

        finally:
            if not self.__is_conman:
                self._labelTxn = TxnRegister().commit_writer_txn(self._labelenv)
                self._dataTxn = TxnRegister().commit_writer_txn(self._dataenv)

        return key

    def remove(self, key):
        '''Remove a piece of metadata from the staging area of the next commit.

        Parameters
        ----------
        key : string
            Metadata name to remove.

        Returns
        -------
        bool
            If the operation was successful or not.
        '''
        if (not key.isascii()) or (not key.isalnum()):
            err = f'key: "{key}" cannot contain non-alphanumeric non-ascii characters.'
            raise ValueError(err)

        if not self.__is_conman:
            self._dataTxn = TxnRegister().begin_writer_txn(self._dataenv)

        try:
            metaRecKey = parsing.metadata_record_db_key_from_raw_key(key)
            delete_succeeded = self._dataTxn.delete(metaRecKey)
            if not delete_succeeded:
                print(f'No metadata with name: {key} stored. no-op')
                return False

            metaRecCountKey = parsing.metadata_count_db_key()
            metaRecCountVal = self._dataTxn.get(metaRecCountKey)
            meta_count = parsing.metadata_count_raw_val_from_db_val(metaRecCountVal)
            meta_count -= 1
            if meta_count == 0:
                self._dataTxn.delete(metaRecCountKey)
            else:
                newMetaRecCountVal = parsing.metadata_count_db_val_from_raw_val(meta_count)
                self._dataTxn.put(metaRecCountKey, newMetaRecCountVal)

        finally:
            if not self.__is_conman:
                self._dataTxn = TxnRegister().commit_writer_txn(self._dataenv)
        return True
