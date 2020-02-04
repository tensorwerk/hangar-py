from collections import Counter
from typing import MutableMapping

import lmdb


class TxnRegisterSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(TxnRegisterSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TxnRegister(metaclass=TxnRegisterSingleton):
    """Singleton to manage transaction thread safety in lmdb databases.

    This is essentailly a reference counting transaction register, lots of room
    for improvement here.
    """

    def __init__(self):
        self.WriterAncestors = Counter()
        self.ReaderAncestors = Counter()
        self.WriterTxn: MutableMapping[lmdb.Environment, lmdb.Transaction] = {}
        self.ReaderTxn: MutableMapping[lmdb.Environment, lmdb.Transaction] = {}

    @property
    def _debug_(self):  # pragma: no cover
        return {
            '__class__': self.__class__,
            'WriterAncestors': self.WriterAncestors,
            'ReaderAncestors': self.ReaderAncestors,
            'WriterTxn': self.WriterTxn,
            'ReaderTxn': self.ReaderTxn,
        }

    def begin_writer_txn(self, lmdbenv: lmdb.Environment,
                         buffer: bool = False) -> lmdb.Transaction:
        """Start a write enabled transaction on the given environment

        If multiple write transactions are requested for the same handle, only
        one instance of the transaction handle will be returened, and will not
        close until all operations on that handle have requested to close

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment to open the transaction on
        buffer : bool, optional
            if buffer objects should be used (the default is False, which does
            not use buffers)

        Returns
        -------
        lmdb.Transaction
            transaction handle to perform operations on
        """
        if self.WriterAncestors[lmdbenv] == 0:
            self.WriterTxn[lmdbenv] = lmdbenv.begin(write=True, buffers=buffer)
        self.WriterAncestors[lmdbenv] += 1
        return self.WriterTxn[lmdbenv]

    def begin_reader_txn(self, lmdbenv: lmdb.Environment,
                         buffer: bool = False) -> lmdb.Transaction:
        """Start a reader only txn for the given environment

        If there a read-only transaction for the same environment already exists
        then the same reader txn handle will be returned, and will not close
        until all operations on that handle have said they are finished.

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment to start the transaction in.
        buffer : bool, optional
            weather a buffer transaction should be used (the default is False,
            which means no buffers are returned)

        Returns
        -------
        lmdb.Transaction
            handle to the lmdb transaction.
        """
        if self.ReaderAncestors[lmdbenv] == 0:
            self.ReaderTxn[lmdbenv] = lmdbenv.begin(write=False, buffers=buffer)
        self.ReaderAncestors[lmdbenv] += 1
        return self.ReaderTxn[lmdbenv]

    def commit_writer_txn(self, lmdbenv: lmdb.Environment) -> bool:
        """Commit changes made in a write-enable transaction handle

        As multiple objects can have references to the same open transaction handle,
        the data is not actually committed until all open transactions have called
        the commit method.

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment handle used to open the transaction

        Raises
        ------
        RuntimeError
            If the internal reference counting gets out of sync

        Returns
        -------
        bool
            True if this operation actually committed, otherwise false
            if other objects have references to the same (open) handle
        """
        ancestors = self.WriterAncestors[lmdbenv]
        if ancestors == 0:
            msg = f'hash ancestors are zero but commit called on {lmdbenv}'
            raise RuntimeError(msg)
        elif ancestors == 1:
            self.WriterTxn[lmdbenv].commit()
            self.WriterTxn.__delitem__(lmdbenv)
            ret = True
        else:
            ret = False
        self.WriterAncestors[lmdbenv] -= 1
        return ret

    def abort_reader_txn(self, lmdbenv: lmdb.Environment) -> bool:
        """Request to close a read-only transaction handle

        As multiple objects can have references to the same open transaction
        handle, the transaction is not actuall aborted until all open transactions
        have called the abort method


        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment handle used to open the transaction

        Raises
        ------
        RuntimeError
            If the internal reference counting gets out of sync.

        Returns
        -------
        bool
            True if this operation actually aborted the transaction,
            otherwise False if other objects have references to the same (open)
            handle.
        """
        ancestors = self.ReaderAncestors[lmdbenv]
        if ancestors == 0:
            raise RuntimeError(f'hash ancestors are zero but abort called')
        elif ancestors == 1:
            self.ReaderTxn[lmdbenv].abort()
            self.ReaderTxn.__delitem__(lmdbenv)
            ret = True
        else:
            ret = False
        self.ReaderAncestors[lmdbenv] -= 1
        return ret
