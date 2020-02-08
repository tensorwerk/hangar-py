from contextlib import contextmanager
from typing import Optional

import lmdb

from ..txnctx import TxnRegister


class AsetTxn(object):
    """Provides context manager ready methods to handle lmdb transactions.

    In order to prevent passing around lmdb.Environment objects, we instantiate
    this class once for each column column and pass weakref proxy handels
    around to reference this object. Calling open / close methods (or using the
    ``with`` style methods) initializes transactions for the appropraite
    environments which are stored in instance attributes for access by the
    caller.
    """

    __slots__ = ('stagehashenv', 'dataenv', 'hashenv', 'hashTxn',
                 'dataTxn', 'stageHashTxn', '_TxnRegister', '__weakref__')

    def __init__(self, dataenv, hashenv, stagehashenv):

        self._TxnRegister = TxnRegister()
        self.stagehashenv = stagehashenv
        self.dataenv = dataenv
        self.hashenv = hashenv

        self.hashTxn: Optional[lmdb.Transaction] = None
        self.dataTxn: Optional[lmdb.Transaction] = None
        self.stageHashTxn: Optional[lmdb.Transaction] = None

    @property
    def _debug_(self):  # pragma: no cover
        return {
            f'__class__': self.__class__,
            f'_TxnRegister': self._TxnRegister._debug_,
            f'dataenv': self.dataenv,
            f'hashenv': self.hashenv,
            f'hashTxn': self.hashTxn,
            f'dataTxn': self.dataTxn,
            f'stageHashTxn': self.stageHashTxn,
        }

    def open_read(self):
        """Manually open read-only transactions, caller responsible for closing.
        """
        self.hashTxn = self._TxnRegister.begin_reader_txn(self.hashenv)
        self.dataTxn = self._TxnRegister.begin_reader_txn(self.dataenv)
        return self

    def close_read(self):
        """Manually close read-only transactions, must be called after manual open.
        """
        self.hashTxn = self._TxnRegister.abort_reader_txn(self.hashenv)
        self.dataTxn = self._TxnRegister.abort_reader_txn(self.dataenv)

    def open_write(self):
        """Manually open write-enabled transactions, caller responsible for closing.
        """
        self.hashTxn = self._TxnRegister.begin_writer_txn(self.hashenv)
        self.dataTxn = self._TxnRegister.begin_writer_txn(self.dataenv)
        self.stageHashTxn = self._TxnRegister.begin_writer_txn(self.stagehashenv)
        return self

    def close_write(self):
        """Manually close write-enabled transactions, must be called after manual open.
        """
        self.hashTxn = self._TxnRegister.commit_writer_txn(self.hashenv)
        self.dataTxn = self._TxnRegister.commit_writer_txn(self.dataenv)
        self.stageHashTxn = self._TxnRegister.commit_writer_txn(self.stagehashenv)

    @contextmanager
    def read(self):
        """Use ``with`` style context manager to open read-only transaction.

        Transaction is automatically closed for the caller irregardless of any
        application exceptions.
        """
        try:
            yield self.open_read()
        finally:
            self.close_read()

    @contextmanager
    def write(self):
        """Use ``with`` style context manager to open write-enabled transaction.

        Transaction is automatically closed for the caller irregardless of any
        application exceptions.
        """
        try:
            yield self.open_write()
        finally:
            self.close_write()


