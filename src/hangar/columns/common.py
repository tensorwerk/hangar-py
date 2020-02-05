from pathlib import Path
from contextlib import contextmanager
from typing import Optional, Tuple, NamedTuple, Collection

import lmdb
import numpy as np

from ..backends import BACKEND_ACCESSOR_MAP, AccessorMapType
from ..txnctx import TxnRegister


class AsetTxn(object):
    """Provides context manager ready methods to handle lmdb transactions.

    In order to prevent passing around lmdb.Environment objects, we instantiate
    this class once for each arrayset arrayset and pass weakref proxy handels
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


def _open_file_handles(used_backends: Collection[str],
                       path: Path,
                       shape: Tuple[int],
                       dtype: np.dtype,
                       mode: str) -> AccessorMapType:
    """Open backend accessor file handles for reading

    Parameters
    ----------
    used_backends : Optional[Sequence[str]]
        backend format codes which should be opened, if ``mode == 'a'``,
        then this argument can be set to an empty list and all available
        backends will be prepared for writing in the column.
    path : Path
        path to the hangar repository on disk
    shape : Tuple[int]
        maximum shape contained data can be sized to; as defined in the
        arrayset schema
    dtype : np.dtype
        data type of the arrays stored in the backend
    mode : str
        one of ['r', 'a'] indicating read or write mode to open backends in.

    Returns
    -------
    AccessorMapType
        dict mapping backend format codes to initialized instances of each
        read-only backend.
    """
    fhandles = {}
    for be, accessor in BACKEND_ACCESSOR_MAP.items():
        if (be in used_backends) or (mode == 'a'):
            if accessor is None:
                continue
            fhandles[be] = accessor(repo_path=path, schema_shape=shape, schema_dtype=dtype)
            fhandles[be].open(mode=mode)
    return fhandles


class UsedBackendInfo(NamedTuple):
    """Describe backends used in a column.

    backends: Collection[str]
        unique backend (format codes) used to store data in column.

    islocal: bool
        indicate if all backend data exists on local machine.
    """
    backends: Collection[str]
    islocal: bool
