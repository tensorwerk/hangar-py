from contextlib import contextmanager
from typing import Optional

import lmdb

from ..txnctx import TxnRegister


class ColumnTxn(object):
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


def open_file_handles(backends, path, mode, schema, *, remote_operation=False):
    """Open backend accessor file handles for reading

    Parameters
    ----------
    backends : Set[str]
        if ``mode == 'r'`` then this should be the used backend format
        codes in the column. if ``mode == 'a'``, then this should be a
        list of the allowed backend format codes this schema can feasably
        write to.
    path : Path
        path to the hangar repository on disk
    mode : str
        one of ['r', 'a'] indicating read or write mode to open backends in.
    schema : ColumnDefinitionTypes
        schema spec so required values can be filled in to backend openers.

    Returns
    -------
    AccessorMapType
        dict mapping backend format codes to initialized instances of each
        read-only backend.
    """
    from ..backends import BACKEND_ACCESSOR_MAP

    fhandles = {}
    for be, accessor in BACKEND_ACCESSOR_MAP.items():
        if be in backends:
            if accessor is None:
                continue

            init_requires = schema._beopts.init_requires
            # TODO rework names for this hack
            kwargs = {}
            for arg in init_requires:
                if arg == 'repo_path':
                    kwargs[arg] = path
                elif arg == 'schema_shape':
                    kwargs[arg] = schema.shape
                elif arg == 'schema_dtype':
                    kwargs[arg] = schema.dtype

            fhandles[be] = accessor(**kwargs)
            fhandles[be].open(mode=mode, remote_operation=remote_operation)

    if mode == 'a':
        if schema.backend in fhandles:
            fhandles[schema.backend].backend_opts = schema.backend_options
    return fhandles
