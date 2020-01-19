from pathlib import Path
import warnings
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from typing import Optional, Tuple, Any, Dict, Sequence, List
from weakref import proxy

import lmdb
import numpy as np
from wrapt import ObjectProxy

from .arrayset_flat import SampleReaderModifier, SampleWriterModifier
from .arrayset_nested import (
    SubsampleReader, SubsampleWriter,
    SubsampleReaderModifier, SubsampleWriterModifier,
)
from ..backends import BACKEND_ACCESSOR_MAP, backend_decoder
from ..records.parsing import hash_data_db_key_from_raw_key, RawArraysetSchemaVal
from ..records.queries import RecordQuery
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

    __slots__ = ('stagehashenv', 'dataenv', 'hashenv',
                 'hashTxn', 'dataTxn', 'stageHashTxn',
                 '_TxnRegister', '__weakref__')

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


class Backend:
    """Set of related functions for parsing backend specs and oppening accessors.
    """

    @staticmethod
    def contains_remote_backend(used_backends: List[str]) -> True:
        has_remote = False
        if not all([bool(be[0] != '5') for be in used_backends]):
            has_remote = True
        return has_remote

    @staticmethod
    def warn_remote(aset_name):
        warnings.warn(
            f'Arrayset: {aset_name} contains `reference-only` samples, with '
            f'actual data residing on a remote server. A `fetch-data` '
            f'operation is required to access these samples.', UserWarning)

    @staticmethod
    def write_open_file_handles(path: Path, shape: Tuple[int],
                                dtype: np.dtype) -> Dict[str, Any]:
        """Open backend accessor file handles for writing.

        Parameters
        ----------
        path : Path
            path to the hangar repository on disk
        shape : Tuple[int]
            maximum shape of data to be written into the backend
        dtype : np.dtype
            data type of arrays to be written into the backend

        Returns
        -------
        Dict[str, Any]
            dict mapping backend format codes to initialized instances of each
            write-ready backend
        """
        fhandles = {}
        for be, accessor in BACKEND_ACCESSOR_MAP.items():
            if accessor is None:
                continue
            fhandles[be] = accessor(repo_path=path, schema_shape=shape, schema_dtype=dtype)
            fhandles[be].open(mode='a')
        return fhandles

    @staticmethod
    def read_open_file_handles(used_backends: Sequence[str], path: Path,
                               shape: Tuple[int],
                               dtype: np.dtype) -> Dict[str, Any]:
        """Open backend accessor file handles for reading

        Parameters
        ----------
        used_backends : Sequence[str]
            backend format codes which should be opened
        path : Path
            path to the hangar repository on disk
        shape : Tuple[int]
            maximum shape contained data can be sized to; as defined in the
            arrayset schema
        dtype : np.dtype
            data type of the arrays stored in the backend

        Returns
        -------
        Dict[str, Any]
            dict mapping backend format codes to initialized instances of each
            read-only backend.
        """
        fhandles = {}
        for be, accessor in BACKEND_ACCESSOR_MAP.items():
            if be in used_backends:
                if accessor is None:
                    continue
                fhandles[be] = accessor(repo_path=path, schema_shape=shape, schema_dtype=dtype)
                fhandles[be].open(mode='r')
        return fhandles


Construct = namedtuple('Construct', ['file_handles', 'modifier'])


class Subsample(Backend):
    """Common methods used to initalize reader or writer arrayset subsample structurs.
    """

    @staticmethod
    def load_sample_keys_and_specs(aset_name, txnctx: AsetTxn):
        sspecs = defaultdict(dict)
        with txnctx.read() as ctx:
            hashTxn = ctx.hashTxn
            asetNamesSpec = RecordQuery(ctx.dataenv).arrayset_data_records(aset_name)
            for asetNames, dataSpec in asetNamesSpec:
                hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
                hash_ref = hashTxn.get(hashKey)
                be_loc = backend_decoder(hash_ref)
                sspecs[asetNames.data_name].update({asetNames.subsample: be_loc})
        return sspecs

    @staticmethod
    def used_backends(sample_specs):
        seen = set()
        for subsamples_specs in sample_specs.values():
            for spec in subsamples_specs.values():
                seen.add(spec.backend)
        return seen

    def _common_setup(self, txnctx: AsetTxn, aset_name):
        _sspecs = self.load_sample_keys_and_specs(aset_name, txnctx)
        used_backends = self.used_backends(_sspecs)
        has_remote_backend = self.contains_remote_backend(used_backends)
        if has_remote_backend:
            self.warn_remote(aset_name)
        return (used_backends, has_remote_backend, _sspecs)

    def generate_reader(self, txnctx: AsetTxn, aset_name: str,
                        path: Path,
                        schema_specs: RawArraysetSchemaVal) -> Construct:
        """Generate instance ready structures for read-only checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        aset_name : str
            name of the arrayset that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema_specs : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        Construct
            initailized structures defining and initializing access to the
            subsample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]

        used_backends, has_remote_backend, _sspecs = self._common_setup(txnctx, aset_name)
        file_handles = self.read_open_file_handles(used_backends, path, shape, dtype)
        file_handles['enter_count'] = 0
        sspecs = {}
        for sample_key, subsample_key_specs in _sspecs.items():
            sspecs[sample_key] = SubsampleReader(
                asetn=aset_name,
                samplen=sample_key,
                be_handles=file_handles,
                specs=subsample_key_specs)

        modifier = SubsampleReaderModifier(aset_name=aset_name,
                                           samples=sspecs,
                                           backend_handles=file_handles,
                                           schema_spec=schema_specs,
                                           repo_path=path)
        return Construct(file_handles=file_handles, modifier=modifier)

    def generate_writer(self, txnctx: AsetTxn, aset_name: str,
                        path: Path,
                        schema_specs: RawArraysetSchemaVal) -> Construct:
        """Generate instance ready structures for write-enabled checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        aset_name : str
            name of the arrayset that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema_specs : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        Construct
            initailized structures defining and initializing access to the
            subsample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]
        default_backend = schema_specs.schema_default_backend
        default_backend_opts = schema_specs.schema_default_backend_opts

        used_backends, has_remote_backend, _sspecs = self._common_setup(txnctx, aset_name)
        file_handles = self.write_open_file_handles(path, shape, dtype)
        file_handles[default_backend].backend_opts = default_backend_opts
        file_handles['enter_count'] = 0
        file_handles['schema_spec'] = schema_specs
        file_handles = ObjectProxy(file_handles)
        file_handle_proxy = proxy(file_handles)
        sspecs = {}
        for sample_key, subsample_key_specs in _sspecs.items():
            sspecs[sample_key] = SubsampleWriter(
                aset_txn_ctx=proxy(txnctx),
                asetn=aset_name,
                samplen=sample_key,
                be_handles=file_handle_proxy,
                specs=subsample_key_specs)

        modifier = SubsampleWriterModifier(aset_txn_ctx=txnctx,
                                           aset_name=aset_name,
                                           samples=sspecs,
                                           backend_handles=file_handles,
                                           schema_spec=schema_specs,
                                           repo_path=path)

        return Construct(file_handles=file_handles, modifier=modifier)


class Sample(Backend):
    """Common methods used to initalize reader or writer arrayset flat sample structures.
    """

    @staticmethod
    def load_sample_keys_and_specs(aset_name, txnctx: AsetTxn):
        sspecs = {}
        with txnctx.read() as ctx:
            hashTxn = ctx.hashTxn
            asetNamesSpec = RecordQuery(ctx.dataenv).arrayset_data_records(aset_name)
            for asetNames, dataSpec in asetNamesSpec:
                hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
                hash_ref = hashTxn.get(hashKey)
                be_loc = backend_decoder(hash_ref)
                sspecs[asetNames.data_name] = be_loc
        return sspecs

    @staticmethod
    def used_backends(sample_specs):
        seen = set()
        for spec in sample_specs.values():
            seen.add(spec.backend)
        return seen

    def _common_setup(self, txnctx: AsetTxn, aset_name):
        sspecs = self.load_sample_keys_and_specs(aset_name, txnctx)
        used_backends = self.used_backends(sspecs)
        has_remote_backend = self.contains_remote_backend(used_backends)
        if has_remote_backend:
            self.warn_remote(aset_name)
        return (used_backends, has_remote_backend, sspecs)

    def generate_reader(self, txnctx: AsetTxn, aset_name: str,
                        path: Path, schema_specs: RawArraysetSchemaVal) -> Construct:
        """Generate instance ready structures for read-only checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        aset_name : str
            name of the arrayset that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema_specs : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        Construct
            initailized structures defining and initializing access to the
            sample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]

        used_backends, has_remote_backend, sspecs = self._common_setup(txnctx, aset_name)
        file_handles = self.read_open_file_handles(used_backends, path, shape, dtype)
        modifier = SampleReaderModifier(aset_name=aset_name,
                                        samples=sspecs,
                                        backend_handles=file_handles,
                                        schema_spec=schema_specs,
                                        repo_path=path)
        return Construct(file_handles=file_handles, modifier=modifier)

    def generate_writer(self, txnctx: AsetTxn, aset_name: str,
                        path: Path, schema_specs: RawArraysetSchemaVal) -> Construct:
        """Generate instance ready structures for write-enabled checkouts.

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        aset_name : str
            name of the arrayset that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema_specs : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        Construct
            initailized structures defining and initializing access to the
            sample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]
        default_backend = schema_specs.schema_default_backend
        default_backend_opts = schema_specs.schema_default_backend_opts

        used_backends, has_remote_backend, sspecs = self._common_setup(txnctx, aset_name)
        file_handles = self.write_open_file_handles(path, shape, dtype)
        file_handles[default_backend].backend_opts = default_backend_opts

        modifier = SampleWriterModifier(aset_txn_ctx=txnctx,
                                        aset_name=aset_name,
                                        samples=sspecs,
                                        backend_handles=file_handles,
                                        schema_spec=schema_specs,
                                        repo_path=path)
        return Construct(file_handles=file_handles, modifier=modifier)
