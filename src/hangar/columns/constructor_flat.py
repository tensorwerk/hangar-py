"""Constructor metaclass for initializing FlatSample columns.
"""
from pathlib import Path
import warnings

import numpy as np

from .common import _open_file_handles, UsedBackendInfo, AsetTxn
from ..backends import BACKEND_IS_LOCAL_MAP, backend_decoder
from ..records.parsing import hash_data_db_key_from_raw_key, RawArraysetSchemaVal
from ..records.queries import RecordQuery


def _load_sample_keys_and_specs(aset_name, txnctx: AsetTxn):
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


def _used_backends(sample_specs) -> UsedBackendInfo:
    seen = set()
    for spec in sample_specs.values():
        seen.add(spec.backend)
    all_be_local = all([BACKEND_IS_LOCAL_MAP[be] for be in seen])
    res = UsedBackendInfo(backends=seen, islocal=all_be_local)
    return res


def _common_setup(txnctx: AsetTxn, aset_name):
    sspecs = _load_sample_keys_and_specs(aset_name, txnctx)
    be_info = _used_backends(sspecs)
    if not be_info.islocal:
        warnings.warn(
            f'Arrayset: {aset_name} contains `reference-only` samples, with '
            f'actual data residing on a remote server. A `fetch-data` '
            f'operation is required to access these samples.', UserWarning)
    return (be_info.backends, sspecs)



class FlatSampleBuilder(type):
    """Metaclass defining constructor methods for FlatSample objects.

    Rather than using @classmethod decorator, we use a metaclass so that the
    instances of the FlatSample class do not have the constructors accessible as
    a bound method. This is important because FlatSample class instances are user
    facing; the ability to construct a new object modifying or accessing repo
    state/data should never be available.
    """

    def _generate_reader(cls,
                         txnctx: AsetTxn,
                         aset_name: str,
                         path: Path,
                         schema_specs: RawArraysetSchemaVal):
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
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]

        used_backends, sspecs = _common_setup(txnctx, aset_name)
        file_handles = _open_file_handles(
            used_backends=used_backends, path=path, shape=shape, dtype=dtype, mode='r')

        return cls(aset_name=aset_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema_spec=schema_specs,
                   repo_path=path,
                   mode='r')

    def _generate_writer(cls,
                         txnctx: AsetTxn,
                         aset_name: str,
                         path: Path,
                         schema_specs: RawArraysetSchemaVal):
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
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]
        default_backend = schema_specs.schema_default_backend
        default_backend_opts = schema_specs.schema_default_backend_opts

        used_backends, sspecs = _common_setup(txnctx, aset_name)
        file_handles = _open_file_handles(
            used_backends=[], path=path, shape=shape, dtype=dtype, mode='a')
        file_handles[default_backend].backend_opts = default_backend_opts

        return cls(aset_txn_ctx=txnctx,
                   aset_name=aset_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema_spec=schema_specs,
                   repo_path=path,
                   mode='a')
