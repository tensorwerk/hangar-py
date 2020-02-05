"""Constructor metaclass for initializing NestedSample columns.
"""
from pathlib import Path
import warnings
from collections import defaultdict
from weakref import proxy

import numpy as np
from wrapt import ObjectProxy

from .common import _open_file_handles, AsetTxn, UsedBackendInfo
from ..backends import BACKEND_IS_LOCAL_MAP, backend_decoder
from ..records.parsing import hash_data_db_key_from_raw_key, RawArraysetSchemaVal
from ..records.queries import RecordQuery


def _load_sample_keys_and_specs(aset_name, txnctx: AsetTxn):
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


def _used_backends(sample_specs) -> UsedBackendInfo:
    seen = set()
    for subsamples_specs in sample_specs.values():
        for spec in subsamples_specs.values():
            seen.add(spec.backend)
    all_be_local = all([BACKEND_IS_LOCAL_MAP[be] for be in seen])
    res = UsedBackendInfo(backends=seen, islocal=all_be_local)
    return res


def _common_setup(txnctx: AsetTxn, aset_name):
    _sspecs = _load_sample_keys_and_specs(aset_name, txnctx)
    be_info = _used_backends(_sspecs)
    if not be_info.islocal:
        warnings.warn(
            f'Arrayset: {aset_name} contains `reference-only` samples, with '
            f'actual data residing on a remote server. A `fetch-data` '
            f'operation is required to access these samples.', UserWarning)
    return (be_info.backends, _sspecs)


class NestedSampleBuilder(type):
    """Metaclass defining constructor methods for NestedSample objects.

    Rather than using @classmethod decorator, we use a metaclass so that the
    instances of the NestedSample class do not have the constructors accessible as
    a bound method. This is important because NestedSample class instances are user
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
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .nested import FlatSubsample

        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]

        used_backends, _sspecs = _common_setup(txnctx, aset_name)
        file_handles = _open_file_handles(
            used_backends=used_backends, path=path, shape=shape, dtype=dtype, mode='r')
        file_handles['enter_count'] = 0
        sspecs = {}

        for sample_key, subsample_key_specs in _sspecs.items():
            sspecs[sample_key] = FlatSubsample(
                asetn=aset_name,
                samplen=sample_key,
                be_handles=file_handles,
                specs=subsample_key_specs,
                mode='r')

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
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .nested import FlatSubsample

        shape = schema_specs.schema_max_shape
        dtype = np.typeDict[schema_specs.schema_dtype]
        default_backend = schema_specs.schema_default_backend
        default_backend_opts = schema_specs.schema_default_backend_opts

        used_backends, _sspecs = _common_setup(txnctx, aset_name)
        file_handles = _open_file_handles(
            used_backends=[], path=path, shape=shape, dtype=dtype, mode='a')
        file_handles[default_backend].backend_opts = default_backend_opts
        file_handles['enter_count'] = 0
        file_handles['schema_spec'] = schema_specs
        file_handles = ObjectProxy(file_handles)
        file_handle_proxy = proxy(file_handles)
        sspecs = {}

        for sample_key, subsample_key_specs in _sspecs.items():
            sspecs[sample_key] = FlatSubsample(
                aset_txn_ctx=proxy(txnctx),
                asetn=aset_name,
                samplen=sample_key,
                be_handles=file_handle_proxy,
                specs=subsample_key_specs,
                mode='a')

        return cls(aset_txn_ctx=txnctx,
                   aset_name=aset_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema_spec=schema_specs,
                   repo_path=path,
                   mode='a')
