"""Constructors for initializing FlatSampleReader and NestedSampleReader columns
"""
import warnings
from _weakref import proxy
from collections import defaultdict
from typing import Union

from wrapt import ObjectProxy

from .common import open_file_handles
from .layout_flat import FlatSampleReader, FlatSampleWriter
from .layout_nested import (
    FlatSubsampleReader, FlatSubsampleWriter,
    NestedSampleReader, NestedSampleWriter,
)
from ..records.queries import RecordQuery
from ..records import hash_data_db_key_from_raw_key
from ..typesystem import NdarrayFixedShape, NdarrayVariableShape, StringVariableShape
from ..backends import BACKEND_IS_LOCAL_MAP, backend_decoder


# --------------- methods common to all column layout types -------------------


KeyType = Union[str, int]

_column_definitions = (NdarrayVariableShape, NdarrayFixedShape, StringVariableShape)


def column_type_object_from_schema(schema: dict):
    for c in _column_definitions:
        try:
            instance = c(**schema)
            return instance
        except (TypeError, ValueError):
            pass
    else:  # N.B. for-else loop (ie. "no-break")
        raise ValueError(f'Could not instantiate column schema object for {schema}')


def _warn_remote(aset_name):
    warnings.warn(
        f'Column: {aset_name} contains `reference-only` samples, with '
        f'actual data residing on a remote server. A `fetch-data` '
        f'operation is required to access these samples.', UserWarning)


# --------- FlatSampleReader constructor metaclass / setup methods ------------------


def _flat_load_sample_keys_and_specs(column_name, txnctx):
    """Load flat sample key / backend location mapping info memory.

    Parameters
    ----------
    column_name: str
        name of the column to load.
    txnctx: ColumnTxn
        transaction context object used to access commit ref info on disk

    Returns
    -------
    Tuple[FlatSampleMapType, Set[str]]
        First element is single level dictionary mapping sample key to backend
        location. Second element is set of all unique backends encountered
        for every data pice in the column.
    """
    seen_bes = set()
    sspecs = {}
    with txnctx.read() as ctx:
        hashTxn = ctx.hashTxn
        asetNamesSpec = RecordQuery(ctx.dataenv).column_data_records(column_name)
        for asetNames, dataSpec in asetNamesSpec:
            hashKey = hash_data_db_key_from_raw_key(dataSpec.digest)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.sample] = be_loc
    seen_bes.update((spc.backend for spc in sspecs.values()))
    return (sspecs, seen_bes)


def generate_flat_column(txnctx, column_name, path, schema, mode):
    """Generate instance ready structures for read-only checkouts

    Parameters
    ----------
    txnctx : ColumnTxn
        transaction context object used to access commit ref info on disk
    column_name : str
        name of the column that the reader constructors are being
        generated for
    path : Path
        path to the repository on disk
    schema : ColumnDefinitionTypes
        schema definition of the column.
    mode: str
        read-only or write-enabled mode. one of ['a', 'r'].

    Returns
    -------
    :class:`~.flat.FlatSampleReader`
        Top level column accessor classes fully initialized for requested
        state. initailized structures defining and initializing access to
        the sample data on disk.
    """
    sspecs, bes = _flat_load_sample_keys_and_specs(column_name, txnctx)
    if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
        _warn_remote(column_name)
    if mode == 'a':
        bes.add(schema.backend)
    file_handles = open_file_handles(backends=bes, path=path, mode=mode, schema=schema)

    if mode == 'r':
        res = FlatSampleReader(columnname=column_name,
                               samples=sspecs,
                               backend_handles=file_handles,
                               schema=schema,
                               repo_path=path,
                               mode=mode)
    elif mode == 'a':
        res = FlatSampleWriter(aset_ctx=txnctx,
                               columnname=column_name,
                               samples=sspecs,
                               backend_handles=file_handles,
                               schema=schema,
                               repo_path=path,
                               mode=mode)
    else:
        raise ValueError(f'mode {mode} is not valid.')

    return res


# --------- NestedSampleReader constructor metaclass / setup methods ----------------


def _nested_load_sample_keys_and_specs(column_name, txnctx):
    """Load nested sample/subsample keys and backend location into memory from disk.

    Parameters
    ----------
    column_name : str
        name of the column to load.
    txnctx : ColumnTxn
        transaction context object used to access commit ref info on disk

    Returns
    -------
    Tuple[NestedSampleMapType, Set[str]]
        First element is nested dictionary where each sample name maps to
        subsample contents dict (associating subsample names with backend
        locations). Second element is set of all unique backends encountered
        for every data pice in the column.
    """
    seen_bes = set()
    sspecs = defaultdict(dict)
    with txnctx.read() as ctx:
        hashTxn = ctx.hashTxn
        asetNamesSpec = RecordQuery(ctx.dataenv).column_data_records(column_name)
        for asetNames, dataSpec in asetNamesSpec:
            hashKey = hash_data_db_key_from_raw_key(dataSpec.digest)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.sample].update({asetNames.subsample: be_loc})
            seen_bes.add(be_loc.backend)
    return (sspecs, seen_bes)


def generate_nested_column(txnctx, column_name, path, schema, mode):
    """Generate instance ready structures for read-only checkouts

    Parameters
    ----------
    txnctx : ColumnTxn
        transaction context object used to access commit ref info on disk
    column_name : str
        name of the column that the reader constructors are being
        generated for
    path : Path
        path to the repository on disk
    schema : ColumnDefinitionTypes
        schema definition of the column.
    mode: str
    read-only or write-enabled mode. one of ['a', 'r'].

    Returns
    -------
    :class:`~.nested.NestedSampleReader`
        Top level column accessor classes fully initialized for requested
        state. Initailized structures defining and initializing access to
        the subsample data on disk.
    """
    specs, bes = _nested_load_sample_keys_and_specs(column_name, txnctx)
    if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
        _warn_remote(column_name)
    if mode == 'a':
        bes.add(schema.backend)
    fhand = open_file_handles(backends=bes, path=path, mode=mode, schema=schema)
    samples = {}
    schema_proxy = proxy(schema)
    fhand['enter_count'] = 0

    if mode == 'r':
        for samp, subspecs in specs.items():
            samples[samp] = FlatSubsampleReader(
                columnname=column_name,
                samplen=samp,
                be_handles=fhand,
                specs=subspecs,
                mode='r')
        res = NestedSampleReader(
            columnname=column_name,
            samples=samples,
            backend_handles=fhand,
            repo_path=path,
            mode='r',
            schema=schema)
    elif mode == 'a':
        fhand = ObjectProxy(fhand)
        fhand_proxy = proxy(fhand)
        for samp, subspecs in specs.items():
            samples[samp] = FlatSubsampleWriter(
                schema=schema_proxy,
                aset_ctx=proxy(txnctx),
                repo_path=path,
                columnname=column_name,
                samplen=samp,
                be_handles=fhand_proxy,
                specs=subspecs,
                mode='a')
        res = NestedSampleWriter(
            aset_ctx=txnctx,
            columnname=column_name,
            samples=samples,
            backend_handles=fhand,
            schema=schema,
            repo_path=path,
            mode='a')
    else:
        raise ValueError(f'mode {mode} is not valid.')

    return res
