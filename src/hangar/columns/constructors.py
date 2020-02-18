"""Constructor metaclass for initializing FlatSample and NestedSample columns

Rather than using @classmethod decorator, we use metaclasses so that the
instances of the the container classes do not have the constructors accessible as
a bound method. This is important because these class instances are user
facing; the ability to construct a new object modifying or accessing repo
state/data should never be outside the hangar core.
"""
import warnings
from _weakref import proxy
from collections import defaultdict
from pathlib import Path
from typing import Collection, Tuple, Dict, Union

import lmdb
from wrapt import ObjectProxy

from .common import AsetTxn
from ..typesystem.ndarray import NdarrayFixedShape, NdarrayVariableShape
from ..typesystem.pystring import StringVariableShape
from ..backends import (
    BACKEND_ACCESSOR_MAP,
    BACKEND_IS_LOCAL_MAP,
    backend_decoder,
    AccessorMapType,
    DataHashSpecsType
)
from ..records.parsing import hash_data_db_key_from_raw_key
from ..records.queries import RecordQuery

# --------------- methods common to all column layout types -------------------

ColumnDefinitionTypes = Union[NdarrayFixedShape, NdarrayVariableShape, StringVariableShape]
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


KeyType = Union[str, int]
FlatSampleMapType = Dict[KeyType, DataHashSpecsType]
NestedSampleMapType = Dict[KeyType, FlatSampleMapType]


def _open_file_handles(backends, path, mode, schema) -> AccessorMapType:
    """Open backend accessor file handles for reading

    Parameters
    ----------
    backends : Collection[str]
        if ``mode == 'r'`` then this should be the used backend format
        codes in the column. if ``mode == 'a'``, then this should be a
        list of the allowed backend format codes this schema can feasably
        write to.
    path : Path
        path to the hangar repository on disk
    mode : str
        one of ['r', 'a'] indicating read or write mode to open backends in.
    schema_spec : ColumnDefinitionTypes
        schema spec so required values can be filled in to backend openers.

    Returns
    -------
    AccessorMapType
        dict mapping backend format codes to initialized instances of each
        read-only backend.
    """
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
            fhandles[be].open(mode=mode)
    return fhandles


def _warn_remote(aset_name):
    warnings.warn(
        f'Arrayset: {aset_name} contains `reference-only` samples, with '
        f'actual data residing on a remote server. A `fetch-data` '
        f'operation is required to access these samples.', UserWarning)


# --------- FlatSample constructor metaclass / setup methods ------------------


def _flat_load_sample_keys_and_specs(
        column_name, txnctx) -> Tuple[FlatSampleMapType, Collection[str]]:
    """Load flat sample key / backend location mapping info memory.

    Parameters
    ----------
    column_name: str
        name of the column to load.
    txnctx: AsetTxn
        transaction context object used to access commit ref info on disk

    Returns
    -------
    Tuple[FlatSampleMapType, Collection[str]]
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
            hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.sample] = be_loc
    seen_bes.update((spc.backend for spc in sspecs.values()))
    return (sspecs, seen_bes)


class FlatSampleBuilder(type):
    """Metaclass defining constructor methods for FlatSample objects.
    """

    def _generate_reader(cls, txnctx, column_name, path, schema):
        """Generate instance ready structures for read-only checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        column_name : str
            name of the column that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema : ColumnDefinitionTypes
            schema definition of the column.

        Returns
        -------
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        sspecs, bes = _flat_load_sample_keys_and_specs(column_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(column_name)
        file_handles = _open_file_handles(
            backends=bes, path=path, mode='r', schema=schema)

        return cls(columnname=column_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema=schema,
                   repo_path=path,
                   mode='r')

    def _generate_writer(cls, txnctx, column_name, path, schema):
        """Generate instance ready structures for write-enabled checkouts.

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        column_name : str
            name of the column that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema : ColumnDefinitionTypes
            schema definition of the column.

        Returns
        -------
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        sspecs, bes = _flat_load_sample_keys_and_specs(column_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(column_name)

        bes.add(schema.backend)
        file_handles = _open_file_handles(
            backends=bes, path=path, mode='a', schema=schema)

        return cls(aset_ctx=txnctx,
                   columnname=column_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema=schema,
                   repo_path=path,
                   mode='a')


# --------- NestedSample constructor metaclass / setup methods ----------------


def _nested_load_sample_keys_and_specs(
        column_name, txnctx) -> Tuple[NestedSampleMapType, Collection[str]]:
    """Load nested sample/subsample keys and backend location into memory from disk.

    Parameters
    ----------
    column_name : str
        name of the column to load.
    txnctx : AsetTxn
        transaction context object used to access commit ref info on disk

    Returns
    -------
    Tuple[NestedSampleMapType, Collection[str]]
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
            hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.sample].update({asetNames.subsample: be_loc})
            seen_bes.add(be_loc.backend)
    return (sspecs, seen_bes)


class NestedSampleBuilder(type):
    """Metaclass defining constructor methods for NestedSample objects.
    """

    def _generate_reader(cls, txnctx, column_name, path, schema):
        """Generate instance ready structures for read-only checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        column_name : str
            name of the column that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema : ColumnDefinitionTypes
            schema definition of the column.

        Returns
        -------
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .layout_nested import FlatSubsample

        specs, bes = _nested_load_sample_keys_and_specs(column_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(column_name)
        fhand = _open_file_handles(
            backends=bes, path=path, mode='r', schema=schema)
        schema_proxy = proxy(schema)
        fhand['enter_count'] = 0
        sample_specs = {}
        for samp, subspecs in specs.items():
            sample_specs[samp] = FlatSubsample(
                columnname=column_name,
                samplen=samp,
                be_handles=fhand,
                specs=subspecs,
                schema=schema_proxy,
                mode='r')

        return cls(columnname=column_name,
                   samples=sample_specs,
                   backend_handles=fhand,
                   schema=schema,
                   repo_path=path,
                   mode='r')

    def _generate_writer(cls, txnctx, column_name, path, schema):
        """Generate instance ready structures for write-enabled checkouts

        Parameters
        ----------
        txnctx : AsetTxn
            transaction context object used to access commit ref info on disk
        column_name : str
            name of the column that the reader constructors are being
            generated for
        path : Path
            path to the repository on disk
        schema : ColumnDefinitionTypes
            schema definition of the column.

        Returns
        -------
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .layout_nested import FlatSubsample

        specs, bes = _nested_load_sample_keys_and_specs(column_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(column_name)
        bes.add(schema.backend)
        fhand = _open_file_handles(backends=bes, path=path, mode='a', schema=schema)
        fhand['enter_count'] = 0
        schema_proxy = proxy(schema)
        fhand = ObjectProxy(fhand)
        fhand_proxy = proxy(fhand)
        samples = {}
        for samp, subspecs in specs.items():
            samples[samp] = FlatSubsample(
                aset_ctx=proxy(txnctx),
                columnname=column_name,
                samplen=samp,
                be_handles=fhand_proxy,
                schema=schema_proxy,
                specs=subspecs,
                mode='a')

        return cls(aset_ctx=txnctx,
                   columnname=column_name,
                   samples=samples,
                   backend_handles=fhand,
                   schema=schema,
                   repo_path=path,
                   mode='a')


# --------------------- column constructor metaclass ------------------------


class ArraysetConstructors(type):
    """Metaclass defining constructor methods for Columns object.
    """

    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        """Class method factory to checkout :class:`Columns` in write mode

        Once you get here, we assume the write lock verification has
        passed, and that write operations are safe to perform.

        Parameters
        ----------
        repo_pth : Path
            directory path to the hangar repository on disk
        hashenv : lmdb.Environment
            environment where tensor data hash records are open in write mode.
        stageenv : lmdb.Environment
            environment where staging records (dataenv) are opened in write mode.
        stagehashenv: lmdb.Environment
            environment where the staged hash records are stored in write mode

        Returns
        -------
        :class:`~column.Columns`
            Interface class with write-enabled attributes activate which contains
            live column data accessors in `write` mode.
        """
        from . import NestedSample, FlatSample

        columns = {}
        txnctx = AsetTxn(stageenv, hashenv, stagehashenv)
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()

        for column_record, schema in stagedSchemaSpecs.items():
            sch = column_type_object_from_schema(schema)
            if column_record.layout == 'nested':
                column = NestedSample._generate_writer(
                    txnctx=txnctx, column_name=column_record.column, path=repo_pth, schema=sch)
            else:
                column = FlatSample._generate_writer(
                    txnctx=txnctx, column_name=column_record.column, path=repo_pth, schema=sch)
            columns[column_record.column] = column

        return cls(mode='a',
                   repo_pth=repo_pth,
                   columns=columns,
                   hashenv=hashenv,
                   dataenv=stageenv,
                   stagehashenv=stagehashenv,
                   txnctx=txnctx)

    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        """Class method factory to checkout :class:`.Columns` in read-only mode

        For read mode, no locks need to be verified, but construction should
        occur through this interface only.

        Parameters
        ----------
        repo_pth : Path
            directory path to the hangar repository on disk
        hashenv : lmdb.Environment
            environment where tensor data hash records are open in read-only mode.
        cmtrefenv : lmdb.Environment
            environment where staging checkout records are opened in read-only mode.

        Returns
        -------
        :class:`~column.Columns`
            Interface class with write-enabled attributes deactivated which
            contains live column data accessors in `read-only` mode.
        """
        from . import NestedSample, FlatSample

        columns = {}
        txnctx = AsetTxn(cmtrefenv, hashenv, None)
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()

        for column_record, schema in cmtSchemaSpecs.items():
            sch = column_type_object_from_schema(schema)
            if column_record.layout == 'nested':
                column = NestedSample._generate_reader(
                    txnctx=txnctx, column_name=column_record.column, path=repo_pth, schema=sch)
            else:
                column = FlatSample._generate_reader(
                    txnctx=txnctx, column_name=column_record.column, path=repo_pth, schema=sch)
            columns[column_record.column] = column

        return cls(mode='r',
                   repo_pth=repo_pth,
                   columns=columns,
                   hashenv=None,
                   dataenv=None,
                   stagehashenv=None,
                   txnctx=None)
