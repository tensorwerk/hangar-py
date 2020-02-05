"""Constructor metaclass for initializing FlatSample and NestedSample columns

Rather than using @classmethod decorator, we use metaclasses so that the
instances of the the container classes do not have the constructors accessible as
a bound method. This is important because these class instances are user
facing; the ability to construct a new object modifying or accessing repo
state/data should never be outside the hangar core.
"""
from _weakref import proxy
from collections import defaultdict
from pathlib import Path
import warnings
from typing import Collection, Tuple, Dict, Union

import lmdb
import numpy as np
from wrapt import ObjectProxy

from .common import AsetTxn
from ..records.parsing import hash_data_db_key_from_raw_key, RawArraysetSchemaVal
from ..records.queries import RecordQuery
from ..backends import (BACKEND_ACCESSOR_MAP, BACKEND_IS_LOCAL_MAP,
                        backend_decoder, AccessorMapType, DataHashSpecsType)


# --------------- methods common to all column layout types -------------------


KeyType = Union[str, int]
FlatSampleMapType = Dict[KeyType, DataHashSpecsType]
NestedSampleMapType = Dict[KeyType, FlatSampleMapType]


def _open_file_handles(used_backends, path, shape, dtype, mode) -> AccessorMapType:
    """Open backend accessor file handles for reading

    Parameters
    ----------
    used_backends : Collection[str]
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
            fhandles[be] = accessor(
                repo_path=path, schema_shape=shape, schema_dtype=dtype)
            fhandles[be].open(mode=mode)
    return fhandles


def _warn_remote(aset_name):
    warnings.warn(
        f'Arrayset: {aset_name} contains `reference-only` samples, with '
        f'actual data residing on a remote server. A `fetch-data` '
        f'operation is required to access these samples.', UserWarning)


# --------- FlatSample constructor metaclass / setup methods ------------------


def _flat_load_sample_keys_and_specs(
        aset_name, txnctx) -> Tuple[FlatSampleMapType, Collection[str]]:
    """Load flat sample key / backend location mapping info memory.

    Parameters
    ----------
    aset_name: str
        name of the arrayset to load.
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
        asetNamesSpec = RecordQuery(ctx.dataenv).arrayset_data_records(aset_name)
        for asetNames, dataSpec in asetNamesSpec:
            hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.data_name] = be_loc
    seen_bes.update((spc.backend for spc in sspecs.values()))
    return (sspecs, seen_bes)


class FlatSampleBuilder(type):
    """Metaclass defining constructor methods for FlatSample objects.
    """

    def _generate_reader(cls, txnctx, aset_name, path, schema_spec):
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
        schema_spec : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        shape = schema_spec.schema_max_shape
        dtype = np.typeDict[schema_spec.schema_dtype]

        sspecs, bes = _flat_load_sample_keys_and_specs(aset_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(aset_name)
        file_handles = _open_file_handles(
            used_backends=bes, path=path, shape=shape, dtype=dtype, mode='r')

        return cls(aset_name=aset_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema_spec=schema_spec,
                   repo_path=path,
                   mode='r')

    def _generate_writer(cls, txnctx, aset_name, path, schema_spec):
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
        schema_spec : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        :class:`~.flat.FlatSample`
            Top level column accessor classes fully initialized for requested
            state. initailized structures defining and initializing access to
            the sample data on disk.
        """
        shape = schema_spec.schema_max_shape
        dtype = np.typeDict[schema_spec.schema_dtype]
        default_backend = schema_spec.schema_default_backend
        default_backend_opts = schema_spec.schema_default_backend_opts

        sspecs, bes = _flat_load_sample_keys_and_specs(aset_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(aset_name)
        file_handles = _open_file_handles(
            used_backends=[], path=path, shape=shape, dtype=dtype, mode='a')
        file_handles[default_backend].backend_opts = default_backend_opts

        return cls(aset_ctx=txnctx,
                   aset_name=aset_name,
                   samples=sspecs,
                   backend_handles=file_handles,
                   schema_spec=schema_spec,
                   repo_path=path,
                   mode='a')


# --------- NestedSample constructor metaclass / setup methods ----------------


def _nested_load_sample_keys_and_specs(
        aset_name, txnctx) -> Tuple[NestedSampleMapType, Collection[str]]:
    """Load nested sample/subsample keys and backend location into memory from disk.

    Parameters
    ----------
    aset_name : str
        name of the arrayset to load.
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
        asetNamesSpec = RecordQuery(ctx.dataenv).arrayset_data_records(aset_name)
        for asetNames, dataSpec in asetNamesSpec:
            hashKey = hash_data_db_key_from_raw_key(dataSpec.data_hash)
            hash_ref = hashTxn.get(hashKey)
            be_loc = backend_decoder(hash_ref)
            sspecs[asetNames.data_name].update({asetNames.subsample: be_loc})
            seen_bes.add(be_loc.backend)
    return (sspecs, seen_bes)


class NestedSampleBuilder(type):
    """Metaclass defining constructor methods for NestedSample objects.
    """

    def _generate_reader(cls, txnctx, aset_name, path, schema_spec):
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
        schema_spec : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .nested import FlatSubsample

        shape = schema_spec.schema_max_shape
        dtype = np.typeDict[schema_spec.schema_dtype]

        specs, bes = _nested_load_sample_keys_and_specs(aset_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(aset_name)
        fhand = _open_file_handles(
            used_backends=bes, path=path, shape=shape, dtype=dtype, mode='r')
        fhand['enter_count'] = 0
        sample_specs = {}
        for samp, subspecs in specs.items():
            sample_specs[samp] = FlatSubsample(
                asetn=aset_name, samplen=samp, be_handles=fhand, specs=subspecs, mode='r')

        return cls(aset_name=aset_name,
                   samples=sample_specs,
                   backend_handles=fhand,
                   schema_spec=schema_spec,
                   repo_path=path,
                   mode='r')

    def _generate_writer(cls, txnctx, aset_name, path, schema_spec):
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
        schema_spec : RawArraysetSchemaVal
            schema definition of the arrayset.

        Returns
        -------
        :class:`~.nested.NestedSample`
            Top level column accessor classes fully initialized for requested
            state. Initailized structures defining and initializing access to
            the subsample data on disk.
        """
        from .nested import FlatSubsample

        shape = schema_spec.schema_max_shape
        dtype = np.typeDict[schema_spec.schema_dtype]
        default_backend = schema_spec.schema_default_backend
        default_backend_opts = schema_spec.schema_default_backend_opts

        specs, bes = _nested_load_sample_keys_and_specs(aset_name, txnctx)
        if not all([BACKEND_IS_LOCAL_MAP[be] for be in bes]):
            _warn_remote(aset_name)
        fhand = _open_file_handles(
            used_backends=[], path=path, shape=shape, dtype=dtype, mode='a')
        fhand[default_backend].backend_opts = default_backend_opts
        fhand['enter_count'] = 0
        fhand['schema_spec'] = schema_spec
        fhand = ObjectProxy(fhand)
        fhand_proxy = proxy(fhand)
        samples = {}
        for samp, subspecs in specs.items():
            samples[samp] = FlatSubsample(
                aset_ctx=proxy(txnctx),
                asetn=aset_name,
                samplen=samp,
                be_handles=fhand_proxy,
                specs=subspecs,
                mode='a')

        return cls(aset_ctx=txnctx,
                   aset_name=aset_name,
                   samples=samples,
                   backend_handles=fhand,
                   schema_spec=schema_spec,
                   repo_path=path,
                   mode='a')


# --------------------- arrayset constructor metaclass ------------------------


class ArraysetConstructors(type):
    """Metaclass defining constructor methods for Arraysets object.
    """

    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        """Class method factory to checkout :class:`Arraysets` in write mode

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
        :class:`~arrayset.Arraysets`
            Interface class with write-enabled attributes activate which contains
            live arrayset data accessors in `write` mode.
        """
        from . import NestedSample, FlatSample

        arraysets = {}
        txnctx = AsetTxn(stageenv, hashenv, stagehashenv)
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()
        for asetn, schema in stagedSchemaSpecs.items():
            if schema.schema_contains_subsamples:
                column = NestedSample._generate_writer(
                    txnctx=txnctx, aset_name=asetn, path=repo_pth, schema_spec=schema)
            else:
                column = FlatSample._generate_writer(
                    txnctx=txnctx, aset_name=asetn, path=repo_pth, schema_spec=schema)
            arraysets[asetn] = column

        return cls(mode='a',
                   repo_pth=repo_pth,
                   arraysets=arraysets,
                   hashenv=hashenv,
                   dataenv=stageenv,
                   stagehashenv=stagehashenv,
                   txnctx=txnctx)

    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        """Class method factory to checkout :class:`.Arraysets` in read-only mode

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
        :class:`~arrayset.Arraysets`
            Interface class with write-enabled attributes deactivated which
            contains live arrayset data accessors in `read-only` mode.
        """
        from . import NestedSample, FlatSample

        arraysets = {}
        txnctx = AsetTxn(cmtrefenv, hashenv, None)
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()

        for asetn, schema in cmtSchemaSpecs.items():
            if schema.schema_contains_subsamples:
                column = NestedSample._generate_reader(
                    txnctx=txnctx, aset_name=asetn, path=repo_pth, schema_spec=schema)
            else:
                column = FlatSample._generate_reader(
                    txnctx=txnctx, aset_name=asetn, path=repo_pth, schema_spec=schema)
            arraysets[asetn] = column

        return cls(mode='r',
                   repo_pth=repo_pth,
                   arraysets=arraysets,
                   hashenv=None,
                   dataenv=None,
                   stagehashenv=None,
                   txnctx=None)
