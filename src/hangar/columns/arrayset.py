from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple, Union, Dict

import lmdb
import numpy as np

from ..backends import parse_user_backend_opts
from ..txnctx import TxnRegister
from ..records.hashmachine import schema_hash_digest
from ..records.parsing import (
    arrayset_record_count_range_key,
    arrayset_record_schema_db_key_from_raw_key,
    arrayset_record_schema_db_val_from_raw_val,
    arrayset_record_schema_raw_val_from_db_val,
    hash_schema_db_key_from_raw_key
)
from ..records.queries import RecordQuery
from ..op_state import writer_checkout_only
from ..utils import is_suitable_user_key, is_ascii
from . import AsetTxn, Sample, Subsample, ModifierTypes, WriterModifierTypes

KeyType = Union[str, int]


class ArraysetConstructors(type):
    """Metaclass defining constructor methods for Arraysets object.

    Rather than using @classmethod decorator, we use a metaclass so that the
    instances of the Arrayset class do not have the constructors accessible as
    a bound method. This is important because Arrayset class instances are user
    facing; the ability to construct a new object modifying or accessing repo
    state/data should never be available.
    """

    def _from_staging_area(cls, repo_pth: Path, hashenv: lmdb.Environment,
                           stageenv: lmdb.Environment,
                           stagehashenv: lmdb.Environment):
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
        :class:`.Arraysets`
            Interface class with write-enabled attributes activate which contains
            live arrayset data accessors in `write` mode.
        """

        arraysets = {}
        txnctx = AsetTxn(stageenv, hashenv, stagehashenv)
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()
        for asetName, schemaSpec in stagedSchemaSpecs.items():
            if schemaSpec.schema_contains_subsamples:
                setup_args = Subsample().generate_writer(
                    txnctx=txnctx,
                    aset_name=asetName,
                    path=repo_pth,
                    schema_specs=schemaSpec)
            else:
                setup_args = Sample().generate_writer(
                    txnctx=txnctx,
                    aset_name=asetName,
                    path=repo_pth,
                    schema_specs=schemaSpec)
            arraysets[asetName] = setup_args.modifier

        return cls('a', repo_pth, arraysets, hashenv, stageenv, stagehashenv, txnctx)

    def _from_commit(cls, repo_pth: Path, hashenv: lmdb.Environment,
                     cmtrefenv: lmdb.Environment):
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
        :class:`.Arraysets`
            Interface class with write-enabled attributes deactivated which
            contains live arrayset data accessors in `read-only` mode.
        """
        arraysets = {}
        txnctx = AsetTxn(cmtrefenv, hashenv, None)
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()

        for asetName, schemaSpec in cmtSchemaSpecs.items():
            if schemaSpec.schema_contains_subsamples:
                setup_args = Subsample().generate_reader(
                    txnctx=txnctx,
                    aset_name=asetName,
                    path=repo_pth,
                    schema_specs=schemaSpec)
            else:
                setup_args = Sample().generate_reader(
                    txnctx=txnctx,
                    aset_name=asetName,
                    path=repo_pth,
                    schema_specs=schemaSpec)

            arraysets[asetName] = setup_args.modifier

        return cls('r', repo_pth, arraysets, None, None, None, None)


"""
Constructor and Interaction Class for Arraysets
--------------------------------------------------
"""


class Arraysets(metaclass=ArraysetConstructors):
    """Common access patterns and initialization/removal of arraysets in a checkout.

    This object is the entry point to all tensor data stored in their
    individual arraysets. Each arrayset contains a common schema which dictates
    the general shape, dtype, and access patters which the backends optimize
    access for. The methods contained within allow us to create, remove, query,
    and access these collections of common tensors.
    """

    def __init__(self,
                 mode: str,
                 repo_pth: Path,
                 arraysets: Dict[str, ModifierTypes],
                 hashenv: Optional[lmdb.Environment] = None,
                 dataenv: Optional[lmdb.Environment] = None,
                 stagehashenv: Optional[lmdb.Environment] = None,
                 txnctx: Optional[AsetTxn] = None):
        """Developer documentation for init method.

        .. warning::

            This class should not be instantiated directly. Instead use the factory
            functions :py:meth:`_from_commit` or :py:meth:`_from_staging` to return
            a pre-initialized class instance appropriately constructed for either a
            read-only or write-enabled checkout.

        Parameters
        ----------
        mode : str
            one of 'r' or 'a' to indicate read or write mode
        repo_pth : Path
            path to the repository on disk
        arraysets : Mapping[str, Union[ArraysetDataReader, ArraysetDataWriter]]
            dictionary of ArraysetData objects
        hashenv : Optional[lmdb.Environment]
            environment handle for hash records
        dataenv : Optional[lmdb.Environment]
            environment handle for the unpacked records. `data` is means to refer to
            the fact that the stageenv is passed in for for write-enabled, and a
            cmtrefenv for read-only checkouts.
        stagehashenv : Optional[lmdb.Environment]
            environment handle for newly added staged data hash records.
        txnctx: Optional[AsetTxn]
            class implementing context managers to handle lmdb transactions
        """
        self._stack = []
        self._is_conman_counter = 0
        self._mode = mode
        self._repo_pth = repo_pth
        self._arraysets = arraysets

        self._hashenv = hashenv
        self._dataenv = dataenv
        self._stagehashenv = stagehashenv
        self._txnctx = txnctx

    def _open(self):
        for v in self._arraysets.values():
            v._open()

    def _close(self):
        for v in self._arraysets.values():
            v._close()

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        self._close()
        for arrayset in self._arraysets.values():
            arrayset._destruct()
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)

    def __getattr__(self, name):
        """Raise permission error after checkout is closed.

         Only runs after a call to :meth:`_destruct`, which is responsible
         for deleting all attributes from the object instance.
        """
        try:
            self.__getattribute__('_mode')  # once checkout is closed, this won't exist.
        except AttributeError:
            err = (f'Unable to operate on past checkout objects which have been '
                   f'closed. No operation occurred. Please use a new checkout.')
            raise PermissionError(err) from None
        return self.__getattribute__(name)

# ------------- Methods Available To Both Read & Write Checkouts ------------------

    def _repr_pretty_(self, p, cycle):
        res = f'Hangar {self.__class__.__name__}\
                \n    Writeable: {bool(0 if self._mode == "r" else 1)}\
                \n    Arrayset Names / Partial Remote References:\
                \n      - ' + '\n      - '.join(
            f'{asetn} / {aset.contains_remote_references}'
            for asetn, aset in self._arraysets.items())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'arraysets={self._arraysets}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        """Let ipython know that any key based access can use the arrayset keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the arrayset keys for access.
        """
        return self.keys()

    def __getitem__(self, key: str) -> ModifierTypes:
        """Dict style access to return the arrayset object with specified key/name.

        Parameters
        ----------
        key : string
            name of the arrayset object to get.

        Returns
        -------
        ModifierTypes
            The object which is returned depends on the mode of checkout
            specified. If the arrayset was checked out with write-enabled,
            return writer object, otherwise return read only object.
        """
        try:
            return self._arraysets[key]
        except KeyError:
            raise KeyError(f'No arrayset exists with name: {key}')

    def __contains__(self, key: str) -> bool:
        """Determine if a arrayset with a particular name is stored in the checkout

        Parameters
        ----------
        key : str
            name of the arrayset to check for

        Returns
        -------
        bool
            True if a arrayset with the provided name exists in the checkout,
            otherwise False.
        """
        return True if key in self._arraysets else False

    def __len__(self) -> int:
        """Get the number of arrayset columns contained in the checkout.
        """
        return len(self._arraysets)

    def __iter__(self) -> Iterable[str]:
        return iter(self._arraysets)

    @property
    def _is_conman(self):
        return bool(self._is_conman_counter)

    def _any_is_conman(self) -> bool:
        """Determine if self or any contains arrayset class is conman.

        Returns
        -------
        bool
            [description]
        """
        res = any([self._is_conman, *[x._is_conman for x in self._arraysets.values()]])
        return res

    def __enter__(self):
        with ExitStack() as stack:
            for asetN in list(self._arraysets.keys()):
                stack.enter_context(self._arraysets[asetN])
            self._is_conman_counter += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._is_conman_counter -= 1
        self._stack.close()

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this arrayset object is write-enabled. Read-only attribute.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> Mapping[str, bool]:
        """Dict of bool indicating data reference locality in each arrayset.

        Returns
        -------
        Mapping[str, bool]
            For each arrayset name key, boolean value where False indicates all
            samples in arrayset exist locally, True if some reference remote
            sources.
        """
        res: Mapping[str, bool] = {}
        for asetn, aset in self._arraysets.items():
            res[asetn] = aset.contains_remote_references
        return res

    @property
    def remote_sample_keys(self) -> Mapping[str, Iterable[Union[int, str]]]:
        """Determine arraysets samples names which reference remote sources.

        Returns
        -------
        Mapping[str, Iterable[Union[int, str]]]
            dict where keys are arrayset names and values are iterables of
            samples in the arrayset containing remote references
        """
        res: Mapping[str, Iterable[Union[int, str]]] = {}
        for asetn, aset in self._arraysets.items():
            res[asetn] = aset.remote_reference_keys
        return res

    def keys(self) -> List[str]:
        """list all arrayset keys (names) in the checkout

        Returns
        -------
        List[str]
            list of arrayset names
        """
        return list(self._arraysets.keys())

    def values(self) -> Iterable[ModifierTypes]:
        """yield all arrayset object instances in the checkout.

        Yields
        -------
        Iterable[ModifierTypes]
            Generator of ArraysetData accessor objects (set to read or write mode
            as appropriate)
        """
        for asetN in list(self._arraysets.keys()):
            asetObj = self._arraysets[asetN]
            yield asetObj

    def items(self) -> Iterable[Tuple[str, ModifierTypes]]:
        """generator providing access to arrayset_name, :class:`Arraysets`

        Yields
        ------
        Iterable[Tuple[str, ModifierTypes]]
            returns two tuple of all all arrayset names/object pairs in the checkout.
        """
        for asetN in list(self._arraysets.keys()):
            asetObj = self._arraysets[asetN]
            yield (asetN, asetObj)

    def get(self, name: str) -> ModifierTypes:
        """Returns a arrayset access object.

        This can be used in lieu of the dictionary style access.

        Parameters
        ----------
        name : str
            name of the arrayset to return

        Returns
        -------
        ModifierTypes
            ArraysetData accessor (set to read or write mode as appropriate) which
            governs interaction with the data

        Raises
        ------
        KeyError
            If no arrayset with the given name exists in the checkout
        """
        return self[name]

    # ------------------------ Writer-Enabled Methods Only ------------------------------

    @writer_checkout_only
    def __delitem__(self, key: str) -> str:
        """remove a arrayset and all data records if write-enabled process.

        Parameters
        ----------
        key : str
            Name of the arrayset to remove from the repository. This will remove
            all records from the staging area (though the actual data and all
            records are still accessible) if they were previously committed

        Returns
        -------
        str
            If successful, the name of the removed arrayset.

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opened in a connection manager.
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any arraysets class is opened in a context manager')
        return self.delete(key)

    @writer_checkout_only
    def init_arrayset(self,
                      name: str,
                      shape: Union[int, Tuple[int]] = None,
                      dtype: np.dtype = None,
                      prototype: np.ndarray = None,
                      variable_shape: bool = False,
                      contains_subsamples: bool = False,
                      *,
                      backend_opts: Optional[Union[str, dict]] = None) -> WriterModifierTypes:
        """Initializes a arrayset in the repository.

        Arrayset columns are created in order to store some arbitrary
        collection of data pieces (arrays). Items need not be related to
        each-other in any direct capacity; the only criteria hangar requires is
        that all pieces of data stored in the arrayset have a compatible schema
        with each-other (more on this below). Each piece of data is indexed by
        some key (either user defined or automatically generated depending on
        the user's preferences). Both single level stores (sample keys mapping
        to data on disk) and nested stores (where some sample key maps to an
        arbitrary number of subsamples, in turn each pointing to some piece of
        store data on disk) are supported.

        All data pieces within a arrayset have the same data type and number of
        dimensions. The size of each dimension can be either fixed (the default
        behavior) or variable per sample. For fixed dimension sizes, all data
        pieces written to the arrayset must have the same shape & size which
        was specified at the time the arrayset column was initialized.
        Alternatively, variable sized arraysets can write data pieces with
        dimensions of any size (up to a specified maximum).


        Parameters
        ----------
        name : str
            The name assigned to this arrayset.
        shape : Union[int, Tuple[int]]
            The shape of the data samples which will be written in this arrayset.
            This argument and the `dtype` argument are required if a `prototype`
            is not provided, defaults to None.
        dtype : :class:`numpy.dtype`
            The datatype of this arrayset. This argument and the `shape` argument
            are required if a `prototype` is not provided., defaults to None.
        prototype : :class:`numpy.ndarray`
            A sample array of correct datatype and shape which will be used to
            initialize the arrayset storage mechanisms. If this is provided, the
            `shape` and `dtype` arguments must not be set, defaults to None.
        variable_shape : bool, optional
            If this is a variable sized arrayset. If true, a the maximum shape is
            set from the provided ``shape`` or ``prototype`` argument. Any sample
            added to the arrayset can then have dimension sizes <= to this
            initial specification (so long as they have the same rank as what
            was specified) defaults to False.
        contains_subsamples : bool, optional
            True if the arrayset column should store data in a nested structure.
            In this scheme, a sample key is used to index an arbitrary number of
            subsamples which map some (sub)key to some piece of data. If False,
            sample keys map directly to a single piece of data; essentially
            acting as a single level key/value store. By default, False.
        backend_opts : Optional[Union[str, dict]], optional
            ADVANCED USERS ONLY, backend format code and filter opts to apply
            to arrayset data. If None, automatically inferred and set based on
            data shape and type. by default None

        Returns
        -------
        WriterModifierTypes
            instance object of the initialized arrayset.

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opened in a connection manager.
        ValueError
            If provided name contains any non ascii letter characters
            characters, or if the string is longer than 64 characters long.
        ValueError
            If required `shape` and `dtype` arguments are not provided in absence of
            `prototype` argument.
        ValueError
            If `prototype` argument is not a C contiguous ndarray.
        LookupError
            If a arrayset already exists with the provided name.
        ValueError
            If rank of maximum tensor shape > 31.
        ValueError
            If zero sized dimension in `shape` argument
        ValueError
            If the specified backend is not valid.
        """
        if self._any_is_conman():
            raise PermissionError('Not allowed while context manager is used.')

        # ------------- Checks for argument validity --------------------------

        try:
            if (not is_suitable_user_key(name)) or (not is_ascii(name)):
                raise ValueError(
                    f'Arrayset name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace). '
                    f'Must be <= 64 characters long')
            if name in self._arraysets:
                raise LookupError(f'Arrayset already exists with name: {name}.')

            if prototype is not None:
                if not isinstance(prototype, np.ndarray):
                    raise ValueError(
                        f'If not `None`, `prototype` argument be `np.ndarray`-like.'
                        f'Invalid value: {prototype} of type: {type(prototype)}')
                elif not prototype.flags.c_contiguous:
                    raise ValueError(f'`prototype` must be "C" contiguous array.')
            elif isinstance(shape, (tuple, list, int)) and (dtype is not None):
                prototype = np.zeros(shape, dtype=dtype)
            else:
                raise ValueError(f'`shape` & `dtype` required if no `prototype` set.')

            if (0 in prototype.shape) or (prototype.ndim > 31):
                raise ValueError(
                    f'Invalid shape specification with ndim: {prototype.ndim} and '
                    f'shape: {prototype.shape}. Array rank > 31 dimensions not '
                    f'allowed AND all dimension sizes must be > 0.')

            beopts = parse_user_backend_opts(backend_opts=backend_opts,
                                             prototype=prototype,
                                             variable_shape=variable_shape)
        except (ValueError, LookupError) as e:
            raise e from None

        # ----------- Determine schema format details -------------------------

        schema_hash = schema_hash_digest(shape=prototype.shape,
                                         size=prototype.size,
                                         dtype_num=prototype.dtype.num,
                                         variable_shape=variable_shape,
                                         contains_subsamples=contains_subsamples,
                                         backend_code=beopts.backend,
                                         backend_opts=beopts.opts)

        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(name)
        asetSchemaVal = arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schema_hash,
            schema_is_var=variable_shape,
            schema_max_shape=prototype.shape,
            schema_dtype=prototype.dtype.num,
            schema_default_backend=beopts.backend,
            schema_default_backend_opts=beopts.opts,
            schema_contains_subsamples=contains_subsamples)

        # -------- set vals in lmdb only after schema is sure to exist --------

        txnctx = AsetTxn(self._dataenv, self._hashenv, self._stagehashenv)
        with txnctx.write() as ctx:
            hashSchemaKey = hash_schema_db_key_from_raw_key(schema_hash)
            hashSchemaVal = asetSchemaVal
            ctx.dataTxn.put(asetSchemaKey, asetSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)

        schemaSpec = arrayset_record_schema_raw_val_from_db_val(asetSchemaVal)
        if contains_subsamples:
            setup_args = Subsample().generate_writer(
                txnctx=self._txnctx,
                aset_name=name,
                path=self._repo_pth,
                schema_specs=schemaSpec)
        else:
            setup_args = Sample().generate_writer(
                txnctx=self._txnctx,
                aset_name=name,
                path=self._repo_pth,
                schema_specs=schemaSpec)
        self._arraysets[name] = setup_args.modifier

        return self.get(name)

    @writer_checkout_only
    def delete(self, arrayset: str) -> str:
        """remove the arrayset and all data contained within it.

        Parameters
        ----------
        arrayset : str
            name of the arrayset to remove

        Returns
        -------
        str
            name of the removed arrayset

        Raises
        ------
        PermissionError
            If any enclosed arrayset is opened in a connection manager.
        KeyError
            If a arrayset does not exist with the provided name
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any arraysets class is opened in a context manager')

        with ExitStack() as stack:
            datatxn = TxnRegister().begin_writer_txn(self._dataenv)
            stack.callback(TxnRegister().commit_writer_txn, self._dataenv)

            if arrayset not in self._arraysets:
                e = KeyError(f'Cannot remove: {arrayset}. Key does not exist.')
                raise e from None

            self._arraysets[arrayset]._close()
            self._arraysets.__delitem__(arrayset)
            with datatxn.cursor() as cursor:
                cursor.first()
                asetRangeKey = arrayset_record_count_range_key(arrayset)
                recordsExist = cursor.set_range(asetRangeKey)
                while recordsExist:
                    k = cursor.key()
                    if k.startswith(asetRangeKey):
                        recordsExist = cursor.delete()
                    else:
                        recordsExist = False

            asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(arrayset)
            datatxn.delete(asetSchemaKey)

        return arrayset
