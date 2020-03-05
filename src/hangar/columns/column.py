"""Constructor and Interaction Class for Columns
"""
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Tuple, Union, Dict

import lmdb

from .common import ColumnTxn
from .constructors import (
    generate_flat_column, generate_nested_column, column_type_object_from_schema
)
from ..records import (
    schema_db_key_from_column,
    schema_hash_db_key_from_digest,
    schema_column_record_from_db_key,
    schema_spec_from_db_val,
    dynamic_layout_data_record_db_start_range_key,
)
from ..records.queries import RecordQuery
from ..op_state import writer_checkout_only
from ..txnctx import TxnRegister

KeyType = Union[str, int]
ModifierTypes = Union['NestedSampleReader', 'FlatSubsampleReader']


class Columns:
    """Common access patterns and initialization/removal of columns in a checkout.

    This object is the entry point to all data stored in their
    individual columns. Each column. contains a common schema which dictates
    the general shape, dtype, and access patters which the backends optimize
    access for. The methods contained within allow us to create, remove, query,
    and access these collections of common data pieces.
    """

    def __init__(self,
                 mode: str,
                 repo_pth: Path,
                 columns: Dict[str, ModifierTypes],
                 hashenv: Optional[lmdb.Environment] = None,
                 dataenv: Optional[lmdb.Environment] = None,
                 stagehashenv: Optional[lmdb.Environment] = None,
                 txnctx: Optional[ColumnTxn] = None):
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
        columns : Mapping[str, Union[ArraysetDataReader, ArraysetDataWriter]]
            dictionary of ArraysetData objects
        hashenv : Optional[lmdb.Environment]
            environment handle for hash records
        dataenv : Optional[lmdb.Environment]
            environment handle for the unpacked records. `data` is means to refer to
            the fact that the stageenv is passed in for for write-enabled, and a
            cmtrefenv for read-only checkouts.
        stagehashenv : Optional[lmdb.Environment]
            environment handle for newly added staged data hash records.
        txnctx: Optional[ColumnTxn]
            class implementing context managers to handle lmdb transactions
        """
        self._stack: Optional[ExitStack] = None
        self._is_conman_counter = 0
        self._mode = mode
        self._repo_pth = repo_pth
        self._columns = columns

        self._hashenv = hashenv
        self._dataenv = dataenv
        self._stagehashenv = stagehashenv
        self._txnctx = txnctx

    def _open(self):
        for v in self._columns.values():
            v._open()

    def _close(self):
        for v in self._columns.values():
            v._close()

    def _destruct(self):
        if isinstance(self._stack, ExitStack):
            self._stack.close()
        self._close()
        for column in self._columns.values():
            column._destruct()
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
        res = f'Hangar {self.__class__.__qualname__}\
                \n    Writeable         : {False if self._mode == "r" else True}\
                \n    Number of Columns : {len(self)}\
                \n    Column Names / Partial Remote References:\
                \n      - ' + '\n      - '.join(
            f'{asetn} / {aset.contains_remote_references}'
            for asetn, aset in self._columns.items())
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'repo_pth={self._repo_pth}, '\
              f'columns={self._columns}, '\
              f'mode={self._mode})'
        return res

    def _ipython_key_completions_(self):
        """Let ipython know that any key based access can use the column keys

        Since we don't want to inherit from dict, nor mess with `__dir__` for
        the sanity of developers, this is the best way to ensure users can
        autocomplete keys.

        Returns
        -------
        list
            list of strings, each being one of the column keys for access.
        """
        return self.keys()

    def __getitem__(self, key: str) -> ModifierTypes:
        """Dict style access to return the column object with specified key/name.

        Parameters
        ----------
        key : string
            name of the column object to get.

        Returns
        -------
        ModifierTypes
            The object which is returned depends on the mode of checkout
            specified. If the column was checked out with write-enabled,
            return writer object, otherwise return read only object.
        """
        try:
            return self._columns[key]
        except KeyError:
            raise KeyError(f'No column exists with name: {key}')

    def __contains__(self, key: str) -> bool:
        """Determine if a column with a particular name is stored in the checkout

        Parameters
        ----------
        key : str
            name of the column to check for

        Returns
        -------
        bool
            True if a column with the provided name exists in the checkout,
            otherwise False.
        """
        return True if key in self._columns else False

    def __len__(self) -> int:
        """Get the number of column columns contained in the checkout.
        """
        return len(self._columns)

    def __iter__(self) -> Iterable[str]:
        return iter(self._columns)

    @property
    def _is_conman(self):
        return bool(self._is_conman_counter)

    def _any_is_conman(self) -> bool:
        """Determine if self or any contains column class is conman.

        Returns
        -------
        bool
            [description]
        """
        res = any([self._is_conman, *[x._is_conman for x in self._columns.values()]])
        return res

    def __enter__(self):
        with ExitStack() as stack:
            for asetN in list(self._columns.keys()):
                stack.enter_context(self._columns[asetN])
            self._is_conman_counter += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._is_conman_counter -= 1
        self._stack.close()

    @property
    def iswriteable(self) -> bool:
        """Bool indicating if this column object is write-enabled. Read-only attribute.
        """
        return False if self._mode == 'r' else True

    @property
    def contains_remote_references(self) -> Mapping[str, bool]:
        """Dict of bool indicating data reference locality in each column.

        Returns
        -------
        Mapping[str, bool]
            For each column name key, boolean value where False indicates all
            samples in column exist locally, True if some reference remote
            sources.
        """
        res = {}
        for asetn, aset in self._columns.items():
            res[asetn] = aset.contains_remote_references
        return res

    @property
    def remote_sample_keys(self) -> Mapping[str, Iterable[Union[int, str]]]:
        """Determine columns samples names which reference remote sources.

        Returns
        -------
        Mapping[str, Iterable[Union[int, str]]]
            dict where keys are column names and values are iterables of
            samples in the column containing remote references
        """
        res = {}
        for asetn, aset in self._columns.items():
            res[asetn] = aset.remote_reference_keys
        return res

    def keys(self) -> List[str]:
        """list all column keys (names) in the checkout

        Returns
        -------
        List[str]
            list of column names
        """
        return list(self._columns.keys())

    def values(self) -> Iterable[ModifierTypes]:
        """yield all column object instances in the checkout.

        Yields
        -------
        Iterable[ModifierTypes]
            Generator of ColumnData accessor objects (set to read or write mode
            as appropriate)
        """
        for asetN in list(self._columns.keys()):
            asetObj = self._columns[asetN]
            yield asetObj

    def items(self) -> Iterable[Tuple[str, ModifierTypes]]:
        """generator providing access to column_name, :class:`Columns`

        Yields
        ------
        Iterable[Tuple[str, ModifierTypes]]
            returns two tuple of all all column names/object pairs in the checkout.
        """
        for asetN in list(self._columns.keys()):
            asetObj = self._columns[asetN]
            yield (asetN, asetObj)

    def get(self, name: str) -> ModifierTypes:
        """Returns a column access object.

        This can be used in lieu of the dictionary style access.

        Parameters
        ----------
        name : str
            name of the column to return

        Returns
        -------
        ModifierTypes
            ColumnData accessor (set to read or write mode as appropriate) which
            governs interaction with the data
        """
        return self[name]

    # -------------------- Writer-Enabled Methods Only ------------------------

    @writer_checkout_only
    def __delitem__(self, key: str) -> str:
        """remove a column and all data records if write-enabled process.

        Parameters
        ----------
        key : str
            Name of the column to remove from the repository. This will remove
            all records from the staging area (though the actual data and all
            records are still accessible) if they were previously committed

        Returns
        -------
        str
            If successful, the name of the removed column.

        Raises
        ------
        PermissionError
            If any enclosed column is opened in a connection manager.
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any columns class is opened in a context manager')
        return self.delete(key)

    @writer_checkout_only
    def delete(self, column: str) -> str:
        """remove the column and all data contained within it.

        Parameters
        ----------
        column : str
            name of the column to remove

        Returns
        -------
        str
            name of the removed column

        Raises
        ------
        PermissionError
            If any enclosed column is opened in a connection manager.
        KeyError
            If a column does not exist with the provided name
        """
        if self._any_is_conman():
            raise PermissionError(
                'Not allowed while any columns class is opened in a context manager')

        with ExitStack() as stack:
            datatxn = TxnRegister().begin_writer_txn(self._dataenv)
            stack.callback(TxnRegister().commit_writer_txn, self._dataenv)

            if column not in self._columns:
                e = KeyError(f'Cannot remove: {column}. Key does not exist.')
                raise e from None

            column_layout = self._columns[column].column_layout
            columnSchemaKey = schema_db_key_from_column(column, layout=column_layout)
            column_record = schema_column_record_from_db_key(columnSchemaKey)
            startRangeKey = dynamic_layout_data_record_db_start_range_key(column_record)

            self._columns[column]._close()
            self._columns.__delitem__(column)
            with datatxn.cursor() as cursor:
                cursor.first()
                recordsExist = cursor.set_range(startRangeKey)
                while recordsExist:
                    k = cursor.key()
                    if k.startswith(startRangeKey):
                        recordsExist = cursor.delete()
                    else:
                        recordsExist = False
            datatxn.delete(columnSchemaKey)

        return column

    @classmethod
    def _from_staging_area(cls, repo_pth, hashenv, stageenv, stagehashenv):
        """INTERNAL USE ONLY

        Class method factory to checkout :class:`Columns` in write mode

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
        columns = {}
        txnctx = ColumnTxn(stageenv, hashenv, stagehashenv)
        query = RecordQuery(stageenv)
        stagedSchemaSpecs = query.schema_specs()

        staged_col_schemas = {}
        with txnctx.read() as r_txn:
            # need to do some conversions here...
            # ref record digest -> hash db key -> schema spec dict -> schema obj
            for column_record, schema_digest_rec in stagedSchemaSpecs.items():
                hashSchemaKey = schema_hash_db_key_from_digest(schema_digest_rec.digest)
                hashSchemaVal = r_txn.hashTxn.get(hashSchemaKey)
                schema_dict = schema_spec_from_db_val(hashSchemaVal)
                schema = column_type_object_from_schema(schema_dict)
                staged_col_schemas[column_record] = schema

        for column_record, schema in staged_col_schemas.items():
            if column_record.layout == 'nested':
                column = generate_nested_column(
                    txnctx=txnctx, column_name=column_record.column,
                    path=repo_pth, schema=schema, mode='a')
            else:
                column = generate_flat_column(
                    txnctx=txnctx, column_name=column_record.column,
                    path=repo_pth, schema=schema, mode='a')
            columns[column_record.column] = column

        return cls(mode='a',
                   repo_pth=repo_pth,
                   columns=columns,
                   hashenv=hashenv,
                   dataenv=stageenv,
                   stagehashenv=stagehashenv,
                   txnctx=txnctx)

    @classmethod
    def _from_commit(cls, repo_pth, hashenv, cmtrefenv):
        """INTERNAL USE ONLY

        Class method factory to checkout :class:`.Columns` in read-only mode

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
        columns = {}
        txnctx = ColumnTxn(cmtrefenv, hashenv, None)
        query = RecordQuery(cmtrefenv)
        cmtSchemaSpecs = query.schema_specs()

        cmt_col_schemas = {}
        with txnctx.read() as r_txn:
            # need to do some conversions here...
            # ref record digest -> hash db key -> schema spec dict -> schema obj
            for column_record, schema_digest_rec in cmtSchemaSpecs.items():
                hashSchemaKey = schema_hash_db_key_from_digest(schema_digest_rec.digest)
                hashSchemaVal = r_txn.hashTxn.get(hashSchemaKey)
                schema_dict = schema_spec_from_db_val(hashSchemaVal)
                schema = column_type_object_from_schema(schema_dict)
                cmt_col_schemas[column_record] = schema

        for column_record, schema in cmt_col_schemas.items():
            if column_record.layout == 'nested':
                column = generate_nested_column(
                    txnctx=txnctx, column_name=column_record.column,
                    path=repo_pth, schema=schema, mode='r')
            else:
                column = generate_flat_column(
                    txnctx=txnctx, column_name=column_record.column,
                    path=repo_pth, schema=schema, mode='r')
            columns[column_record.column] = column

        return cls(mode='r',
                   repo_pth=repo_pth,
                   columns=columns,
                   hashenv=None,
                   dataenv=None,
                   stagehashenv=None,
                   txnctx=None)
