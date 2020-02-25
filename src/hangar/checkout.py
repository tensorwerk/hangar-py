import atexit
from pathlib import Path
import weakref
from contextlib import suppress, ExitStack
from uuid import uuid4
from typing import Optional, Union

import numpy as np
import lmdb

from .mixins import GetMixin, CheckoutDictIteration
from .columns import (
    AsetTxn,
    Columns,
    MetadataReader,
    MetadataWriter,
    generate_nested_column,
    generate_flat_column,
)
from .diff import ReaderUserDiff, WriterUserDiff
from .merger import select_merge_algorithm
from .records import commiting, hashs, heads
from .typesystem import NdarrayFixedShape, NdarrayVariableShape, StringVariableShape
from .utils import is_suitable_user_key, is_ascii
from .records import (
    schema_db_key_from_column,
    schema_hash_record_db_val_from_spec,
    schema_hash_db_key_from_digest,
    schema_record_db_val_from_digest,
)


class ReaderCheckout(GetMixin, CheckoutDictIteration):
    """Checkout the repository as it exists at a particular branch.

    This class is instantiated automatically from a repository checkout
    operation. This object will govern all access to data and interaction methods
    the user requests.

        >>> co = repo.checkout()
        >>> isinstance(co, ReaderCheckout)
        True

    If a commit hash is provided, it will take precedent over the branch name
    parameter. If neither a branch not commit is specified, the staging
    environment's base branch ``HEAD`` commit hash will be read.

        >>> co = repo.checkout(commit='foocommit')
        >>> co.commit_hash
        'foocommit'
        >>> co.close()
        >>> co = repo.checkout(branch='testbranch')
        >>> co.commit_hash
        'someothercommithashhere'
        >>> co.close()

    Unlike :class:`WriterCheckout`, any number of :class:`ReaderCheckout`
    objects can exist on the repository independently. Like the
    ``write-enabled`` variant, the :meth:`close` method should be called after
    performing the necessary operations on the repo. However, as there is no
    concept of a ``lock`` for ``read-only`` checkouts, this is just to free up
    memory resources, rather than changing recorded access state.

    In order to reduce the chance that the python interpreter is shut down
    without calling :meth:`close`,  - a common mistake during ipython / jupyter
    sessions - an `atexit <https://docs.python.org/3/library/atexit.html>`_
    hook is registered to :meth:`close`. If properly closed by the user, the
    hook is unregistered after completion with no ill effects. So long as a the
    process is NOT terminated via non-python ``SIGKILL``, fatal internal python
    error, or or special ``os exit`` methods, cleanup will occur on interpreter
    shutdown and resources will be freed. If a non-handled termination method
    does occur, the implications of holding resources varies on a per-OS basis.
    While no risk to data integrity is observed, repeated misuse may require a
    system reboot in order to achieve expected performance characteristics.
    """

    def __init__(self,
                 base_path: Path, labelenv: lmdb.Environment,
                 dataenv: lmdb.Environment, hashenv: lmdb.Environment,
                 branchenv: lmdb.Environment, refenv: lmdb.Environment,
                 commit: str):
        """Developer documentation of init method.

        Parameters
        ----------
        base_path : Path
            directory path to the Hangar repository on disk
        labelenv : lmdb.Environment
            db where the label dat is stored
        dataenv : lmdb.Environment
            db where the checkout record data is unpacked and stored.
        hashenv : lmdb.Environment
            db where the hash records are stored.
        branchenv : lmdb.Environment
            db where the branch records are stored.
        refenv : lmdb.Environment
            db where the commit references are stored.
        commit : str
            specific commit hash to checkout
        """
        self._commit_hash = commit
        self._repo_path = base_path
        self._labelenv = labelenv
        self._dataenv = dataenv
        self._hashenv = hashenv
        self._branchenv = branchenv
        self._refenv = refenv
        self._enter_count = 0
        self._stack: Optional[ExitStack] = None

        self._metadata = MetadataReader(
            mode='r',
            repo_pth=self._repo_path,
            dataenv=self._dataenv,
            labelenv=self._labelenv)
        self._columns = Columns._from_commit(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            cmtrefenv=self._dataenv)
        self._differ = ReaderUserDiff(
            commit_hash=self._commit_hash,
            branchenv=self._branchenv,
            refenv=self._refenv)
        atexit.register(self.close)

    def _repr_pretty_(self, p, cycle):
        """pretty repr for printing in jupyter notebooks
        """
        self._verify_alive()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : False\
                \n    Commit Hash  : {self._commit_hash}\
                \n    Num Columns  : {len(self)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self._verify_alive()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'labelenv={self._labelenv} '\
              f'dataenv={self._dataenv} '\
              f'hashenv={self._hashenv} '\
              f'commit={self._commit_hash})'
        return res

    def __enter__(self):
        self._verify_alive()
        with ExitStack() as stack:
            if self._enter_count == 0:
                stack.enter_context(self._columns)
                stack.enter_context(self._metadata)
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1

    def _verify_alive(self):
        """Validates that the checkout object has not been closed

        Raises
        ------
        PermissionError
            if the checkout was previously close
        """
        try:
            self._columns
        except AttributeError:
            err = PermissionError(
                f'Unable to operate on past checkout objects which have been '
                f'closed. No operation occurred. Please use a new checkout.')
            raise err from None

    @property
    def _is_conman(self):
        self._verify_alive()
        return bool(self._enter_count)

    @property
    def columns(self) -> Columns:
        """Provides access to column interaction object.

        Can be used to either return the columns accessor for all elements or
        a single column instance by using dictionary style indexing.

            >>> co = repo.checkout(write=False)
            >>> len(co.columns)
            1
            >>> print(co.columns.keys())
            ['foo']
            >>> fooAset = co.columns['foo']
            >>> fooAset.dtype
            np.fooDtype
            >>> asets = co.columns
            >>> fooAset = asets['foo']
            >>> fooAset.dtype
            np.fooDtype
            >>> fooAset = asets.get('foo')
            >>> fooAset.dtype
            np.fooDtype

        .. seealso::

            The class :class:`~.columns.column.Columns` contains all methods
            accessible by this property accessor

        Returns
        -------
        :class:`~.columns.column.Columns`
            the columns object which behaves exactly like a
            columns accessor class but which can be invalidated when the writer
            lock is released.
        """
        self._verify_alive()
        return self._columns

    @property
    def metadata(self) -> MetadataReader:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`~hangar.columns.metadata.MetadataReader` contains all methods
            accessible by this property accessor

        Returns
        -------
        MetadataReader
            weakref proxy to the metadata object which behaves exactly like a
            metadata class but which can be invalidated when the writer lock is
            released.
        """
        self._verify_alive()
        return self._metadata

    @property
    def diff(self) -> ReaderUserDiff:
        """Access the differ methods for a read-only checkout.

        .. seealso::

            The class :class:`ReaderUserDiff` contains all methods accessible
            by this property accessor

        Returns
        -------
        ReaderUserDiff
            weakref proxy to the differ object (and contained methods) which behaves
            exactly like the differ class but which can be invalidated when the
            writer lock is released.
        """
        self._verify_alive()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def commit_hash(self) -> str:
        """Commit hash this read-only checkout's data is read from.

            >>> co = repo.checkout()
            >>> co.commit_hash
            foohashdigesthere

        Returns
        -------
        str
            commit hash of the checkout
        """
        self._verify_alive()
        return self._commit_hash

    def close(self) -> None:
        """Gracefully close the reader checkout object.

        Though not strictly required for reader checkouts (as opposed to
        writers), closing the checkout after reading will free file handles and
        system resources, which may improve performance for repositories with
        multiple simultaneous read checkouts.
        """
        self._verify_alive()
        if isinstance(self._stack, ExitStack):
            self._stack.close()

        self._columns._destruct()
        self._metadata._destruct()
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)
        atexit.unregister(self.close)
        return


# --------------- Write enabled checkout ---------------------------------------


class WriterCheckout(GetMixin, CheckoutDictIteration):
    """Checkout the repository at the head of a given branch for writing.

    This is the entry point for all writing operations to the repository, the
    writer class records all interactions in a special ``"staging"`` area,
    which is based off the state of the repository as it existed at the
    ``HEAD`` commit of a branch.

        >>> co = repo.checkout(write=True)
        >>> co.branch_name
        'master'
        >>> co.commit_hash
        'masterheadcommithash'
        >>> co.close()

    At the moment, only one instance of this class can write data to the
    staging area at a time. After the desired operations have been completed,
    it is crucial to call :meth:`close` to release the writer lock. In
    addition, after any changes have been made to the staging area, the branch
    ``HEAD`` cannot be changed. In order to checkout another branch ``HEAD``
    for writing, you must either :meth:`commit` the changes, or perform a
    hard-reset of the staging area to the last commit via
    :meth:`reset_staging_area`.

    In order to reduce the chance that the python interpreter is shut down
    without calling :meth:`close`, which releases the writer lock - a common
    mistake during ipython / jupyter sessions - an `atexit
    <https://docs.python.org/3/library/atexit.html>`_ hook is registered to
    :meth:`close`. If properly closed by the user, the hook is unregistered
    after completion with no ill effects. So long as a the process is NOT
    terminated via non-python SIGKILL, fatal internal python error, or or
    special os exit methods, cleanup will occur on interpreter shutdown and the
    writer lock will be released. If a non-handled termination method does
    occur, the :meth:`~.Repository.force_release_writer_lock` method must be
    called manually when a new python process wishes to open the writer
    checkout.
    """

    def __init__(self,
                 repo_pth: Path,
                 branch_name: str,
                 labelenv: lmdb.Environment,
                 hashenv: lmdb.Environment,
                 refenv: lmdb.Environment,
                 stageenv: lmdb.Environment,
                 branchenv: lmdb.Environment,
                 stagehashenv: lmdb.Environment,
                 mode: str = 'a'):
        """Developer documentation of init method.

        Parameters
        ----------
        repo_pth : Path
            local file path of the repository.
        branch_name : str
            name of the branch whose ``HEAD`` commit will for the starting state
            of the staging area.
        labelenv : lmdb.Environment
            db where the label dat is stored
        hashenv lmdb.Environment
            db where the hash records are stored.
        refenv : lmdb.Environment
            db where the commit record data is unpacked and stored.
        stageenv : lmdb.Environment
            db where the stage record data is unpacked and stored.
        branchenv : lmdb.Environment
            db where the head record data is unpacked and stored.
        stagehashenv: lmdb.Environment
            db where the staged hash record data is stored.
        mode : str, optional
            open in write or read only mode, default is 'a' which is write-enabled.
        """
        self._enter_count = 0
        self._repo_path: Path = repo_pth
        self._branch_name = branch_name
        self._writer_lock = str(uuid4())
        self._stack: Optional[ExitStack] = None

        self._refenv = refenv
        self._hashenv = hashenv
        self._labelenv = labelenv
        self._stageenv = stageenv
        self._branchenv = branchenv
        self._stagehashenv = stagehashenv

        self._columns: Optional[Columns] = None
        self._differ: Optional[WriterUserDiff] = None
        self._metadata: Optional[MetadataWriter] = None
        self._setup()
        atexit.register(self.close)

    def _repr_pretty_(self, p, cycle):
        """pretty repr for printing in jupyter notebooks
        """
        self._verify_alive()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : True\
                \n    Base Branch  : {self._branch_name}\
                \n    Num Columns  : {len(self)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self._verify_alive()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'branch_name={self._branch_name} ' \
              f'labelenv={self._labelenv} '\
              f'hashenv={self._hashenv} '\
              f'refenv={self._refenv} '\
              f'stageenv={self._stageenv} '\
              f'branchenv={self._branchenv})\n'
        return res

    def __enter__(self):
        self._verify_alive()
        with ExitStack() as stack:
            if self._enter_count == 0:
                stack.enter_context(self._columns)
                stack.enter_context(self._metadata)
            self._enter_count += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._stack.close()
        self._enter_count -= 1

    @property
    def _is_conman(self):
        self._verify_alive()
        return bool(self._enter_count)

    def _verify_alive(self):
        """Ensures that this class instance holds the writer lock in the database.

        Raises
        ------
        PermissionError
            If the checkout was previously closed (no :attr:``_writer_lock``)
            or if the writer lock value does not match that recorded in the
            branch db
        """
        try:
            self._writer_lock
        except AttributeError:
            with suppress(AttributeError):
                self._columns._destruct()
                del self._columns
            with suppress(AttributeError):
                self._metadata._destruct()
                del self._metadata
            with suppress(AttributeError):
                del self._differ
            err = f'Unable to operate on past checkout objects which have been '\
                  f'closed. No operation occurred. Please use a new checkout.'
            raise PermissionError(err) from None

        try:
            heads.acquire_writer_lock(self._branchenv, self._writer_lock)
        except PermissionError as e:
            with suppress(AttributeError):
                self._columns._destruct()
                del self._columns
            with suppress(AttributeError):
                self._metadata._destruct()
                del self._metadata
            with suppress(AttributeError):
                del self._differ
            raise e from None

    def _setup(self):
        """setup the staging area appropriately for a write enabled checkout.

        On setup, we cannot be sure what branch the staging area was previously
        checked out on, and we cannot be sure if there are any 'uncommitted
        changes' in the staging area (ie. the staging area is ``DIRTY``). The
        setup methods here ensure that we can safety make any changes to the
        staging area without overwriting uncommitted changes, and then perform
        the setup steps to checkout staging area state at that point in time.

        Raises
        ------
        ValueError
            if there are changes previously made in the staging area which were
            based on one branch's ``HEAD``, but a different branch was specified to
            be used for the base of this checkout.
        """
        self._verify_alive()
        current_head = heads.get_staging_branch_head(self._branchenv)
        currentDiff = WriterUserDiff(stageenv=self._stageenv,
                                     refenv=self._refenv,
                                     branchenv=self._branchenv,
                                     branch_name=current_head)
        if currentDiff.status() == 'DIRTY':
            if current_head != self._branch_name:
                e = ValueError(
                    f'Unable to check out branch: {self._branch_name} for writing '
                    f'as the staging area has uncommitted changes on branch: '
                    f'{current_head}. Please commit or stash uncommitted changes '
                    f'before checking out a different branch for writing.')
                self.close()
                raise e
        else:
            if current_head != self._branch_name:
                try:
                    cmt = heads.get_branch_head_commit(
                        branchenv=self._branchenv, branch_name=self._branch_name)
                except ValueError as e:
                    self.close()
                    raise e
                commiting.replace_staging_area_with_commit(
                    refenv=self._refenv, stageenv=self._stageenv, commit_hash=cmt)
                heads.set_staging_branch_head(
                    branchenv=self._branchenv, branch_name=self._branch_name)

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._columns = Columns._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

    def __setitem__(self, index, value):
        """Syntax for setting items.

        Checkout object can be thought of as a "dataset" ("dset") mapping a view
        of samples across columns:

            >>> dset = repo.checkout(branch='master', write=True)
            >>>
            >>> # Add single sample to single column
            >>> dset['foo', 1] = np.array([1])
            >>> dset['foo', 1]
            array([1])
            >>>
            >>> # Add multiple samples to single column
            >>> dset['foo', [1, 2, 3]] = [np.array([1]), np.array([2]), np.array([3])]
            >>> dset['foo', [1, 2, 3]]
            [array([1]), array([2]), array([3])]
            >>>
            >>> # Add single sample to multiple columns
            >>> dset[['foo', 'bar'], 1] = [np.array([1]), np.array([11])]
            >>> dset[:, 1]
            ArraysetData(foo=array([1]), bar=array([11]))

        Parameters
        ----------
        index: Union[Iterable[str], Iterable[str, int]]
            Please see detailed explanation above for full options.The first
            element (or collection) specified must be ``str`` type and correspond
            to an column name(s). The second element (or collection) are keys
            corresponding to sample names which the data should be written to.

            Unlike the :meth:`__getitem__` method, only ONE of the ``column``
            name(s) or ``sample`` key(s) can specify multiple elements at the same
            time. Ie. If multiple ``columns`` are specified, only one sample key
            can be set, likewise if multiple ``samples`` are specified, only one
            ``column`` can be specified. When specifying multiple ``columns``
            or ``samples``, each data piece to be stored must reside as individual
            elements (``np.ndarray``) in a List or Tuple. The number of keys and
            the number of values must match exactly.

        value: Union[:class:`numpy.ndarray`, Iterable[:class:`numpy.ndarray`]]
            Data to store in the specified columns/sample keys. When
            specifying multiple ``columns`` or ``samples``, each data piece
            to be stored must reside as individual elements (``np.ndarray``) in
            a List or Tuple. The number of keys and the number of values must
            match exactly.

        Notes
        -----

        *  No slicing syntax is supported for either columns or samples. This
           is in order to ensure explicit setting of values in the desired
           fields/keys

        *  Add multiple samples to multiple columns not yet supported.
        """
        self._verify_alive()
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if not isinstance(index, (tuple, list)):
                raise ValueError(f'Idx: {index} does not specify column(s) AND sample(s)')
            elif len(index) > 2:
                raise ValueError(f'Index of len > 2 invalid. To multi-set, pass in lists')
            asetsIdx, sampleNames = index

            # Parse Columns
            if isinstance(asetsIdx, str):
                asets = [self._columns._columns[asetsIdx]]
            elif isinstance(asetsIdx, (tuple, list)):
                asets = [self._columns._columns[aidx] for aidx in asetsIdx]
            else:
                raise TypeError(f'Column idx: {asetsIdx} of type: {type(asetsIdx)}')
            nAsets = len(asets)

            # Parse sample names
            if isinstance(sampleNames, (str, int)):
                sampleNames = [sampleNames]
            elif not isinstance(sampleNames, (list, tuple)):
                raise TypeError(f'Sample names: {sampleNames} type: {type(sampleNames)}')
            nSamples = len(sampleNames)

            # Verify asets
            if (nAsets > 1) and (nSamples > 1):
                raise SyntaxError(
                    'Not allowed to specify BOTH multiple samples AND multiple'
                    'columns in `set` operation in current Hangar implementation')

            elif (nAsets == 1) and (nSamples == 1):
                aset = asets[0]
                sampleName = sampleNames[0]
                aset[sampleName] = value

            elif nAsets >= 2:
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f'Value: {value} not list/tuple of np.ndarray')
                elif not (len(value) == nAsets):
                    raise ValueError(f'Num values: {len(value)} != num columns {nAsets}')
                for aset, val in zip(asets, value):
                    isCompat = aset._schema.verify_data_compatible(val)
                    if not isCompat.compatible:
                        raise ValueError(isCompat.reason)
                for sampleName in sampleNames:
                    for aset, val in zip(asets, value):
                        aset[sampleName] = val

            else:  # nSamples >= 2
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f'Value: {value} not list/tuple of np.ndarray')
                elif not (len(value) == nSamples):
                    raise ValueError(f'Num values: {len(value)} != num samples {nSamples}')
                for aset in asets:
                    for val in value:
                        isCompat = aset._schema.verify_data_compatible(val)
                        if not isCompat.compatible:
                            raise ValueError(isCompat.reason)
                for aset in asets:
                    for sampleName, val in zip(sampleNames, value):
                        aset[sampleName] = val
                return None

    @property
    def columns(self) -> Columns:
        """Provides access to column interaction object.

        Can be used to either return the columns accessor for all elements or
        a single column instance by using dictionary style indexing.

            >>> co = repo.checkout(write=True)
            >>> asets = co.columns
            >>> len(asets)
            0
            >>> fooAset = co.add_ndarray_column('foo', shape=(10, 10), dtype=np.uint8)
            >>> len(co.columns)
            1
            >>> len(co)
            1
            >>> list(co.columns.keys())
            ['foo']
            >>> list(co.keys())
            ['foo']
            >>> fooAset = co.columns['foo']
            >>> fooAset.dtype
            np.fooDtype
            >>> fooAset = asets.get('foo')
            >>> fooAset.dtype
            np.fooDtype
            >>> 'foo' in co.columns
            True
            >>> 'bar' in co.columns
            False

        .. seealso::

            The class :class:`~.columns.column.Columns` contains all methods
            accessible by this property accessor

        Returns
        -------
        :class:`~.columns.column.Columns`
            the columns object which behaves exactly like a columns accessor
            class but which can be invalidated when the writer lock is
            released.
        """
        self._verify_alive()
        return self._columns

    @property
    def metadata(self) -> MetadataWriter:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`~.columns.metadata.MetadataWriter` contains
            all methods accessible by this property accessor

        Returns
        -------
        MetadataWriter
            the metadata object which behaves exactly like a metadata class but
            which can be invalidated when the writer lock is released.
        """
        self._verify_alive()
        return self._metadata

    @property
    def diff(self) -> WriterUserDiff:
        """Access the differ methods which are aware of any staged changes.

        .. seealso::

            The class :class:`hangar.diff.WriterUserDiff` contains all methods
            accessible by this property accessor

        Returns
        -------
        WriterUserDiff
            weakref proxy to the differ object (and contained methods) which
            behaves exactly like the differ class but which can be invalidated
            when the writer lock is released.
        """
        self._verify_alive()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def branch_name(self) -> str:
        """Branch this write enabled checkout's staging area was based on.

        Returns
        -------
        str
            name of the branch whose commit ``HEAD`` changes are staged from.
        """
        self._verify_alive()
        return self._branch_name

    @property
    def commit_hash(self) -> str:
        """Commit hash which the staging area of `branch_name` is based on.

        Returns
        -------
        str
            commit hash
        """
        self._verify_alive()
        cmt = heads.get_branch_head_commit(branchenv=self._branchenv,
                                           branch_name=self._branch_name)
        return cmt

    def add_str_column(self,
                       name: str,
                       contains_subsamples: bool = False,
                       *,
                       backend: Optional[str] = None,
                       backend_options: Optional[dict] = None):
        """Initializes a :class:`str` container column

        Columns are created in order to store some arbitrary collection of data
        pieces. In this case, we store :class:`str` data. Items need not be
        related to each-other in any direct capacity; the only criteria hangar
        requires is that all pieces of data stored in the column have a
        compatible schema with each-other (more on this below). Each piece of
        data is indexed by some key (either user defined or automatically
        generated depending on the user's preferences). Both single level
        stores (sample keys mapping to data on disk) and nested stores (where
        some sample key maps to an arbitrary number of subsamples, in turn each
        pointing to some piece of store data on disk) are supported.

        All data pieces within a column have the same data type. For
        :class:`str` columns, there is no distinction between
        ``'variable_shape'`` and ``'fixed_shape'`` schema types. Values are
        allowed to take on a value of any size so long as the datatype and
        contents are valid for the schema definition.

        Parameters
        ----------
        name : str
            Name assigned to the column
        contains_subsamples : bool, optional
            True if the column column should store data in a nested structure.
            In this scheme, a sample key is used to index an arbitrary number
            of subsamples which map some (sub)key to a piece of data. If False,
            sample keys map directly to a single piece of data; essentially
            acting as a single level key/value store. By default, False.
        backend : Optional[str], optional
            ADVANCED USERS ONLY, backend format code to use for column data. If
            None, automatically inferred and set based on data shape and type.
            by default None
        backend_options : Optional[dict], optional
            ADVANCED USERS ONLY, filter opts to apply to column data. If None,
            automatically inferred and set based on data shape and type.
            by default None

        Returns
        -------
        :class:`~.columns.column.Columns`
            instance object of the initialized column.
        """
        if self.columns._any_is_conman() or self._is_conman:
            raise PermissionError('Not allowed while context manager is used.')

        # ------------- Checks for argument validity --------------------------

        try:
            if (not is_suitable_user_key(name)) or (not is_ascii(name)):
                raise ValueError(
                    f'Column name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace). '
                    f'Must be <= 64 characters long')

            if name in self._columns:
                raise LookupError(f'Column already exists with name: {name}.')

            if not isinstance(contains_subsamples, bool):
                raise ValueError(f'contains_subsamples argument must be bool, '
                                 f'not type {type(contains_subsamples)}')

        except (ValueError, LookupError) as e:
            raise e from None

        layout = 'nested' if contains_subsamples else 'flat'
        schema = StringVariableShape(
            dtype=str, column_layout=layout, backend=backend, backend_options=backend_options)

        schema_digest = schema.schema_hash_digest()
        columnSchemaKey = schema_db_key_from_column(name, layout=layout)
        columnSchemaVal = schema_record_db_val_from_digest(schema_digest)
        hashSchemaKey = schema_hash_db_key_from_digest(schema_digest)
        hashSchemaVal = schema_hash_record_db_val_from_spec(schema.schema)

        # -------- set vals in lmdb only after schema is sure to exist --------

        txnctx = AsetTxn(self._stageenv, self._hashenv, self._stagehashenv)
        with txnctx.write() as ctx:
            ctx.dataTxn.put(columnSchemaKey, columnSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)

        if contains_subsamples:
            setup_args = generate_nested_column(
                txnctx=txnctx, column_name=name,
                path=self._repo_path, schema=schema, mode='a')
        else:
            setup_args = generate_flat_column(
                txnctx=txnctx, column_name=name,
                path=self._repo_path, schema=schema, mode='a')
        self.columns._columns[name] = setup_args
        return self[name]

    def add_ndarray_column(self,
                           name: str,
                           shape: Optional[Union[int, tuple]] = None,
                           dtype: Optional[np.dtype] = None,
                           prototype: Optional[np.ndarray] = None,
                           variable_shape: bool = False,
                           contains_subsamples: bool = False,
                           *,
                           backend: Optional[str] = None,
                           backend_options: Optional[dict] = None):
        """Initializes a :class:`numpy.ndarray` container column.

        Columns are created in order to store some arbitrary collection of data
        pieces. In this case, we store :class:`numpy.ndarray` data. Items need
        not be related to each-other in any direct capacity; the only criteria
        hangar requires is that all pieces of data stored in the column have a
        compatible schema with each-other (more on this below). Each piece of
        data is indexed by some key (either user defined or automatically
        generated depending on the user's preferences). Both single level
        stores (sample keys mapping to data on disk) and nested stores (where
        some sample key maps to an arbitrary number of subsamples, in turn each
        pointing to some piece of store data on disk) are supported.

        All data pieces within a column have the same data type and number of
        dimensions. The size of each dimension can be either fixed (the default
        behavior) or variable per sample. For fixed dimension sizes, all data
        pieces written to the column must have the same shape & size which was
        specified at the time the column column was initialized. Alternatively,
        variable sized columns can write data pieces with dimensions of any
        size (up to a specified maximum).

        Parameters
        ----------
        name : str
            The name assigned to this column.
        shape : Union[int, Tuple[int]]
            The shape of the data samples which will be written in this column.
            This argument and the `dtype` argument are required if a `prototype`
            is not provided, defaults to None.
        dtype : :class:`numpy.dtype`
            The datatype of this column. This argument and the `shape` argument
            are required if a `prototype` is not provided., defaults to None.
        prototype : :class:`numpy.ndarray`
            A sample array of correct datatype and shape which will be used to
            initialize the column storage mechanisms. If this is provided, the
            `shape` and `dtype` arguments must not be set, defaults to None.
        variable_shape : bool, optional
            If this is a variable sized column. If true, a the maximum shape is
            set from the provided ``shape`` or ``prototype`` argument. Any sample
            added to the column can then have dimension sizes <= to this
            initial specification (so long as they have the same rank as what
            was specified) defaults to False.
        contains_subsamples : bool, optional
            True if the column column should store data in a nested structure.
            In this scheme, a sample key is used to index an arbitrary number of
            subsamples which map some (sub)key to some piece of data. If False,
            sample keys map directly to a single piece of data; essentially
            acting as a single level key/value store. By default, False.
        backend : Optional[str], optional
            ADVANCED USERS ONLY, backend format code to use for column data. If
            None, automatically inferred and set based on data shape and type.
            by default None
        backend_options : Optional[dict], optional
            ADVANCED USERS ONLY, filter opts to apply to column data. If None,
            automatically inferred and set based on data shape and type.
            by default None

        Returns
        -------
        :class:`~.columns.column.Columns`
            instance object of the initialized column.
        """
        if self.columns._any_is_conman() or self._is_conman:
            raise PermissionError('Not allowed while context manager is used.')

        # ------------- Checks for argument validity --------------------------

        try:
            if (not is_suitable_user_key(name)) or (not is_ascii(name)):
                raise ValueError(
                    f'Column name provided: `{name}` is invalid. Can only contain '
                    f'alpha-numeric or "." "_" "-" ascii characters (no whitespace). '
                    f'Must be <= 64 characters long')
            if name in self.columns:
                raise LookupError(f'Column already exists with name: {name}.')

            if not isinstance(contains_subsamples, bool):
                raise ValueError(f'contains_subsamples is not bool type')

            if prototype is not None:
                if (shape is not None) or (dtype is not None):
                    raise ValueError(f'cannot set both prototype and shape/dtype args.')
            else:
                prototype = np.zeros(shape, dtype=dtype)

            # these shape and dtype vars will be used as downstream input
            # (now that they have been sanitized for really crazy input values).
            dtype = prototype.dtype
            shape = prototype.shape
            if not all([x > 0 for x in shape]):
                raise ValueError(f'all dimensions must be sized greater than zero')

        except (ValueError, LookupError) as e:
            raise e from None

        column_layout = 'nested' if contains_subsamples else 'flat'
        if variable_shape:
            schema = NdarrayVariableShape(dtype=dtype, shape=shape, column_layout=column_layout,
                                          backend=backend, backend_options=backend_options)
        else:
            schema = NdarrayFixedShape(dtype=dtype, shape=shape, column_layout=column_layout,
                                       backend=backend, backend_options=backend_options)

        schema_digest = schema.schema_hash_digest()
        columnSchemaKey = schema_db_key_from_column(name, layout=column_layout)
        columnSchemaVal = schema_record_db_val_from_digest(schema_digest)
        hashSchemaKey = schema_hash_db_key_from_digest(schema_digest)
        hashSchemaVal = schema_hash_record_db_val_from_spec(schema.schema)

        # -------- set vals in lmdb only after schema is sure to exist --------

        txnctx = AsetTxn(self._stageenv, self._hashenv, self._stagehashenv)
        with txnctx.write() as ctx:
            ctx.dataTxn.put(columnSchemaKey, columnSchemaVal)
            ctx.hashTxn.put(hashSchemaKey, hashSchemaVal, overwrite=False)

        if contains_subsamples:
            setup_args = generate_nested_column(
                txnctx=txnctx, column_name=name,
                path=self._repo_path, schema=schema, mode='a')
        else:
            setup_args = generate_flat_column(
                txnctx=txnctx, column_name=name,
                path=self._repo_path, schema=schema, mode='a')

        self.columns._columns[name] = setup_args
        return self[name]

    def merge(self, message: str, dev_branch: str) -> str:
        """Merge the currently checked out commit with the provided branch name.

        If a fast-forward merge is possible, it will be performed, and the
        commit message argument to this function will be ignored.

        Parameters
        ----------
        message : str
            commit message to attach to a three-way merge
        dev_branch : str
            name of the branch which should be merge into this branch
            (ie `master`)

        Returns
        -------
        str
            commit hash of the new commit for the `master` branch this checkout
            was started from.
        """
        self._verify_alive()
        commit_hash = select_merge_algorithm(
            message=message,
            branchenv=self._branchenv,
            stageenv=self._stageenv,
            refenv=self._refenv,
            stagehashenv=self._stagehashenv,
            master_branch=self._branch_name,
            dev_branch=dev_branch,
            repo_path=self._repo_path,
            writer_uuid=self._writer_lock)

        for asetHandle in self._columns.values():
            with suppress(KeyError):
                asetHandle._close()

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._columns = Columns._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

        return commit_hash

    def commit(self, commit_message: str) -> str:
        """Commit the changes made in the staging area on the checkout branch.

        Parameters
        ----------
        commit_message : str, optional
            user proved message for a log of what was changed in this commit.
            Should a fast forward commit be possible, this will NOT be added to
            fast-forward ``HEAD``.

        Returns
        -------
        str
            The commit hash of the new commit.

        Raises
        ------
        RuntimeError
            If no changes have been made in the staging area, no commit occurs.
        """
        self._verify_alive()

        open_asets = []
        for column in self._columns.values():
            if column._is_conman:
                open_asets.append(column.column)
        open_meta = self._metadata._is_conman

        try:
            if open_meta:
                self._metadata.__exit__()
            for asetn in open_asets:
                self._columns[asetn].__exit__()

            if self._differ.status() == 'CLEAN':
                e = RuntimeError('No changes made in staging area. Cannot commit.')
                raise e from None

            self._columns._close()
            commit_hash = commiting.commit_records(message=commit_message,
                                                   branchenv=self._branchenv,
                                                   stageenv=self._stageenv,
                                                   refenv=self._refenv,
                                                   repo_path=self._repo_path)
            # purge recs then reopen file handles so that we don't have to invalidate
            # previous weakproxy references like if we just called :meth:``_setup```
            hashs.clear_stage_hash_records(self._stagehashenv)
            self._columns._open()

        finally:
            for asetn in open_asets:
                self._columns[asetn].__enter__()
            if open_meta:
                self._metadata.__enter__()

        return commit_hash

    def reset_staging_area(self) -> str:
        """Perform a hard reset of the staging area to the last commit head.

        After this operation completes, the writer checkout will automatically
        close in the typical fashion (any held references to :attr:``column``
        or :attr:``metadata`` objects will finalize and destruct as normal), In
        order to perform any further operation, a new checkout needs to be
        opened.

        .. warning::

            This operation is IRREVERSIBLE. all records and data which are note
            stored in a previous commit will be permanently deleted.

        Returns
        -------
        str
            Commit hash of the head which the staging area is reset to.

        Raises
        ------
        RuntimeError
            If no changes have been made to the staging area, No-Op.
        """
        self._verify_alive()
        print(f'Hard reset requested with writer_lock: {self._writer_lock}')

        if self._differ.status() == 'CLEAN':
            e = RuntimeError(f'No changes made in staging area. No reset necessary.')
            raise e from None

        if isinstance(self._stack, ExitStack):
            self._stack.close()
        if hasattr(self._columns, '_destruct'):
            self._columns._destruct()
        if hasattr(self._metadata, '_destruct'):
            self._metadata._destruct()

        hashs.remove_stage_hash_records_from_hashenv(self._hashenv, self._stagehashenv)
        hashs.clear_stage_hash_records(self._stagehashenv)
        hashs.backends_remove_in_process_data(self._repo_path)

        branch_head = heads.get_staging_branch_head(self._branchenv)
        head_commit = heads.get_branch_head_commit(self._branchenv, branch_head)
        commiting.replace_staging_area_with_commit(refenv=self._refenv,
                                                   stageenv=self._stageenv,
                                                   commit_hash=head_commit)

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._columns = Columns._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)
        return head_commit

    def close(self) -> None:
        """Close all handles to the writer checkout and release the writer lock.

        Failure to call this method after the writer checkout has been used
        will result in a lock being placed on the repository which will not
        allow any writes until it has been manually cleared.
        """
        self._verify_alive()
        if isinstance(self._stack, ExitStack):
            self._stack.close()

        if hasattr(self._columns, '_destruct'):
            self._columns._destruct()
        if hasattr(self._metadata, '_destruct'):
            self._metadata._destruct()
        heads.release_writer_lock(self._branchenv, self._writer_lock)
        for attr in list(self.__dict__.keys()):
            delattr(self, attr)
        atexit.unregister(self.close)
        return
