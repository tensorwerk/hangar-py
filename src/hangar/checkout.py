import atexit
import os
import weakref
from contextlib import suppress
from functools import partial
from os.path import join as pjoin
from uuid import uuid4

import lmdb

from . import constants as c
from .arrayset import Arraysets
from .diff import ReaderUserDiff, WriterUserDiff
from .merger import select_merge_algorithm
from .metadata import MetadataReader, MetadataWriter
from .records import commiting, hashs, heads
from .utils import cm_weakref_obj_proxy


class ReaderCheckout(object):
    """Checkout the repository as it exists at a particular branch.

    If a commit hash is provided, it will take precedent over the branch name
    parameter. If neither a branch not commit is specified, the staging
    environment's base branch ``HEAD`` commit hash will be read.

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
                 base_path: os.PathLike, labelenv: lmdb.Environment,
                 dataenv: lmdb.Environment, hashenv: lmdb.Environment,
                 branchenv: lmdb.Environment, refenv: lmdb.Environment,
                 commit: str):
        """Developer documentation of init method.

        Parameters
        ----------
        base_path : str
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

        self._metadata = MetadataReader(
            mode='r',
            repo_pth=self._repo_path,
            dataenv=self._dataenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_commit(
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
        self.__verify_checkout_alive()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : False\
                \n    Commit Hash  : {self._commit_hash}\
                \n    Num Arraysets : {len(self._arraysets)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self.__verify_checkout_alive()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'labelenv={self._labelenv} '\
              f'dataenv={self._dataenv} '\
              f'hashenv={self._hashenv} '\
              f'commit={self._commit_hash})'
        return res

    def __verify_checkout_alive(self):
        """Validates that the checkout object has not been closed

        Raises
        ------
        PermissionError
            if the checkout was previously close
        """
        p_hasattr = partial(hasattr, self)
        if not all(map(p_hasattr, ['_metadata', '_arraysets', '_differ'])):
            e = PermissionError(
                f'Unable to operate on past checkout objects which have been '
                f'closed. No operation occurred. Please use a new checkout.')
            raise e from None

    @property
    def arraysets(self) -> Arraysets:
        """Provides access to arrayset interaction object.

        .. seealso::

            The class :class:`~.arrayset.Arraysets` contains all methods
            accessible by this property accessor

        Returns
        -------
        Arraysets
            weakref proxy to the arraysets object which behaves exactly like a
            arraysets accessor class but which can be invalidated when the writer
            lock is released.
        """
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._arraysets)
        return wr

    @property
    def metadata(self) -> MetadataReader:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`hangar.metadata.MetadataReader` contains all methods
            accessible by this property accessor

        Returns
        -------
        MetadataReader
            weakref proxy to the metadata object which behaves exactly like a
            metadata class but which can be invalidated when the writer lock is
            released.
        """
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

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
        self.__verify_checkout_alive()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def commit_hash(self) -> str:
        """Commit hash this read-only checkout's data is read from.

        Returns
        -------
        string
            commit hash of the checkout
        """
        self.__verify_checkout_alive()
        return self._commit_hash

    def close(self) -> None:
        """Gracefully close the reader checkout object.

        Though not strictly required for reader checkouts (as opposed to
        writers), closing the checkout after reading will free file handles and
        system resources, which may improve performance for repositories with
        multiple simultaneous read checkouts.
        """
        self.__verify_checkout_alive()
        with suppress(AttributeError):
            self._arraysets._close()

        for asetn in (self._arraysets._arraysets.keys()):
            for attr in list(self._arraysets._arraysets[asetn].__dir__()):
                with suppress(AttributeError, TypeError):
                    delattr(self._arraysets._arraysets[asetn], attr)

        for attr in list(self._arraysets.__dir__()):
            with suppress(AttributeError, TypeError):
                # adding `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                delattr(self._arraysets, f'_self_{attr}')

        for attr in list(self._metadata.__dir__()):
            with suppress(AttributeError, TypeError):
                # adding `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                delattr(self._metadata, f'_self_{attr}')

        del self._arraysets
        del self._metadata
        del self._differ
        del self._commit_hash
        del self._repo_path
        del self._labelenv
        del self._dataenv
        del self._hashenv
        del self._branchenv
        del self._refenv
        atexit.unregister(self.close)
        return


# --------------- Write enabled checkout ---------------------------------------


class WriterCheckout(object):
    """Checkout the repository at the head of a given branch for writing.

    This is the entry point for all writing operations to the repository, the
    writer class records all interactions in a special ``"staging"`` area,
    which is based off the state of the repository as it existed at the
    ``HEAD`` commit of a branch.

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
    occur, the :py:meth:`~.Repository.force_release_writer_lock` method must be
    called manually when a new python process wishes to open the writer
    checkout.
    """

    def __init__(self,
                 repo_pth: os.PathLike,
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
        repo_pth : str
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
        stagenv : lmdb.Environment
            db where the stage record data is unpacked and stored.
        branchenv : lmdb.Environment
            db where the head record data is unpacked and stored.
        stagehashenv: lmdb.Environment
            db where the staged hash record data is stored.
        mode : str, optional
            open in write or read only mode, default is 'a' which is write-enabled.
        """
        self._repo_path = repo_pth
        self._branch_name = branch_name
        self._writer_lock = str(uuid4())

        self._refenv = refenv
        self._hashenv = hashenv
        self._labelenv = labelenv
        self._stageenv = stageenv
        self._branchenv = branchenv
        self._stagehashenv = stagehashenv
        self._repo_stage_path = pjoin(self._repo_path, c.DIR_DATA_STAGE)
        self._repo_store_path = pjoin(self._repo_path, c.DIR_DATA_STORE)

        self._arraysets: Arraysets = None
        self._differ: WriterUserDiff = None
        self._metadata: MetadataWriter = None
        self.__setup()
        atexit.register(self.close)

    def _repr_pretty_(self, p, cycle):
        """pretty repr for printing in jupyter notebooks
        """
        self.__acquire_writer_lock()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : True\
                \n    Base Branch  : {self._branch_name}\
                \n    Num Arraysets : {len(self._arraysets)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self.__acquire_writer_lock()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'branch_name={self._branch_name} ' \
              f'labelenv={self._labelenv} '\
              f'hashenv={self._hashenv} '\
              f'refenv={self._refenv} '\
              f'stageenv={self._stageenv} '\
              f'branchenv={self._branchenv})\n'
        return res

    @property
    def arraysets(self) -> Arraysets:
        """Provides access to arrayset interaction object.

        .. seealso::

            The class :class:`~.arrayset.Arraysets` contains all methods accessible
            by this property accessor

        Returns
        -------
        Arraysets
            weakref proxy to the arraysets object which behaves exactly like a
            arraysets accessor class but which can be invalidated when the writer
            lock is released.
        """
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._arraysets)
        return wr

    @property
    def metadata(self) -> MetadataWriter:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`hangar.metadata.MetadataWriter` contains all methods
            accessible by this property accessor

        Returns
        -------
        MetadataWriter
            weakref proxy to the metadata object which behaves exactly like a
            metadata class but which can be invalidated when the writer lock is
            released.
        """
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

    @property
    def diff(self) -> WriterUserDiff:
        """Access the differ methods which are aware of any staged changes.

        .. seealso::

            The class :class:`hangar.diff.WriterUserDiff` contains all methods
            accessible by this property accessor

        Returns
        -------
        WriterUserDiff
            weakref proxy to the differ object (and contained methods) which behaves
            exactly like the differ class but which can be invalidated when the
            writer lock is released.
        """
        self.__acquire_writer_lock()
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
        self.__acquire_writer_lock()
        return self._branch_name

    @property
    def commit_hash(self) -> str:
        """Commit hash which the staging area of `branch_name` is based on.

        Returns
        -------
        string
            commit hash
        """
        self.__acquire_writer_lock()
        cmt = heads.get_branch_head_commit(branchenv=self._branchenv,
                                           branch_name=self._branch_name)
        return cmt

    def merge(self, message: str, dev_branch: str) -> str:
        """Merge the currently checked out commit with the provided branch name.

        If a fast-forward merge is possible, it will be performed, and the
        commit message argument to this function will be ignored.

        Parameters
        ----------
        message : str
            commit message to attach to a three-way merge
        dev_branch : str
            name of the branch which should be merge into this branch (`master`)

        Returns
        -------
        str
            commit hash of the new commit for the `master` branch this checkout
            was started from.
        """
        self.__acquire_writer_lock()
        commit_hash = select_merge_algorithm(
            message=message,
            branchenv=self._branchenv,
            stageenv=self._stageenv,
            refenv=self._refenv,
            stagehashenv=self._stagehashenv,
            master_branch_name=self._branch_name,
            dev_branch_name=dev_branch,
            repo_path=self._repo_path,
            writer_uuid=self._writer_lock)

        for asetHandle in self._arraysets.values():
            with suppress(KeyError):
                asetHandle._close()

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_staging_area(
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

    def __acquire_writer_lock(self):
        """Ensures that this class instance holds the writer lock in the database.

        Raises
        ------
        PermissionError
            If the checkout was previously closed (no :attr:``_writer_lock``) or if
            the writer lock value does not match that recorded in the branch db
        """
        try:
            self._writer_lock
        except AttributeError:
            with suppress(AttributeError):
                del self._arraysets
            with suppress(AttributeError):
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
                del self._arraysets
            with suppress(AttributeError):
                del self._metadata
            with suppress(AttributeError):
                del self._differ
            raise e from None

    def __setup(self):
        """setup the staging area appropriately for a write enabled checkout.

        On setup, we cannot be sure what branch the staging area was previously
        checked out on, and we cannot be sure if there are any `uncommitted
        changes` in the staging area (ie. the staging area is `DIRTY`). The
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
        self.__acquire_writer_lock()
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
                cmt = heads.get_branch_head_commit(
                    branchenv=self._branchenv, branch_name=self._branch_name)
                commiting.replace_staging_area_with_commit(
                    refenv=self._refenv, stageenv=self._stageenv, commit_hash=cmt)
                heads.set_staging_branch_head(
                    branchenv=self._branchenv, branch_name=self._branch_name)

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

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
        string
            The commit hash of the new commit.

        Raises
        ------
        RuntimeError
            If no changes have been made in the staging area, no commit occurs.
        """
        self.__acquire_writer_lock()

        open_asets = []
        for arrayset in self._arraysets.values():
            if arrayset._is_conman:
                open_asets.append(arrayset.name)
        open_meta = self._metadata._is_conman

        try:
            if open_meta:
                self._metadata.__exit__()
            for asetn in open_asets:
                self._arraysets[asetn].__exit__()

            if self._differ.status() == 'CLEAN':
                e = RuntimeError('No changes made in staging area. Cannot commit.')
                raise e from None

            self._arraysets._close()
            commit_hash = commiting.commit_records(message=commit_message,
                                                   branchenv=self._branchenv,
                                                   stageenv=self._stageenv,
                                                   refenv=self._refenv,
                                                   repo_path=self._repo_path)
            # purge recs then reopen file handles so that we don't have to invalidate
            # previous weakproxy references like if we just called :meth:``__setup```
            hashs.clear_stage_hash_records(self._stagehashenv)
            self._arraysets._open()

        finally:
            for asetn in open_asets:
                self._arraysets[asetn].__enter__()
            if open_meta:
                self._metadata.__enter__()

        return commit_hash

    def reset_staging_area(self) -> str:
        """Perform a hard reset of the staging area to the last commit head.

        After this operation completes, the writer checkout will automatically
        close in the typical fashion (any held references to :attr:``arrayset``
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
        self.__acquire_writer_lock()
        print(f'Hard reset requested with writer_lock: {self._writer_lock}')

        if self._differ.status() == 'CLEAN':
            e = RuntimeError(f'No changes made in staging area. No reset necessary.')
            raise e from None

        self._arraysets._close()
        hashs.remove_stage_hash_records_from_hashenv(self._hashenv, self._stagehashenv)
        hashs.clear_stage_hash_records(self._stagehashenv)
        hashs.delete_in_process_data(self._repo_path)

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
        self._arraysets = Arraysets._from_staging_area(
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

        Failure to call this method after the writer checkout has been used will
        result in a lock being placed on the repository which will not allow any
        writes until it has been manually cleared.
        """
        self.__acquire_writer_lock()

        if hasattr(self, '_arraysets') and (getattr(self, '_arraysets') is not None):
            self._arraysets._close()

            for asetn in (self._arraysets._arraysets.keys()):
                for attr in list(self._arraysets._arraysets[asetn].__dir__()):
                    with suppress(AttributeError, TypeError):
                        delattr(self._arraysets._arraysets[asetn], attr)

            for attr in list(self._arraysets.__dir__()):
                with suppress(AttributeError, TypeError):
                    # prepending `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                    delattr(self._arraysets, f'_self_{attr}')

        if hasattr(self, '_metadata') and (getattr(self, '_arraysets') is not None):
            for attr in list(self._metadata.__dir__()):
                with suppress(AttributeError, TypeError):
                    # prepending `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                    delattr(self._metadata, f'_self_{attr}')

        with suppress(AttributeError):
            del self._arraysets
        with suppress(AttributeError):
            del self._metadata
        with suppress(AttributeError):
            del self._differ

        heads.release_writer_lock(self._branchenv, self._writer_lock)

        del self._refenv
        del self._hashenv
        del self._labelenv
        del self._stageenv
        del self._branchenv
        del self._stagehashenv
        del self._repo_path
        del self._writer_lock
        del self._branch_name
        del self._repo_stage_path
        del self._repo_store_path
        atexit.unregister(self.close)
        return
