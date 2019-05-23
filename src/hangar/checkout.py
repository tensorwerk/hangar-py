import logging
from os.path import join as pjoin
from uuid import uuid4
import weakref

import numpy as np

from . import config
from .dataset import Datasets
from .diff import ReaderUserDiff, WriterUserDiff
from .merger import select_merge_algorithm
from .metadata import MetadataReader
from .metadata import MetadataWriter
from .records import commiting
from .records import hashs
from .records import heads
from .utils import cm_weakref_obj_proxy

logger = logging.getLogger(__name__)

STORE_DATA_DIR = config.get('hangar.repository.store_data_dir')
STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')


class ReaderCheckout(object):
    '''Checkout the repository as it exists at a particular branch.

    if a commit hash is provided, it will take precedent over the branch name
    parameter. If neither a branch not commit is specified, the staging
    environment's base branch HEAD commit hash will be read.

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
    '''

    def __init__(self, base_path, labelenv, dataenv, hashenv, branchenv, refenv, commit):
        self._commit_hash = commit
        self._repo_path = base_path
        self._labelenv = labelenv
        self._dataenv = dataenv
        self._hashenv = hashenv
        self._branchenv = branchenv
        self._refenv = refenv

        self._metadata = MetadataReader(
            dataenv=self._dataenv,
            labelenv=self._labelenv)
        self._datasets = Datasets._from_commit(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            cmtrefenv=self._dataenv)
        self._differ = ReaderUserDiff(
            commit_hash=self._commit_hash,
            branchenv=self._branchenv,
            refenv=self._refenv)

    def _repr_pretty_(self, p, cycle):
        '''pretty repr for printing in jupyter notebooks
        '''
        self.__verify_checkout_alive()
        res = f'\n Hangar {self.__class__.__name__}\
                \n     Writer       : False\
                \n     Commit Hash  : {self._commit_hash}\
                \n     Num Datasets : {len(self._datasets)}\
                \n     Num Metadata : {len(self._metadata)}\n'
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
        '''Validates that the checkout object has not been closed

        Raises
        ------
        PermissionError
            if the checkout was previously close
        '''
        try:
            assert hasattr(self, '_metadata')
            assert hasattr(self, '_datasets')
        except AssertionError:
            err = f'Unable to operate on past checkout objects which have been '\
                  f'closed. No operation occured. Please use a new checkout.'
            raise PermissionError(err) from None

    @property
    def datasets(self):
        '''Provides access to dataset interaction object.

        .. seealso::

            The class :class:`hangar.dataset.Datasets` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the datasets object which behaves exactally like a
            datasets accessor class but which can be invalidated when the writer
            lock is released.
        '''
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._datasets)
        return wr

    @property
    def metadata(self):
        '''Provides access to metadata interaction object.

        .. seealso::

            The class :class:`hangar.metadata.MetadataReader` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the metadata object which behaves exactally like a
            metadata class but which can be invalidated when the writer lock is
            released.
        '''
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

    @property
    def diff(self):
        '''Access the differ methods for a read-only checkout.

        .. seealso::

            The class :class:`hangar.diff.ReaderUserDiff` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the differ object (and contained methods) which behaves
            exactally like the differ class but which can be invalidated when the
            writer lock is released.
        '''
        self.__verify_checkout_alive()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def commit_hash(self):
        '''Commit hash this read-only checkout's data is read from.

        Returns
        -------
        string
            commit hash of the checkout
        '''
        self.__verify_checkout_alive()
        return self._commit_hash

    def close(self):
        '''Gracefully close the reader checkout object.

        Though not strictly required for reader checkouts (as opposed to
        writers), closing the checkout after reading will free file handles and
        system resources, which may improve performance for repositories with
        multiple simultaneous read checkouts.
        '''
        self.__verify_checkout_alive()
        for dsetHandle in self._datasets.values():
            dsetHandle._close()
        del self._datasets
        del self._metadata
        del self._differ


# --------------- Write enabled checkout ---------------------------------------


class WriterCheckout(object):
    '''Checkout the repository at the head of a given branch for writing.

    This is the entry point for all writing operations to the repository, the
    writer class records all interactions in a special "staging" area, which is
    based off the state of the repository as it existed at the HEAD commit of a
    branch.

    At the moment, only one instance of this class can write data to the staging
    area at a time. After the desired operations have been completed, it is
    crucial to call :py:meth:`close` to release the writer lock. In addition,
    after any changes have been made to the staging area, the branch HEAD cannot
    be changed. In order to checkout another branch HEAD for writing, you must
    either commit the changes, or perform a hard-reset of the staging area to
    the last commit.

    Parameters
    ----------
    repo_pth : str
        local file path of the repository.
    branch_name : str
        name of the branch whose HEAD commit will for the starting state
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
    '''

    def __init__(self, repo_pth, branch_name, labelenv, hashenv, refenv,
                 stageenv, branchenv, stagehashenv, mode='a'):
        self._branch_name = branch_name
        self._writer_lock = str(uuid4())
        self._repo_path = repo_pth
        self._repo_stage_path = pjoin(self._repo_path, STAGE_DATA_DIR)
        self._repo_store_path = pjoin(self._repo_path, STORE_DATA_DIR)
        self._labelenv = labelenv
        self._stageenv = stageenv
        self._hashenv = hashenv
        self._refenv = refenv
        self._branchenv = branchenv
        self._stagehashenv = stagehashenv
        self._datasets: Datasets = {}
        self._metadata: MetadataWriter = None
        self._differ: WriterUserDiff = None
        self.__setup()

    def _repr_pretty_(self, p, cycle):
        '''pretty repr for printing in jupyter notebooks
        '''
        self.__acquire_writer_lock()
        res = f'\n Hangar {self.__class__.__name__}\
                \n     Writer       : True\
                \n     Base Branch  : {self._branch_name}\
                \n     Num Datasets : {len(self._datasets)}\
                \n     Num Metadata : {len(self._metadata)}\n'
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
    def datasets(self):
        '''Provides access to dataset interaction object.

        .. seealso::

            The class :class:`hangar.dataset.Datasets` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the datasets object which behaves exactally like a
            datasets accessor class but which can be invalidated when the writer
            lock is released.
        '''
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._datasets)
        return wr

    @property
    def metadata(self):
        '''Provides access to metadata interaction object.

        .. seealso::

            The class :class:`hangar.metadata.MetadataWriter` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the metadata object which behaves exactally like a
            metadata class but which can be invalidated when the writer lock is
            released.
        '''
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

    @property
    def diff(self):
        '''Access the differ methods which are aware of any staged changes.

        .. seealso::

            The class :class:`hangar.diff.WriterUserDiff` contains all methods accessible
            by this property accessor

        Returns
        -------
        weakref.proxy
            weakref proxy to the differ object (and contained methods) which behaves
            exactally like the differ class but which can be invalidated when the
            writer lock is released.
        '''
        self.__acquire_writer_lock()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def branch_name(self):
        '''Branch this write enabled checkout's staging area was based on.

        Returns
        -------
        string
            name of the branch whose commit HEAD changes are staged from.
        '''
        self.__acquire_writer_lock()
        return self._branch_name

    @property
    def commit_hash(self):
        '''Commit hash which the staging area of :ref:`branch_name` is based on.

        Returns
        -------
        string
            commit hash
        '''
        self.__acquire_writer_lock()
        commit_hash = heads.get_branch_head_commit(
            branchenv=self._branchenv, branch_name=self._branch_name)
        return commit_hash

    def merge(self, message: str, dev_branch: str) -> str:
        '''Merge the currently checked out commit with the provided branch name.

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
        '''
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

        for dsetHandle in self._datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass

        self._metadata = MetadataWriter(
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._datasets = Datasets._from_staging_area(
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
        '''Ensures that this class instance holds the writer lock in the database.
        '''
        try:
            self._writer_lock
        except AttributeError:
            try:
                del self._datasets
            except AttributeError:
                pass
            try:
                del self._metadata
            except AttributeError:
                pass
            try:
                del self._differ
            except AttributeError:
                pass
            err = f'Unable to operate on past checkout objects which have been '\
                  f'closed. No operation occured. Please use a new checkout.'
            logger.error(err, exc_info=0)
            raise PermissionError(err) from None

        try:
            heads.acquire_writer_lock(self._branchenv, self._writer_lock)
        except PermissionError as e:
            try:
                del self._datasets
            except AttributeError:
                pass
            try:
                del self._metadata
            except AttributeError:
                pass
            try:
                del self._differ
            except AttributeError:
                pass
            logger.error(e, exc_info=0)
            raise e from None

    def __setup(self):
        '''setup the staging area appropriatly for a write enabled checkout.

        On setup, we cannot be sure what branch the staging area was previously
        checked out on, and we cannot be sure if there are any `uncommitted
        changes` in the staging area (ie. the staging area is `DIRTY`). The
        setup methods here ensure that we can safetly make any changes to the
        staging area without overwriting uncommitted changes, and then perform
        the setup steps to checkout staging area state at that point in time.

        Raises
        ------
        ValueError
            if there are changes previously made in the staging area which were
            based on one branch's HEAD, but a different branch was specified to
            be used for the base of this checkout.
        '''
        self.__acquire_writer_lock()
        current_head = heads.get_staging_branch_head(self._branchenv)
        currentDiff = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=current_head)
        if currentDiff.status() == 'DIRTY':
            if current_head != self._branch_name:
                err = f'Unable to check out branch: {self._branch_name} for writing as '\
                      f'the staging area has uncommitted changes on branch: {current_head}. '\
                      f'Please commit or stash uncommited changes before checking out a '\
                      f'different branch for writing.'
                self.close()
                logger.error(err, exc_info=1)
                raise ValueError(err)
        else:
            if current_head != self._branch_name:
                commit_hash = heads.get_branch_head_commit(
                    branchenv=self._branchenv,
                    branch_name=self._branch_name)

                commiting.replace_staging_area_with_commit(
                    refenv=self._refenv,
                    stageenv=self._stageenv,
                    commit_hash=commit_hash)

                heads.set_staging_branch_head(
                    branchenv=self._branchenv,
                    branch_name=self._branch_name)

        self._metadata = MetadataWriter(
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._datasets = Datasets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

    def commit(self, commit_message):
        '''commit the changes made in the staging area.

        This creates a new commit on the checkout branch.

        Parameters
        ----------
        commit_message : str, optional
            log of what was changed in this commit, defaults to ''

        Returns
        -------
        string
            The commit hash of the new commit

        Raises
        ------
        RuntimeError
            If no changes have been made in the staging area, no commit occurs.
        '''
        self.__acquire_writer_lock()
        logger.info(f'Commit operation requested with message: {commit_message}')
        if self._differ.status() == 'CLEAN':
            msg = f'HANGAR RUNTIME ERROR: No changes made in staging area. Cannot commit.'
            e = RuntimeError(msg)
            logger.error(e, exc_info=False)
            raise e

        for dsetHandle in self._datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        hashs.remove_unused_dataset_hdf5(self._repo_path, self._stagehashenv)

        commit_hash = commiting.commit_records(
            message=commit_message,
            branchenv=self._branchenv,
            stageenv=self._stageenv,
            refenv=self._refenv,
            repo_path=self._repo_path)

        hashs.clear_stage_hash_records(self._stagehashenv)
        # reopen the hdf5 file handles so that we don't have to invalidate
        # previous weakproxy references like if we just called `__setup`
        for dsetH in self._datasets.values():
            dsetH._setup_file_access()

        logger.info(f'Commit completed. Commit hash: {commit_hash}')
        return commit_hash

    def reset_staging_area(self):
        '''Perform a hard reset of the staging area to the last commit head.

        After this operation is performed, the writer checkout will be
        automatically closed in the typical fashion (any held references to
        `dataset` or `metadata` objects will finalize and destruct as normal),
        In order to perform any further operation, a new checkout needs to be
        opened.

        .. warning::

            This operation is IRREVERSIBLE. all records and data which are note
            stored in a previous commit will be premenantly deleted.

        Returns
        -------
        string
            commit hash of the head which the staging area is reset to.

        Raises
        ------
        RuntimeError
            If no changes have been made to the staging area, no reset is necessary.
        '''
        self.__acquire_writer_lock()
        logger.info(f'Hard reset requested with writer_lock: {self._writer_lock}')

        if self._differ.status() == 'CLEAN':
            msg = f'HANGAR RUNTIME ERROR: No changes made in staging area. No reset necessary.'
            e = RuntimeError(msg)
            logger.error(e, exc_info=False)
            raise e

        for dsetHandle in self._datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        hashs.remove_stage_hash_records_from_hashenv(self._hashenv, self._stagehashenv)
        hashs.clear_stage_hash_records(self._stagehashenv)
        hashs.remove_unused_dataset_hdf5(self._repo_path, self._stagehashenv)

        branch_head = heads.get_staging_branch_head(self._branchenv)
        head_commit = heads.get_branch_head_commit(self._branchenv, branch_head)
        commiting.replace_staging_area_with_commit(
            refenv=self._refenv,
            stageenv=self._stageenv,
            commit_hash=head_commit)

        logger.info(f'Hard reset completed, staging area head commit: {head_commit}')
        self.close()
        return head_commit

    def close(self):
        '''Close all handles to the writer checkout and release the writer lock.

        Failure to call this method after the writer checkout has been used will
        result in a lock being placed on the repository which will not allow any
        writes until it has been manually cleared.
        '''
        self.__acquire_writer_lock()
        for dsetHandle in self._datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        del self._datasets
        del self._metadata
        del self._differ
        logger.info(f'writer checkout of {self._branch_name} closed')
        heads.release_writer_lock(self._branchenv, self._writer_lock)
        del self._writer_lock
