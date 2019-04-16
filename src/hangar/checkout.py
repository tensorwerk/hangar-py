import logging
from os.path import join as pjoin
from uuid import uuid4

from . import config
from .dataset import Datasets
from .diff import staging_area_status
from .metadata import MetadataReader
from .metadata import MetadataWriter
from .records import commiting
from .records import hashs
from .records import heads

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
    hashenv lmdb.Environment
        db where the hash records are stored.
    commit : str
        specific commit hash to checkout
    '''

    def __init__(self, base_path, labelenv, dataenv, hashenv, commit):
        self._commit_hash = commit
        self._repo_path = base_path
        self._labelenv = labelenv
        self._dataenv = dataenv
        self._hashenv = hashenv
        self.metadata = MetadataReader(dataenv=self._dataenv, labelenv=self._labelenv)
        self.datasets = Datasets._from_commit(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            cmtrefenv=self._dataenv)

    def _repr_pretty_(self, p, cycle):
        res = f'\n Hangar {self.__class__.__name__}\
                \n     Writer       : False\
                \n     Commit Hash  : {self._commit_hash}\n'
        p.text(res)

    def __repr__(self):
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'labelenv={self._labelenv} '\
              f'dataenv={self._dataenv} '\
              f'hashenv={self._hashenv} '\
              f'commit={self._commit_hash})'
        return res

    def close(self):
        '''Gracefully close the reader checkout object.

        Though not strictly required for reader checkouts (as opposed to
        writers), closing the checkout after reading will free file handles and
        system resources, which may improve performance for repositories with
        multiple simultaneous read checkouts.
        '''
        for dsetHandle in self.datasets.values():
            dsetHandle._close()


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
        self.__writer_lock = str(uuid4())
        self._repo_path = repo_pth
        self._repo_stage_path = pjoin(self._repo_path, STAGE_DATA_DIR)
        self._repo_store_path = pjoin(self._repo_path, STORE_DATA_DIR)
        self._labelenv = labelenv
        self._stageenv = stageenv
        self._hashenv = hashenv
        self._refenv = refenv
        self._branchenv = branchenv
        self._stagehashenv = stagehashenv
        self.datasets = None
        self.metadata = None
        self.__setup()

    def _repr_pretty_(self, p, cycle):
        self.__acquire_writer_lock()
        res = f'\n Hangar {self.__class__.__name__}\
                \n     Writer       : True\
                \n     Base Branch  : {self._branch_name}\n'
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

    def __acquire_writer_lock(self):
        '''Ensures that this class instance holds the writer lock in the database.
        '''
        try:
            self.__writer_lock
        except AttributeError:
            self.datasets = None
            self.metadata = None
            err = f'Unable to operate on past checkout objects which have been '\
                  f'closed. No operation occured. Please use a new checkout.'
            logger.error(err, exc_info=1)
            raise PermissionError(err)

        try:
            heads.acquire_writer_lock(self._branchenv, self.__writer_lock)
        except PermissionError as e:
            self.datasets = None
            self.metadata = None
            logger.error(e, exc_info=1)
            raise e

    def __setup(self):
        self.__acquire_writer_lock()
        staged_status = staging_area_status(self._stageenv, self._refenv, self._branchenv)
        current_head = heads.get_staging_branch_head(self._branchenv)
        if staged_status == 'DIRTY':
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

        self.metadata = MetadataWriter(dataenv=self._stageenv, labelenv=self._labelenv)
        self.datasets = Datasets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)

    def commit(self, commit_message=''):
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
        False
            If no changes could be made
        '''
        self.__acquire_writer_lock()
        if staging_area_status(self._stageenv, self._refenv, self._branchenv) == 'CLEAN':
            logger.info('No changes made to the repository. Cannot commit')
            return False

        for dsetHandle in self.datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        hashs.remove_unused_dataset_hdf_files(self._repo_path, self._stagehashenv)

        commit_hash = commiting.commit_records(
            message=commit_message,
            branchenv=self._branchenv,
            stageenv=self._stageenv,
            refenv=self._refenv,
            repo_path=self._repo_path)

        hashs.clear_stage_hash_records(self._stagehashenv)
        self.__setup()
        return commit_hash

    def reset_staging_area(self):
        '''Perform a hard reset of the staging area to the last commit head.

        .. warning::

            This operation is irreversable. all records and data will be deleted.

        Returns
        -------
        bool
            if the operation was successful
        '''
        logger.info(f'Hard reset of staging area requested with writer_lock: {self.__writer_lock}')

        self.__acquire_writer_lock()
        if staging_area_status(self._stageenv, self._refenv, self._branchenv) == 'CLEAN':
            logger.info('No changes made to the repository. No reset is necessary.')
            return False

        for dsetHandle in self.datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        hashs.remove_stage_hash_records_from_hashenv(self._hashenv, self._stagehashenv)
        hashs.clear_stage_hash_records(self._stagehashenv)
        hashs.remove_unused_dataset_hdf_files(self._repo_path, self._stagehashenv)

        branch_head = heads.get_staging_branch_head(self._branchenv)
        head_commit = heads.get_branch_head_commit(self._branchenv, branch_head)
        commiting.replace_staging_area_with_commit(
            refenv=self._refenv,
            stageenv=self._stageenv,
            commit_hash=head_commit)

        logger.info(f'Hard reset completed')
        self.__setup()
        return True

    def close(self):
        '''Close all handles to the writer checkout and release the writer lock.

        Failure to call this method after the writer checkout has been used will
        result in a lock being placed on the repository which will not allow any
        writes until it has been manually cleared.
        '''
        heads.release_writer_lock(self._branchenv, self.__writer_lock)
        for dsetHandle in self.datasets.values():
            try:
                dsetHandle._close()
            except KeyError:
                pass
        self.__dict__.__delitem__(f'_{self.__class__.__name__}__writer_lock')
        logger.info(f'writer checkout of {self._branch_name} closed')
        # self.close_lmdb_environments()
