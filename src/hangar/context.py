import logging
import os
import shutil
import platform
import tempfile
from typing import MutableMapping
from collections import Counter
from os.path import dirname
from os.path import join as pjoin

import lmdb
import yaml

from . import config
from .diagnostics.ecosystem import get_versions

logger = logging.getLogger(__name__)


class TxnRegisterSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(TxnRegisterSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class TxnRegister(metaclass=TxnRegisterSingleton):
    '''Singleton to manage transaction thread safety in lmdb databases.

    This is essentailly a reference counting transaction register, lots of room
    for improvement here.
    '''

    def __init__(self):
        self.WriterAncestors = Counter()
        self.ReaderAncestors = Counter()
        self.WriterTxn: MutableMapping[lmdb.Environment, lmdb.Transaction] = {}
        self.ReaderTxn: MutableMapping[lmdb.Environment, lmdb.Transaction] = {}

    def begin_writer_txn(self, lmdbenv: lmdb.Environment,
                         buffer: bool = False) -> lmdb.Transaction:
        '''Start a write enabled transaction on the given environment

        If multiple write transactions are requested for the same handle, only
        one instance of the transaction handle will be returened, and will not
        close until all operations on that handle have requested to close

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment to open the transaction on
        buffer : bool, optional
            if buffer objects should be used (the default is False, which does
            not use buffers)

        Returns
        -------
        lmdb.Transaction
            transaction handle to perform operations on
        '''
        if self.WriterAncestors[lmdbenv] == 0:
            self.WriterTxn[lmdbenv] = lmdbenv.begin(write=True, buffers=buffer)
        self.WriterAncestors[lmdbenv] += 1
        return self.WriterTxn[lmdbenv]

    def begin_reader_txn(self, lmdbenv: lmdb.Environment,
                         buffer: bool = False) -> lmdb.Transaction:
        '''Start a reader only txn for the given environment

        If there a read-only transaction for the same environment already exists
        then the same reader txn handle will be returned, and will not close
        until all operations on that handle have said they are finished.

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment to start the transaction in.
        buffer : bool, optional
            weather a buffer transaction should be used (the default is False,
            which means no buffers are returned)

        Returns
        -------
        lmdb.Transaction
            handle to the lmdb transaction.
        '''
        if self.ReaderAncestors[lmdbenv] == 0:
            self.ReaderTxn[lmdbenv] = lmdbenv.begin(write=False, buffers=buffer)
        self.ReaderAncestors[lmdbenv] += 1
        return self.ReaderTxn[lmdbenv]

    def commit_writer_txn(self, lmdbenv: lmdb.Environment) -> bool:
        '''Commit changes made in a write-enable transaction handle

        As multiple objects can have references to the same open transaction handle,
        the data is not actually committed until all open transactions have called
        the commit method.

        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment handle used to open the transaction

        Raises
        ------
        RuntimeError
            If the internal reference counting gets out of sync

        Returns
        -------
        bool
            True if this operation actually committed, otherwise false
            if other objects have references to the same (open) handle
        '''
        ancestors = self.WriterAncestors[lmdbenv]
        if ancestors == 0:
            msg = f'hash ancestors are zero but commit called on {lmdbenv}'
            err = RuntimeError(msg)
            logger.error(msg)
            raise err
        elif ancestors == 1:
            self.WriterTxn[lmdbenv].commit()
            self.WriterTxn.__delitem__(lmdbenv)
            ret = True
        else:
            ret = False
        self.WriterAncestors[lmdbenv] -= 1
        return ret

    def abort_reader_txn(self, lmdbenv: lmdb.Environment) -> bool:
        '''Request to close a read-only transaction handle

        As multiple objects can have references to the same open transaction
        handle, the transaction is not actuall aborted until all open transactions
        have called the abort method


        Parameters
        ----------
        lmdbenv : lmdb.Environment
            the environment handle used to open the transaction

        Raises
        ------
        RuntimeError
            If the internal reference counting gets out of sync.

        Returns
        -------
        bool
            True if this operation actually aborted the transaction,
            otherwise False if other objects have references to the same (open)
            handle.
        '''
        ancestors = self.ReaderAncestors[lmdbenv]
        if ancestors == 0:
            raise RuntimeError(f'hash ancestors are zero but commit called')
        elif ancestors == 1:
            self.ReaderTxn[lmdbenv].abort()
            self.ReaderTxn.__delitem__(lmdbenv)
            ret = True
        else:
            ret = False
        self.ReaderAncestors[lmdbenv] -= 1
        return ret


'''
Todo, refactor to avoid the need for these imports to be below TxnRegister,
if they aren't right now, we get circular imports...
'''

from .records import commiting, heads


class EnvironmentsSingleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        repo_path = kwargs['repo_path']
        if repo_path not in cls._instances:
            cls._instances[repo_path] = super(EnvironmentsSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[repo_path]


class Environments(metaclass=EnvironmentsSingleton):

    def __init__(self, repo_path: str):

        self.repo_path: str = repo_path
        self.refenv: lmdb.Environment = None
        self.hashenv: lmdb.Environment = None
        self.stageenv: lmdb.Environment = None
        self.branchenv: lmdb.Environment = None
        self.labelenv: lmdb.Environment = None
        self.stagehashenv: lmdb.Environment = None
        self.cmtenv: MutableMapping[str, lmdb.Environment] = {}
        self._startup()

    @property
    def repo_is_initialized(self):
        '''Property to check if the repository is initialized, read-only attribute

        Returns
        -------
        bool
            True if repo environments are initialized, False otherwise
        '''
        environmentInitialized = isinstance(self.refenv, lmdb.Environment)
        return environmentInitialized

    def _startup(self) -> bool:
        '''When first access to the Repo starts, attempt to open the lmdb.Environments.

        This function is designed to fail if a repository does not exist at the
        :py:attribute:`repo_path` which is specified, so the user can explicitly choose
        to initialize the repo. Once opened, the lmdb environments should not be
        closed until the program terminates.

        Returns
        -------
        bool
            False if no repository exists at the given path, otherwise True

        '''
        if not os.path.isdir(self.repo_path):
            msg = f'HANGAR RUNTIME WARNING: no repository exists at {self.repo_path}, '\
                  f'please use `init_repo` function'
            logger.warning(msg)
            return False

        config.refresh(paths=[self.repo_path])
        self._open_environments()
        return True

    def _init_repo(self, user_name: str, user_email: str, remove_old: bool = False) -> str:
        '''Create a new hangar repositiory at the specified environment path.

        Parameters
        ----------
        user_name : str
            Name of the repository user.
        user_email : str
            Email address of the respository user.
        remove_old : bool, optional(default value = False)
            DEVELOPER USE ONLY --- Remove all data and records stored in the
            repository if this opetion is enabled, defaults to False.

        Returns
        -------
        str
            The path to the newly created repository on disk.

        Raises
        ------
        OSError
            If a hangar repository exists at the specified path, and `remove_old`
            was not set to ``True``.
        '''
        if os.path.isdir(self.repo_path):
            if remove_old is True:
                shutil.rmtree(self.repo_path)
            else:
                raise OSError(f'invariant dir: {self.repo_path} already exists')

        os.makedirs(pjoin(self.repo_path, config.get('hangar.repository.store_data_dir')))
        os.makedirs(pjoin(self.repo_path, config.get('hangar.repository.stage_data_dir')))
        os.makedirs(pjoin(self.repo_path, config.get('hangar.repository.remote_data_dir')))
        os.makedirs(pjoin(self.repo_path, config.get('hangar.repository.data_dir')))
        logger.info(f'Hangar Repo initialized at: {self.repo_path}')

        config.ensure_file(
            source=pjoin(dirname(__file__), 'config_hangar.yml'), destination=self.repo_path)
        config.ensure_file(
            source=pjoin(dirname(__file__), 'config_user.yml'), destination=self.repo_path)

        with open(pjoin(self.repo_path, 'config_user.yml')) as f:
            userConf = yaml.safe_load(f)

        userConf['user']['name'] = user_name
        userConf['user']['email'] = user_email

        with open(os.path.join(self.repo_path, 'config_user.yml'), 'w') as f:
            yaml.safe_dump(userConf, f, default_flow_style=False)

        config.refresh(paths=[self.repo_path])
        logger.debug(f'{get_versions()}')

        self._open_environments()
        heads.create_branch(self.branchenv, 'master', '')
        heads.set_staging_branch_head(self.branchenv, 'master')
        return self.repo_path

    def checkout_commit(self, branch_name: str = '', commit: str = '') -> str:
        '''Set up environments for read only access & with records referencing commit hash dbs.

        Parameters
        ----------
        repo_pth : str
            path to the repository directory on the local disk
        branch_name : str, optional
            name of the branch to read, defaults to ''
        commit : str, optional
            name of the commit to read, defaults to ''

        Returns
        -------
        str
            commit hash which was checked out
        '''
        if commit != '':
            commit_hash = commit
            txt = f' * Checking out COMMIT: {commit_hash}'
        elif branch_name != '':
            commit_hash = heads.get_branch_head_commit(self.branchenv, branch_name)
            txt = f' * Checking out BRANCH: {branch_name} with current HEAD: {commit_hash}'
        else:
            head_branch = heads.get_staging_branch_head(self.branchenv)
            commit_hash = heads.get_branch_head_commit(self.branchenv, head_branch)
            txt = f'\n Warning: Neither BRANCH or COMMIT specified.'\
                  f'\n * Checking out writing HEAD BRANCH: {head_branch} at COMMIT: {commit_hash}'
        logger.info(txt)

        # On UNIX-like system, an open process still retains ability to interact
        # with disk space allocated to a file when it is removed from disk.
        # Windows does not, and will not allow file to be removed if a process
        # is interacting with it. While the CM form is cleaner, this hack allows
        # similar usage on Windows platforms.

        LMDB_CONFIG = config.get('hangar.lmdb')
        if platform.system() != 'Windows':
            with tempfile.TemporaryDirectory() as tempD:
                tmpDF = os.path.join(tempD, f'{commit_hash}.lmdb')
                logger.debug(f'checkout creating temp in CM: {tmpDF}')
                tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
                commiting.unpack_commit_ref(self.refenv, tmpDB, commit_hash)
                self.cmtenv[commit_hash] = tmpDB
        else:
            tempD = tempfile.mkdtemp()
            tmpDF = os.path.join(tempD, f'{commit_hash}.lmdb')
            logger.debug(f'checkout creating temp: {tmpDF}')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
            commiting.unpack_commit_ref(self.refenv, tmpDB, commit_hash)
            self.cmtenv[commit_hash] = tmpDB

        return commit_hash

    def _open_environments(self):
        '''Open the standard lmdb databases at the repo path.

        If any commits are checked out (in an unpacked state), read those in as well.
        '''
        LMDB_CONFIG = config.get('hangar.lmdb')

        BRANCH_DB_PTH = config.get('hangar.repository.branch_db_pth')
        HASH_DB_PTH = config.get('hangar.repository.hash_db_pth')
        LABEL_DB_PTH = config.get('hangar.repository.label_db_pth')
        REF_DB_PTH = config.get('hangar.repository.ref_db_pth')
        STAGE_DB_PTH = config.get('hangar.repository.stage_db_pth')
        STAGE_HASH_DB_PTH = config.get('hangar.repository.stage_hash_db_pth')

        self.refenv = lmdb.open(path=pjoin(self.repo_path, REF_DB_PTH), **LMDB_CONFIG)
        self.hashenv = lmdb.open(path=pjoin(self.repo_path, HASH_DB_PTH), **LMDB_CONFIG)
        self.stageenv = lmdb.open(path=pjoin(self.repo_path, STAGE_DB_PTH), **LMDB_CONFIG)
        self.branchenv = lmdb.open(path=pjoin(self.repo_path, BRANCH_DB_PTH), **LMDB_CONFIG)
        self.labelenv = lmdb.open(path=pjoin(self.repo_path, LABEL_DB_PTH), **LMDB_CONFIG)
        self.stagehashenv = lmdb.open(path=pjoin(self.repo_path, STAGE_HASH_DB_PTH), **LMDB_CONFIG)

    def _close_environments(self):

        self.refenv.close()
        self.hashenv.close()
        self.stageenv.close()
        self.branchenv.close()
        self.labelenv.close()
        self.stagehashenv.close()
        for env in self.cmtenv.values():
            if platform.system() == 'Windows':
                envpth = env.path()
                env.close()
                os.remove(envpth)
            else:
                env.close()
