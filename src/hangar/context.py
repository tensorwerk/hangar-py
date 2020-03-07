import configparser
import os
from pathlib import Path
import platform
import shutil
import tempfile
import warnings
from typing import MutableMapping, Optional

import lmdb

from . import __version__
from .constants import (
    CONFIG_USER_NAME,
    DIR_DATA_REMOTE,
    DIR_DATA_STAGE,
    DIR_DATA_STORE,
    DIR_DATA,
    LMDB_BRANCH_NAME,
    LMDB_HASH_NAME,
    LMDB_REF_NAME,
    LMDB_SETTINGS,
    LMDB_STAGE_HASH_NAME,
    LMDB_STAGE_REF_NAME,
    README_FILE_NAME,
)
from .records.commiting import unpack_commit_ref
from .records.heads import (
    create_branch,
    get_branch_head_commit,
    get_staging_branch_head,
    set_staging_branch_head,
)
from .records.parsing import repo_version_raw_spec_from_raw_string
from .records.vcompat import (
    is_repo_software_version_compatible,
    set_repository_software_version,
    startup_check_repo_version,
)
from .utils import readme_contents, is_64bits


class Environments(object):

    def __init__(self, pth: Path):

        self.repo_path: Path = pth
        self.refenv: Optional[lmdb.Environment] = None
        self.hashenv: Optional[lmdb.Environment] = None
        self.stageenv: Optional[lmdb.Environment] = None
        self.branchenv: Optional[lmdb.Environment] = None
        self.stagehashenv: Optional[lmdb.Environment] = None
        self.cmtenv: MutableMapping[str, lmdb.Environment] = {}
        self._startup()

    @property
    def repo_is_initialized(self) -> bool:
        """Property to check if the repository is initialized, read-only attribute

        Returns
        -------
        bool
            True if repo environments are initialized, False otherwise
        """
        ret = True if isinstance(self.refenv, lmdb.Environment) else False
        return ret

    def _startup(self) -> bool:
        """When first access to the Repo starts, attempt to open the db envs.

        This function is designed to fail if a repository does not exist at the
        :py:attribute:`repo_path` which is specified, so the user can
        explicitly choose to initialize the repo. Once opened, the lmdb
        environments should not be closed until the program terminates.

        Returns
        -------
        bool False if no repository exists at the given path, otherwise True

        Warns
        -----
        UserWarning Should the repository not exist at the provided repo path.

        Raises
        ------
        RuntimeError If the repository version is not compatible with the
        current software.
        """

        if not self.repo_path.joinpath(LMDB_BRANCH_NAME).is_file():
            msg = f'No repository exists at {self.repo_path}, please use `repo.init()` method'
            warnings.warn(msg, UserWarning)
            return False

        if not is_64bits():
            raise OSError(f'Hangar cannot run on 32 bit machines')

        repo_ver = startup_check_repo_version(self.repo_path)
        curr_ver = repo_version_raw_spec_from_raw_string(v_str=__version__)
        if not is_repo_software_version_compatible(repo_ver, curr_ver):
            msg = f'repository written version: {repo_ver} is not comatible '\
                  f'with the current Hangar software version: {curr_ver}'
            raise RuntimeError(msg)

        self._open_environments()
        return True

    def init_repo(self,
                  user_name: str,
                  user_email: str,
                  remove_old: bool = False) -> Path:
        """Create a new hangar repositiory at the specified environment path.

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
        Path
            The path to the newly created repository on disk.

        Raises
        ------
        OSError
            If a hangar repository exists at the specified path, and `remove_old`
            was not set to ``True``.
        """
        if self.repo_path.joinpath(LMDB_BRANCH_NAME).is_file():
            if remove_old is True:
                shutil.rmtree(str(self.repo_path))
            else:
                raise OSError(f'Hangar Directory: {self.repo_path} already exists')

        self.repo_path.mkdir()
        self.repo_path.joinpath(DIR_DATA_STORE).mkdir()
        self.repo_path.joinpath(DIR_DATA_STAGE).mkdir()
        self.repo_path.joinpath(DIR_DATA_REMOTE).mkdir()
        self.repo_path.joinpath(DIR_DATA).mkdir()
        print(f'Hangar Repo initialized at: {self.repo_path}')

        userConf = {'USER': {'name': user_name, 'email': user_email}}
        CFG = configparser.ConfigParser()
        CFG.read_dict(userConf)
        with self.repo_path.joinpath(CONFIG_USER_NAME).open('w') as f:
            CFG.write(f)

        readmeTxt = readme_contents(user_name, user_email)
        with self.repo_path.joinpath(README_FILE_NAME).open('w') as f:
            f.write(readmeTxt.getvalue())

        self._open_environments()
        set_repository_software_version(branchenv=self.branchenv, ver_str=__version__)
        create_branch(self.branchenv, 'master', '')
        set_staging_branch_head(self.branchenv, 'master')
        return self.repo_path

    def checkout_commit(self, branch_name: str = '', commit: str = '') -> str:
        """Set up db environment with unpacked commit ref records.

        Parameters
        ----------
        branch_name : str, optional
            name of the branch to read, defaults to ''
        commit : str, optional
            name of the commit to read, defaults to ''

        Returns
        -------
        str
            commit hash which was checked out
        """
        if commit != '':
            commit_hash = commit
            txt = f' * Checking out COMMIT: {commit_hash}'
        elif branch_name != '':
            commit_hash = get_branch_head_commit(self.branchenv, branch_name)
            txt = f' * Checking out BRANCH: {branch_name} with current HEAD: {commit_hash}'
        else:
            head_branch = get_staging_branch_head(self.branchenv)
            commit_hash = get_branch_head_commit(self.branchenv, head_branch)
            txt = f'\n Neither BRANCH or COMMIT specified.'\
                  f'\n * Checking out writing HEAD BRANCH: {head_branch}'
        print(txt)

        # On UNIX-like system, an open process still retains ability to
        # interact with disk space allocated to a file when it is removed from
        # disk. Windows does not, and will not allow file to be removed if a
        # process is interacting with it. While the CM form is cleaner, this
        # hack allows similar usage on Windows platforms.

        if platform.system() != 'Windows':
            with tempfile.TemporaryDirectory() as tempD:
                tmpDF = os.path.join(tempD, f'{commit_hash}.lmdb')
                tmpDB = lmdb.open(path=tmpDF, **LMDB_SETTINGS)
                unpack_commit_ref(self.refenv, tmpDB, commit_hash)
                self.cmtenv[commit_hash] = tmpDB
        else:
            tempD = tempfile.mkdtemp()
            tmpDF = os.path.join(tempD, f'{commit_hash}.lmdb')
            tmpDB = lmdb.open(path=tmpDF, **LMDB_SETTINGS)
            unpack_commit_ref(self.refenv, tmpDB, commit_hash)
            self.cmtenv[commit_hash] = tmpDB

        return commit_hash

    def _open_environments(self):
        """Open the standard lmdb databases at the repo path.

        If any commits are checked out (in an unpacked state), read those in as
        well.
        """
        ref_pth = str(self.repo_path.joinpath(LMDB_REF_NAME))
        hash_pth = str(self.repo_path.joinpath(LMDB_HASH_NAME))
        stage_pth = str(self.repo_path.joinpath(LMDB_STAGE_REF_NAME))
        branch_pth = str(self.repo_path.joinpath(LMDB_BRANCH_NAME))
        stagehash_pth = str(self.repo_path.joinpath(LMDB_STAGE_HASH_NAME))

        self.refenv = lmdb.open(path=ref_pth, **LMDB_SETTINGS)
        self.hashenv = lmdb.open(path=hash_pth, **LMDB_SETTINGS)
        self.stageenv = lmdb.open(path=stage_pth, **LMDB_SETTINGS)
        self.branchenv = lmdb.open(path=branch_pth, **LMDB_SETTINGS)
        self.stagehashenv = lmdb.open(path=stagehash_pth, **LMDB_SETTINGS)

    def _close_environments(self):

        self.refenv.close()
        self.hashenv.close()
        self.stageenv.close()
        self.branchenv.close()
        self.stagehashenv.close()
        for env in self.cmtenv.values():
            if platform.system() == 'Windows':
                envpth = env.path()
                env.close()
                os.remove(envpth)
            else:
                env.close()
