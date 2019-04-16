import os
from typing import Optional
import logging

from tqdm.auto import tqdm

from . import config, diff
from .checkout import ReaderCheckout, WriterCheckout
from .context import Environments
from .diagnostics import graphing
from .records import heads, parsing, summarize, commiting
from .remote.hangar_client import HangarClient


logger = logging.getLogger(__name__)


class Repository(object):
    '''Launching point for all user operations in a Hangar repository.

    All interaction, including the ability to initialize a repo, checkout a
    commit (for either reading or writing), create a branch, merge branches, or
    generally view the contents or state of the local repository starts here.
    Just provide this class instance with a path to an existing Hangar
    repository, or to a directory one should be initialized, and all required
    data for starting your work on the repo will automatically be populated.

    Parameters
    ----------
    path : str
        local directory path where the Hangar repository exists (or initialized)
    '''

    def __init__(self, path):
        path = os.path.join(path, config.get('hangar.repository.hangar_dir_name'))
        path = os.path.expanduser(path)
        self._env = Environments(repo_path=path)
        self._repo_path = self._env.repo_path
        self._client: Optional[HangarClient] = None

    def _repr_pretty_(self, p, cycle):
        '''provide a pretty-printed repr for ipython based user interaction.

        Parameters
        ----------
        p : printer
            io stream printer type object which is provided via ipython
        cycle : bool
            if the pretty-printer detects a cycle or infinite loop. Not a
            concern here since we just output the text and return, no looping
            required.

        '''
        res = f'\
            \n Hangar {self.__class__.__name__}\
            \n     Repository Path  : {self._repo_path}\
            \n     Writer-Lock Free : {heads.writer_lock_held(self._env.branchenv)}\n'
        p.text(res)

    def __repr__(self):
        '''Override the default repr to show useful information to developers.

        Note: the pprint repr (ipython enabled) is seperately defined in
        :py:meth:`_repr_pretty_`. We specialize because we assume that anyone
        operating in a terminal-based interpreter is probably a more advanced
        developer-type, and expects traditional repr information instead of a
        user facing summary of the repo. Though if we're wrong, go ahead and
        feel free to reassign the attribute :) won't hurt our feelings, promise.

        Returns
        -------
        string
            formated representation of the object
        '''
        res = f'{self.__class__}(path={self._repo_path})'
        return res

    def checkout(self, write=False, *, branch_name='master', commit=''):
        '''Checkout the repo at some point in time in either `read` or `write` mode.

        Only one writer instance can exist at a time. Write enabled checkout
        must must create a staging area from the HEAD commit of a branch. On the
        contrary, any number of reader checkouts can exist at the same time and
        can specify either a branch name or a commit hash.

        Parameters
        ----------
        write : bool, optional
            Specify if the checkout is write capable, defaults to False
        branch_name : str, optional
            name of the branch to checkout. This utilizes the state of the repo
            as it existed at the branch HEAD commit when this checkout object
            was instantiated, defaults to 'master'
        commit : str, optional
            specific hash of a commit to use for the checkout (instead of a
            branch HEAD commit). This argument takes precedent over a branch
            name parameter if it is set. Note: this only will be used in
            non-writeable checkouts, defaults to ''

        Returns
        -------
        object
            Checkout object which can be used to interact with the repository
            data
        '''
        try:
            if write is True:
                co = WriterCheckout(
                    repo_pth=self._repo_path,
                    branch_name=branch_name,
                    labelenv=self._env.labelenv,
                    hashenv=self._env.hashenv,
                    refenv=self._env.refenv,
                    stageenv=self._env.stageenv,
                    branchenv=self._env.branchenv,
                    stagehashenv=self._env.stagehashenv)
                return co
            else:
                commit_hash = self._env.checkout_commit(
                    branch_name=branch_name, commit=commit)

                co = ReaderCheckout(
                    base_path=self._repo_path,
                    labelenv=self._env.labelenv,
                    dataenv=self._env.cmtenv[commit_hash],
                    hashenv=self._env.hashenv,
                    commit=commit_hash)
                return co
        except (RuntimeError, ValueError) as e:
            logger.error(e, exc_info=1, extra=self._env.__dict__)
            return None

    def clone(self, user_name: str, user_email: str, remote_address: str,
              *, remove_old: bool = False) -> str:
        '''Download a remote repository to the local disk.

        The clone method implemented here is very similar to a `git clone`
        operation. This method will pull all commit records, history, and data
        which are parents of the remote's `master` branch head commit. If a
        :class:`hangar.repository.Repository` exists at the specified directory,
        the operation will fail.

        Parameters
        ----------
        user_name : str
            Name of the person who will make commits to the repository. This
            information is recorded permanently in the commit records.
        user_email : str
            Email address of the repository user. This information is recorded
            permenantly in any commits created.
        remote_address : str
            location where the
            :class:`hangar.remote.hangar_server.HangarServer` process is
            running and accessable by the clone user.
        remove_old : bool, optional, kwarg only
            DANGER! DEVELOPMENT USE ONLY! If enabled, a
            :class:`hangar.repository.Repository` existing on disk at the same
            path as the requested clone location will be completly removed and
            replaced with the newly cloned repo. (the default is False, which
            will not modify any contents on disk and which will refuse to create
            a repository at a given location if one already exists there.)

        Returns
        -------
        str
            Name of the master branch for the newly cloned repository.
        '''
        self.init(user_name=user_name, user_email=user_email, remove_old=remove_old)
        self.add_remote(remote_name='origin', remote_address=remote_address)
        branch_name = self.fetch(remote_name='origin', branch_name='master', concat_branch_names=False)
        return branch_name

    def fetch(self, remote_name: str, branch_name: str,
              *, concat_branch_names: bool = True) -> str:
        '''Retrieve new commits made on a remote repository branch.

        This is symantecally identical to a `git fetch` command. Any new commits
        along the branch will be retrived, but placed on an isolated branch to
        the local copy (ie. ``remote_name/branch_name``). In order to unify
        histories, simply merge the remote branch into the local branch.

        Parameters
        ----------
        remote_name : str
            name of the remote repository to fetch from (ie. ``origin``)
        branch_name : str
            name of the branch to fetch the commit references for.
        concat_branch_names : bool, optional, kwarg only
            DEVELOPER USE ONLY! TODO: remove this...

        Returns
        -------
        str
            Name of the branch which stores the retrieved commits.
        '''
        address = heads.get_remote_address(branchenv=self._env.branchenv, name=remote_name)
        self._client = HangarClient(envs=self._env, address=address)

        try:
            c_bcommit = heads.get_branch_head_commit(self._env.branchenv, branch_name)
            c_bhistory = summarize.list_history(self._env.refenv, self._env.branchenv, branch_name=branch_name)
            c_bhistory['order']

            s_branch = self._client.fetch_branch_record(branch_name)
            if s_branch.error.code == 0:
                s_bcommit = s_branch.rec.commit
                if s_bcommit == c_bcommit:
                    logger.info(f'server head: {s_bcommit} == client head: {c_bcommit}. No-op')
                    return
                # TODO: way to reject divergences upstream
                elif s_bcommit in c_bhistory:
                    logger.warning(f'REJECTED: server head: {s_bcommit} in client history')
                    return False

        except ValueError:
            s_branch = self._client.fetch_branch_record(branch_name)

        m_all_schemas, m_all_labels = [], []
        res = self._client.fetch_find_missing_commits(branch_name)
        m_commits = res.commits
        for commit in m_commits:
            try:
                schema_res = self._client.fetch_find_missing_schemas(commit)
                missing_schemas = schema_res.schema_digests
                m_all_schemas.extend(missing_schemas)
            except AttributeError:
                pass
            missing_labels = self._client.fetch_find_missing_labels(commit)
            m_all_labels.extend(missing_labels)

        m_schemas, m_labels = set(m_all_schemas), set(m_all_labels)
        for schema in tqdm(m_schemas, desc='Fetch Schemas:'):
            self._client.fetch_schema(schema)
        with self._client.fs as fs:
            fbar, savebar = 0, 0
            ret = 'AGAIN'
            while ret == 'AGAIN':
                m_all_data = []
                for commit in m_commits:
                    missing_hashes = self._client.fetch_find_missing_hash_records(commit)
                    m_all_data.extend(missing_hashes)
                m_data = set(m_all_data)
                ret, fbar, savebar = self._client.fetch_data(m_data, fs, fbar, savebar)
        for label in tqdm(m_labels, desc='Fetch Labels'):
            self._client.fetch_label(label)
        for commit in tqdm(m_commits, desc='Fetch Commits'):
            self._client.fetch_commit_record(commit)

        commiting.move_process_data_to_store(self._repo_path, remote_operation=True)

        if concat_branch_names is True:
            fetch_branch_name = f'{remote_name}/{branch_name}'
        else:
            fetch_branch_name = f'{branch_name}'

        try:
            heads.create_branch(
                branchenv=self._env.branchenv,
                branch_name=fetch_branch_name,
                base_commit=s_branch.rec.commit)
        except ValueError:
            heads.set_branch_head_commit(
                branchenv=self._env.branchenv,
                branch_name=fetch_branch_name,
                commit_hash=s_branch.rec.commit)

        self._client.channel.close()
        return fetch_branch_name

    def push(self, remote_name: str, branch_name: str) -> bool:
        '''push changes made on a local repository to a remote repository.

        This method is symantically identical to a ``git push`` operation.
        Any local updates will be sent to the remote repository.

        .. note::

            The current implementation is not capable of performing a
            ``force push`` operation. As such, remote branches with diverged
            histories to the local repo must be retrieved, locally merged,
            then re-pushed. This feature will be added in the near future.

        Parameters
        ----------
        remote_name : str
            name of the remote repository to make the push on.
        branch_name : str
            Name of the branch to push to the remote. If the branch name does
            not exist on the remote, the it will be created

        Returns
        -------
        bool
            True if the operation succeeded, Otherwise False
        '''
        address = heads.get_remote_address(branchenv=self._env.branchenv, name=remote_name)
        self._client = HangarClient(envs=self._env, address=address)

        c_bcommit = heads.get_branch_head_commit(self._env.branchenv, branch_name)
        c_bhistory = summarize.list_history(
            refenv=self._env.refenv,
            branchenv=self._env.branchenv,
            branch_name=branch_name)

        s_branch = self._client.fetch_branch_record(branch_name)
        if s_branch.error.code == 0:
            s_bcommit = s_branch.rec.commit
            if s_bcommit == c_bcommit:
                logger.warning(f'server head: {s_bcommit} == client head: {c_bcommit}. No-op')
                return False
            elif (s_bcommit not in c_bhistory['order']) and (s_bcommit != ''):
                logger.warning(f'REJECTED: server branch has commits not present on client')
                return False

        m_all_schemas, m_all_data, m_all_labels = [], [], []
        res = self._client.push_find_missing_commits(branch_name)
        m_commits = res.commits
        for commit in m_commits:
            try:
                schema_res = self._client.push_find_missing_schemas(commit)
                missing_schemas = schema_res.schema_digests
                m_all_schemas.extend(missing_schemas)
            except AttributeError:
                pass
            missing_hashes = self._client.push_find_missing_hash_records(commit)
            m_all_data.extend(missing_hashes)
            missing_labels = self._client.push_find_missing_labels(commit)
            m_all_labels.extend(missing_labels)

        m_schemas, m_data, m_labels = set(m_all_schemas), set(m_all_data), set(m_all_labels)
        for schema in tqdm(m_schemas, desc='Push Schema:'):
            self._client.push_schema(schema)
        self._client.push_data(m_data)
        for label in tqdm(m_labels, desc='Push Labels:'):
            self._client.push_label(label)
        for commit in tqdm(m_commits, desc='Push Commits:'):
            self._client.push_commit_record(commit)

        self._client.push_branch_record('master')
        self._client.channel.close()
        return True

    def add_remote(self, remote_name: str, remote_address: str) -> bool:
        '''Add a remote to the repository accessible by `name` at `address`.

        Parameters
        ----------
        remote_name : str
            the name which should be used to refer to the remote server (ie:
            'origin')
        remote_address : str
            the IP:PORT where the hangar server is running

        Returns
        -------
        bool
            True if successful, False if a remote already exists with the
            provided name
        '''
        succ = heads.add_remote(
            branchenv=self._env.branchenv,
            name=remote_name,
            address=remote_address)
        return succ

    def remove_remote(self, remote_name: str) -> str:
        '''Remove a remote repository from the branch records

        Parameters
        ----------
        remote_name : str
            name of the remote to remove the reference to

        Raises
        ------
        KeyError
            If a remote with the provided name does not exist

        Returns
        -------
        str
            The channel address which was removed at the given remote name
        '''
        try:
            rm_address = heads.remove_remote(
                branchenv=self._env.branchenv, name=remote_name)
        except KeyError:
            err = f'No remote reference with name: {remote_name}'
            raise KeyError(err)

        return rm_address

    def init(self, user_name, user_email, remove_old=False):
        '''Initialize a Hangar repositor at the specified directory path.

        This function must be called before a checkout can be performed.

        Parameters
        ----------
        user_name : str
            Name of the repository user.
        user_email : str
            Email address of the respository user.
        remove_old : bool, optional
            DEVELOPER USE ONLY -- remove and reinitialize a Hangar
            repository at the given path, defaults to False

        Returns
        -------
        str
            the full directory path where the Hangar repository was
            initialized on disk.
        '''
        pth = self._env._init_repo(
            user_name=user_name, user_email=user_email, remove_old=remove_old)
        return pth

    def log(self, branch_name=None, commit_hash=None, *, return_contents=False):
        '''Alias for lmdb_utils.list_history() call

        Parameters
        ----------
        branch_name : str
            The name of the branch to start the log process from. (Default value
            = None)
        commit_hash : str
            The commit hash to start the log process from. (Default value = None)

        Returns
        -------
        dict
            Dict containing the commit ancestor graph, and all specifications.
        '''
        res = summarize.list_history(
            refenv=self._env.refenv,
            branchenv=self._env.branchenv,
            branch_name=branch_name,
            commit_hash=commit_hash)

        if return_contents:
            return res
        else:
            g = graphing.Graph()
            g.show_nodes(
                dag=res['ancestors'],
                spec=res['specs'],
                start=res['head'],
                order=res['order'])

    def summary(self, *, branch_name='', commit='', return_contents=False):
        '''Alias for lmdb_utils.summary() call

        Parameters
        ----------
        branch_name : str
            A specific branch name whose head commit will be used as the summary
            point (Default value = '')
        commit : str
            A specific commit hash which should be used as the summary point.
            (Default value = '')
        return_contents : bool
            If true, return a full log of what records are in the repository at
            the summary point. (Default value = False)

        Returns
        -------
        dict
            contents of the entire repository (if `return_contents=True`)
        '''
        ppbuf, res = summarize.summary(self._env, branch_name=branch_name, commit=commit)
        if return_contents is True:
            return res
        else:
            print(ppbuf.getvalue())

    def status(self):
        '''status of the staging area, dirty or clean

        Returns
        -------
        str
            status of the staging area. One of "DIRTY" or "CLEAN"
        '''
        res = diff.staging_area_status(
            stageenv=self._env.stageenv,
            refenv=self._env.refenv,
            branchenv=self._env.branchenv)
        return res

    def _details(self):
        '''DEVELOPER USE ONLY: Dump some details about the underlying db structure to disk.
        '''
        summarize.details(self._env)
        return

    def merge(self, message, master_branch, dev_branch):
        '''Not Implemented

        Parameters
        ----------
        message: str
            Commit message to use for this merge.
        master_branch : str
            name of the master branch to merge into
        dev_branch : str
            name of the dev/feature branch to merge

        Returns
        -------
        str
            Hashof the commit which is written if possible.
        '''
        commit_hash = diff.select_merge_algorithm(
            message=message,
            branchenv=self._env.branchenv,
            stageenv=self._env.stageenv,
            refenv=self._env.refenv,
            stagehashenv=self._env.stagehashenv,
            master_branch_name=master_branch,
            dev_branch_name=dev_branch,
            repo_path=self._repo_path)

        return commit_hash

    def diff_commit(self, master_commit, dev_commit):
        '''Returns the diff of two commit hashes

        Parameters
        ----------
        master_commit : str
            commit hash to serve as the "master" branch
        dev_commit : str
            commit hash to serve as the "dev" branch

        Returns
        -------
        list
            list of all changes in the repository between the two commits
            (adds/changes/removes)
        '''
        dif = diff.diff_commits(
            refenv=self._env.refenv,
            masterHEAD=master_commit,
            devHEAD=dev_commit)
        return dif

    def diff_branch(self, master_branch, dev_branch):
        '''Returns the diff of the head commits between a master and dev branch.

        Parameters
        ----------
        master_branch : str
            name of the master branch - must exist in the repository
        dev_branch : str
            name of the dev branch - must exist in the repository

        Returns
        -------
        list
            list of all changes in the repository between the two branches
            (adds/changes/removes)
        '''
        masterHEAD = heads.get_branch_head_commit(
            branchenv=self._env.branchenv, branch_name=master_branch)
        devHEAD = heads.get_branch_head_commit(
            branchenv=self._env.branchenv, branch_name=dev_branch)

        dif = diff.diff_commits(
            refenv=self._env.refenv, masterHEAD=masterHEAD, devHEAD=devHEAD)
        return dif

    def diff_staged_changes(self):
        '''Find the diff of all changes in the staging area

        Returns
        -------
        list
            list of all changes in the repository between current stage area and
            the last branch head commit.
        '''
        dif = diff.diff_staged_changes(
            refenv=self._env.refenv,
            stageenv=self._env.stageenv,
            branchenv=self._env.branchenv)
        return dif

    def create_branch(self, branch_name, base_commit=None):
        '''create a branch with the provided name from a certain commit.

        If no base commit hash is specified, the current writer branch HEAD
        commit is used as the base_commit hash for the branch. Note that
        creating a branch does not actually create a checkout object for
        interaction with the data. to interact you must use the repository
        checkout method to properly initialize a read (or write) enabled
        checkout object.

        Parameters
        ----------
        branch_name : str
            name to assign to the new branch
        base_commit : str, optional
            commit hash to start the branch root at. if not specified, the
            writer branch HEAD commit at the time of execution will be used,
            defaults to None

        Returns
        -------
        bool
            if the operation was successful.
        '''
        didCreateBranch = heads.create_branch(
            branchenv=self._env.branchenv,
            branch_name=branch_name,
            base_commit=base_commit)
        return didCreateBranch

    def remove_branch(self, branch_name):
        '''Not Implemented
        '''
        raise NotImplementedError()

    def list_branch_names(self):
        '''list all branch names created in the repository.

        Returns
        -------
        list of str
            the branch names recorded in the repository
        '''
        branches = heads.get_branch_names(self._env.branchenv)
        return branches

    def force_release_writer_lock(self):
        '''Force release the lock left behind by an unclosed writer-checkout

        .. warning::

            *NEVER USE THIS METHOD IF WRITER PROCESS IS CURRENTLY ACTIVE.* At the time
            of writing, the implications of improper/malicious use of this are not
            understood, and there is a a risk of of undefined behavior or (potentially)
            data corruption.

            At the moment, the responsibility to close a write-enabled checkout is
            placed entirely on the user. If the `close()` method is not called
            before the program terminates, a new checkout with write=True will fail.
            The lock can only be released via a call to this method.

        .. note::

            This entire mechanism is subject to review/replacement in the future.

        Returns
        -------
        bool
            if the operation was successful.
        '''
        forceReleaseSentinal = parsing.repo_writer_lock_force_release_sentinal()
        success = heads.release_writer_lock(self._env.branchenv, forceReleaseSentinal)
        return success
