import os
import weakref
import warnings
from typing import Union, Optional, List

from . import merger
from . import constants as c
from .remotes import Remotes
from .context import Environments
from .diagnostics import graphing, ecosystem
from .records import heads, parsing, summarize, vcompat
from .checkout import ReaderCheckout, WriterCheckout
from .utils import is_valid_directory_path, is_suitable_user_key


class Repository(object):
    """Launching point for all user operations in a Hangar repository.

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
    exists : bool, optional
        True if a Hangar repository should exist at the given directory path.
        Should no Hangar repository exists at that location, a UserWarning will
        be raised indicating that the :meth:`init` method needs to be called.

        False if the provided path does not need to (but optionally can) contain a
        Hangar repository.  if a Hangar repository does not exist at that path, the
        usual UserWarning will be suppressed.

        In both cases, the path must exist and the user must have sufficient OS
        permissions to write to that location. Default = True
    """

    def __init__(self, path: os.PathLike, exists: bool = True):

        try:
            usr_path = is_valid_directory_path(path)
        except (TypeError, NotADirectoryError, PermissionError) as e:
            raise e from None

        repo_pth = os.path.join(usr_path, c.DIR_HANGAR)
        if exists is False:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                envs = Environments(pth=repo_pth)
        else:
            envs = Environments(pth=repo_pth)

        self._repo_path: os.PathLike = repo_pth
        self._env: Environments = envs
        self._remote: Remotes = Remotes(self._env)

    def _repr_pretty_(self, p, cycle):
        """provide a pretty-printed repr for ipython based user interaction.

        Parameters
        ----------
        p : printer
            io stream printer type object which is provided via ipython
        cycle : bool
            if the pretty-printer detects a cycle or infinite loop. Not a
            concern here since we just output the text and return, no looping
            required.

        """
        self.__verify_repo_initialized()
        res = f'Hangar {self.__class__.__name__}\
               \n    Repository Path  : {self.path}\
               \n    Writer-Lock Free : {heads.writer_lock_held(self._env.branchenv)}\n'
        p.text(res)

    def __repr__(self):
        """Override the default repr to show useful information to developers.

        Note: the pprint repr (ipython enabled) is separately defined in
        :py:meth:`_repr_pretty_`. We specialize because we assume that anyone
        operating in a terminal-based interpreter is probably a more advanced
        developer-type, and expects traditional repr information instead of a
        user facing summary of the repo. Though if we're wrong, go ahead and
        feel free to reassign the attribute :) won't hurt our feelings, promise.

        Returns
        -------
        string
            formatted representation of the object
        """
        res = f'{self.__class__}(path={self._repo_path})'
        return res

    def __verify_repo_initialized(self):
        """Internal method to verify repo initialized before operations occur

        Raises
        ------
        RuntimeError
            If the repository db environments have not been initialized at the
            specified repo path.
        """
        if not self._env.repo_is_initialized:
            msg = f'Repository at path: {self._repo_path} has not been initialized. '\
                  f'Please run the `init_repo()` function'
            raise RuntimeError(msg)

    @property
    def remote(self) -> Remotes:
        """Accessor to the methods controlling remote interactions.

        .. seealso::

           :class:`Remotes` for available methods of this property

        Returns
        -------
        Remotes
            Accessor object methods for controlling remote interactions.
        """
        proxy = weakref.proxy(self._remote)
        return proxy

    @property
    def path(self) -> os.PathLike:
        """Return the path to the repository on disk, read-only attribute

        Returns
        -------
        os.PathLike
            path to the specified repository, not including `.hangar` directory
        """
        self.__verify_repo_initialized()
        return os.path.dirname(self._repo_path)

    @property
    def writer_lock_held(self) -> bool:
        """Check if the writer lock is currently marked as held. Read-only attribute.

        Returns
        -------
        bool
            True is writer-lock is held, False if writer-lock is free.
        """
        self.__verify_repo_initialized()
        return not heads.writer_lock_held(self._env.branchenv)

    @property
    def version(self) -> str:
        """Find the version of Hangar software the repository is written with

        Returns
        -------
        str
            semantic version of major, minor, micro version of repo software version.
        """
        self.__verify_repo_initialized()
        res = vcompat.get_repository_software_version_str(self._env.branchenv)
        return res

    def checkout(self,
                 write: bool = False,
                 *,
                 branch: str = 'master',
                 commit: str = '') -> Union[ReaderCheckout, WriterCheckout]:
        """Checkout the repo at some point in time in either `read` or `write` mode.

        Only one writer instance can exist at a time. Write enabled checkout
        must must create a staging area from the ``HEAD`` commit of a branch. On
        the contrary, any number of reader checkouts can exist at the same time
        and can specify either a branch name or a commit hash.

        Parameters
        ----------
        write : bool, optional
            Specify if the checkout is write capable, defaults to False
        branch : str, optional
            name of the branch to checkout. This utilizes the state of the repo
            as it existed at the branch ``HEAD`` commit when this checkout object
            was instantiated, defaults to 'master'
        commit : str, optional
            specific hash of a commit to use for the checkout (instead of a
            branch ``HEAD`` commit). This argument takes precedent over a branch
            name parameter if it is set. Note: this only will be used in
            non-writeable checkouts, defaults to ''

        Raises
        ------
        ValueError
            If the value of `write` argument is not boolean

        Returns
        -------
        Union[ReaderCheckout, WriterCheckout]
            Checkout object which can be used to interact with the repository
            data
        """
        self.__verify_repo_initialized()
        try:
            if write is True:
                co = WriterCheckout(
                    repo_pth=self._repo_path,
                    branch_name=branch,
                    labelenv=self._env.labelenv,
                    hashenv=self._env.hashenv,
                    refenv=self._env.refenv,
                    stageenv=self._env.stageenv,
                    branchenv=self._env.branchenv,
                    stagehashenv=self._env.stagehashenv)
                return co
            elif write is False:
                commit_hash = self._env.checkout_commit(
                    branch_name=branch, commit=commit)
                co = ReaderCheckout(
                    base_path=self._repo_path,
                    labelenv=self._env.labelenv,
                    dataenv=self._env.cmtenv[commit_hash],
                    hashenv=self._env.hashenv,
                    branchenv=self._env.branchenv,
                    refenv=self._env.refenv,
                    commit=commit_hash)
                return co
            else:
                raise ValueError("Argument `write` only takes True or False as value")
        except (RuntimeError, ValueError) as e:
            raise e from None

    def clone(self, user_name: str, user_email: str, remote_address: str,
              *, remove_old: bool = False) -> str:
        """Download a remote repository to the local disk.

        The clone method implemented here is very similar to a `git clone`
        operation. This method will pull all commit records, history, and data
        which are parents of the remote's `master` branch head commit. If a
        :class:`Repository` exists at the specified directory,
        the operation will fail.

        Parameters
        ----------
        user_name : str
            Name of the person who will make commits to the repository. This
            information is recorded permanently in the commit records.
        user_email : str
            Email address of the repository user. This information is recorded
            permanently in any commits created.
        remote_address : str
            location where the
            :class:`hangar.remote.server.HangarServer` process is
            running and accessible by the clone user.
        remove_old : bool, optional, kwarg only
            DANGER! DEVELOPMENT USE ONLY! If enabled, a
            :class:`hangar.repository.Repository` existing on disk at the same
            path as the requested clone location will be completely removed and
            replaced with the newly cloned repo. (the default is False, which
            will not modify any contents on disk and which will refuse to create
            a repository at a given location if one already exists there.)

        Returns
        -------
        str
            Name of the master branch for the newly cloned repository.
        """
        self.init(user_name=user_name, user_email=user_email, remove_old=remove_old)
        self._remote.add(name='origin', address=remote_address)
        branch = self._remote.fetch(remote='origin', branch='master')
        HEAD = heads.get_branch_head_commit(self._env.branchenv, branch_name=branch)
        heads.set_branch_head_commit(self._env.branchenv, 'master', HEAD)
        with warnings.catch_warnings(record=False):
            warnings.simplefilter('ignore', category=UserWarning)
            co = self.checkout(write=True, branch='master')
            co.reset_staging_area()
            co.close()
        return 'master'

    def init(self,
             user_name: str,
             user_email: str,
             *,
             remove_old: bool = False) -> os.PathLike:
        """Initialize a Hangar repository at the specified directory path.

        This function must be called before a checkout can be performed.

        Parameters
        ----------
        user_name : str
            Name of the repository user account.
        user_email : str
            Email address of the repository user account.
        remove_old : bool, kwarg-only
            DEVELOPER USE ONLY -- remove and reinitialize a Hangar
            repository at the given path, Default = False

        Returns
        -------
        os.PathLike
            the full directory path where the Hangar repository was
            initialized on disk.
        """
        pth = self._env._init_repo(
            user_name=user_name, user_email=user_email, remove_old=remove_old)
        return pth

    def log(self,
            branch: str = None,
            commit: str = None,
            *,
            return_contents: bool = False,
            show_time: bool = False,
            show_user: bool = False) -> Optional[dict]:
        """Displays a pretty printed commit log graph to the terminal.

        .. note::

            For programatic access, the return_contents value can be set to true
            which will retrieve relevant commit specifications as dictionary
            elements.

        Parameters
        ----------
        branch : str, optional
            The name of the branch to start the log process from. (Default value
            = None)
        commit : str, optional
            The commit hash to start the log process from. (Default value = None)
        return_contents : bool, optional, kwarg only
            If true, return the commit graph specifications in a dictionary
            suitable for programatic access/evaluation.
        show_time : bool, optional, kwarg only
            If true and return_contents is False, show the time of each commit
            on the printed log graph
        show_user : bool, optional, kwarg only
            If true and return_contents is False, show the committer of each
            commit on the printed log graph
        Returns
        -------
        Optional[dict]
            Dict containing the commit ancestor graph, and all specifications.
        """
        self.__verify_repo_initialized()
        res = summarize.list_history(
            refenv=self._env.refenv,
            branchenv=self._env.branchenv,
            branch_name=branch,
            commit_hash=commit)

        if return_contents:
            return res
        else:
            branchMap = heads.commit_hash_to_branch_name_map(branchenv=self._env.branchenv)
            g = graphing.Graph()
            g.show_nodes(dag=res['ancestors'],
                         spec=res['specs'],
                         branch=branchMap,
                         start=res['head'],
                         order=res['order'],
                         show_time=show_time,
                         show_user=show_user)

    def summary(self, *, branch: str = '', commit: str = '',
                return_contents: bool = False) -> Optional[dict]:
        """Print a summary of the repository contents to the terminal

        .. note::

            Programatic access is provided by the return_contents argument.

        Parameters
        ----------
        branch : str, optional
            A specific branch name whose head commit will be used as the summary
            point (Default value = '')
        commit : str, optional
            A specific commit hash which should be used as the summary point.
            (Default value = '')
        return_contents : bool
            If true, return a full log of what records are in the repository at
            the summary point. (Default value = False)

        Returns
        -------
        Optional[dict]
            contents of the entire repository (if `return_contents=True`)
        """
        self.__verify_repo_initialized()
        ppbuf, res = summarize.summary(self._env, branch=branch, commit=commit)
        if return_contents is True:
            return res
        else:
            print(ppbuf.getvalue())

    def _details(self) -> None:  # pragma: noqa
        """DEVELOPER USE ONLY: Dump some details about the underlying db structure to disk.
        """
        print(summarize.details(self._env.branchenv).getvalue())
        print(summarize.details(self._env.refenv).getvalue())
        print(summarize.details(self._env.hashenv).getvalue())
        print(summarize.details(self._env.labelenv).getvalue())
        print(summarize.details(self._env.stageenv).getvalue())
        print(summarize.details(self._env.stagehashenv).getvalue())
        for commit, commitenv in self._env.cmtenv.items():
            print(summarize.details(commitenv).getvalue())
        return

    def _ecosystem_details(self) -> dict:
        """DEVELOPER USER ONLY: log and return package versions on the system.
        """
        eco = ecosystem.get_versions()
        return eco

    def merge(self, message: str, master_branch: str, dev_branch: str) -> str:
        """Perform a merge of the changes made on two branches.

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
            Hash of the commit which is written if possible.
        """
        self.__verify_repo_initialized()
        commit_hash = merger.select_merge_algorithm(
            message=message,
            branchenv=self._env.branchenv,
            stageenv=self._env.stageenv,
            refenv=self._env.refenv,
            stagehashenv=self._env.stagehashenv,
            master_branch_name=master_branch,
            dev_branch_name=dev_branch,
            repo_path=self._repo_path)

        return commit_hash

    def create_branch(self, name: str, base_commit: str = None) -> str:
        """create a branch with the provided name from a certain commit.

        If no base commit hash is specified, the current writer branch ``HEAD``
        commit is used as the base_commit hash for the branch. Note that
        creating a branch does not actually create a checkout object for
        interaction with the data. to interact you must use the repository
        checkout method to properly initialize a read (or write) enabled
        checkout object.

        Parameters
        ----------
        name : str
            name to assign to the new branch
        base_commit : str, optional
            commit hash to start the branch root at. if not specified, the
            writer branch ``HEAD`` commit at the time of execution will be used,
            defaults to None

        Returns
        -------
        str
            name of the branch which was created
        """
        self.__verify_repo_initialized()
        if not is_suitable_user_key(name):
            e = ValueError(
                f'Branch name provided: `{name}` invalid. Must only contain '
                f'alpha-numeric or "." "_" "-" ascii characters.')
            raise e from None
        didCreateBranch = heads.create_branch(
            branchenv=self._env.branchenv,
            name=name,
            base_commit=base_commit)
        return didCreateBranch

    def remove_branch(self, name):
        """Not Implemented
        """
        self.__verify_repo_initialized()
        raise NotImplementedError()

    def list_branches(self) -> List[str]:
        """list all branch names created in the repository.

        Returns
        -------
        list of str
            the branch names recorded in the repository
        """
        self.__verify_repo_initialized()
        branches = heads.get_branch_names(self._env.branchenv)
        return branches

    def force_release_writer_lock(self) -> bool:
        """Force release the lock left behind by an unclosed writer-checkout

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
        """
        self.__verify_repo_initialized()
        forceReleaseSentinal = parsing.repo_writer_lock_force_release_sentinal()
        success = heads.release_writer_lock(self._env.branchenv, forceReleaseSentinal)
        return success