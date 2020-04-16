from pathlib import Path
import weakref
import warnings
from typing import Union, Optional, List

from .merger import select_merge_algorithm
from .constants import DIR_HANGAR
from .remotes import Remotes
from .context import Environments
from .diagnostics import ecosystem, integrity
from .records import heads, parsing, summarize, vcompat, commiting
from .checkout import ReaderCheckout, WriterCheckout
from .diff import DiffAndConflicts, ReaderUserDiff
from .utils import (
    is_valid_directory_path,
    is_suitable_user_key,
    is_ascii,
    folder_size,
    format_bytes
)


class Repository(object):
    """Launching point for all user operations in a Hangar repository.

    All interaction, including the ability to initialize a repo, checkout a
    commit (for either reading or writing), create a branch, merge branches, or
    generally view the contents or state of the local repository starts here.
    Just provide this class instance with a path to an existing Hangar
    repository, or to a directory one should be initialized, and all required
    data for starting your work on the repo will automatically be populated.

        >>> from hangar import Repository
        >>> repo = Repository('foo/path/to/dir')

    Parameters
    ----------
    path : Union[str, os.PathLike]
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

    def __init__(self, path: Union[str, Path], exists: bool = True):

        if isinstance(path, (str, bytes)):
            path = Path(path)

        try:
            usr_path = is_valid_directory_path(path)
        except (TypeError, NotADirectoryError, PermissionError) as e:
            raise e from None

        repo_pth = usr_path.joinpath(DIR_HANGAR)
        if exists is False:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                envs = Environments(pth=repo_pth)
        else:
            envs = Environments(pth=repo_pth)

        self._repo_path: Path = repo_pth
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
    def path(self) -> str:
        """Return the path to the repository on disk, read-only attribute

        Returns
        -------
        str
            path to the specified repository, not including `.hangar` directory
        """
        self.__verify_repo_initialized()
        return str(self._repo_path.parent)

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
        res = vcompat.get_repository_software_version_spec(self._env.branchenv)
        return str(res)

    @property
    def initialized(self) -> bool:
        """
        Check if the repository has been initialized or not

        Returns
        -------
        bool
            True if repository has been initialized.
        """
        return self._env.repo_is_initialized

    @property
    def size_nbytes(self) -> int:
        """Disk space used by the repository returned in number of bytes.

            >>> repo.size_nbytes
            1234567890
            >>> print(type(repo.size_nbytes))
            <class 'int'>

        Returns
        -------
        int
            number of bytes used by the repository on disk.
        """
        self.__verify_repo_initialized()
        return folder_size(self._repo_path, recurse=True)

    @property
    def size_human(self) -> str:
        """Disk space used by the repository returned in human readable string.

            >>> repo.size_human
            '1.23 GB'
            >>> print(type(repo.size_human))
            <class 'str'>

        Returns
        -------
        str
            disk space used by the repository formated in human readable text.
        """
        self.__verify_repo_initialized()
        nbytes = folder_size(self._repo_path, recurse=True)
        return format_bytes(nbytes)

    def checkout(self,
                 write: bool = False,
                 *,
                 branch: str = '',
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
            was instantiated, defaults to ''
        commit : str, optional
            specific hash of a commit to use for the checkout (instead of a
            branch ``HEAD`` commit). This argument takes precedent over a branch
            name parameter if it is set. Note: this only will be used in
            non-writeable checkouts, defaults to ''

        Raises
        ------
        ValueError
            If the value of `write` argument is not boolean
        ValueError
            If ``commit`` argument is set to any value when ``write=True``.
            Only ``branch`` argument is allowed.

        Returns
        -------
        Union[ReaderCheckout, WriterCheckout]
            Checkout object which can be used to interact with the repository
            data
        """
        self.__verify_repo_initialized()
        try:
            if write is True:
                if commit != '':
                    raise ValueError(
                        f'Only `branch` argument can be set if `write=True`. '
                        f'Setting `commit={commit}` not allowed.')
                if branch == '':
                    branch = heads.get_staging_branch_head(self._env.branchenv)
                co = WriterCheckout(
                    repo_pth=self._repo_path,
                    branch_name=branch,
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
             remove_old: bool = False) -> str:
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
        str
            the full directory path where the Hangar repository was
            initialized on disk.
        """
        pth = self._env.init_repo(user_name=user_name,
                                  user_email=user_email,
                                  remove_old=remove_old)
        return str(pth)

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
        res = summarize.log(branchenv=self._env.branchenv,
                            refenv=self._env.refenv,
                            branch=branch,
                            commit=commit,
                            return_contents=return_contents,
                            show_time=show_time,
                            show_user=show_user)
        return res

    def summary(self, *, branch: str = '', commit: str = '') -> None:
        """Print a summary of the repository contents to the terminal

        Parameters
        ----------
        branch : str, optional
            A specific branch name whose head commit will be used as the summary
            point (Default value = '')
        commit : str, optional
            A specific commit hash which should be used as the summary point.
            (Default value = '')
        """
        self.__verify_repo_initialized()
        ppbuf = summarize.summary(self._env, branch=branch, commit=commit)
        print(ppbuf.getvalue())
        return None

    def _details(self, *, line_limit=100, line_length=100) -> None:  # pragma: no cover
        """DEVELOPER USE ONLY: Dump some details about the underlying db structure to disk.
        """
        print(summarize.details(
            self._env.branchenv, line_limit=line_limit, line_length=line_length).getvalue())
        print(summarize.details(
            self._env.refenv, line_limit=line_limit, line_length=line_length).getvalue())
        print(summarize.details(
            self._env.hashenv, line_limit=line_limit, line_length=line_length).getvalue())
        print(summarize.details(
            self._env.stageenv, line_limit=line_limit, line_length=line_length).getvalue())
        print(summarize.details(
            self._env.stagehashenv, line_limit=line_limit, line_length=line_length).getvalue())
        for commit, commitenv in self._env.cmtenv.items():
            print(summarize.details(
                commitenv, line_limit=line_limit, line_length=line_length).getvalue())
        return

    def _ecosystem_details(self) -> dict:
        """DEVELOPER USER ONLY: log and return package versions on the system.
        """
        eco = ecosystem.get_versions()
        return eco

    def diff(self, master: str, dev: str) -> DiffAndConflicts:
        """Calculate diff between master and dev branch/commits.

        Diff is calculated as if we are to merge "dev" into "master"

        Parameters
        ----------
        master: str
            branch name or commit hash digest to use as the "master" which
            changes made in "dev" are compared to.
        dev: str
            branch name or commit hash digest to use as the "dev"
            (ie. "feature") branch which changes have been made to
            which are to be compared to the contents of "master".

        Returns
        -------
        DiffAndConflicts
            Standard output diff structure.
        """
        current_branches = self.list_branches()

        # assert branch / commit specified by "master" exists and
        # standardize into "digest" rather than "branch name" arg type
        if master in current_branches:
            masterHEAD = heads.get_branch_head_commit(
                branchenv=self._env.branchenv, branch_name=master)
        else:
            cmtExists = commiting.check_commit_hash_in_history(
                refenv=self._env.refenv, commit_hash=master)
            if not cmtExists:
                raise ValueError(f'`master` {master} is not valid branch/commit.')
            masterHEAD = master

        # same check & transform for "dev" branch/commit arg.
        if dev in current_branches:
            devHEAD = heads.get_branch_head_commit(
                branchenv=self._env.branchenv, branch_name=dev)
        else:
            cmtExists = commiting.check_commit_hash_in_history(
                refenv=self._env.refenv, commit_hash=dev)
            if not cmtExists:
                raise ValueError(f'`dev` {dev} is not valid branch/commit.')
            devHEAD = dev

        # create differ object and generate results...
        diff = ReaderUserDiff(commit_hash=masterHEAD,
                              branchenv=self._env.branchenv,
                              refenv=self._env.refenv)
        res = diff.commit(dev_commit_hash=devHEAD)
        return res


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
        commit_hash = select_merge_algorithm(
            message=message,
            branchenv=self._env.branchenv,
            stageenv=self._env.stageenv,
            refenv=self._env.refenv,
            stagehashenv=self._env.stagehashenv,
            master_branch=master_branch,
            dev_branch=dev_branch,
            repo_path=self._repo_path)

        return commit_hash

    def create_branch(self, name: str, base_commit: str = None) -> heads.BranchHead:
        """create a branch with the provided name from a certain commit.

        If no base commit hash is specified, the current writer branch ``HEAD``
        commit is used as the ``base_commit`` hash for the branch. Note that
        creating a branch does not actually create a checkout object for
        interaction with the data. to interact you must use the repository
        checkout method to properly initialize a read (or write) enabled
        checkout object.

            >>> from hangar import Repository
            >>> repo = Repository('foo/path/to/dir')

            >>> repo.create_branch('testbranch')
                BranchHead(name='testbranch', digest='b66b...a8cc')
            >>> repo.list_branches()
                ['master', 'testbranch']
            >>> co = repo.checkout(write=True, branch='testbranch')
            >>> # add data ...
            >>> newDigest = co.commit('added some stuff')

            >>> repo.create_branch('new-changes', base_commit=newDigest)
                BranchHead(name='new-changes', digest='35kd...3254')
            >>> repo.list_branches()
                ['master', 'new-changes', 'testbranch']

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
        :class:`~.heads.BranchHead`
            NamedTuple[str, str] with fields for ``name`` and ``digest`` of the
            branch created (if the operation was successful)

        Raises
        ------
        ValueError
            If the branch name provided contains characters outside of alpha-numeric
            ascii characters and ".", "_", "-" (no whitespace), or is > 64 characters.
        ValueError
            If the branch already exists.
        RuntimeError
            If the repository does not have at-least one commit on the "default"
            (ie. ``master``) branch.
        """
        self.__verify_repo_initialized()
        if (not is_ascii(name)) or (not is_suitable_user_key(name)):
            err = ValueError(
                f'Branch name provided: {name} invalid. Must contain only alpha-numeric '
                f'or "." "_" "-" ascii characters. And be <= 64 Characters')
            raise err from None
        createdBranch = heads.create_branch(
            branchenv=self._env.branchenv,
            name=name,
            base_commit=base_commit)
        return createdBranch

    def remove_branch(self, name: str, *, force_delete: bool = False) -> heads.BranchHead:
        """Permanently delete a branch pointer from the repository history.

        Since a branch (by definition) is the name associated with the HEAD
        commit of a historical path, the default behavior of this method is to
        throw an exception (no-op) should the ``HEAD`` not be referenced as an
        ancestor (or at least as a twin) of a separate branch which is
        currently *ALIVE*. If referenced in another branch's history, we are
        assured that all changes have been merged and recorded, and that this
        pointer can be safely deleted without risk of damage to historical
        provenance or (eventual) loss to garbage collection.

            >>> from hangar import Repository
            >>> repo = Repository('foo/path/to/dir')

            >>> repo.create_branch('first-testbranch')
            BranchHead(name='first-testbranch', digest='9785...56da')
            >>> repo.create_branch('second-testbranch')
            BranchHead(name='second-testbranch', digest='9785...56da')
            >>> repo.list_branches()
            ['master', 'first-testbranch', 'second-testbranch']
            >>> # Make a commit to advance a branch
            >>> co = repo.checkout(write=True, branch='first-testbranch')
            >>> # add data ...
            >>> co.commit('added some stuff')
            '3l253la5hna3k3a553256nak35hq5q534kq35532'
            >>> co.close()

            >>> repo.remove_branch('second-testbranch')
            BranchHead(name='second-testbranch', digest='9785...56da')

        A user may manually specify to delete an un-merged branch, in which
        case the ``force_delete`` keyword-only argument should be set to
        ``True``.

            >>> # check out master and try to remove 'first-testbranch'
            >>> co = repo.checkout(write=True, branch='master')
            >>> co.close()

            >>> repo.remove_branch('first-testbranch')
            Traceback (most recent call last):
                ...
            RuntimeError: ("The branch first-testbranch is not fully merged. "
            "If you are sure you want to delete it, re-run with "
            "force-remove parameter set.")
            >>> # Now set the `force_delete` parameter
            >>> repo.remove_branch('first-testbranch', force_delete=True)
            BranchHead(name='first-testbranch', digest='9785...56da')

        It is important to note that *while this method will handle all safety
        checks, argument validation, and performs the operation to permanently
        delete a branch name/digest pointer, **no commit refs along the history
        will be deleted from the Hangar database**.* Most of the history contains
        commit refs which must be safe in other branch histories, and recent
        commits may have been used as the base for some new history. As such, even
        if some of the latest commits leading up to a deleted branch ``HEAD`` are
        orphaned (unreachable), the records (and all data added in those commits)
        will remain on the disk.

        In the future, we intend to implement a garbage collector which will remove
        orphan commits which have not been modified for some set amount of time
        (probably on the order of a few months), but this is not implemented at the
        moment.

        Should an accidental forced branch deletion occur, *it is possible to
        recover* and create a new branch head pointing to the same commit. If
        the commit digest of the removed branch ``HEAD`` is known, its as simple as
        specifying a name and the ``base_digest`` in the normal
        :meth:`create_branch` method. If the digest is unknown, it will be a
        bit more work, but some of the developer facing introspection tools /
        routines could be used to either manually or (with minimal effort)
        programmatically find the orphan commit candidates. If you find
        yourself having accidentally deleted a branch, and must get it back,
        please reach out on the `Github Issues
        <https://github.com/tensorwerk/hangar-py/issues>`__ page. We'll gladly
        explain more in depth and walk you through the process in any way we
        can help!

        Parameters
        ----------
        name : str
            name of the branch which should be deleted. This branch must exist, and
            cannot refer to a remote tracked branch (ie. origin/devbranch), please
            see exception descriptions for other parameters determining validity of
            argument
        force_delete : bool, optional
            If True, remove the branch pointer even if the changes are un-merged in
            other branch histories. May result in orphaned commits which may be
            time-consuming to recover if needed, by default False

        Returns
        -------
        :class:`~.heads.BranchHead`
            NamedTuple[str, str] with fields for `name` and `digest` of the branch
            pointer deleted.

        Raises
        ------
        ValueError
            If a branch with the provided name does not exist locally
        PermissionError
            If removal of the branch would result in a repository with zero local
            branches.
        PermissionError
            If a write enabled checkout is holding the writer-lock at time of this
            call.
        PermissionError
            If the branch to be removed was the last used in a write-enabled
            checkout, and whose contents form the base of the staging area.
        RuntimeError
            If the branch has not been fully merged into other branch histories,
            and ``force_delete`` option is not ``True``.
        """
        self.__verify_repo_initialized()
        res = heads.remove_branch(branchenv=self._env.branchenv,
                                  refenv=self._env.refenv,
                                  name=name,
                                  force_delete=force_delete)
        return res

    def list_branches(self) -> List[str]:
        """list all branch names created in the repository.

        Returns
        -------
        List[str]
            the branch names recorded in the repository
        """
        self.__verify_repo_initialized()
        branches = heads.get_branch_names(self._env.branchenv)
        return branches

    def verify_repo_integrity(self) -> bool:
        """Verify the integrity of the repository data on disk.

        Runs a full cryptographic verification of repository contents in order
        to ensure the integrity of all data and history recorded on disk.

        .. note::

            This proof may take a significant amount of time to run for
            repositories which:

            1. store significant quantities of data on disk.
            2. have a very large number of commits in their history.

            As a brief explanation for why these are the driving factors behind
            processing time:

            1. Every single piece of data in the repositories history must be read
               from disk, cryptographically hashed, and compared to the expected
               value. There is no exception to this rule; regardless of when a piece
               of data was added / removed from an column, or for how many (or how
               few) commits some sample exists in. The integrity of the commit tree at
               any point after some piece of data is added to the repo can only be
               validated if it - and all earlier data pieces - are proven to be intact
               and unchanged.

               Note: This does not mean that the verification is repeatedly
               performed for every commit some piece of data is stored in. Each
               data piece is read from disk and verified only once, regardless of
               how many commits some piece of data is referenced in.

            2. Each commit reference (defining names / contents of a commit) must be
               decompressed and parsed into a usable data structure. We scan across
               all data digests referenced in the commit and ensure that the
               corresponding data piece is known to hangar (and validated as
               unchanged). The commit refs (along with the corresponding user records,
               message, and parent map), are then re-serialized and cryptographically
               hashed for comparison to the expected value. While this process is
               fairly efficient for a single commit, it must be repeated for each
               commit in the repository history, and may take a non-trivial amount of
               time for repositories with thousands of commits.

        While the two points above are the most time consuming operations,
        there are many more checks which are performed alongside them as part
        of the full verification run.

        Returns
        -------
        bool
            True if integrity verification is successful, otherwise False; in
            this case, a message describing the offending component will be
            printed to stdout.
        """
        self.__verify_repo_initialized()
        heads.acquire_writer_lock(self._env.branchenv, 'VERIFY_PROCESS')
        try:
            integrity.run_verification(
                branchenv=self._env.branchenv,
                hashenv=self._env.hashenv,
                refenv=self._env.refenv,
                repo_path=self._env.repo_path)
        finally:
            heads.release_writer_lock(self._env.branchenv, 'VERIFY_PROCESS')
        return True

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
