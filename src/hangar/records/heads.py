import warnings
from collections import defaultdict
from typing import NamedTuple

import lmdb

from .parsing import (
    remote_db_key_from_raw_key,
    remote_db_val_from_raw_val,
    remote_raw_key_from_db_key,
    remote_raw_val_from_db_val,
    repo_branch_head_db_key_from_raw_key,
    repo_branch_head_db_val_from_raw_val,
    repo_branch_head_raw_key_from_db_key,
    repo_branch_head_raw_val_from_db_val,
    repo_head_db_key,
    repo_head_db_val_from_raw_val,
    repo_head_raw_val_from_db_val,
    repo_writer_lock_db_key,
    repo_writer_lock_db_val_from_raw_val,
    repo_writer_lock_force_release_sentinal,
    repo_writer_lock_sentinal_db_val,
)
from ..constants import K_REMOTES, K_BRANCH
from ..txnctx import TxnRegister


class BranchHead(NamedTuple):
    name: str
    digest: str


"""
Write operation enabled lock methods
------------------------------------

Any operation which wants to interact with the main storage services in a
write-enabled way must acquire a lock to perform the operation. See docstrings
below for more info
"""


def writer_lock_held(branchenv):
    """Check to see if the writer lock is free before attempting to acquire it.

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment where the writer lock is stored

    Returns
    -------
    bool
        True if the lock is available to take, False if it is currently held.
    """
    writerLockKey = repo_writer_lock_db_key()
    writerLockSentinalVal = repo_writer_lock_sentinal_db_val()
    branchtxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        currentWriterLockVal = branchtxn.get(writerLockKey)
        if currentWriterLockVal == writerLockSentinalVal:
            lockAvailable = True
        elif currentWriterLockVal is None:
            # on first initialization, writer lock key/val is not set.
            lockAvailable = True
        else:
            lockAvailable = False
    finally:
        TxnRegister().abort_reader_txn(branchenv)
    return lockAvailable


def acquire_writer_lock(branchenv, writer_uuid):
    """Attempt to acquire the writer lock for a write-enabled checkout object.

    If the writer_uuid matches the recorded value, or the lock is available (or
    uninitialized entirely in the case of a brand-new repository), the lock will
    be updated with the requested uuid, and no other write-enabled checkout can
    be started until it is either released, or a force reset is performed (in
    the event of a system crash or user error.)

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment where the writer lock is stored
    writer_uuid : str
        uuid generated when a write enabled checkout instance starts

    Returns
    -------
    bool
        success of the operation, which will be validated by the writer class as
        a safety net incase the upstream in the event some user code tries to
        catch the exception.Z

    Raises
    ------
    PermissionError
        If the lock can not be acquired

    """
    writerLockKey = repo_writer_lock_db_key()
    writerLockSentinalVal = repo_writer_lock_sentinal_db_val()
    requestWriterLockVal = repo_writer_lock_db_val_from_raw_val(writer_uuid)

    branchtxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        currentWriterLockVal = branchtxn.get(writerLockKey)
        if currentWriterLockVal == requestWriterLockVal:
            success = True
        elif currentWriterLockVal == writerLockSentinalVal:
            branchtxn.put(writerLockKey, requestWriterLockVal)
            success = True
        elif currentWriterLockVal is None:
            # on first initialization, writer lock key/val is not set.
            branchtxn.put(writerLockKey, requestWriterLockVal)
            success = True
        else:
            err = 'Cannot acquire the writer lock. Only one instance of a writer checkout '\
                  'can be active at a time. If the last checkout of this repository did '\
                  'not properly close, or a crash occurred, the lock must be manually freed '\
                  'before another writer can be instantiated.'
            raise PermissionError(err)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return success


def release_writer_lock(branchenv, writer_uuid):
    """Internal method to release a writer lock held by a specified uuid.

    This method also accept the force-release sentinel by a caller in the
    writer_uuid field. If the writer_uuid does not match the lock value (and the
    force sentinel is not used), then a runtime error will be thrown and no-op
    performed

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment where the lock key/val lives
    writer_uuid : str
        uuid of the requested releaser

    Returns
    -------
    bool
        if the operation was successful or now

    Raises
    ------
    RuntimeError
        if the request uuid does not match the lock value.
    """
    writerLockKey = repo_writer_lock_db_key()
    forceReleaseSentinal = repo_writer_lock_force_release_sentinal()
    lockSentinalVal = repo_writer_lock_sentinal_db_val()
    requestWriterLockVal = repo_writer_lock_db_val_from_raw_val(writer_uuid)

    txn = TxnRegister().begin_writer_txn(branchenv)
    try:
        currentLockVal = txn.get(writerLockKey)
        if writer_uuid == forceReleaseSentinal:
            warnings.warn('Writer lock force successfully force released.', ResourceWarning)
            txn.put(writerLockKey, lockSentinalVal)
            success = True
        elif currentLockVal == requestWriterLockVal:
            txn.put(writerLockKey, lockSentinalVal)
            success = True
        elif currentLockVal == lockSentinalVal:
            warnings.warn('The lock is already available, no release is necessary.', UserWarning)
            success = True
        else:
            err = f'FATAL ERROR Requested release of writer lock: {currentLockVal} by '\
                  f'non-valid requestor: {requestWriterLockVal} -- How did this happen?'
            raise RuntimeError(err)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return success


"""
Methods to interact with the branch head records
------------------------------------------------

.. todo::
   Need a delete branch operation.
"""

# ---------------- branch creation and deletion operations ------------------------------


def create_branch(branchenv, name, base_commit) -> BranchHead:
    """Internal operations used to create a branch.

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment of the branch db
    name : str
        Name of the branch to create, if a branch with this name exists no
        operation  will occur and a `ValueError` will be thrown.
    base_commit : str
        The commit to start this branch from.

    Returns
    -------
    BranchHead
        NamedTuple[str, str] with fields for `name` and `digest` of the branch
        created (if the operation was successful)

    Raises
    ------
    ValueError
        If the branch already exists, no-op and raise this.
    RuntimeError
        If the repository does not have at-least one commit on the `default`
        (ie. `master`) branch.
    """
    if base_commit is None:
        headBranch = get_staging_branch_head(branchenv)
        base_commit = get_branch_head_commit(branchenv, headBranch)
        if (headBranch == 'master') and (base_commit == ''):
            msg = 'At least one commit must be made in the repository on the `default` '\
                  '(`master`) branch before new branches can be created'
            raise RuntimeError(msg)

    branchHeadKey = repo_branch_head_db_key_from_raw_key(name)
    branchHeadVal = repo_branch_head_db_val_from_raw_val(base_commit)

    branchtxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        success = branchtxn.put(branchHeadKey, branchHeadVal, overwrite=False)
        if success is False:
            err = f'A branch with the name {name} already exists, please specify'\
                  f'a different name or delete the branch.'
            raise ValueError(err)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return BranchHead(name=name, digest=base_commit)


def remove_branch(branchenv: lmdb.Environment,
                  refenv: lmdb.Environment,
                  name: str,
                  *,
                  force_delete: bool = False) -> BranchHead:
    """Remove a branch head pointer after verifying validity and safety

    Parameters
    ----------
    branchenv : lmdb.Environment
        db containing the branch head specs
    refenv : lmdb.Environment
        db containing the commit refs
    name : str
        name of the branch which should be deleted.
    force_delete : bool, optional
        If True, remove the branch pointer even if the changes are un-merged in
        other branch histories. by default False

    Returns
    -------
    BranchHead
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
    from .commiting import get_commit_ancestors_graph

    all_branches = get_branch_names(branchenv)
    alive_branches = [x for x in all_branches if '/' not in x]  # exclude remotes
    if name not in alive_branches:
        raise ValueError(f'Branch: {name} does not exist')

    alive_branches.remove(name)
    if len(alive_branches) == 0:
        msg = f'Not allowed to remove all branches from a repository! '\
              f'Operation aborted without completing removal of branch: {name}'
        raise PermissionError(msg)

    if writer_lock_held(branchenv) is False:
        msg = f'Cannot remove branch when a `write-enabled` checkout is active. '\
              f're-run after committing/closing the writer.'
        raise PermissionError(msg)

    staging_branch = get_staging_branch_head(branchenv)
    if staging_branch == name:
        msg = f'Branch: {name} cannot be deleted while acting as the base for '\
              f'contents of the staging area. re-run after checking out a '\
              f'different branch in `write` mode.'
        raise PermissionError(msg)

    HEAD = get_branch_head_commit(branchenv, name)
    if not force_delete:
        for branch in alive_branches:
            b_head = get_branch_head_commit(branchenv, branch)
            b_ancestors = get_commit_ancestors_graph(refenv, starting_commit=b_head)
            if HEAD in b_ancestors:
                break
        else:  # N.B. for-else conditional (ie. "no break")
            msg = f'The branch {name} is not fully merged. If you are sure '\
                  f'you want to delete it, re-run with force-remove parameter set'
            raise RuntimeError(msg)

    branchtxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        branchHeadKey = repo_branch_head_db_key_from_raw_key(name)
        branchtxn.delete(branchHeadKey)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return BranchHead(name=name, digest=HEAD)


# ------------- set and get with staging area HEAD branch name --------------------------


def get_staging_branch_head(branchenv):
    """Get the name of the current staging area HEAD branch

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment for the branch references

    Returns
    -------
    str
        name of the staging HEAD branch
    """
    headKey = repo_head_db_key()
    txn = TxnRegister().begin_reader_txn(branchenv)
    try:
        headBranchVal = txn.get(headKey)
    finally:
        TxnRegister().abort_reader_txn(branchenv)
    headBranch = repo_head_raw_val_from_db_val(headBranchVal)
    return headBranch


def set_staging_branch_head(branchenv, branch_name):
    """Set the writer HEAD to a branch name. Does not modify staging area contents.

    A writer-checkout must specify a branch name to use as it's ancestor. We do
    not allow a writer (or staging area) to exist in a "Detached HEAD" state. In
    order to make modifications starting from a specific commit, the user must
    create a branch with that commit hash as the specified "base".

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment of the branch db.
    branch_name : str
        name of the branch to checkout.

    Returns
    -------
    bool
        if the operation was successful.

    Raises
    ------
    ValueError
        If the specified branch name does not exist.
    """
    headKey = repo_head_db_key()
    requestedHeadVal = repo_head_db_val_from_raw_val(branch_name)
    requestedBranchKey = repo_branch_head_db_key_from_raw_key(branch_name)

    branchtxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        branchNameExists = branchtxn.get(requestedBranchKey, default=False)
        if branchNameExists is False:
            err = f'No branch with the name: {branch_name} exists, no-op performed'
            raise ValueError(err)
        else:
            branchtxn.put(headKey, requestedHeadVal)
            success = True
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return success


# ------------- get and set a named branch HEAD commit hash --------------------------===


def get_branch_head_commit(branchenv, branch_name):
    """Find the commit hash which corresponds to the HEAD of a particular branch.

    Parameters
    ----------
    branchenv: lmdb.Environment
        lmdb environment for the branch spec
    branch_name: str
        name of the branch to find the head commit hash for

    Returns
    -------
    str
        the commit hash of the branch head

    Raises
    ------
    ValueError
        if `branch_name` does not exist in the repository
    """
    requestedBranchKey = repo_branch_head_db_key_from_raw_key(branch_name)
    branchtxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        branchNameVal = branchtxn.get(requestedBranchKey, default=False)
        if branchNameVal is False:
            err = f'branch with name: {branch_name} does not exist. cannot get head.'
            raise ValueError(err)
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    commit_hash = repo_branch_head_raw_val_from_db_val(branchNameVal)
    return commit_hash


def set_branch_head_commit(branchenv, branch_name, commit_hash):
    """Update an existing branch HEAD to point to a new commit hash.

    Does not update stage or refenv contents. If the current HEAD of the branch
    == the new commit hash, no operation will occur and an exception will be
    thrown.

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment where the branch records are kept
    branch_name : string
        Name of the branch to update the HEAD commit of
    commit_hash : string
        Commit hash to update the branch HEAD to point to.

    Returns
    -------
    string
        Commit hash of the new branch head if the operation was successful.

    Raises
    ------
    ValueError
        If the current HEAD is the same as the new commit hash.
    """
    currentHeadCommit = get_branch_head_commit(branchenv=branchenv, branch_name=branch_name)
    if currentHeadCommit == commit_hash:
        err = f'Current branch: {branch_name} HEAD: {currentHeadCommit} is same as the '\
              f'requested updated HEAD: {commit_hash}, no-op performed'
        raise ValueError(err)

    branchtxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        branchHeadKey = repo_branch_head_db_key_from_raw_key(branch_name)
        branchHeadVal = repo_branch_head_db_val_from_raw_val(commit_hash)
        branchtxn.put(branchHeadKey, branchHeadVal)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return commit_hash


def get_branch_names(branchenv):
    """get a list of all branches in the repository.

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment storing the branch records.

    Returns
    -------
    list of str
        list of branch names active in the repository.
    """
    branchStartKey = K_BRANCH.encode()  # TODO: This is odd, why??
    branchNames = []
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        with branchTxn.cursor() as cursor:
            cursor.first()
            branchRangeExists = cursor.set_range(branchStartKey)
            while branchRangeExists:
                branchKey = cursor.key()
                if branchKey.startswith(branchStartKey):
                    name = repo_branch_head_raw_key_from_db_key(branchKey)
                    branchNames.append(name)
                    branchRangeExists = cursor.next()
                else:
                    branchRangeExists = False
        cursor.close()
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    return branchNames


def commit_hash_to_branch_name_map(branchenv: lmdb.Environment) -> dict:
    """Determine branch names which map to commit hashs

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the branch references are stored

    Returns
    -------
    dict
        keys are commit hash strings, values are list of branch names (strings)
        whose HEAD are at the key commit
    """
    outMap = defaultdict(list)
    branchNames = get_branch_names(branchenv=branchenv)
    for branchName in branchNames:
        branchHEAD = get_branch_head_commit(branchenv=branchenv, branch_name=branchName)
        outMap[branchHEAD].append(branchName)

    return outMap


# ----------------------------- Remotes ---------------------------------------


def add_remote(branchenv: lmdb.Environment, name: str, address: str) -> bool:
    """add a remote server reference to the repository.

    This method does not check that the remote is actually accessible, rather it
    just records the reference. If a remote with the same name already exists,
    no change will occur.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the branch (and remote) references are stored.
    name : str
        name of the remote to add the address for
    address : str
        IP:PORT where the remote server can be accessed

    Returns
    -------
    bool
        True if the new reference was saved, False if not.
    """
    dbKey = remote_db_key_from_raw_key(name)
    dbVal = remote_db_val_from_raw_val(address)

    branchTxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        succ = branchTxn.put(dbKey, dbVal, overwrite=False)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    return succ


def get_remote_address(branchenv: lmdb.Environment, name: str) -> str:
    """Retrieve the IO:PORT of the remote server for a given name

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the branch (and remote) references are stored
    name : str
        name of the remote to fetch

    Raises
    ------
    KeyError
        if a remote with the provided name does not exist

    Returns
    -------
    str
        IP:PORT of the recorded remote server.
    """
    dbKey = remote_db_key_from_raw_key(name)
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        dbVal = branchTxn.get(dbKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    if dbVal is False:
        msg = f'No remote with the name: {name} exists in the repo.'
        raise KeyError(msg)
    else:
        remote_address = remote_raw_val_from_db_val(dbVal)
        return remote_address


def remove_remote(branchenv: lmdb.Environment, name: str) -> str:
    """remove a remote reference with the provided name.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the branch (and remote) records are stored.
    name : str
        name of the remote to remove from the repo

    Raises
    ------
    ValueError
        if a remote with the provided name does not exist

    Returns
    -------
    str
        IP:PORT of the remote with provided name (which was removed)
    """
    dbKey = remote_db_key_from_raw_key(name)
    branchTxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        dbVal = branchTxn.pop(dbKey)
    finally:
        TxnRegister().commit_writer_txn(branchenv)

    if dbVal is None:
        msg = f'No remote with the name: {name} exists in the repo.'
        raise ValueError(msg)

    remote_address = remote_raw_val_from_db_val(dbVal)
    return remote_address


def get_remote_names(branchenv):
    """get a list of all remotes in the repository.

    Parameters
    ----------
    branchenv : lmdb.Environment
        lmdb environment storing the branch records.

    Returns
    -------
    list of str
        list of remote names active in the repository.
    """
    remoteStartKey = K_REMOTES.encode()
    remoteNames = []
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        with branchTxn.cursor() as cursor:
            cursor.first()
            remoteRangeExists = cursor.set_range(remoteStartKey)
            while remoteRangeExists:
                remoteKey = cursor.key()
                if remoteKey.startswith(remoteStartKey):
                    name = remote_raw_key_from_db_key(remoteKey)
                    remoteNames.append(name)
                    remoteRangeExists = cursor.next()
                else:
                    remoteRangeExists = False
        cursor.close()
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    return remoteNames
