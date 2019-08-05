import hashlib
import os
import tempfile
import time
from os.path import join as pjoin
import shutil

import lmdb
import yaml

from . import heads, parsing
from .. import constants as c
from ..context import TxnRegister
from .queries import RecordQuery
from ..utils import symlink_rel


'''
Reading commit specifications and parents.
------------------------------------------
'''


def expand_short_commit_digest(refenv: lmdb.Environment, commit_hash: str) -> str:
    '''Find the a full commit hash from a short version provided by the user

    Parameters
    ----------
    refenv : lmdb.Environment
        db where the commit references are stored
    commit_hash : str
        short commit hash to search for in the repository

    Returns
    -------
    str
        full commit hash if short maps to a unique digest in the repo history

    Raises
    ------
    KeyError
        If the short commit hash can reference two full commit digests
    KeyError
        if no expanded commit digest is found starting with the short version.
    '''
    reftxn = TxnRegister().begin_reader_txn(refenv)
    commitParentStart = parsing.commit_parent_db_key_from_raw_key(commit_hash)
    with reftxn.cursor() as cursor:
        shortHashExists = cursor.set_range(commitParentStart)
        if shortHashExists is True:
            commitKey = cursor.key()
            commit_key = parsing.commit_parent_raw_key_from_db_key(commitKey)
            if commit_key.startswith(commit_hash) is False:
                raise KeyError(f'No expanded commit hash found for short: {commit_hash}')
            cursor.next()
            cursor.next()
            nextHashExist = cursor.next()
            if nextHashExist is False:
                return commit_key
            nextCommitKey = cursor.key()
            next_commit_key = parsing.commit_parent_raw_key_from_db_key(nextCommitKey)
            if next_commit_key.startswith(commit_hash) is True:
                raise KeyError(f'Non unique short commit hash: {commit_hash}')
            else:
                return commit_key
        else:
            raise KeyError(f'No expanded commit hash found for short: {commit_hash}')


def check_commit_hash_in_history(refenv, commit_hash):
    '''Check if a commit hash exists in the repository history

    Parameters
    ----------
    refenv : lmdb.Environment
        refenv where the commit history is stored
    commit_hash : str
        hash of the commit to check for existence

    Returns
    -------
    bool
        True if exists, otherwise False
    '''
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit_hash)
        commitParentVal = reftxn.get(commitParentKey, default=False)
        isCommitInHistory = True if commitParentVal is not False else False
    finally:
        TxnRegister().abort_reader_txn(refenv)
    return isCommitInHistory


def get_commit_spec(refenv, commit_hash):
    '''Get the commit specifications of a particular hash.

    Parameters
    ----------
    refenv : lmdb.Environment
        refenv where the specs are stored
    commit_hash : str
        commit hash to query

    Returns
    -------
    namedtuple
        named tuple with all the commit specs included

    Raises
    ------
    ValueError
        if no commit exists with the provided hash
    '''
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        parentCommitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit_hash)
        parentCommitSpecVal = reftxn.get(parentCommitSpecKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if parentCommitSpecVal is False:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    parentCommitSpec = parsing.commit_spec_raw_val_from_db_val(parentCommitSpecVal)
    return parentCommitSpec


def get_commit_ancestors(refenv, commit_hash):
    '''find the ancestors of a particular commit hash.

    Parameters
    ----------
    refenv : lmdb.Environment
        lmdb environment where the commit refs are stored
    commit_hash : string
        commit hash to find the ancestors for

    Returns
    -------
    namedtuple
        Namedtuple describing is_merge_commit, master_ancester, &
        child_ancestor (in the even of merge commit)

    Raises
    ------
    ValueError
        if no commit exists with the provided hash
    '''

    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        parentCommitKey = parsing.commit_parent_db_key_from_raw_key(commit_hash)
        parentCommitVal = reftxn.get(parentCommitKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if parentCommitVal is False:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    parentCommitAncestors = parsing.commit_parent_raw_val_from_db_val(parentCommitVal)
    return parentCommitAncestors


def get_commit_ancestors_graph(refenv, starting_commit):
    '''returns a DAG of all commits starting at some hash pointing to the repo root.

    Parameters
    ----------
    refenv : lmdb.Environment
        lmdb environment where the commit refs are stored
    starting_commit : string
        commit hash to start creating the DAG from

    Returns
    -------
    dict
        a dictionary where each key is a commit hash encountered along the way,
        and it's value is a list containing either one or two elements which
        identify the child commits of that parent hash.
    '''
    parent_commit = starting_commit
    commit_graph = {}
    seen = set(starting_commit)
    more_work = []
    end_commit = False

    if parent_commit == '':
        end_commit = True

    while end_commit is not True:
        childCommit = get_commit_ancestors(refenv, parent_commit)

        if ((childCommit.master_ancestor == '') or (childCommit.master_ancestor in seen)):
            end_commit = True
            commit_graph[parent_commit] = [childCommit.master_ancestor]
            if len(more_work) != 0:
                master_commit = more_work.pop(0)
                end_commit = False
            else:
                continue

        elif childCommit.is_merge_commit is True:
            master_commit = childCommit.master_ancestor
            dev_commit = childCommit.dev_ancestor
            more_work.append(dev_commit)
            commit_graph[parent_commit] = [master_commit, dev_commit]
            seen.add(master_commit)
            seen.add(dev_commit)

        else:
            master_commit = childCommit.master_ancestor
            commit_graph[parent_commit] = [master_commit]
            seen.add(master_commit)

        parent_commit = master_commit

    return commit_graph


'''
Methods for reading packed commit data and reconstructing an unpacked format.
-----------------------------------------------------------------------------
'''


def get_commit_ref(refenv, commit_hash):
    '''Read the commit data record references from a specific commit.

    This only returns a list of tuples with binary encoded key/value pairs.

    Parameters
    ----------
    refenv : lmdb.Envirionment`
        lmdb environment where the references are stored
    commit_hash : string
        hash of the commit to retrieve.

    Returns
    -------
    tuple
        tuple of tuples containing encoded key/value pairs of the data
        records

    Raises
    ------
    ValueError
        if no commit exists with the provided hash
    '''
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit_hash)
        commitRefVal = reftxn.get(commitRefKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if commitRefVal is False:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    commitRefs = parsing.commit_ref_raw_val_from_db_val(commitRefVal)
    return commitRefs


def get_commit_ref_contents(refenv, commit_hash):
    '''hackish way to convert lmdb from compressed structure to in memory without a new env.

    .. todo:: Completly refactor this mess...

    Parameters
    ----------
    refenv : lmdb.Environment
        lmdb environment where the commit refs are stored
    commit_hash : str
        hash of the commit to get the contents of

    Returns
    -------
    dict
        nested dict
    '''
    with tempfile.TemporaryDirectory() as tempD:
        tmpDF = os.path.join(tempD, 'test.lmdb')
        tmpDB = lmdb.open(path=tmpDF, **c.LMDB_SETTINGS)
        unpack_commit_ref(refenv, tmpDB, commit_hash)
        outDict = RecordQuery(tmpDB).all_records()
        tmpDB.close()

    return outDict


def unpack_commit_ref(refenv, cmtrefenv, commit_hash):
    '''unpack a commit record ref into a new key/val db for reader checkouts.

    Parameters
    ----------
    refenv : lmdb.Environment
        environment handle open for reading in the refenv
    cmtrefenv : lmdb.Environment
        environment handle open for writing on disk. this db must be empty.
    commit_hash : str
        hash of the commit to read in from refs and unpack in a checkout.
    '''

    commitRefs = get_commit_ref(refenv=refenv, commit_hash=commit_hash)
    cmttxn = TxnRegister().begin_writer_txn(cmtrefenv)
    try:
        with cmttxn.cursor() as cursor:
            cursor.first()
            cursor.putmulti(commitRefs, append=True)
        try:
            cursor.close()
        except Exception as e:
            msg = 'could not close cursor cmttxn: {cmttxn} commit_hash: {commit_hash}'
            e.args = (*e.args, msg)
            raise e
    finally:
        TxnRegister().commit_writer_txn(cmtrefenv)

    return


'''
Methods to write new commits
----------------------------

The functions below act to:
    - Reading and formating all record data from the staging area.
    - Determining the ancestor(s) of the new commit
    - Specify commit details (message, time, commiter-info, etc.)
    - Coordinate record hashing
    - Write the commit record
    - Update the branch head to point to the new commit hash
'''

# ---------------- Functions to format the writen values of a commit --------------------


def _commit_ancestors(branchenv: lmdb.Environment,
                      *,
                      is_merge_commit: bool = False,
                      master_branch_name: str = '',
                      dev_branch_name: str = '') -> bytes:
    '''Format the commit parent db value, finding HEAD commits automatically.

    This method handles formating for both regular & merge commits through the
    the keyword only arguments.

    Parameters
    ----------
    branchenv : lmdb.Environment
        Lmdb envrionment where branch data is located. If not merge commit, head
        branch and commit will be found.
    is_merge_commit : bool, optional
        If this is a merge commit or now, defaults to False
    master_branch_name : str, optional
        If merge commit, the master branch name must be specified, and the
        branch HEAD commit hash will be determined automatically, defaults to ''
    dev_branch_name : str, optional
        If merge commit, the dev branch name must be specified, and the branch
        HEAD commit hash will be determined automatically, defaults to ''

    Returns
    -------
    bytes
        Commit parent db value formatted appropriately based on the repo state and
        any specified arguments.
    '''
    if not is_merge_commit:
        masterBranch = heads.get_staging_branch_head(branchenv)
        master_ancestor = heads.get_branch_head_commit(branchenv, masterBranch)
        dev_ancestor = ''
    else:
        master_ancestor = heads.get_branch_head_commit(branchenv, master_branch_name)
        dev_ancestor = heads.get_branch_head_commit(branchenv, dev_branch_name)

    commitParentVal = parsing.commit_parent_db_val_from_raw_val(
        master_ancestor=master_ancestor,
        dev_ancestor=dev_ancestor,
        is_merge_commit=is_merge_commit)

    return commitParentVal


def _commit_spec(message: str, user: str, email: str) -> bytes:
    '''Format the commit specification according to the supplied username and email.

    This method currently only acts as a pass through to the parsing options
    (with time filled in).

    Parameters
    ----------
    message : string
        Commit message sent in by the user.
    user : str, optional
        Name of the commiter
    email : str, optional
        Email of the committer

    Returns
    -------
    bytes
        Formatted value for the specification field of the commit.
    '''
    commitSpecVal = parsing.commit_spec_db_val_from_raw_val(
        commit_time=time.time(),
        commit_message=message,
        commit_user=user,
        commit_email=email)
    return commitSpecVal


def _commit_ref(stageenv: lmdb.Environment) -> bytes:
    '''Query and format all staged data records, and format it for ref storage.

    Parameters
    ----------
    stageenv : lmdb.Environment
        lmdb environment where the staged record data is actually stored.

    Returns
    -------
    bytes
        Serialized and compressed version of all staged record data.

    '''
    querys = RecordQuery(dataenv=stageenv)
    allRecords = tuple(querys._traverse_all_records())
    commitRefVal = parsing.commit_ref_db_val_from_raw_val(allRecords)
    return commitRefVal


# -------------------- Format ref k/v pairs and write the commit to disk ----------------


def commit_records(message, branchenv, stageenv, refenv, repo_path,
                   *, is_merge_commit=False, merge_master=None, merge_dev=None):
    '''Commit all staged records to the repository, updating branch HEAD as needed.

    This method is intended to work for both merge commits as well as regular
    ancestor commits.

    Parameters
    ----------
    message : string
        Message the user accociates with what has been added, removed, or
        changed in this commit. Must not be empty.
    branchenv : lmdb.Environment
        lmdb environment where branch records are stored.
    stageenv : lmdb.Environment
        lmdb environment where the staged data records are stored in
        uncompressed format.
    refenv : lmdb.Environment
        lmdb environment where the commit ref records are stored.
    is_merge_commit : bool, optional
        Is the commit a merge commit or not? defaults to False
    merge_master : string, optional
        If merge commit, specify the name of the master branch, defaults to None
    merge_dev : string, optional
        If merge commit, specify the name of the dev branch, defaults to None

    Returns
    -------
    string
        Commit hash of the newly added commit
    '''
    commitParentVal = _commit_ancestors(branchenv=branchenv,
                                         is_merge_commit=is_merge_commit,
                                         master_branch_name=merge_master,
                                         dev_branch_name=merge_dev)

    user_info_pth = pjoin(repo_path, 'config_user.yml')
    with open(user_info_pth) as f:
        user_info = yaml.safe_load(f.read()) or {}

    USER_NAME = user_info['name']
    USER_EMAIL = user_info['email']
    if (USER_NAME is None) or (USER_EMAIL is None):
        raise RuntimeError(f'Username and Email are required. Please configure.')

    commitSpecVal = _commit_spec(message=message, user=USER_NAME, email=USER_EMAIL)
    commitRefVal = _commit_ref(stageenv=stageenv)

    hasher = hashlib.blake2b(digest_size=20)
    hasher.update(commitParentVal)
    hasher.update(commitSpecVal)
    hasher.update(commitRefVal)
    commit_hash = hasher.hexdigest()

    commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit_hash)
    commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit_hash)
    commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit_hash)

    reftxn = TxnRegister().begin_writer_txn(refenv)
    try:
        reftxn.put(commitSpecKey, commitSpecVal, overwrite=False)
        reftxn.put(commitParentKey, commitParentVal, overwrite=False)
        reftxn.put(commitRefKey, commitRefVal, overwrite=False)
    finally:
        TxnRegister().commit_writer_txn(refenv)

    # possible seperate function
    move_process_data_to_store(repo_path)
    if is_merge_commit is False:
        headBranchName = heads.get_staging_branch_head(branchenv)
        heads.set_branch_head_commit(branchenv, headBranchName, commit_hash)
    else:
        heads.set_staging_branch_head(branchenv=branchenv, branch_name=merge_master)
        heads.set_branch_head_commit(branchenv, merge_master, commit_hash)

    return commit_hash


# --------------------- staging setup, may need to move this elsewhere ------------------


def replace_staging_area_with_commit(refenv, stageenv, commit_hash):
    '''DANGER ZONE: Delete the stage db and replace it with a copy of a commit environent.

    .. warning::

        In the current implementation, this method will not validate that it is safe
        to do this operation. All validation logic must be handled upstream.

    Parameters
    ----------
    refenv : [type]
        lmdb environment opened to the long term storage commit env
    stageenv : lmdb.Environment
        lmdb environment opened to the staging area.
    commit_hash : str
        commit hash to read from the refenv and replace the stage contents with.
    '''
    stagetxn = TxnRegister().begin_writer_txn(stageenv)
    with stagetxn.cursor() as cursor:
        positionExists = cursor.first()
        while positionExists:
            positionExists = cursor.delete()
    cursor.close()
    TxnRegister().commit_writer_txn(stageenv)

    unpack_commit_ref(refenv=refenv, cmtrefenv=stageenv, commit_hash=commit_hash)
    return


def replace_staging_area_with_refs(stageenv, sorted_content):
    '''DANGER ZONE: Delete all stage db records and replace it with specified data.

    .. warning::

        In the current implementation, this method will not validate that it is safe
        to do this operation. All validation logic must be handled upstream.

    Parameters
    ----------
    stageenv : lmdb.Enviornment
        staging area db to replace all data in.
    sorted_content : iterable of tuple
        iterable containing two-tuple of byte encoded record data to place in the
        stageenv db. index 0 -> db key; index 1 -> db val, it is assumed that the
        order of the tuples is lexigraphically sorted by index 0 values, if not,
        this will result in unknown behavior.
    '''
    stagetxn = TxnRegister().begin_writer_txn(stageenv)
    with stagetxn.cursor() as cursor:
        positionExists = cursor.first()
        while positionExists:
            positionExists = cursor.delete()
    cursor.close()
    TxnRegister().commit_writer_txn(stageenv)

    cmttxn = TxnRegister().begin_writer_txn(stageenv)
    try:
        with cmttxn.cursor() as cursor:
            cursor.first()
            cursor.putmulti(sorted_content, append=True)
        cursor.close()
    finally:
        TxnRegister().commit_writer_txn(stageenv)


def move_process_data_to_store(repo_path: str, *, remote_operation: bool = False):
    '''Move symlinks to hdf5 files from process directory to store directory

    In process writes never directly access files in the data directory.
    Instead, when the file is created is is symlinked to either the remote data
    or stage data directory. All access is handled through this intermediate
    symlink in order to prevent any ability to overwpackedeven if
    there are major errors in the hash records). Once the write operation
    packedor staging, or completion of fetch for remote), this
    method is called to move the symlinks from the write enabled directory to
    the (read only, fully-committed) storage dir.

    Parameters
    ----------
    repo_path : str
        path to the repository on dir
    remote_operation : bool, optional
        If this operation is occuring from a remote fetch operation. (the
        default is False, which means that all changes will occur in the staging
        area)

    '''
    store_dir = pjoin(repo_path, c.DIR_DATA_STORE)

    if not remote_operation:
        process_dir = pjoin(repo_path, c.DIR_DATA_STAGE)
    else:
        process_dir = pjoin(repo_path, c.DIR_DATA_REMOTE)

    dirs_to_make, symlinks_to_make = [], []
    for root, dirs, files in os.walk(process_dir):
        for d in dirs:
            dirs_to_make.append(os.path.relpath(pjoin(root, d), process_dir))
        for f in files:
            store_file_pth = pjoin(store_dir, os.path.relpath(pjoin(root, f), process_dir))
            link_file_pth = os.path.normpath(pjoin(root, os.readlink(pjoin(root, f))))
            symlinks_to_make.append((link_file_pth, store_file_pth))

    for d in dirs_to_make:
        dpth = pjoin(store_dir, d)
        if not os.path.isdir(dpth):
            os.makedirs(dpth)
    for src, dest in symlinks_to_make:
        symlink_rel(src, dest)

    # reset before releasing control.
    shutil.rmtree(process_dir)
    os.makedirs(process_dir)


def list_all_commits(refenv):
    '''returns a list of all commits stored in the repostiory

    Parameters
    ----------
    refenv : lmdb.Environment
        db where all commit data is stored

    Returns
    -------
    list
        list of all commit digests.
    '''
    refTxn = TxnRegister().begin_reader_txn(refenv)
    try:
        commits = set()
        with refTxn.cursor() as cursor:
            cursor.first()
            for k in cursor.iternext(keys=True, values=False):
                commitKey = k.decode()[:40]
                commits.add(commitKey)
            cursor.close()
    finally:
        TxnRegister().abort_reader_txn(refenv)

    return list(commits)
