import hashlib
import os
import tempfile
import time
from os.path import join as pjoin

import lmdb

from . import heads, parsing
from .. import config
from ..context import TxnRegister
from .queries import RecordQuery


'''
Reading commit specifications and parents.
------------------------------------------
'''


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
    refenv : lmdb.Environment`
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
    refenv : lmdb.Environment`
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
    '''returns a DAG of all commits which start at some hash as they point to the repo root.

    Parameters
    ----------
    refenv : lmdb.Environment`
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
    refenv : lmdb.Environment`
        lmdb environment where the commit refs are stored
    commit_hash : str
        hash of the commit to get the contents of

    Returns
    -------
    dict
        nested dict
    '''
    LMDB_CONFIG = config.get('hangar.lmdb')

    with tempfile.TemporaryDirectory() as tempD:
        tmpDF = os.path.join(tempD, 'test.lmdb')
        tmpDB = lmdb.open(path=tmpDF, **LMDB_CONFIG)
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
        except Exception:
            print('could not close cursor')
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


def __commit_ancestors(branchenv, *, is_merge_commit=False, master_branch_name='', dev_branch_name=''):
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
    bytestring
        Commit parent db value formated appropriatly based on the repo state and
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


def __commit_spec(message, user, email):
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
    bytestring
        Formated value for the specification field of the commit.
    '''
    commitSpecVal = parsing.commit_spec_db_val_from_raw_val(
        commit_time=time.time(),
        commit_message=message,
        commit_user=user,
        commit_email=email)

    return commitSpecVal


def __commit_ref(stageenv):
    '''Query and format all staged data records, and format it for ref storage.

    Parameters
    ----------
    stageenv : lmdb.Environment`
        lmdb environment where the staged record data is actually stored.

    Returns
    -------
    bytes
        Serialized and compressed version of all staged record data.

    '''
    querys = RecordQuery(dataenv=stageenv)
    allRecords = querys._traverse_all_records()
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
    branchenv : lmdb.Environment`
        lmdb environment where branch records are stored.
    stageenv : lmdb.Environment`
        lmdb environment where the staged data records are stored in
        uncompressed format.
    refenv : lmdb.Environment`
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
    commitParentVal = __commit_ancestors(
        branchenv=branchenv,
        is_merge_commit=is_merge_commit,
        master_branch_name=merge_master,
        dev_branch_name=merge_dev)

    USER_NAME = config.get('user.name')
    USER_EMAIL = config.get('user.email')
    if (USER_NAME is None) or (USER_EMAIL is None):
        raise RuntimeError(f'Username and Email are required. Please configure.')

    commitSpecVal = __commit_spec(message=message, user=USER_NAME, email=USER_EMAIL)
    commitRefVal = __commit_ref(stageenv=stageenv)

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
    symlink in order to prevent any ability to overwrite commit data (even if
    there are major errors in the hash records). Once the write operation
    complets (commit for staging, or completion of fetch for remote), this
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
    STORE_DATA_DIR = config.get('hangar.repository.store_data_dir')
    store_dir = pjoin(repo_path, STORE_DATA_DIR)

    if remote_operation:
        REMOTE_DATA_DIR = config.get('hangar.repository.remote_data_dir')
        process_dir = pjoin(repo_path, REMOTE_DATA_DIR)
    else:
        STAGE_DATA_DIR = config.get('hangar.repository.stage_data_dir')
        process_dir = pjoin(repo_path, STAGE_DATA_DIR)

    p_schema_dirs = [x for x in os.listdir(process_dir) if x.startswith('hdf_')]
    for p_schema_dir in p_schema_dirs:
        schema_dir_pth = pjoin(process_dir, p_schema_dir)
        p_files = [x for x in os.listdir(schema_dir_pth) if x.endswith('.hdf5')]
        store_schema_dir_pth = pjoin(store_dir, p_schema_dir)

        for p_file in p_files:
            if not os.path.isdir(store_schema_dir_pth):
                os.makedirs(store_schema_dir_pth)
            p_fp = pjoin(schema_dir_pth, p_file)
            p_relpth = os.readlink(p_fp)
            store_fp = pjoin(store_schema_dir_pth, p_file)
            os.symlink(p_relpth, store_fp)
            os.remove(p_fp)

        os.removedirs(schema_dir_pth)

    os.makedirs(process_dir, exist_ok=True)


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
