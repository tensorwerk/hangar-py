import configparser
import os
import shutil
import tempfile
import time
from contextlib import contextmanager, closing
from pathlib import Path

import lmdb

from .heads import (
    get_branch_head_commit,
    get_staging_branch_head,
    set_branch_head_commit,
    set_staging_branch_head,
)
from .parsing import (
    cmt_final_digest,
    commit_parent_db_key_from_raw_key,
    commit_parent_db_val_from_raw_val,
    commit_parent_raw_key_from_db_key,
    commit_parent_raw_val_from_db_val,
    commit_ref_db_key_from_raw_key,
    commit_ref_db_val_from_raw_val,
    commit_ref_raw_val_from_db_val,
    commit_spec_db_key_from_raw_key,
    commit_spec_db_val_from_raw_val,
    commit_spec_raw_val_from_db_val,
    DigestAndBytes,
)
from ..constants import (
    CONFIG_USER_NAME,
    DIR_DATA_REMOTE,
    DIR_DATA_STAGE,
    DIR_DATA_STORE,
    LMDB_SETTINGS,
    SEP_KEY,
)
from ..txnctx import TxnRegister

"""
Reading commit specifications and parents.
------------------------------------------
"""


def expand_short_commit_digest(refenv: lmdb.Environment, commit_hash: str) -> str:
    """Find the a full commit hash from a short version provided by the user

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
    """
    reftxn = TxnRegister().begin_reader_txn(refenv)
    commitParentStart = commit_parent_db_key_from_raw_key(commit_hash)
    with reftxn.cursor() as cursor:
        shortHashExists = cursor.set_range(commitParentStart)
        if shortHashExists is True:
            commitKey = cursor.key()
            commit_key = commit_parent_raw_key_from_db_key(commitKey)
            cursor.next()
            cursor.next()
            nextHashExist = cursor.next()
            if nextHashExist is False:
                return commit_key
            nextCommitKey = cursor.key()
            next_commit_key = commit_parent_raw_key_from_db_key(nextCommitKey)
            if next_commit_key.startswith(commit_hash) is True:
                raise KeyError(f'Non unique short commit hash: {commit_hash}')
            else:
                return commit_key
        else:
            raise KeyError(f'No matching commit hash found starting with: {commit_hash}')


def check_commit_hash_in_history(refenv, commit_hash):
    """Check if a commit hash exists in the repository history

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
    """
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        commitParentKey = commit_parent_db_key_from_raw_key(commit_hash)
        commitParentVal = reftxn.get(commitParentKey, default=False)
        isCommitInHistory = True if commitParentVal is not False else False
    finally:
        TxnRegister().abort_reader_txn(refenv)
    return isCommitInHistory


def get_commit_spec(refenv, commit_hash):
    """Get the commit specifications of a particular hash.

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
    """
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        parentCommitSpecKey = commit_spec_db_key_from_raw_key(commit_hash)
        parentCommitSpecVal = reftxn.get(parentCommitSpecKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if parentCommitSpecVal is False:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    parentCommitSpec = commit_spec_raw_val_from_db_val(parentCommitSpecVal)
    return parentCommitSpec.user_spec


def get_commit_ancestors(refenv, commit_hash):
    """find the ancestors of a particular commit hash.

    Parameters
    ----------
    refenv : lmdb.Environment
        lmdb environment where the commit refs are stored
    commit_hash : string
        commit hash to find the ancestors for

    Returns
    -------
    namedtuple
        Namedtuple describing is_merge_commit, master_ancestor, &
        child_ancestor (in the even of merge commit)

    Raises
    ------
    ValueError
        if no commit exists with the provided hash
    """

    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        parentCommitKey = commit_parent_db_key_from_raw_key(commit_hash)
        parentCommitVal = reftxn.get(parentCommitKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if parentCommitVal is False:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    parentCommitAncestors = commit_parent_raw_val_from_db_val(parentCommitVal)
    return parentCommitAncestors.ancestor_spec


def get_commit_ancestors_graph(refenv, starting_commit):
    """returns a DAG of all commits starting at some hash pointing to the repo root.

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
    """
    parent_commit = starting_commit
    commit_graph = {}
    seen = set(starting_commit)
    more_work = []
    end_commit = False

    if parent_commit == '':
        end_commit = True

    while end_commit is not True:
        childCommit = get_commit_ancestors(refenv, parent_commit)

        if (childCommit.master_ancestor == '') or (childCommit.master_ancestor in seen):
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


"""
Methods for reading packed commit data and reconstructing an unpacked format.
-----------------------------------------------------------------------------
"""


def get_commit_ref(refenv, commit_hash):
    """Read the commit data record references from a specific commit.

    This only returns a list of tuples with binary encoded key/value pairs.

    Parameters
    ----------
    refenv : lmdb.Environment`
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
    """
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        cmtRefKey = commit_ref_db_key_from_raw_key(commit_hash)
        cmtSpecKey = commit_spec_db_key_from_raw_key(commit_hash)
        cmtParentKey = commit_parent_db_key_from_raw_key(commit_hash)

        cmtRefVal = reftxn.get(cmtRefKey, default=False)
        cmtSpecVal = reftxn.get(cmtSpecKey, default=False)
        cmtParentVal = reftxn.get(cmtParentKey, default=False)
    except lmdb.BadValsizeError:
        raise ValueError(f'No commit exists with the hash: {commit_hash}')
    finally:
        TxnRegister().abort_reader_txn(refenv)

    if (cmtRefVal is False) or (cmtSpecVal is False) or (cmtParentVal is False):
        raise ValueError(f'No commit exists with the hash: {commit_hash}')

    commitRefs = commit_ref_raw_val_from_db_val(cmtRefVal)
    commitSpecs = commit_spec_raw_val_from_db_val(cmtSpecVal)
    commitParent = commit_parent_raw_val_from_db_val(cmtParentVal)

    calculatedDigest = cmt_final_digest(
        parent_digest=commitParent.digest,
        spec_digest=commitSpecs.digest,
        refs_digest=commitRefs.digest)

    if calculatedDigest != commit_hash:
        raise IOError(
            f'Data Corruption Detected. On retrieval of stored references for '
            f'commit_hash: {commit_hash} validation of commit record/contents '
            f'integrity failed. Calculated digest: {calculatedDigest} != '
            f'expected: {commit_hash}. Please alert the Hangar development team to '
            f'this error if possible.')

    return commitRefs.db_kvs


def unpack_commit_ref(refenv, cmtrefenv, commit_hash):
    """unpack a commit record ref into a new key/val db for reader checkouts.

    This method also validates that the record data (parent, spec, and refs)
    have not been corrupted on disk (ie)

    Parameters
    ----------
    refenv : lmdb.Environment
        environment handle open for reading in the refenv
    cmtrefenv : lmdb.Environment
        environment handle open for writing on disk. this db must be empty.
    commit_hash : str
        hash of the commit to read in from refs and unpack in a checkout.
    """

    commitRefs = get_commit_ref(refenv=refenv, commit_hash=commit_hash)
    cmttxn = TxnRegister().begin_writer_txn(cmtrefenv)
    try:
        with cmttxn.cursor() as cursor:
            cursor.first()
            cursor.putmulti(commitRefs, append=True)
        try:
            cursor.close()
        except Exception as e:
            msg = f'could not close cursor cmttxn {cmttxn} commit_hash {commit_hash}'
            e.args = (*e.args, msg)
            raise e
    finally:
        TxnRegister().commit_writer_txn(cmtrefenv)

    return


@contextmanager
def tmp_cmt_env(refenv: lmdb.Environment, commit_hash: str):
    """create temporary unpacked lmdb environment from compressed structure

    Parameters
    ----------
    refenv : lmdb.Environment
        lmdb environment where the commit refs are stored
    commit_hash : str
        hash of the commit to get the contents of

    Returns
    -------
    lmdb.Environment
        environment with all db contents from ``commit`` unpacked
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpDF = os.path.join(tmpdir, 'test.lmdb')
        with closing(
                lmdb.open(tmpDF, sync=False, writemap=True, **LMDB_SETTINGS)
        ) as tmpDB:
            unpack_commit_ref(refenv, tmpDB, commit_hash)
            yield tmpDB


"""
Methods to write new commits
----------------------------

The functions below act to:
    - Reading and formatting all record data from the staging area.
    - Determining the ancestor(s) of the new commit
    - Specify commit details (message, time, committer-info, etc.)
    - Coordinate record hashing
    - Write the commit record
    - Update the branch head to point to the new commit hash
"""

# ---------------- Functions to format the written values of a commit --------------------


def _commit_ancestors(branchenv: lmdb.Environment,
                      *,
                      is_merge_commit: bool = False,
                      master_branch: str = '',
                      dev_branch: str = '') -> DigestAndBytes:
    """Format the commit parent db value, finding HEAD commits automatically.

    This method handles formatting for both regular & merge commits through the
    the keyword only arguments.

    Parameters
    ----------
    branchenv : lmdb.Environment
        Lmdb environment where branch data is located. If not merge commit, head
        branch and commit will be found.
    is_merge_commit : bool, optional
        If this is a merge commit or now, defaults to False
    master_branch : str, optional
        If merge commit, the master branch name must be specified, and the
        branch HEAD commit hash will be determined automatically, defaults to ''
    dev_branch : str, optional
        If merge commit, the dev branch name must be specified, and the branch
        HEAD commit hash will be determined automatically, defaults to ''

    Returns
    -------
    DigestAndBytes
        Commit parent db value and digest of commit parent val formatted
        appropriately based on the repo state and any specified arguments.
    """
    if not is_merge_commit:
        masterBranch = get_staging_branch_head(branchenv)
        master_ancestor = get_branch_head_commit(branchenv, masterBranch)
        dev_ancestor = ''
    else:
        master_ancestor = get_branch_head_commit(branchenv, master_branch)
        dev_ancestor = get_branch_head_commit(branchenv, dev_branch)

    commitParentVal = commit_parent_db_val_from_raw_val(
        master_ancestor=master_ancestor,
        dev_ancestor=dev_ancestor,
        is_merge_commit=is_merge_commit)

    return commitParentVal


def _commit_spec(message: str, user: str, email: str) -> DigestAndBytes:
    """Format the commit specification according to the supplied username and email.

    This method currently only acts as a pass through to the parsing options
    (with time filled in).

    Parameters
    ----------
    message : string
        Commit message sent in by the user.
    user : str, optional
        Name of the committer
    email : str, optional
        Email of the committer

    Returns
    -------
    DigestAndBytes
        Formatted value for the specification field of the commit and digest of
        spec.
    """
    spec_db = commit_spec_db_val_from_raw_val(commit_time=time.time(),
                                              commit_message=message,
                                              commit_user=user,
                                              commit_email=email)
    return spec_db


def _commit_ref(stageenv: lmdb.Environment) -> DigestAndBytes:
    """Query and format all staged data records, and format it for ref storage.

    Parameters
    ----------
    stageenv : lmdb.Environment
        lmdb environment where the staged record data is actually stored.

    Returns
    -------
    DigestAndBytes
        Serialized and compressed version of all staged record data along with
        digest of commit refs.
    """
    from .queries import RecordQuery  # needed to avoid cyclic import

    querys = RecordQuery(dataenv=stageenv)
    allRecords = tuple(querys._traverse_all_records())
    res = commit_ref_db_val_from_raw_val(allRecords)
    return res


# -------------------- Format ref k/v pairs and write the commit to disk ----------------


def commit_records(message, branchenv, stageenv, refenv, repo_path: Path,
                   *, is_merge_commit=False, merge_master=None, merge_dev=None):
    """Commit all staged records to the repository, updating branch HEAD as needed.

    This method is intended to work for both merge commits as well as regular
    ancestor commits.

    Parameters
    ----------
    message : string
        Message the user associates with what has been added, removed, or
        changed in this commit. Must not be empty.
    branchenv : lmdb.Environment
        lmdb environment where branch records are stored.
    stageenv : lmdb.Environment
        lmdb environment where the staged data records are stored in
        uncompressed format.
    refenv : lmdb.Environment
        lmdb environment where the commit ref records are stored.
    repo_path : Path
        path to the hangar repository on disk
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
    """
    cmtParent = _commit_ancestors(branchenv=branchenv,
                                  is_merge_commit=is_merge_commit,
                                  master_branch=merge_master,
                                  dev_branch=merge_dev)

    user_info_pth = Path(repo_path, CONFIG_USER_NAME)
    CFG = configparser.ConfigParser()
    CFG.read(user_info_pth)

    USER_NAME = CFG['USER']['name']
    USER_EMAIL = CFG['USER']['email']
    if (USER_NAME is None) or (USER_EMAIL is None):
        raise RuntimeError(f'Username and Email are required. Please configure.')

    cmtSpec = _commit_spec(message=message, user=USER_NAME, email=USER_EMAIL)
    cmtRefs = _commit_ref(stageenv=stageenv)

    commit_hash = cmt_final_digest(parent_digest=cmtParent.digest,
                                   spec_digest=cmtSpec.digest,
                                   refs_digest=cmtRefs.digest)

    commitSpecKey = commit_spec_db_key_from_raw_key(commit_hash)
    commitParentKey = commit_parent_db_key_from_raw_key(commit_hash)
    commitRefKey = commit_ref_db_key_from_raw_key(commit_hash)

    reftxn = TxnRegister().begin_writer_txn(refenv)
    try:
        reftxn.put(commitSpecKey, cmtSpec.raw, overwrite=False)
        reftxn.put(commitParentKey, cmtParent.raw, overwrite=False)
        reftxn.put(commitRefKey, cmtRefs.raw, overwrite=False)
    finally:
        TxnRegister().commit_writer_txn(refenv)

    # possible separate function
    move_process_data_to_store(repo_path)
    if is_merge_commit is False:
        headBranchName = get_staging_branch_head(branchenv)
        set_branch_head_commit(branchenv, headBranchName, commit_hash)
    else:
        set_staging_branch_head(branchenv=branchenv, branch_name=merge_master)
        set_branch_head_commit(branchenv, merge_master, commit_hash)

    return commit_hash


# --------------------- staging setup, may need to move this elsewhere ------------------


def replace_staging_area_with_commit(refenv, stageenv, commit_hash):
    """DANGER ZONE: Delete the stage db and replace it with a copy of a commit environment.

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
    """
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
    """DANGER ZONE: Delete all stage db records and replace it with specified data.

    .. warning::

        In the current implementation, this method will not validate that it is safe
        to do this operation. All validation logic must be handled upstream.

    Parameters
    ----------
    stageenv : lmdb.Environment
        staging area db to replace all data in.
    sorted_content : iterable of tuple
        iterable containing two-tuple of byte encoded record data to place in
        the stageenv db. index 0 -> db key; index 1 -> db val, it is assumed
        that the order of the tuples is lexicographically sorted by index 0
        values, if not, this will result in unknown behavior.
    """
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


def move_process_data_to_store(repo_path: Path, *, remote_operation: bool = False):
    """Move symlinks to hdf5 files from process directory to store directory

    In process writes never directly access files in the data directory.
    Instead, when the file is created is is symlinked to either the remote data
    or stage data directory. All access is handled through this intermediate
    symlink in order to prevent any ability to overwrite (even if there are
    major errors in the hash records). Once the write operation is packed in
    the staging or remote area, this method is called to move the symlinks from
    the write enabled directory to the (read only, fully-committed) storage
    dir.

    Parameters
    ----------
    repo_path : Path
        path to the repository on dir
    remote_operation : bool, optional
        If this operation is occurring from a remote fetch operation. (the
        default is False, which means that all changes will occur in the
        staging area)

    """
    store_dir = Path(repo_path, DIR_DATA_STORE)

    type_dir = DIR_DATA_REMOTE if remote_operation else DIR_DATA_STAGE
    process_dir = Path(repo_path, type_dir)

    store_fps = []
    for be_pth in process_dir.iterdir():
        if be_pth.is_dir():
            for fpth in be_pth.iterdir():
                if fpth.is_file() and not fpth.stem.startswith('.'):
                    store_fps.append(store_dir.joinpath(be_pth.name, fpth.name))

    for fpth in store_fps:
        if not fpth.parent.is_dir():
            fpth.parent.mkdir()
        fpth.touch()

    # reset before releasing control.
    shutil.rmtree(process_dir)
    process_dir.mkdir(exist_ok=False)


def list_all_commits(refenv):
    """returns a list of all commits stored in the repository

    Parameters
    ----------
    refenv : lmdb.Environment
        db where all commit data is stored

    Returns
    -------
    list
        list of all commit digests.
    """
    refTxn = TxnRegister().begin_reader_txn(refenv)
    try:
        commits = set()
        with refTxn.cursor() as cursor:
            cursor.first()
            for k in cursor.iternext(keys=True, values=False):
                commitKey, *_ = k.decode().split(SEP_KEY)
                commits.add(commitKey)
            cursor.close()
    finally:
        TxnRegister().abort_reader_txn(refenv)

    return list(commits)
