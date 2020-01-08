"""Merge Methods

In the current implementation only fast-forward and a competent, but limited,
three-way merge algorithm are implemented. All user facing API calls should be
funneled through the :function:`select_merge_algorithm` function

.. note::

    In the current implementation, it is not possible to stop a merge in progress or
    to revert a bad merge commit. All revert like operations should be made by
    creating new branches from the last "good" state, after which new merge
    operations can be attempted (if desired.)
"""
from pathlib import Path

import lmdb

from .diff import WriterUserDiff, diff_envs, find_conflicts
from .records.commiting import (
    tmp_cmt_env,
    replace_staging_area_with_commit,
    replace_staging_area_with_refs,
    commit_records,
)
from .records.hashs import clear_stage_hash_records, backends_remove_in_process_data
from .records.heads import (
    get_staging_branch_head,
    get_branch_head_commit,
    set_staging_branch_head,
    set_branch_head_commit,
    release_writer_lock,
    acquire_writer_lock,
)


def select_merge_algorithm(message: str,
                           branchenv: lmdb.Environment,
                           stageenv: lmdb.Environment,
                           refenv: lmdb.Environment,
                           stagehashenv: lmdb.Environment,
                           master_branch: str,
                           dev_branch: str,
                           repo_path: Path,
                           *,
                           writer_uuid: str = 'MERGE_PROCESS') -> str:
    """Entry point to perform a merge.

    Automatically selects algorithm and does the operation if no conflicts are
    found. This call requires that the staging area status be "CLEAN", if
    a "DIRTY" staging environment is found, an RuntimeError will be thrown.

    Parameters
    ----------
    message : str
        user message describing the commit
    branchenv : lmdb.Environment
        where the branch references are stored
    stageenv : lmdb.Environment
        where the staging area is open
    refenv : lmdb.Environment
        where commit history is stored
    stagehashenv: lmdb.Environment
        where the stage hash environment data is stored
    master_branch : str
        name of the branch to serve as a merge master
    dev_branch : str
        name of the branch to use as the feature branch
    repo_path: Path
        path to the repository on disk
    writer_uuid : str, optional, kwarg only
        if the merge method is called from the repo level, the default writer
        lock `MERGE_PROCESS` is used to ensure that a writer is active. If
        called from within a write-enabled checkout, the writer lock is set to
        the writer_uuid of the writer checkout so that the lock can be acquired.

    Raises
    ------
    RuntimeError
        if the staging area is not `CLEAN` of other changes
    PermissionError
        if the writer lock is currently held
    ValueError
        if a conflict is found in a three way merge, no operation will be performed.

    Returns
    -------
    str
        commit hash of the merge if this was a successful operation.
    """
    current_head = get_staging_branch_head(branchenv)
    wDiffer = WriterUserDiff(stageenv=stageenv,
                             branchenv=branchenv,
                             refenv=refenv,
                             branch_name=current_head)
    if wDiffer.status() != 'CLEAN':
        e = RuntimeError(
            'Changes are currently pending in the staging area To avoid mangled '
            'histories, the staging area must exist in a clean state Please '
            'reset or commit any changes before the merge operation')
        raise e from None

    try:
        acquire_writer_lock(branchenv=branchenv, writer_uuid=writer_uuid)
    except PermissionError as e:
        raise e from None

    try:
        mHEAD = get_branch_head_commit(branchenv, branch_name=master_branch)
        dHEAD = get_branch_head_commit(branchenv, branch_name=dev_branch)
        branchHistory = wDiffer._determine_ancestors(mHEAD=mHEAD, dHEAD=dHEAD)

        if branchHistory.canFF is True:
            print('Selected Fast-Forward Merge Strategy')
            success = _fast_forward_merge(
                branchenv=branchenv,
                stageenv=stageenv,
                refenv=refenv,
                stagehashenv=stagehashenv,
                master_branch=master_branch,
                new_masterHEAD=branchHistory.devHEAD,
                repo_path=repo_path)
        else:
            print('Selected 3-Way Merge Strategy')
            success = _three_way_merge(
                message=message,
                master_branch=master_branch,
                masterHEAD=branchHistory.masterHEAD,
                dev_branch=dev_branch,
                devHEAD=branchHistory.devHEAD,
                ancestorHEAD=branchHistory.ancestorHEAD,
                branchenv=branchenv,
                stageenv=stageenv,
                refenv=refenv,
                stagehashenv=stagehashenv,
                repo_path=repo_path)

    except ValueError as e:
        raise e from None

    finally:
        if writer_uuid == 'MERGE_PROCESS':
            release_writer_lock(branchenv=branchenv, writer_uuid=writer_uuid)

    return success


# ------------------ Fast Forward Merge Methods -------------------------------


def _fast_forward_merge(branchenv: lmdb.Environment,
                        stageenv: lmdb.Environment,
                        refenv: lmdb.Environment,
                        stagehashenv: lmdb.Environment,
                        master_branch: str,
                        new_masterHEAD: str,
                        repo_path: Path) -> str:
    """Update branch head pointer to perform a fast-forward merge.

    This method does not check that it is safe to do this operation, all
    verification should happen before this point is reached

    Parameters
    ----------
    branchenv : lmdb.Environment
        db with the branch head pointers
    stageenv : lmdb.Environment
        db where the staging area records are stored.
    refenv : lmdb.Environment
        db where the merge commit records are stored.
    stagehashenv: lmdb.Environment
        db where the staged hash records are stored
    master_branch : str
        name of the merge_master branch which should be updated
    new_masterHEAD : str
        commit hash to update the master_branch name to point to.
    repo_path: Path
        path to the repository on disk.

    Returns
    -------
    str
        if successful, returns the commit hash the master branch name was
        updated to.
    """
    try:
        replace_staging_area_with_commit(
            refenv=refenv, stageenv=stageenv, commit_hash=new_masterHEAD)

        outBranchName = set_branch_head_commit(
            branchenv=branchenv, branch_name=master_branch, commit_hash=new_masterHEAD)
        set_staging_branch_head(branchenv=branchenv, branch_name=master_branch)

        backends_remove_in_process_data(repo_path=repo_path)
        clear_stage_hash_records(stagehashenv=stagehashenv)

    except ValueError as e:
        raise e from None

    return outBranchName


# ----------------------- Three-Way Merge Methods -----------------------------


def _three_way_merge(message: str,
                     master_branch: str,
                     masterHEAD: str,
                     dev_branch: str,
                     devHEAD: str,
                     ancestorHEAD: str,
                     branchenv: lmdb.Environment,
                     stageenv: lmdb.Environment,
                     refenv: lmdb.Environment,
                     stagehashenv: lmdb.Environment,
                     repo_path: Path) -> str:
    """Merge strategy with diff/patch computed from changes since last common ancestor.

    Parameters
    ----------
    message : str
        commit message to apply to this merge commit (specified by the user)
    master_branch : str
        name of the merge master branch
    masterHEAD : str
        commit hash of the merge master HEAD
    dev_branch : str
        name of the merge dev branch
    devHEAD : str
        commit hash of the merge dev HEAD
    ancestorHEAD : str
        commit hash of the nearest common ancestor which the merge_master and
        merge_dev branches both share in their commit history.
    branchenv : lmdb.Environment
        db where the branch head records are stored
    stageenv : lmdb.Environment
        db where the staging area records are stored.
    refenv : lmdb.Environment
        db where the merge commit records are stored.
    stagehashenv: lmdb.Environment
        db where the staged hash records are stored
    repo_path: Path
        path to the repository on disk.

    Returns
    -------
    str
        commit hash of the new merge commit if the operation was successful.

    Raises
    ------
    ValueError
        If a conflict is found, the operation will abort before completing.
    """
    with tmp_cmt_env(refenv, ancestorHEAD) as aEnv, tmp_cmt_env(
            refenv, masterHEAD) as mEnv, tmp_cmt_env(refenv, devHEAD) as dEnv:

        m_diff = diff_envs(aEnv, mEnv)
        d_diff = diff_envs(aEnv, dEnv)
        conflict = find_conflicts(m_diff, d_diff)
        if conflict.conflict is True:
            msg = f'HANGAR VALUE ERROR:: Merge ABORTED with conflict: {conflict}'
            raise ValueError(msg) from None

        with mEnv.begin(write=True) as txn:
            for k, _ in d_diff.deleted:
                txn.delete(k)
            for k, v in d_diff.mutated:
                txn.put(k, v, overwrite=True)
            for k, v in d_diff.added:
                txn.put(k, v, overwrite=True)

        dbcont = []
        with mEnv.begin(write=False) as txn:
            with txn.cursor() as cur:
                cur.first()
                for kv in cur.iternext(keys=True, values=True):
                    dbcont.append(kv)

    backends_remove_in_process_data(repo_path=repo_path)
    replace_staging_area_with_refs(stageenv=stageenv, sorted_content=dbcont)

    commit_hash = commit_records(
        message=message,
        branchenv=branchenv,
        stageenv=stageenv,
        refenv=refenv,
        repo_path=repo_path,
        is_merge_commit=True,
        merge_master=master_branch,
        merge_dev=dev_branch)

    clear_stage_hash_records(stagehashenv=stagehashenv)
    return commit_hash
