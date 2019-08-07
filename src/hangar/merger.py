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
import os

import lmdb

from .diff import ThreeWayCommitDiffer, WriterUserDiff, ReaderUserDiff
from .records import commiting, hashs, heads, parsing


def select_merge_algorithm(message: str,
                           branchenv: lmdb.Environment,
                           stageenv: lmdb.Environment,
                           refenv: lmdb.Environment,
                           stagehashenv: lmdb.Environment,
                           master_branch_name: str,
                           dev_branch_name: str,
                           repo_path: str,
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
    master_branch_name : str
        name of the branch to serve as a merge master
    dev_branch_name : str
        name of the branch to use as the feature branch
    repo_path: str
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
    current_head = heads.get_staging_branch_head(branchenv)
    wDiffer = WriterUserDiff(stageenv=stageenv,
                             branchenv=branchenv,
                             refenv=refenv,
                             branch_name=current_head)
    if wDiffer.status() != 'CLEAN':
        e = RuntimeError(
            'HANGAR RUNTIME ERROR: Changes are currently pending in the staging area ',
            'To avoid mangled histories, the staging area must exist in a clean state ',
            'Please reset or commit any changes before the merge operation')
        raise e from None

    try:
        heads.acquire_writer_lock(branchenv=branchenv, writer_uuid=writer_uuid)
    except PermissionError as e:
        raise e from None

    try:
        mHEAD = heads.get_branch_head_commit(branchenv, branch_name=master_branch_name)
        dHEAD = heads.get_branch_head_commit(branchenv, branch_name=dev_branch_name)
        rDiffer = ReaderUserDiff(commit_hash=mHEAD, branchenv=branchenv, refenv=refenv)
        branchHistory = rDiffer._determine_ancestors(mHEAD=mHEAD, dHEAD=dHEAD)

        if branchHistory.canFF is True:
            print('Selected Fast-Forward Merge Strategy')
            success = _fast_forward_merge(
                branchenv=branchenv,
                stageenv=stageenv,
                refenv=refenv,
                stagehashenv=stagehashenv,
                master_branch=master_branch_name,
                new_masterHEAD=branchHistory.devHEAD,
                repo_path=repo_path)
        else:
            print('Selected 3-Way Merge Strategy')
            success = _three_way_merge(
                message=message,
                master_branch_name=master_branch_name,
                masterHEAD=branchHistory.masterHEAD,
                dev_branch_name=dev_branch_name,
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
            heads.release_writer_lock(branchenv=branchenv, writer_uuid=writer_uuid)

    return success


# ------------------ Fast Forward Merge Methods -------------------------------


def _fast_forward_merge(branchenv: lmdb.Environment,
                        stageenv: lmdb.Environment,
                        refenv: lmdb.Environment,
                        stagehashenv: lmdb.Environment,
                        master_branch: str,
                        new_masterHEAD: str,
                        repo_path: os.PathLike) -> str:
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
    repo_path: os.PathLike
        path to the repository on disk.

    Returns
    -------
    str
        if successful, returns the commit hash the master branch name was
        updated to.
    """
    try:
        commiting.replace_staging_area_with_commit(
            refenv=refenv, stageenv=stageenv, commit_hash=new_masterHEAD)

        outBranchName = heads.set_branch_head_commit(
            branchenv=branchenv, branch_name=master_branch, commit_hash=new_masterHEAD)
        heads.set_staging_branch_head(branchenv=branchenv, branch_name=master_branch)

        hashs.delete_in_process_data(repo_path=repo_path)
        hashs.clear_stage_hash_records(stagehashenv=stagehashenv)

    except ValueError as e:
        raise e from None

    return outBranchName


# ----------------------- Three-Way Merge Methods -----------------------------


def _three_way_merge(message: str,
                     master_branch_name: str,
                     masterHEAD: str,
                     dev_branch_name: str,
                     devHEAD: str,
                     ancestorHEAD: str,
                     branchenv: lmdb.Environment,
                     stageenv: lmdb.Environment,
                     refenv: lmdb.Environment,
                     stagehashenv: lmdb.Environment,
                     repo_path: os.PathLike) -> str:
    """Merge strategy with diff/patch computed from changes since last common ancestor.

    Parameters
    ----------
    message : str
        commit message to apply to this merge commit (specified by the user)
    master_branch_name : str
        name of the merge master branch
    masterHEAD : str
        commit hash of the merge master HEAD
    dev_branch_name : str
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
    repo_path: os.PathLike
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
    aCont = commiting.get_commit_ref_contents(refenv=refenv, commit_hash=ancestorHEAD)
    mCont = commiting.get_commit_ref_contents(refenv=refenv, commit_hash=masterHEAD)
    dCont = commiting.get_commit_ref_contents(refenv=refenv, commit_hash=devHEAD)

    try:
        mergeContents = _compute_merge_results(a_cont=aCont, m_cont=mCont, d_cont=dCont)
    except ValueError as e:
        raise e from None

    fmtCont = _merge_dict_to_lmdb_tuples(patchedRecs=mergeContents)
    hashs.delete_in_process_data(repo_path=repo_path)
    commiting.replace_staging_area_with_refs(stageenv=stageenv, sorted_content=fmtCont)

    commit_hash = commiting.commit_records(
        message=message,
        branchenv=branchenv,
        stageenv=stageenv,
        refenv=refenv,
        repo_path=repo_path,
        is_merge_commit=True,
        merge_master=master_branch_name,
        merge_dev=dev_branch_name)

    hashs.clear_stage_hash_records(stagehashenv=stagehashenv)
    return commit_hash


def _merge_changes(changes: dict, m_dict: dict) -> dict:
    """Common class which can merge changes between two branches.

    This class does NOT CHECK FOR MERGE CONFLICTS, and will result in UNDEFINED
    BEHAVIOR if conflicts are present. All validations must be performed
    upstream of this method.

    Parameters
    ----------
    changes : dict
        mapping containing changes made on `master` and `dev` branch.
    m_dict : dict
        record structure of interest as it exists at the tip of the `master`
        branch. Acts as the starting point from which to merge changes on `dev`
        into.

    Returns
    -------
    dict
        record structure similar to `m_dict` input with changes merged into
        it which were made on the `dev` branch.
    """
    m_added = changes['master']['additions']
    m_unchanged = changes['master']['unchanged']
    # m_removed = changes['master']['removals']
    m_mutated = changes['master']['mutations']
    d_added = changes['dev']['additions']
    # d_unchanged = changes['dev']['unchanged']
    d_removed = changes['dev']['removals']
    d_mutated = changes['dev']['mutations']

    for dK in d_added:
        if dK not in m_added:
            m_dict[dK] = d_added[dK]

    for dK in d_removed:
        if dK in m_unchanged:
            del m_dict[dK]

    for dK in d_mutated:
        if dK not in m_mutated:
            m_dict[dK] = d_mutated[dK]

    return m_dict


def _compute_merge_results(a_cont, m_cont, d_cont):
    """Compute the diff of a 3-way merge and patch historical contents to get new state

    Parameters
    ----------
    a_cont : dict
        contents of the latest common ancestor of the `master` and `dev` HEAD commits.
    m_cont : dict
        contents of the `master` HEAD commit.
    d_cont : dict
        contents of the `dev` HEAD commit.

    Returns
    -------
    dict
        nested dict specifying arraysets and metadata record specs of the new
        merge commit.

    Raises
    ------
    ValueError
        If a conflict is found, the operation is aborted
    """
    # conflict checking
    cmtDiffer = ThreeWayCommitDiffer(a_cont, m_cont, d_cont)
    confs = cmtDiffer.determine_conflicts()
    if confs['conflict_found'] is True:
        msg = f'HANGAR VALUE ERROR:: Merge ABORTED with conflict: {confs}'
        raise ValueError(msg) from None

    # merging: arrayset schemas
    m_schema_dict = {}
    for asetn in m_cont['arraysets']:
        m_schema_dict[asetn] = m_cont['arraysets'][asetn]['schema']
    o_schema_dict = _merge_changes(cmtDiffer.arrayset_changes(), m_schema_dict)

    # merging: arrayset samples
    o_data_dict = {}
    sample_changes = cmtDiffer.sample_changes()
    for asetn in o_schema_dict:
        if asetn not in m_cont['arraysets']:
            o_data_dict[asetn] = d_cont['arraysets'][asetn]['data']
            continue
        else:
            m_asetn_data_dict = m_cont['arraysets'][asetn]['data']

        if asetn not in d_cont['arraysets']:
            o_data_dict[asetn] = m_cont['arraysets'][asetn]['data']
            continue

        aset_sample_changes = {
            'master': sample_changes['master'][asetn],
            'dev': sample_changes['dev'][asetn],
        }
        o_data_dict[asetn] = _merge_changes(aset_sample_changes, m_asetn_data_dict)

    # merging: metadata
    o_meta_dict = _merge_changes(cmtDiffer.meta_changes(), m_cont['metadata'])

    # collect all merge results into final data structure
    outDict = {}
    outDict['metadata'] = o_meta_dict
    outDict['arraysets'] = {}
    for asetn, asetSchema in o_schema_dict.items():
        outDict['arraysets'][asetn] = {
            'schema': asetSchema,
            'data': o_data_dict[asetn]
        }
    return outDict


def _merge_dict_to_lmdb_tuples(patchedRecs):
    """Create a lexicographically sorted iterable of (key/val tuples) from a dict.

    .. note::

        This  method is currently designed to parse the output of
        :function:`~commiting.get_commit_ref_contents` and will break if that output
        format, or the output of :function:`_compute_merge_results` changes.

    Parameters
    ----------
    patchedRecs : dict
        nested dict which specifies all records for arraysets & metadata

    Returns
    -------
    iterable
        iterable of tuples formatted correctly to serve an a drop in replacement
        for the staging environment, with elements lexicographically sorted so
        that an lmdb `putmulti` operation can be performed with `append=True`.
    """
    entries = []
    numAsetsKey = parsing.arrayset_total_count_db_key()
    numAsetsVal = parsing.arrayset_total_count_db_val_from_raw_val(
        number_of_asets=len(patchedRecs['arraysets'].keys()))
    entries.append((numAsetsKey, numAsetsVal))

    for asetn in patchedRecs['arraysets'].keys():
        schemaSpec = patchedRecs['arraysets'][asetn]['schema']
        schemaKey = parsing.arrayset_record_schema_db_key_from_raw_key(asetn)
        schemaVal = parsing.arrayset_record_schema_db_val_from_raw_val(
            schema_hash=schemaSpec.schema_hash,
            schema_is_var=schemaSpec.schema_is_var,
            schema_max_shape=schemaSpec.schema_max_shape,
            schema_dtype=schemaSpec.schema_dtype,
            schema_is_named=schemaSpec.schema_is_named,
            schema_default_backend=schemaSpec.schema_default_backend)
        entries.append((schemaKey, schemaVal))

        dataRecs = patchedRecs['arraysets'][asetn]['data']
        numDataRecs = len(dataRecs.keys())
        numDataKey = parsing.arrayset_record_count_db_key_from_raw_key(asetn)
        numDataVal = parsing.arrayset_record_count_db_val_from_raw_val(numDataRecs)
        entries.append((numDataKey, numDataVal))

        for dataRawK, dataRawV in dataRecs.items():
            dataRecKey = parsing.data_record_db_key_from_raw_key(
                aset_name=dataRawK.aset_name,
                data_name=dataRawK.data_name)
            dataRecVal = parsing.data_record_db_val_from_raw_val(
                data_hash=dataRawV.data_hash)
            entries.append((dataRecKey, dataRecVal))

    numMetaKey = parsing.metadata_count_db_key()
    numMetaVal = parsing.metadata_count_db_val_from_raw_val(len(patchedRecs['metadata'].keys()))
    entries.append((numMetaKey, numMetaVal))

    for metaRecRawKey, metaRecRawVal in patchedRecs['metadata'].items():
        metaRecKey = parsing.metadata_record_db_key_from_raw_key(metaRecRawKey.meta_name)
        metaRecVal = parsing.metadata_record_db_val_from_raw_val(metaRecRawVal.meta_hash)
        entries.append((metaRecKey, metaRecVal))

    sortedEntries = sorted(entries, key=lambda x: x[0])
    return sortedEntries
