from collections import namedtuple

import dictdiffer

from .context import TxnRegister
from .records import commiting
from .records import hashs
from .records import heads
from .records import parsing
from .records.queries import RecordQuery

# ------------------- historical analysis methods --------------------------------


HisoryDiffStruct = namedtuple(
    'HistoryDiffStruct', ['masterHEAD', 'devHEAD', 'ancestorHEAD', 'canFF'])


def _determine_ancestors(branchenv, refenv, merge_master, merge_dev):
    '''Search the commit history to determine the closest common ancestor.

    The closest common ancestor is important because it serves as the "merge
    base" in a 3-way merge stratagy. This is a very nieve implementation, but it
    works well enough right now for simple branch histories.

    Parameters
    ----------
    branchenv : lmdb.Environment
        environment where the branch refs are stored
    refenv : lmdb.Environment
        refenv to follow the commit history
    merge_master : str
        name of the master branch for the merge commit
    merge_dev : str
        name of the dev branch for the merge commit

    Returns
    -------
    namedtuple
        indicating the masterHEAD, devHEAD, ancestorHEAD, and canFF which
        tells if this is a fast-forwardable commit.
    '''
    mHEAD = heads.get_branch_head_commit(branchenv, merge_master)
    dHEAD = heads.get_branch_head_commit(branchenv, merge_dev)

    mAncestors = commiting.get_commit_ancestors_graph(refenv, mHEAD)
    dAncestors = commiting.get_commit_ancestors_graph(refenv, dHEAD)

    cAncestors = set(mAncestors.keys()).intersection(set(dAncestors.keys()))
    canFF = True if mHEAD in cAncestors else False

    ancestorOrder = []
    for ancestor in cAncestors:
        timeOfCommit = commiting.get_commit_spec(refenv, ancestor).commit_time
        ancestorOrder.append((ancestor, timeOfCommit))

    ancestorOrder.sort(key=lambda t: t[1], reverse=True)
    commonAncestor = ancestorOrder[0][0]

    res = HisoryDiffStruct(
        masterHEAD=mHEAD,
        devHEAD=dHEAD,
        ancestorHEAD=commonAncestor,
        canFF=canFF)
    return res


# ------------------- check the status of the staging area ---------------------


def staging_area_status(stageenv, refenv, branchenv):
    '''Determine if changes have been made in the staging area

    If the contents of the staging area and it's parent commit are the same, the
    status is said to be "CLEAN". If even one dataset or metadata record has
    changed however, the status is "DIRTY".

    Parameters
    ----------
    stageenv : lmdb.Environment
        staging area record environment
    refenv : lmdb.Environment
        commit ref record environment
    branchenv : lmdb.Environment
        commit branch record environment

    Returns
    -------
    str
        CLEAN" if no changes have been made, otherwise "DIRTY"

    '''
    head_branch = heads.get_staging_branch_head(branchenv)
    head_commit = heads.get_branch_head_commit(branchenv, head_branch)
    if head_commit == '':
        base_refs = ()
    else:
        base_refs = commiting.get_commit_ref(refenv, head_commit)

    stage_refs = tuple(RecordQuery(stageenv)._traverse_all_records())
    if base_refs != stage_refs:
        status = 'DIRTY'
    else:
        status = 'CLEAN'

    return status


# --------------------------- Diff Methods ------------------------------------


def diff_commits(refenv, masterHEAD, devHEAD):
    '''Return the diff of two commits

    Parameters
    ----------
    refenv : lmdb.Environment
        db where the commit references are stored
    masterHEAD : str
        commit hash of the master head
    devHEAD : str
        commit hash of the dev head

    Returns
    -------
    list
        list of add, changes, and removed data
    '''
    m_contents = commiting.get_commit_ref_contents(refenv, masterHEAD)
    d_contents = commiting.get_commit_ref_contents(refenv, devHEAD)

    ancestorToDevDiff = dictdiffer.diff(m_contents, d_contents)
    return list(ancestorToDevDiff)


def diff_staged_changes(refenv, stageenv, branchenv):
    '''Return the diff of the staging area contents to the staging base HEAD

    Parameters
    ----------
    refenv : lmdb.Environment
        db where the commit references are stored
    stageenv : lmdb.Environment
        db where all staging environment is stored
    branchenv : lmdb.Environment
        db where all branch reference data is stored

    Returns
    -------
    list
        list of add, changes, and removed data records.
    '''
    head_branch = heads.get_staging_branch_head(branchenv)
    head_commit = heads.get_branch_head_commit(branchenv, head_branch)
    if head_commit == '':
        base_refs = ()
    else:
        base_refs = commiting.get_commit_ref_contents(refenv, head_commit)

    stage_refs = RecordQuery(stageenv).all_records()
    stageToHeadDiff = dictdiffer.diff(base_refs, stage_refs)
    return list(stageToHeadDiff)


'''
Merge Methods
-------------

In the current implementation, only fast-forward and a very simple three-way merge
algorithm are implemented. All user facing API calls should be funnled through the
:function:`select_merge_algorithm` function

.. note::

    In the current implementation, it is not possible to stop a merge in progress or
    to revert a bad merge commit. All revert like operations should be made by
    creating new branchs from the last "good" state, after which new merge
    operations can be attempted (if desired.)
'''


def select_merge_algorithm(message, branchenv, stageenv, refenv,
                           stagehashenv,
                           master_branch_name, dev_branch_name,
                           repo_path):
    '''Entry point to perform a merge.

    Automatically selects algorithm and does the operation if no conflicts are
    found. This call requires that the staging area status be "CLEAN", if
    a "DIRTY" staging environment is found, an RuntimeError will be thrown.

    Parameters
    ----------
    message : str
        user message describing the commit
    branchenv : `lmdb.Envrionment`
        where the branch references are stored
    stageenv : `lmdb.Environment`
        where the staging area is open
    refenv : `lmdb.Evironment`
        where commit history is stored
    stagehashenv: `lmdb.Environment`
        where the stage hash environment data is stored
    master_branch_name : str
        name of the branch to serve as a merge master
    dev_branch_name : str
        name of the branch to use as the feature branch
    repo_path: str
        path to the repository on disk

    Raises
    ------
    RuntimeError
        if the staging area is not `CLEAN` of other changes
    PermissionError
        if the writer lock is currently held

    Returns
    -------
    str
        commit hash of the merge if this was a successful operation.
    '''
    # ensure the writer lock is held and that the staging area is in a 'CLEAN' state.
    stageStatus = staging_area_status(
        stageenv=stageenv,
        refenv=refenv,
        branchenv=branchenv)

    if stageStatus != 'CLEAN':
        raise RuntimeError(
            'ERROR: Changes are currently pending in the staging area. To prevent'
            'potential diverging histories, the staging area must exist in a clean state.'
            'Please stash or commit any pending changes before performing a merge'
            'operation.')

    lockHeld = heads.acquire_writer_lock(branchenv=branchenv, writer_uuid='MERGE_PROCESS')
    if lockHeld is False:
        raise PermissionError(
            'ERROR: The writer lock could not be acquired. Please close any pending write '
            'operations and retry the merge operation.')

    try:
        branchDiff = _determine_ancestors(
            branchenv=branchenv,
            refenv=refenv,
            merge_master=master_branch_name,
            merge_dev=dev_branch_name)

        if branchDiff.canFF is True:
            print('Selected Fast-Forward Merge Stratagy')
            success = _fast_forward_merge(
                branchenv=branchenv,
                master_branch=master_branch_name,
                new_masterHEAD=branchDiff.devHEAD)
        else:
            print('Selected 3-Way Merge Strategy')
            success = _three_way_merge(
                message=message,
                master_branch_name=master_branch_name,
                masterHEAD=branchDiff.masterHEAD,
                dev_branch_name=dev_branch_name,
                devHEAD=branchDiff.devHEAD,
                ancestorHEAD=branchDiff.ancestorHEAD,
                branchenv=branchenv,
                stageenv=stageenv,
                refenv=refenv,
                stagehashenv=stagehashenv,
                repo_path=repo_path)
    finally:
        heads.release_writer_lock(branchenv=branchenv, writer_uuid='MERGE_PROCESS')

    return success


# ------------------ Fast Forward Merge Methods -------------------------------

def _fast_forward_merge(branchenv, master_branch, new_masterHEAD):
    '''Update branch head pointer to perform a fast-forward merge.

    This method does not check that it is safe to do this operation, all
    verification should happen before this point is reached

    Parameters
    ----------
    branchenv : lmdb.Environment
        db with the branch head pointers
    master_branch : str
        name of the merge_master branch which should be updated
    new_masterHEAD : str
        commit hash to update the master_branch name to point to.

    Returns
    -------
    str
        if successful, returns the commit hash the master branch name was
        updated to.
    '''
    try:
        success = heads.set_branch_head_commit(
            branchenv=branchenv,
            branch_name=master_branch,
            commit_hash=new_masterHEAD)

    except ValueError as e:
        print(e)
        raise

    return success


# ----------------------- Three-Way Merge Methods -----------------------------


def _three_way_merge(message, master_branch_name, masterHEAD, dev_branch_name,
                     devHEAD, ancestorHEAD, branchenv, stageenv, refenv, stagehashenv,
                     repo_path):
    '''Merge stratagy with diff/patch computed from changes since last common ancestor.

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
    repo_path: str
        path to the repository on disk.

    Returns
    -------
    str
        commit hash of the new merge commit if the operation was successful.

    Notes
    -----

    The current implementation of the three-way merge essentially does the following:

    1)  Unpacks a struct containing records as the existed in `master`, `dev`
        and `ancestor` commits.
    2)  Create "diff" of changes from `ancestor` -> `dev`.
    3)  "Patch" `master` records with adds/removals/changes recorded in the "diff"
    4)  If no hard conflicts result, delete staging area, replace with contents of
        patch
    5)  Compute new commit hash, and commit staging area records
    '''

    mergeContents = _compute_merge_results(refenv, ancestorHEAD, masterHEAD, devHEAD)
    formatedContents = _merge_dict_to_lmdb_tuples(mergeContents)
    hashs.remove_unused_dataset_hdf_files(repo_path, stagehashenv)
    _overwrite_stageenv(stageenv, formatedContents)

    commit_hash = commiting.commit_records(
        message=message,
        branchenv=branchenv,
        stageenv=stageenv,
        refenv=refenv,
        repo_path=repo_path,
        is_merge_commit=True,
        merge_master=master_branch_name,
        merge_dev=dev_branch_name)

    hashs.clear_stage_hash_records(stagehashenv)
    return commit_hash


def _overwrite_stageenv(stageenv, sorted_tuple_output):
    '''Delete all records in a db and replace it with specified data.

    This method does not validate that it is safe to perform this operation, all
    checking needs to be performed before this point is reached

    Parameters
    ----------
    stageenv : lmdb.Enviornment
        staging area db to replace all data in.
    sorted_tuple_output : iterable of tuple
        iterable containing two-tuple of byte encoded record data to place in
        the stageenv db. index 0 -> db key; index 1 -> db val, it is assumed
        that the order of the tuples is lexigraphically sorted by index 0
        values, if not, this will result in unknown behavior.
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
            cursor.putmulti(sorted_tuple_output, append=True)
        try:
            cursor.close()
        except Exception:
            print('could not close cursor')
    finally:
        TxnRegister().commit_writer_txn(stageenv)


def _compute_merge_results(refenv, ancestorHEAD, masterHEAD, devHEAD):
    '''Compute the diff of a 3-way merge and patch historical contents to get new state

    .. warning::

        This method is not robust, and will require a require in the future.

    Parameters
    ----------
    refenv : lmdb.Environment
        db where the commit records are stored
    ancestorHEAD : str
        commit hash of the common ancestor of dev and master merge branch
    masterHEAD : str
        commit hash of the merge master branch head
    devHEAD : str
        commit hash of the merge dev branch head

    Returns
    -------
    dict
        nested dict specifying datasets and metadata record specs of the new
        merge commit.
    '''
    a_contents = commiting.get_commit_ref_contents(refenv, ancestorHEAD)
    m_contents = commiting.get_commit_ref_contents(refenv, masterHEAD)
    d_contents = commiting.get_commit_ref_contents(refenv, devHEAD)

    ancestorToDevDiff = dictdiffer.diff(a_contents, d_contents)
    patchedMergeContents = dictdiffer.patch(ancestorToDevDiff, m_contents)
    return patchedMergeContents


def _merge_dict_to_lmdb_tuples(patchedRecs):
    '''Create a lexographically sorted iterable of (key/val tuples) from a dict.

    .. note::

        This  method is currently designed to parse the output of
        :function:`~commiting.get_commit_ref_contents` and will break if that output
        format, or the output of :function:`_compute_merge_results` changes.

    Parameters
    ----------
    patchedRecs : dict
        nested dict which specifies all records for datasets & metdata

    Returns
    -------
    iterable
        iterable of tuples formated correctly to serve an a drop in replacement
        for the staging environment, with elements lexographically sorted so
        that an lmdb `putmulti` operation can be performed with `append=True`.
    '''
    entries = []
    numDsetsKey = parsing.dataset_total_count_db_key()
    numDsetsVal = parsing.dataset_total_count_db_val_from_raw_val(
        number_of_dsets=len(patchedRecs['datasets'].keys()))
    entries.append((numDsetsKey, numDsetsVal))

    for dsetn in patchedRecs['datasets'].keys():
        schemaSpec = patchedRecs['datasets'][dsetn]['schema']
        schemaKey = parsing.dataset_record_schema_db_key_from_raw_key(dsetn)
        schemaVal = parsing.dataset_record_schema_db_val_from_raw_val(
            schema_uuid=schemaSpec.schema_uuid,
            schema_hash=schemaSpec.schema_hash,
            schema_is_var=schemaSpec.schema_is_var,
            schema_max_shape=schemaSpec.schema_max_shape,
            schema_dtype=schemaSpec.schema_dtype,
            schema_is_named=schemaSpec.schema_is_named)
        entries.append((schemaKey, schemaVal))

        dataRecs = patchedRecs['datasets'][dsetn]['data']
        numDataRecs = len(dataRecs.keys())
        numDataKey = parsing.dataset_record_count_db_key_from_raw_key(dsetn)
        numDataVal = parsing.dataset_record_count_db_val_from_raw_val(numDataRecs)
        entries.append((numDataKey, numDataVal))

        for dataRawK, dataRawV in dataRecs.items():
            dataRecKey = parsing.data_record_db_key_from_raw_key(
                dset_name=dataRawK.dset_name,
                data_name=dataRawK.data_name)
            dataRecVal = parsing.data_record_db_val_from_raw_val(
                data_hash=dataRawV.data_hash)
            entries.append((dataRecKey, dataRecVal))

    numMetaKey = parsing.metadata_count_db_key()
    numMetaVal = parsing.metadata_count_db_val_from_raw_val(
        len(patchedRecs['metadata'].keys()))
    entries.append((numMetaKey, numMetaVal))

    for metaRecRawKey, metaRecRawVal in patchedRecs['metadata'].items():
        metaRecKey = parsing.metadata_record_db_key_from_raw_key(metaRecRawKey)
        metaRecVal = parsing.metadata_record_db_val_from_raw_val(metaRecRawVal)
        entries.append((metaRecKey, metaRecVal))

    sortedEntries = sorted(entries, key=lambda x: x[0])
    return sortedEntries
