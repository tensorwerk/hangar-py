import logging

from .context import TxnRegister
from .diff import ThreeWayCommitDiffer, WriterUserDiff
from .records import commiting, hashs, heads, parsing

logger = logging.getLogger(__name__)

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
    writerDiffer = WriterUserDiff(
        stageenv=stageenv, branchenv=branchenv, refenv=refenv, branch_name=dev_branch_name)
    if writerDiffer.status != 'CLEAN':
        msg = 'HANGAR RUNTIME ERROR: Changes are currently pending in the staging area '\
              'To avoid mangled histories, the staging area must exist in a clean state '\
              'Please reset or commit any changes before the merge operation'
        e = RuntimeError(msg)
        logger.error(e, exc_info=False)
        raise e from None

    try:
        heads.acquire_writer_lock(branchenv=branchenv, writer_uuid='MERGE_PROCESS')
    except PermissionError as e:
        logger.error(e, exc_info=False)
        raise e from None

    try:
        dHEAD = heads.get_branch_head_commit(branchenv, dev_branch_name)
        branchHistory = writerDiffer._determine_ancestors(
            mHEAD=writerDiffer.commit_hash, dHEAD=dHEAD)

        if branchHistory.canFF is True:
            logger.info('Selected Fast-Forward Merge Stratagy')
            success = _fast_forward_merge(
                branchenv=branchenv,
                master_branch=master_branch_name,
                new_masterHEAD=branchHistory.devHEAD)
        else:
            logger.info('Selected 3-Way Merge Strategy')
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
        logger.error(e, exc_info=False)
        raise e from None
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
    a_cont = commiting.get_commit_ref_contents(refenv, ancestorHEAD)
    m_cont = commiting.get_commit_ref_contents(refenv, masterHEAD)
    d_cont = commiting.get_commit_ref_contents(refenv, devHEAD)

    mergeContents = _compute_merge_results(a_cont, m_cont, d_cont)
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
        cursor.close()
    finally:
        TxnRegister().commit_writer_txn(stageenv)


def _merge_changes(changes: dict, m_dict: dict) -> dict:
    '''Common class which can merge changes between two branches.

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
    '''
    m_added = changes['master']['additions']
    m_unchanged = changes['master']['unchanged']
    m_removed = changes['master']['removals']
    m_mutated = changes['master']['mutations']
    d_added = changes['dev']['additions']
    d_unchanged = changes['dev']['unchanged']
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
    '''Compute the diff of a 3-way merge and patch historical contents to get new state

    .. warning::

        This method is not robust, and will require a require in the future.

    Parameters
    ----------
    a_cont : dict
        contents of the lastest common ancestor of the `master` and `dev` HEAD commits.
    m_cont : dict
        contents of the `master` HEAD commit.
    d_cont : dict
        contents of the `dev` HEAD commit.

    Returns
    -------
    dict
        nested dict specifying datasets and metadata record specs of the new
        merge commit.
    '''
    # conflict checking
    cmtDiffer = ThreeWayCommitDiffer(a_cont, m_cont, d_cont)
    confs = cmtDiffer.determine_conflicts()
    if confs['conflict_found'] is True:
        msg = f'HANGAR VALUE ERROR:: Merge ABORTED with conflict: {confs}'
        raise ValueError(msg)

    # merging: dataset schemas
    m_schema_dict = {}
    for dsetn in m_cont['datasets']:
        m_schema_dict[dsetn] = m_cont['datasets'][dsetn]['schema']
    o_schema_dict = _merge_changes(cmtDiffer.dataset_changes(), m_schema_dict)

    # merging: dataset samples
    o_data_dict = {}
    sample_changes = cmtDiffer.sample_changes()
    for dsetn in o_schema_dict:
        if dsetn not in m_cont['datasets']:
            o_data_dict[dsetn] = d_cont['datasets'][dsetn]['data']
            continue
        else:
            m_dsetn_data_dict = m_cont['datasets'][dsetn]['data']

        dset_sample_changes = {
            'master': sample_changes['master'][dsetn],
            'dev': sample_changes['dev'][dsetn],
        }
        o_data_dict[dsetn] = _merge_changes(dset_sample_changes, m_dsetn_data_dict)

    # merging: metadata
    o_meta_dict = _merge_changes(cmtDiffer.meta_changes(), m_cont['metadata'])

    # collect all merge results into final data structure
    outDict = {}
    outDict['metadata'] = o_meta_dict
    outDict['datasets'] = {}
    for dsetn, dsetSchema in o_schema_dict.items():
        outDict['datasets'][dsetn] = {
            'schema': dsetSchema,
            'data': o_data_dict[dsetn]
        }
    return outDict


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
