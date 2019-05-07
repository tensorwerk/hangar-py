import logging
from collections import namedtuple
import copy

from .context import TxnRegister
from .records import commiting
from .records import hashs
from .records import heads
from .records import parsing
from .records.queries import RecordQuery

logger = logging.getLogger(__name__)

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


def diff_branches(branchenv, refenv, master_branch, dev_branch):
    '''Find diff of two branch heads formatted as if dev was to merge into master.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the branch head references are stored
    refenv : lmdb.Environment
        db where the commit references are stored
    master_branch : str
        name of the master branch
    dev_branch : str
        name of the dev branch

    Returns
    -------
    dict
        dict describing additions, removals, mutations, and no-change pieces
        for metadata, datasets, and samples in the diff.
    '''
    branchDiff = _determine_ancestors(
        branchenv=branchenv,
        refenv=refenv,
        merge_master=master_branch,
        merge_dev=dev_branch)

    a_cont = commiting.get_commit_ref_contents(refenv, branchDiff.ancestorHEAD)
    m_cont = commiting.get_commit_ref_contents(refenv, branchDiff.masterHEAD)
    d_cont = commiting.get_commit_ref_contents(refenv, branchDiff.devHEAD)

    ancestorToDevDiffer = CommitDiffer(a_cont, m_cont, d_cont)
    res = ancestorToDevDiffer.all_changes(include_master=False)
    return res


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

    ancestorToDevDiffer = CommitDiffer(m_contents, d_contents, m_contents)
    res = ancestorToDevDiffer.all_changes(include_dev=False)
    return res


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
    stageToHeadDiffer = CommitDiffer(base_refs, stage_refs, base_refs)
    res = stageToHeadDiffer.all_changes(include_dev=False)
    return res


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
    stgStatus = staging_area_status(stageenv=stageenv, refenv=refenv, branchenv=branchenv)
    if stgStatus != 'CLEAN':
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
        branchDiff = _determine_ancestors(
            branchenv=branchenv,
            refenv=refenv,
            merge_master=master_branch_name,
            merge_dev=dev_branch_name)

        if branchDiff.canFF is True:
            logger.info('Selected Fast-Forward Merge Stratagy')
            success = _fast_forward_merge(
                branchenv=branchenv,
                master_branch=master_branch_name,
                new_masterHEAD=branchDiff.devHEAD)
        else:
            logger.info('Selected 3-Way Merge Strategy')
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
        cursor.close()
    finally:
        TxnRegister().commit_writer_txn(stageenv)


def _evaluate_conflicts(differ) -> dict:
    '''Evaluate and collect all possible conflicts in a repo differ instance

    Parameters
    ----------
    differ : CommitDiffer
        instance initialized with branch commit contents.

    Returns
    -------
    dict
        containing conflict info in `dset`, `meta`, `sample` and `counflict_found`
        boolean field.
    '''

    dset_confs = differ.dataset_conflicts()
    meta_confs = differ.meta_conflicts()
    sample_confs = differ.sample_conflicts()

    conflictFound = False
    for dsetn, confval in sample_confs.items():
        if confval['conflict'] is True:
            conflictFound = True
    if (dset_confs['conflict'] is True) or (meta_confs['conflict'] is True):
        conflictFound = True

    confs = {
        'dset': dset_confs,
        'meta': meta_confs,
        'sample': sample_confs,
        'conflict_found': conflictFound,
    }
    return confs


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
    a_cont = commiting.get_commit_ref_contents(refenv, ancestorHEAD)
    m_cont = commiting.get_commit_ref_contents(refenv, masterHEAD)
    d_cont = commiting.get_commit_ref_contents(refenv, devHEAD)

    cmtDiffer = CommitDiffer(a_cont, m_cont, d_cont)
    confs = _evaluate_conflicts(cmtDiffer)
    if confs['conflict_found'] is True:
        msg = f'HANGAR VALUE ERROR:: Merge ABORTED with conflict: {confs}'
        raise ValueError(msg)

    all_dsets = cmtDiffer.dataset_changes()
    mas = all_dsets['additions']['master']  # master add dset
    das = all_dsets['additions']['dev']     # dev add dset
    mas.update(das)                   # combined
    mrs = all_dsets['removals']['master']   # master remove dset
    drs = all_dsets['removals']['dev']      # dev remove dset
    crs = mrs.union(drs)                    # combined
    mms = all_dsets['mutations']['master']  # master mutate dset
    dms = all_dsets['mutations']['dev']     # dev mutate dset
    mms.update(dms)                   # combined

    merge_cont = copy.deepcopy(a_cont)
    for dsetn in crs:
        merge_cont['datasets'].__delitem__(dsetn)
    for dsetn, schema in mms.items():
        merge_cont['datasets'].__delitem__(dsetn)
        merge_cont['datasets'][dsetn] = {'schema': schema, 'data': {}}
    for dsetn, schemaval in mas.items():
        merge_cont['datasets'][dsetn] = {'schema': schemaval, 'data': {}}

    for dsetn in cmtDiffer.sampdiff:
        for sampDif in cmtDiffer.sampdiff[dsetn].values():
            for remkey in sampDif.removals:
                try:
                    merge_cont['datasets'][dsetn]['data'].__delitem__(remkey)
                except KeyError:
                    pass
            for mutkey in sampDif.mutations:
                try:
                    merge_cont['datasets'][dsetn]['data'][mutkey] = sampDif.d_data[mutkey]
                except KeyError:
                    pass
            for addkey in sampDif.additions:
                try:
                    merge_cont['datasets'][dsetn]['data'][addkey] = sampDif.d_data[addkey]
                except KeyError:
                    pass

    metadif = cmtDiffer.meta_changes()
    metaremovals = metadif['removals']
    metamutations = metadif['mutations']
    metaadditions = metadif['additions']

    for metaspec in metaremovals.values():
        for k in metaspec:
            try:
                merge_cont['metadata'].__delitem__(k)
            except KeyError:
                pass
    for metaspec in metamutations.values():
        for k, v in metaspec.items():
            merge_cont['metadata'][k] = v
    for metaspec in metaadditions.values():
        for k, v in metaspec.items():
            merge_cont['metadata'][k] = v

    return merge_cont


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


# -----------------------------------------------------------------------------


MetaRecord = namedtuple('MetaRecord', field_names=['meta_key', 'meta_hash'])

SamplesDataRecord = namedtuple(
    'SamplesDataRecord', field_names=['dset_name', 'data_name', 'data_hash'])

DatasetSchemaRecord = namedtuple('DatasetSchemaRecord',
                                 field_names=[
                                     'dset_name', 'schema_hash',
                                     'schema_dtype', 'schema_is_var',
                                     'schema_max_shape', 'schema_is_named'
                                 ])


def meta_record_dict_to_nt(record_dict: dict) -> set:
    records = set()
    for k, v in record_dict.items():
        records.add(MetaRecord(meta_key=k, meta_hash=v))
    return records


class MetadataDiffer(object):

    def __init__(self, ancestor_meta: dict, dev_meta: dict):

        self.a_meta = ancestor_meta
        self.d_meta = dev_meta
        self.a_meta_keys = set(ancestor_meta.keys())
        self.d_meta_keys = set(dev_meta.keys())

        self.additions = None
        self.removals = None
        self.unchanged = None
        self.mutations = None

        self.run()

    def run(self):
        self.additions = self.d_meta_keys.difference(self.a_meta_keys)
        self.removals = self.a_meta_keys.difference(self.d_meta_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = self.a_meta_keys.intersection(self.d_meta_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_meta[k]
            d_unchanged_kv[k] = self.d_meta[k]
        arecords = meta_record_dict_to_nt(a_unchanged_kv)
        drecords = meta_record_dict_to_nt(d_unchanged_kv)

        self.mutations = set([m.meta_key for m in arecords.difference(drecords)])
        self.unchanged = potential_unchanged.difference(self.mutations)

    def diff_out(self):
        out = {
            'additions': self.additions,
            'removals': self.removals,
            'mutations': self.mutations,
            'unchanged': self.unchanged,
        }
        return out

    def kv_diff_out(self):
        out = {
            'additions': {k: self.d_meta[k] for k in self.additions},
            'removals': {k: self.a_meta[k] for k in self.removals},
            'mutations': {k: self.d_meta[k] for k in self.mutations},
            'unchanged': {k: self.a_meta[k] for k in self.unchanged},
        }
        return out


# ----------------------


def samples_record_dict_to_nt(record_dict: dict) -> set:
    records = set()
    for k, v in record_dict.items():
        rec = SamplesDataRecord(
            dset_name=k.dset_name,
            data_name=k.data_name,
            data_hash=v.data_hash
        )
        records.add(rec)
    return records


class SampleDiffer(object):

    def __init__(self, dset_name: str, ancestor_data: dict, dev_data: dict):

        self.dset_name = dset_name
        self.a_data = ancestor_data
        self.d_data = dev_data

        self.additions = None
        self.removals = None
        self.unchanged = None
        self.mutations = None

        self.run()

    def run(self):
        a_keys = set(self.a_data.keys())
        d_keys = set(self.d_data.keys())
        self.additions = d_keys.difference(a_keys)
        self.removals = a_keys.difference(d_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = a_keys.intersection(d_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_data[k]
            d_unchanged_kv[k] = self.d_data[k]
        arecords = samples_record_dict_to_nt(a_unchanged_kv)
        drecords = samples_record_dict_to_nt(d_unchanged_kv)
        muts = arecords.difference(drecords)

        self.mutations = set()
        for m in muts:
            rec = parsing.RawDataRecordKey(dset_name=m.dset_name, data_name=m.data_name)
            self.mutations.add(rec)
        self.unchanged = potential_unchanged.difference(self.mutations)

    def diff_out(self):
        out = {
            'additions': self.additions,
            'removals': self.removals,
            'mutations': self.mutations,
            'unchanged': self.unchanged,
        }
        return out

    def kv_diff_out(self):
        out = {
            'additions': {k: self.d_data[k] for k in self.additions},
            'removals': {k: self.a_data[k] for k in self.removals},
            'mutations': {k: self.d_data[k] for k in self.mutations},
            'unchanged': {k: self.a_data[k] for k in self.unchanged},
        }
        return out

# -------------------------------


def schema_record_dict_to_nt(record_dict: dict) -> set:
    records = set()
    for k, v in record_dict.items():
        rec = DatasetSchemaRecord(
            dset_name=k,
            schema_hash=v.schema_hash,
            schema_dtype=v.schema_dtype,
            schema_is_var=v.schema_is_var,
            schema_max_shape=tuple(v.schema_max_shape),
            schema_is_named=v.schema_is_named
        )
        records.add(rec)
    return records


class DatasetDiffer(object):

    def __init__(self, ancestor_dsets: dict, dev_dsets: dict):

        self.a_dssch = self._isolate_dset_schemas(ancestor_dsets)
        self.d_dssch = self._isolate_dset_schemas(dev_dsets)

        self.additions = None
        self.removals = None
        self.unchanged = None
        self.mutations = None

        self.run()

    def _isolate_dset_schemas(self, dataset_specs: dict) -> dict:
        schemas_dict = {}
        for k, v in dataset_specs.items():
            schemas_dict[k] = v['schema']
        return schemas_dict

    def run(self):
        a_keys = set(self.a_dssch.keys())
        d_keys = set(self.d_dssch.keys())
        self.additions = d_keys.difference(a_keys)
        self.removals = a_keys.difference(d_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = a_keys.intersection(d_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_dssch[k]
            d_unchanged_kv[k] = self.d_dssch[k]
        arecords = schema_record_dict_to_nt(a_unchanged_kv)
        drecords = schema_record_dict_to_nt(d_unchanged_kv)
        self.mutations = set([m.dset_name for m in arecords.difference(drecords)])

        self.unchanged = potential_unchanged.difference(self.mutations)

    def diff_out(self):
        out = {
            'additions': self.additions,
            'removals': self.removals,
            'mutations': self.mutations,
            'unchanged': self.unchanged,
        }
        return out

    def kv_diff_out(self):
        out = {
            'additions': {k: self.d_dssch[k] for k in self.additions},
            'removals': {k: self.a_dssch[k] for k in self.removals},
            'mutations': {k: self.d_dssch[k] for k in self.mutations},
            'unchanged': {k: self.a_dssch[k] for k in self.unchanged},
        }
        return out


class CommitDiffer(object):

    def __init__(self, ancestor_contents: dict, master_contents: dict, dev_contents: dict):

        self.acont = ancestor_contents  # ancestor contents
        self.mcont = master_contents    # master contents
        self.dcont = dev_contents       # dev contents

        self.am_dset_diff: DatasetDiffer = None   # ancestor -> master dset diff
        self.ad_dset_diff: DatasetDiffer = None   # ancestor -> dev dset diff
        self.am_meta_diff: MetadataDiffer = None  # ancestor -> master metadata diff
        self.ad_meta_diff: MetadataDiffer = None  # ancestor -> dev metadata diff
        self.am_samp_diff = {}
        self.ad_samp_diff = {}

        self.run()

    def run(self):

        self.meta_diff()
        self.dataset_diff()
        self.sample_diff()

# ----------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------

    def meta_diff(self):

        self.am_meta_diff = MetadataDiffer(
            ancestor_meta=self.acont['metadata'],
            dev_meta=self.mcont['metadata'])
        self.ad_meta_diff = MetadataDiffer(
            ancestor_meta=self.acont['metadata'],
            dev_meta=self.dcont['metadata'])

    def meta_conflicts(self):

        # addition conflicts
        meta_conflicts_t1 = []  # added in master & dev with different values
        addition_keys = self.am_meta_diff.additions.intersection(self.ad_meta_diff.additions)
        for meta_key in addition_keys:
            m_hash = self.am_meta_diff.d_meta[meta_key]
            d_hash = self.ad_meta_diff.d_meta[meta_key]
            if m_hash != d_hash:
                meta_conflicts_t1.append(meta_key)

        # removal conflicts
        meta_conflicts_t21 = []  # removed in master, mutated in dev
        meta_conflicts_t22 = []  # removed in dev, mutated in master

        am_removal_keys = self.am_meta_diff.removals.intersection(self.ad_meta_diff.mutations)
        meta_conflicts_t21.extend(am_removal_keys)

        ad_removal_keys = self.ad_meta_diff.removals.intersection(self.am_meta_diff.mutations)
        meta_conflicts_t22.extend(ad_removal_keys)

        # mutation conflicts
        meta_conflicts_t311 = []  # mutated in master & dev to different values
        meta_conflicts_t312 = []  # mutated in master, removed in dev
        meta_conflicts_t322 = []  # mutated in dev, removed in master
        for meta_key in self.am_meta_diff.mutations:
            if meta_key in self.ad_meta_diff.mutations:
                m_hash = self.am_meta_diff.d_meta[meta_key]
                d_hash = self.ad_meta_diff.d_meta[meta_key]
                if m_hash != d_hash:
                    meta_conflicts_t311.append(meta_key)
            elif meta_key in self.ad_meta_diff.removals:
                meta_conflicts_t312.append(meta_key)

        for meta_key in self.ad_meta_diff.mutations:
            if meta_key in self.am_meta_diff.removals:
                meta_conflicts_t322.append(meta_key)

        out = {
            't1': meta_conflicts_t1,
            't21': meta_conflicts_t21,
            't22': meta_conflicts_t22,
            't311': meta_conflicts_t311,
            't312': meta_conflicts_t312,
            't322': meta_conflicts_t322
        }
        conflictFound = False
        for v in out.values():
            if len(v) != 0:
                conflictFound = True
                break
        out['conflict'] = conflictFound
        return out

    def meta_changes(self):
        out = {
            'master': self.am_meta_diff.kv_diff_out(),
            'dev': self.ad_meta_diff.kv_diff_out(),
        }
        return out

    # ----------------------------------------------------------------
    # Datasets
    # ----------------------------------------------------------------

    def dataset_diff(self):

        self.am_dset_diff = DatasetDiffer(
            ancestor_dsets=self.acont['datasets'],
            dev_dsets=self.mcont['datasets'])

        self.ad_dset_diff = DatasetDiffer(
            ancestor_dsets=self.acont['datasets'],
            dev_dsets=self.dcont['datasets'])

    def dataset_conflicts(self):

        # addition conflicts
        dset_conflicts_t1 = []  # added in master & dev with different values
        for dsetn in self.am_dset_diff.additions:
            if dsetn in self.ad_dset_diff.additions:
                m_srec = schema_record_dict_to_nt(self.am_dset_diff.d_dssch[dsetn])
                d_srec = schema_record_dict_to_nt(self.ad_dset_diff.d_dssch[dsetn])
                if m_srec != d_srec:
                    dset_conflicts_t1.append(dsetn)

        # removal conflicts
        dset_conflicts_t21 = []  # removed in master, mutated in dev
        dset_conflicts_t22 = []  # removed in dev, mutated in master
        for dsetn in self.am_dset_diff.removals:
            if dsetn in self.ad_dset_diff.mutations:
                dset_conflicts_t21.append(dsetn)
        for dsetn in self.ad_dset_diff.removals:
            if dsetn in self.am_dset_diff.mutations:
                dset_conflicts_t22.append(dsetn)

        # mutation conflicts
        dset_conflicts_t311 = []  # mutated in master & dev to different values
        dset_conflicts_t312 = []  # mutated in master, removed in dev
        dset_conflicts_t322 = []  # mutated in dev, removed in master
        for dsetn in self.am_dset_diff.mutations:
            if dsetn in self.ad_dset_diff.mutations:
                m_srec = schema_record_dict_to_nt(self.am_dset_diff.d_dssch[dsetn])
                d_srec = schema_record_dict_to_nt(self.ad_dset_diff.d_dssch[dsetn])
                if m_srec != d_srec:
                    dset_conflicts_t311.append(dsetn)
            elif dsetn in self.ad_dset_diff.removals:
                dset_conflicts_t312.append(dsetn)

        for dsetn in self.ad_dset_diff.mutations:
            if dsetn in self.am_dset_diff.removals:
                dset_conflicts_t322.append(dsetn)

        out = {
            't1': dset_conflicts_t1,
            't21': dset_conflicts_t21,
            't22': dset_conflicts_t22,
            't311': dset_conflicts_t311,
            't312': dset_conflicts_t312,
            't322': dset_conflicts_t322
        }
        conflictFound = False
        for v in out.values():
            if len(v) != 0:
                conflictFound = True
                break
        out['conflict'] = conflictFound
        return out

    def dataset_changes(self):
        out = {
            'master': self.am_dset_diff.kv_diff_out(),
            'dev': self.ad_dset_diff.kv_diff_out(),
        }
        return out

    # ----------------------------------------------------------------
    # Samples
    # ----------------------------------------------------------------

    def sample_diff(self):

        # ------------ ancestor -> master changes --------------------

        m_dsets = self.am_dset_diff.unchanged.union(self.am_dset_diff.additions).union(self.am_dset_diff.mutations)
        for dset_name in m_dsets:
            m_dset_data = self.mcont['datasets'][dset_name]['data']
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        for dset_name in self.am_dset_diff.removals:
            a_dset_data = self.acont['datasets'][dset_name]['data']
            m_dset_data = {}
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        # ------------ ancestor -> dev changes --------------------

        d_dsets = self.ad_dset_diff.unchanged.union(self.ad_dset_diff.additions).union(self.ad_dset_diff.mutations)
        for dset_name in d_dsets:
            d_dset_data = self.dcont['datasets'][dset_name]['data']
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

        for dset_name in self.ad_dset_diff.removals:
            a_dset_data = self.acont['datasets'][dset_name]['data']
            d_dset_data = {}
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

    def sample_conflicts(self):

        out = {}
        all_dset_names = set(self.ad_samp_diff.keys()).union(set(self.am_samp_diff.keys()))
        for dsetn in all_dset_names:
            if dsetn in self.am_samp_diff:
                mdiff = self.am_samp_diff[dsetn]
            else:
                mdiff = SampleDiffer(dsetn, {}, {})

            if dsetn in self.ad_samp_diff:
                ddiff = self.ad_samp_diff[dsetn]
            else:
                ddiff = SampleDiffer(dsetn, {}, {})

            # addition conflicts
            samp_conflicts_t1 = []  # added in master & dev with different values
            for samp in mdiff.additions:
                if samp in ddiff.additions:
                    m_rec = mdiff.d_data[samp]
                    d_rec = ddiff.d_data[samp]
                    if m_rec != d_rec:
                        samp_conflicts_t1.append(samp)

            # removal conflicts
            samp_conflicts_t21 = []  # removed in master, mutated in dev
            samp_conflicts_t22 = []  # removed in dev, mutated in master
            for samp in mdiff.removals:
                if samp in ddiff.mutations:
                    samp_conflicts_t21.append(samp)
            for samp in ddiff.removals:
                if samp in mdiff.mutations:
                    samp_conflicts_t22.append(samp)

            # mutation conflicts
            samp_conflicts_t311 = []  # mutated in master & dev to different values
            samp_conflicts_t312 = []  # mutated in master, removed in dev
            samp_conflicts_t322 = []  # mutated in dev, removed in master
            for samp in mdiff.mutations:
                if samp in ddiff.mutations:
                    m_rec = mdiff.d_data[samp]
                    d_rec = ddiff.d_data[samp]
                    if m_rec != d_rec:
                        samp_conflicts_t311.append(samp)
                elif samp in ddiff.removals:
                    samp_conflicts_t312.append(samp)

            for samp in ddiff.mutations:
                if samp in mdiff.removals:
                    samp_conflicts_t322.append(samp)

            out[dsetn] = {
                't1': samp_conflicts_t1,
                't21': samp_conflicts_t21,
                't22': samp_conflicts_t22,
                't311': samp_conflicts_t311,
                't312': samp_conflicts_t312,
                't322': samp_conflicts_t322
            }
            conflictFound = False
            for v in out[dsetn].values():
                if len(v) != 0:
                    conflictFound = True
                    break
            out[dsetn]['conflict'] = conflictFound

        return out

    def sample_changes(self):
        out = {
            'master': {dsetn: self.am_samp_diff[dsetn].kv_diff_out() for dsetn in self.am_samp_diff},
            'dev': {dsetn: self.ad_samp_diff[dsetn].kv_diff_out() for dsetn in self.ad_samp_diff},
        }
        return out

    def all_changes(self, include_master: bool = True, include_dev: bool = True) -> dict:

        meta = self.meta_changes()
        dsets = self.dataset_changes()
        samples = self.sample_changes()

        if not include_master:
            meta.__delitem__('master')
            dsets.__delitem__('master')
            samples.__delitem__('master')
        elif not include_dev:
            meta.__delitem__('dev')
            dsets.__delitem__('dev')
            samples.__delitem__('dev')

        res = {
            'metadata': meta,
            'datasets': dsets,
            'samples': samples,
        }
        return res
