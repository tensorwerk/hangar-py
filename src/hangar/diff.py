import logging
from typing import MutableMapping
from collections import namedtuple
from functools import partial

from .records import commiting, heads, parsing
from .records.queries import RecordQuery

logger = logging.getLogger(__name__)

HistoryDiffStruct = namedtuple(
    'HistoryDiffStruct', ['masterHEAD', 'devHEAD', 'ancestorHEAD', 'canFF'])


class BaseUserDiff(object):

    def __init__(self, branchenv, refenv, *args, **kwargs):

        self._branchenv = branchenv
        self._refenv = refenv

    def _determine_ancestors(self, mHEAD: str, dHEAD: str) -> HistoryDiffStruct:
        '''Search the commit history to determine the closest common ancestor.

        The closest common ancestor is important because it serves as the "merge
        base" in a 3-way merge stratagy. This is a very nieve implementation, but it
        works well enough right now for simple branch histories.

        Parameters
        ----------
        mHEAD : str
            full commit hash to use as the `master` branch head commit
        dHEAD : str
            full commit hash to use as the `dev` branch head commit

        Returns
        -------
        namedtuple
            indicating the masterHEAD, devHEAD, ancestorHEAD, and canFF which
            tells if this is a fast-forwardable commit.
        '''

        mAncestors = commiting.get_commit_ancestors_graph(self._refenv, mHEAD)
        dAncestors = commiting.get_commit_ancestors_graph(self._refenv, dHEAD)
        cAncestors = set(mAncestors.keys()).intersection(set(dAncestors.keys()))
        canFF = True if mHEAD in cAncestors else False

        ancestorOrder = []
        for ancestor in cAncestors:
            timeOfCommit = commiting.get_commit_spec(self._refenv, ancestor).commit_time
            ancestorOrder.append((ancestor, timeOfCommit))

        ancestorOrder.sort(key=lambda t: t[1], reverse=True)
        commonAncestor = ancestorOrder[0][0]
        res = HistoryDiffStruct(
            masterHEAD=mHEAD, devHEAD=dHEAD, ancestorHEAD=commonAncestor, canFF=canFF)
        return res

    def _diff_three_way_cmts(self, a_cont: dict, m_cont: dict, d_cont: dict) -> tuple:
        ancestorToDevDiffer = ThreeWayCommitDiffer(a_cont, m_cont, d_cont)
        res = ancestorToDevDiffer.all_changes()
        conflicts = ancestorToDevDiffer.determine_conflicts()
        return (res, conflicts)

    def _diff_ff_cmts(self, m_cont: dict, d_cont: dict) -> tuple:
        ancestorToDevDiffer = ThreeWayCommitDiffer(m_cont, d_cont, m_cont)
        res = ancestorToDevDiffer.all_changes(include_dev=False)
        conflicts = ancestorToDevDiffer.determine_conflicts()
        return (res, conflicts)


class ReaderUserDiff(BaseUserDiff):

    def __init__(self, commit_hash, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._commit_hash = commit_hash

    @property
    def commit_hash(self):
        '''Return the commit_hash used for the reader checkout. Read-only attribute.

        Unlike the analogous method on `write-enabled` checkouts, the reader
        version cannot move across time, and is a fixed value set at checkout
        time.

        Returns
        -------
        string
            commit hash of the checkout references.
        '''
        return self._commit_hash

    def commit(self, dev_commit_hash: str) -> tuple:
        '''Compute changes and conflicts for diff between HEAD and a commit hash.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comarison.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        '''
        if not commiting.check_commit_hash_in_history(self._refenv, dev_commit_hash):
            msg = f'HANGAR VALUE ERROR: dev_commit_hash: {dev_commit_hash} does not exist'
            raise ValueError(msg)

        hist = self._determine_ancestors(self._commit_hash, dev_commit_hash)
        m_cont = commiting.get_commit_ref_contents(self._refenv, hist.masterHEAD)
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(m_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, m_cont, d_cont)

    def branch(self, dev_branch_name: str) -> tuple:
        '''Compute changes and conflicts for diff between HEAD and a branch name.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            If the specified `dev_branch_name` does not exist.
        '''
        branchNames = heads.get_branch_names(self._branchenv)
        if dev_branch_name in branchNames:
            dHEAD = heads.get_branch_head_commit(self._branchenv, dev_branch_name)
        else:
            msg = f'HANGAR VALUE ERROR: dev_branch_name: {dev_branch_name} invalid branch name'
            raise ValueError(msg)

        hist = self._determine_ancestors(self._commit_hash, dHEAD)
        m_cont = commiting.get_commit_ref_contents(self._refenv, hist.masterHEAD)
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(m_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, m_cont, d_cont)


class WriterUserDiff(BaseUserDiff):

    def __init__(self, stageenv, branch_name, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._stageenv = stageenv
        self._branch_name = branch_name

    @property
    def commit_hash(self):
        '''Calculate the current base commit for the write-enabled checkout object

        Though read only, it's interesting to note that this is actually
        calculated on the fly in order to allow a write-enabled checkout to
        commit and move it's HEAD reference along without the value here being
        out of date.

        Returns
        -------
        string
            commit hash of the base commit used to set up the staging area
        '''
        commit_hash = heads.get_branch_head_commit(
            branchenv=self._branchenv, branch_name=self._branch_name)
        return commit_hash

    @property
    def branch_name(self):
        '''Branch name of the writer-checkout setting up this method. Read-only.

        Returns
        -------
        string
            branch name which the write-enabled checkout is committing on.
        '''
        return self._branch_name

    def status(self):
        '''Determine if changes have been made in the staging area

        If the contents of the staging area and it's parent commit are the same,
        the status is said to be "CLEAN". If even one dataset or metadata record
        has changed however, the status is "DIRTY".

        Returns
        -------
        str
            "CLEAN" if no changes have been made, otherwise "DIRTY"
        '''
        head_commit = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        if head_commit == '':
            base_refs = ()
        else:
            base_refs = commiting.get_commit_ref(self._refenv, head_commit)

        stage_refs = tuple(RecordQuery(self._stageenv)._traverse_all_records())
        status = 'DIRTY' if (base_refs != stage_refs) else 'CLEAN'
        return status

    def commit(self, dev_commit_hash: str) -> tuple:
        '''Compute changes and conflicts for diff between HEAD and a commit hash.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comarison.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        '''
        if not commiting.check_commit_hash_in_history(self._refenv, dev_commit_hash):
            msg = f'HANGAR VALUE ERROR: dev_commit_hash: {dev_commit_hash} does not exist'
            raise ValueError(msg)

        hist = self._determine_ancestors(self.commit_hash, dev_commit_hash)
        stage_cont = RecordQuery(self._stageenv).all_records()
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(stage_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, stage_cont, d_cont)

    def branch(self, dev_branch_name: str) -> tuple:
        '''Compute changes and conflicts for diff between HEAD and a branch name.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            If the specified `dev_branch_name` does not exist.
        '''
        branchNames = heads.get_branch_names(self._branchenv)
        if dev_branch_name in branchNames:
            dHEAD = heads.get_branch_head_commit(self._branchenv, dev_branch_name)
        else:
            msg = f'HANGAR VALUE ERROR: dev_branch_name: {dev_branch_name} invalid branch name'
            raise ValueError(msg)

        hist = self._determine_ancestors(self.commit_hash, dHEAD)
        stage_cont = RecordQuery(self._stageenv).all_records()
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(stage_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, stage_cont, d_cont)

    def staged(self):
        '''Return the diff of the staging area contents to the staging base HEAD

        Returns
        -------
        dict
            contains all `addition`, `mutation`, `removals` and `unchanged` datasets
            schema, samples, and metadata references in the current staging area.
        '''
        base_cont = commiting.get_commit_ref_contents(self._refenv, self.commit_hash)
        stage_cont = RecordQuery(self._stageenv).all_records()

        ancestorToDevDiffer = ThreeWayCommitDiffer(base_cont, stage_cont, base_cont)
        res = ancestorToDevDiffer.all_changes(include_dev=False)
        conflicts = ancestorToDevDiffer.determine_conflicts()
        return (res, conflicts)


# --------------------- Differ Primitives ------------------------------------


class DifferBase(object):
    '''Low level class implementing methods common to all record differ objects

    Parameters
    ----------
    mut_func : func
        creates a nt descriptor to enable set operations on k/v record pairs.
    '''

    def __init__(self, mut_func):

        self._mut_func = mut_func

        self._a_data: dict = {}
        self._d_data: dict = {}
        self._a_data_keys: set = set()
        self._d_data_keys: set = set()

        self.additions: set = set()
        self.removals: set = set()
        self.unchanged: set = set()
        self.mutations: set = set()

    @property
    def a_data(self):
        '''getter for the ancestor data dict
        '''
        return self._a_data

    @a_data.setter
    def a_data(self, value):
        '''setter for the ancestor data dict
        '''
        self._a_data = value
        self._a_data_keys = set(value.keys())

    @property
    def a_data_keys(self):
        '''Getter for the ancestor data keys, set when a_data is set
        '''
        return self._a_data_keys

    @property
    def d_data(self):
        '''getter for the dev data dict
        '''
        return self._d_data

    @d_data.setter
    def d_data(self, value):
        '''setter for the dev data dict
        '''
        self._d_data = value
        self._d_data_keys = set(value.keys())

    @property
    def d_data_keys(self):
        '''Getter for the dev data keys, set when d_data is set
        '''
        return self._d_data_keys

    def compute(self):
        '''perform the computation
        '''
        self.additions = self.d_data_keys.difference(self.a_data_keys)
        self.removals = self.a_data_keys.difference(self.d_data_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = self.a_data_keys.intersection(self.d_data_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_data[k]
            d_unchanged_kv[k] = self.d_data[k]

        self.mutations = self._mut_func(
            a_unchanged_kv=a_unchanged_kv,
            d_unchanged_kv=d_unchanged_kv)
        self.unchanged = potential_unchanged.difference(self.mutations)

    def kv_diff_out(self):
        '''summary of the changes between ancestor and dev data

        Returns
        -------
        dict
            dict listing all changes via the form: `additions`, `removals`,
            `mutations`, `unchanged`.
        '''
        out = {
            'additions': {k: self.d_data[k] for k in self.additions},
            'removals': {k: self.a_data[k] for k in self.removals},
            'mutations': {k: self.d_data[k] for k in self.mutations},
            'unchanged': {k: self.a_data[k] for k in self.unchanged},
        }
        return out


# -------------------------- Metadata Differ ----------------------------------


MetaRecord = namedtuple('MetaRecord', field_names=['meta_key', 'meta_hash'])


def _meta_mutation_finder(a_unchanged_kv: dict, d_unchanged_kv: dict) -> set:
    '''Determine mutated metadata records between an ancestor and dev commit

    Parameters
    ----------
    a_unchanged_kv : dict
        dict containing metadata names as keys and hash values as samples for
        the ancestor commit
    d_unchanged_kv : dict
        dict containing metadata names as keys and hash values as samples for
        the dev commit

    Returns
    -------
    set
        metadata names (keys in the input dicts) which changed hash value from
        ancestor to dev.
    '''
    def meta_nt_func(record_dict: dict) -> set:
        records = set()
        for k, v in record_dict.items():
            records.add(MetaRecord(meta_key=k, meta_hash=v))
        return records

    arecords, drecords = meta_nt_func(a_unchanged_kv), meta_nt_func(d_unchanged_kv)
    mutations = set([m.meta_key for m in arecords.difference(drecords)])
    return mutations

# -------------------- Dataset Schemas Differ ---------------------------------


DatasetSchemaRecord = namedtuple('DatasetSchemaRecord',
                                 field_names=[
                                     'dset_name', 'schema_hash',
                                     'schema_dtype', 'schema_is_var',
                                     'schema_max_shape', 'schema_is_named'
                                 ])


def _isolate_dset_schemas(dataset_specs: dict) -> dict:
    '''Isolate only the schema specification from a full dataset records dict

    Parameters
    ----------
    dataset_specs : dict
        dict containing both `schema` and `data` record specification for any
        number of datasets

    Returns
    -------
    dict
        containing keys for dataset names and values of the schema specification
    '''
    schemas_dict = {}
    for k, v in dataset_specs.items():
        schemas_dict[k] = v['schema']
    return schemas_dict


def _schema_dict_to_nt(record_dict: dict) -> set:
    '''Convert schema records specification dict into set of named tuples

    Parameters
    ----------
    record_dict : dict
        dict containing keys for dataset names and values as nested dicts of the
        schema specification

    Returns
    -------
    set
        of nametuples each recording a dataset schema specification
    '''
    records = set()
    for k, v in record_dict.items():
        rec = DatasetSchemaRecord(
            dset_name=k,
            schema_hash=v.schema_hash,
            schema_dtype=v.schema_dtype,
            schema_is_var=v.schema_is_var,
            schema_max_shape=tuple(v.schema_max_shape),
            schema_is_named=v.schema_is_named)
        records.add(rec)
    return records


def _schema_mutation_finder(sch_nt_func, a_unchanged_kv: dict,
                                   d_unchanged_kv: dict) -> set:
    '''Determine mutated dataset schemas betwwen an ancestor and dev commit

    Parameters
    ----------
    sch_nt_func : function
        function to be used to convert the schema specification dict into set of
        named tuples
    a_unchanged_kv : dict
        containing dataset names as keys and nested dictionary (specifying
        schema parameters) as the value for the ancestor commit
    d_unchanged_kv : dict
        containing dataset names as keys and nested dictionary (specifying
        schema parameters) as the value for the dev commit

    Returns
    -------
    set
        of mutated dataset names whose schemas mutated
    '''
    arecords, drecords = sch_nt_func(a_unchanged_kv), sch_nt_func(d_unchanged_kv)
    mutations = set([m.dset_name for m in arecords.difference(drecords)])
    return mutations

# ---------------------- Sample Differ ----------------------------------------


SamplesDataRecord = namedtuple(
    'SamplesDataRecord', field_names=['dset_name', 'data_name', 'data_hash'])


def _samples_mutation_finder(a_unchanged_kv: dict, d_unchanged_kv: dict) -> set:
    '''Determine mutated sample records between an ancestor and dev commit

    Parameters
    ----------
    a_unchanged_kv : dict
        of dset & sample names / hash values for ancestor commit
    d_unchanged_kv : dict
        of dset & sample names / hash values for dev commit

    Returns
    -------
    set
        of named tuples each specifying dset & sample name for mutated sample
        records
    '''

    def samp_nt_func(record_dict: dict) -> set:
        records = set()
        for k, v in record_dict.items():
            rec = SamplesDataRecord(
                dset_name=k.dset_name, data_name=k.data_name, data_hash=v.data_hash)
            records.add(rec)
        return records

    mutations = set()
    arecords, drecords = samp_nt_func(a_unchanged_kv), samp_nt_func(d_unchanged_kv)
    for m in arecords.difference(drecords):
        rec = parsing.RawDataRecordKey(dset_name=m.dset_name, data_name=m.data_name)
        mutations.add(rec)
    return mutations

# ------------------------- Commit Differ -------------------------------------


ConflictRecords = namedtuple(
    'ConflictRecords', field_names=['t1', 't21', 't22', 't3', 'conflict'])
ConflictRecords.__doc__ = 'Four types of conflicts are accessible through this object.'
ConflictRecords.t1.__doc__ = 'Addition of key in master AND dev with different values.'
ConflictRecords.t21.__doc__ = 'Removed key in master, mutated value in dev.'
ConflictRecords.t22.__doc__ = 'Removed key in dev, mutated value in master.'
ConflictRecords.t3.__doc__ = 'Mutated key in both master AND dev to different values.'
ConflictRecords.conflict.__doc__ = 'Bool indicating if any type of conflict is present.'


class ThreeWayCommitDiffer(object):

    def __init__(self, ancestor_contents: dict, master_contents: dict, dev_contents: dict):

        self.acont = ancestor_contents  # ancestor contents
        self.mcont = master_contents    # master contents
        self.dcont = dev_contents       # dev contents

        self.am_dsetD: DifferBase = None   # ancestor -> master dset diff
        self.ad_dsetD: DifferBase = None   # ancestor -> dev dset diff
        self.am_metaD: DifferBase = None  # ancestor -> master metadata diff
        self.ad_metaD: DifferBase = None  # ancestor -> dev metadata diff
        self.am_sampD: MutableMapping[str, DifferBase] = {}
        self.ad_sampD: MutableMapping[str, DifferBase] = {}

        self._meta_diff()
        self._dataset_diff()
        self._sample_diff()

    # -------------------- Metadata Diff / Conflicts --------------------------

    def _meta_diff(self):

        self.am_metaD = DifferBase(mut_func=_meta_mutation_finder)
        self.am_metaD.a_data = self.acont['metadata']
        self.am_metaD.d_data = self.mcont['metadata']
        self.am_metaD.compute()

        self.ad_metaD = DifferBase(mut_func=_meta_mutation_finder)
        self.ad_metaD.a_data = self.acont['metadata']
        self.ad_metaD.d_data = self.dcont['metadata']
        self.ad_metaD.compute()

    def meta_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out, tempt1, tempt3 = {}, [], []

        # addition conflicts
        addition_keys = self.am_metaD.additions.intersection(self.ad_metaD.additions)
        for meta_key in addition_keys:
            if self.am_metaD.d_data[meta_key] != self.ad_metaD.d_data[meta_key]:
                tempt1.append(meta_key)
        out['t1'] = tempt1

        # removal conflicts
        out['t21'] = self.am_metaD.removals.intersection(self.ad_metaD.mutations)
        out['t22'] = self.ad_metaD.removals.intersection(self.am_metaD.mutations)

        # mutation conflicts
        mutation_keys = self.am_metaD.mutations.intersection(self.ad_metaD.mutations)
        for meta_key in mutation_keys:
            if self.am_metaD.d_data[meta_key] != self.ad_metaD.d_data[meta_key]:
                tempt3.append(meta_key)
        out['t3'] = tempt3

        for k in list(out.keys()):
            out[k] = tuple(out[k])
        out['conflict'] = any([bool(len(x)) for x in out.values()])
        res = ConflictRecords(**out)
        return res

    def meta_changes(self):
        out = {
            'master': self.am_metaD.kv_diff_out(),
            'dev': self.ad_metaD.kv_diff_out(),
        }
        return out

    # -------------------- Dataset Diff / Conflicts ---------------------------

    def _dataset_diff(self):

        mutFunc = partial(_schema_mutation_finder, sch_nt_func=_schema_dict_to_nt)
        self.am_dsetD = DifferBase(mut_func=mutFunc)
        self.am_dsetD.a_data = _isolate_dset_schemas(self.acont['datasets'])
        self.am_dsetD.d_data = _isolate_dset_schemas(self.mcont['datasets'])
        self.am_dsetD.compute()

        self.ad_dsetD = DifferBase(mut_func=mutFunc)
        self.ad_dsetD.a_data = _isolate_dset_schemas(self.acont['datasets'])
        self.ad_dsetD.d_data = _isolate_dset_schemas(self.dcont['datasets'])
        self.ad_dsetD.compute()

    def dataset_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out, tempt1, tempt3 = {}, [], []

        # addition conflicts
        addition_keys = self.am_dsetD.additions.intersection(self.ad_dsetD.additions)
        for dsetn in addition_keys:
            m_srec = _schema_dict_to_nt(self.am_dsetD.d_data[dsetn])
            d_srec = _schema_dict_to_nt(self.ad_dsetD.d_data[dsetn])
            if m_srec != d_srec:
                tempt1.append(dsetn)
        out['t1'] = tempt1

        # removal conflicts
        out['t21'] = self.am_dsetD.removals.intersection(self.ad_dsetD.mutations)
        out['t22'] = self.ad_dsetD.removals.intersection(self.am_dsetD.mutations)

        # mutation conflicts
        mutation_keys = self.am_dsetD.mutations.intersection(self.ad_dsetD.mutations)
        for dsetn in mutation_keys:
            m_srec = _schema_dict_to_nt(self.am_dsetD.d_data[dsetn])
            d_srec = _schema_dict_to_nt(self.ad_dsetD.d_data[dsetn])
            if m_srec != d_srec:
                tempt3.append(dsetn)
        out['t3'] = tempt3

        for k in list(out.keys()):
            out[k] = tuple(out[k])
        out['conflict'] = any([bool(len(x)) for x in out.values()])
        res = ConflictRecords(**out)
        return res

    def dataset_changes(self):
        out = {
            'master': self.am_dsetD.kv_diff_out(),
            'dev': self.ad_dsetD.kv_diff_out(),
        }
        return out

    # -------------------- Sample Diff / Conflicts ----------------------------

    def _sample_diff(self):

        # ---------------- ancestor -> master changes -------------------------

        m_dsets = self.am_dsetD.unchanged.union(
            self.am_dsetD.additions).union(self.am_dsetD.mutations)
        for dset_name in m_dsets:
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.am_sampD[dset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.am_sampD[dset_name].a_data = a_dset_data
            self.am_sampD[dset_name].d_data = self.mcont['datasets'][dset_name]['data']
            self.am_sampD[dset_name].compute()

        for dset_name in self.am_dsetD.removals:
            self.am_sampD[dset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.am_sampD[dset_name].a_data = self.acont['datasets'][dset_name]['data']
            self.am_sampD[dset_name].d_data = {}
            self.am_sampD[dset_name].compute()

        # ---------------- ancestor -> dev changes ----------------------------

        d_dsets = self.ad_dsetD.unchanged.union(
            self.ad_dsetD.additions).union(self.ad_dsetD.mutations)
        for dset_name in d_dsets:
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            self.ad_sampD[dset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.ad_sampD[dset_name].a_data = a_dset_data
            self.ad_sampD[dset_name].d_data = self.dcont['datasets'][dset_name]['data']
            self.ad_sampD[dset_name].compute()

        for dset_name in self.ad_dsetD.removals:
            self.ad_sampD[dset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.ad_sampD[dset_name].a_data = self.acont['datasets'][dset_name]['data']
            self.ad_sampD[dset_name].d_data = {}
            self.ad_sampD[dset_name].compute()

    def sample_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out = {}
        all_dset_names = set(self.ad_sampD.keys()).union(set(self.am_sampD.keys()))
        for dsetn in all_dset_names:
            # When dataset IN `dev` OR `master` AND NOT IN ancestor OR `other`.
            try:
                mdiff = self.am_sampD[dsetn]
            except KeyError:
                mdiff = DifferBase(mut_func=_samples_mutation_finder)
            try:
                ddiff = self.ad_sampD[dsetn]
            except KeyError:
                ddiff = DifferBase(mut_func=_samples_mutation_finder)

            samp_conflicts_t1, samp_conflicts_t3 = [], []

            # addition conflicts
            addition_keys = mdiff.additions.intersection(ddiff.additions)
            for samp in addition_keys:
                if mdiff.d_data[samp] != ddiff.d_data[samp]:
                    samp_conflicts_t1.append(samp)

            # removal conflicts
            samp_conflicts_t21 = mdiff.removals.intersection(ddiff.mutations)
            samp_conflicts_t22 = ddiff.removals.intersection(mdiff.mutations)

            # mutation conflicts
            mutation_keys = mdiff.mutations.intersection(ddiff.mutations)
            for samp in mutation_keys:
                if mdiff.d_data[samp] != ddiff.d_data[samp]:
                    samp_conflicts_t3.append(samp)

            tempOut = {
                't1': tuple(samp_conflicts_t1),
                't21': tuple(samp_conflicts_t21),
                't22': tuple(samp_conflicts_t22),
                't3': tuple(samp_conflicts_t3)
            }
            for k in list(tempOut.keys()):
                tempOut[k] = tuple(tempOut[k])
            tempOut['conflict'] = any([bool(len(x)) for x in tempOut.values()])
            tout = ConflictRecords(**tempOut)
            out[dsetn] = tout

        return out

    def sample_changes(self):
        out = {
            'master': {dsetn: self.am_sampD[dsetn].kv_diff_out() for dsetn in self.am_sampD},
            'dev': {dsetn: self.ad_sampD[dsetn].kv_diff_out() for dsetn in self.ad_sampD},
        }
        return out

    # ----------------------------- Summary Methods ---------------------------

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

    def determine_conflicts(self):
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
        dset_confs = self.dataset_conflicts()
        meta_confs = self.meta_conflicts()
        sample_confs = self.sample_conflicts()

        conflictFound = any(
            [dset_confs.conflict, meta_confs.conflict,
             *[confval.conflict for confval in sample_confs.values()]])

        confs = {
            'dset': dset_confs,
            'meta': meta_confs,
            'sample': sample_confs,
            'conflict_found': conflictFound,
        }
        return confs
