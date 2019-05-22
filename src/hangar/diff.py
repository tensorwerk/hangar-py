import logging
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
            masterHEAD=mHEAD,
            devHEAD=dHEAD,
            ancestorHEAD=commonAncestor,
            canFF=canFF)
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


MetaRecord = namedtuple('MetaRecord', field_names=['meta_key', 'meta_hash'])

SamplesDataRecord = namedtuple(
    'SamplesDataRecord', field_names=['dset_name', 'data_name', 'data_hash'])

DatasetSchemaRecord = namedtuple('DatasetSchemaRecord',
                                 field_names=[
                                     'dset_name', 'schema_hash',
                                     'schema_dtype', 'schema_is_var',
                                     'schema_max_shape', 'schema_is_named'
                                 ])


class DifferBase(object):
    '''Low level class implementing methods common to all record differ objects

    Parameters
    ----------
    ancestor_data : dict
        key/value pairs making up records of the ancestor data
    dev_data : dict
        key/value pairs making up records of the dev data.
    '''

    def __init__(self, ancestor_data: dict, dev_data: dict):
        self.a_data = ancestor_data
        self.d_data = dev_data
        self.a_data_keys = set(self.a_data.keys())
        self.d_data_keys = set(self.d_data.keys())

        self.additions: set = None
        self.removals: set = None
        self.unchanged: set = None
        self.mutations: set = None

    def compute(self, mutation_finder_partial):
        '''perform the computation

        Parameters
        ----------
        mutation_finder_partial : func
            creates a nt descriptor to enable set operations on k/v record pairs.
        '''
        self.additions = self.d_data_keys.difference(self.a_data_keys)
        self.removals = self.a_data_keys.difference(self.d_data_keys)

        a_unchanged_kv, d_unchanged_kv = {}, {}
        potential_unchanged = self.a_data_keys.intersection(self.d_data_keys)
        for k in potential_unchanged:
            a_unchanged_kv[k] = self.a_data[k]
            d_unchanged_kv[k] = self.d_data[k]

        self.mutations = mutation_finder_partial(
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


class MetadataDiffer(DifferBase):
    '''Specifialized differ class for metadata records.

    Parameters
    ----------
        **kwargs:
            See args of :class:`DifferBase`
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.compute(self.meta_record_mutation_finder)

    @staticmethod
    def meta_record_mutation_finder(a_unchanged_kv, d_unchanged_kv):

        def meta_nt_func(record_dict: dict) -> set:
            records = set()
            for k, v in record_dict.items():
                records.add(MetaRecord(meta_key=k, meta_hash=v))
            return records

        arecords, drecords = meta_nt_func(a_unchanged_kv), meta_nt_func(d_unchanged_kv)
        mutations = set([m.meta_key for m in arecords.difference(drecords)])
        return mutations


# -------------------- Dataset Schemas Differ ---------------------------------


class DatasetDiffer(DifferBase):
    '''Differ class specifialized for dataset schemas

    Parameters
    ----------
    ancestor_data : dict
        object containing both `data` and `schemas` keys, from
        which only `schemas` will be used in diff
    dev_data : dict
        object containing both `data` and `schemas` keys, from
        which only `schemas` will be used in diff
    '''

    def __init__(self, ancestor_data, dev_data, *args, **kwargs):
        a_schemas = self._isolate_dset_schemas(ancestor_data)
        d_schemas = self._isolate_dset_schemas(dev_data)
        super().__init__(a_schemas, d_schemas, *args, **kwargs)
        self.compute(partial(self.schema_record_mutation_finder,
                             schema_nt_func=self.schema_record_dict_to_nt))

    @staticmethod
    def _isolate_dset_schemas(dataset_specs: dict) -> dict:
        schemas_dict = {}
        for k, v in dataset_specs.items():
            schemas_dict[k] = v['schema']
        return schemas_dict

    @staticmethod
    def schema_record_dict_to_nt(record_dict: dict) -> set:
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

    @staticmethod
    def schema_record_mutation_finder(schema_nt_func, a_unchanged_kv, d_unchanged_kv):
        arecords, drecords = schema_nt_func(a_unchanged_kv), schema_nt_func(d_unchanged_kv)
        mutations = set([m.dset_name for m in arecords.difference(drecords)])
        return mutations


# ---------------------- Sample Differ ----------------------------------------


class SampleDiffer(DifferBase):
    '''Specialized Differ class for dataset samples.

    Parameters
    ----------
        dset_name: str
            name of the dataset whose samples are being comapared.
        **kwargs:
            See args of :class:`DifferBase`
    '''
    def __init__(self, dset_name: str, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.dset_name = dset_name
        self.compute(self.samples_record_mutation_finder)

    @staticmethod
    def samples_record_mutation_finder(a_unchanged_kv, d_unchanged_kv):

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
ConflictRecords.__doc__ = 'Four types of conflicts (t1, t21, t22, t3) are defined and accessible through this object.'
ConflictRecords.t1.__doc__ = 'Additions in both master and dev setting same key to different values.'
ConflictRecords.t21.__doc__ = 'Removed key in master, mutated value in dev.'
ConflictRecords.t22.__doc__ = 'Removed key in dev, mutated value in master.'
ConflictRecords.t3.__doc__ = 'Mutated in both master and dev to non-matching values.'
ConflictRecords.conflict.__doc__ = 'Bool indicating if any type of conflict is present; included as convenience method.'


class ThreeWayCommitDiffer(object):

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

        self._run()

    def _run(self):

        self.meta_diff()
        self.dataset_diff()
        self.sample_diff()

# ----------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------

    def meta_diff(self):

        self.am_meta_diff = MetadataDiffer(
            ancestor_data=self.acont['metadata'], dev_data=self.mcont['metadata'])
        self.ad_meta_diff = MetadataDiffer(
            ancestor_data=self.acont['metadata'], dev_data=self.dcont['metadata'])

    def meta_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out, tempt1, tempt3 = {}, [], []

        # addition conflicts
        addition_keys = self.am_meta_diff.additions.intersection(self.ad_meta_diff.additions)
        for meta_key in addition_keys:
            m_hash = self.am_meta_diff.d_data[meta_key]
            d_hash = self.ad_meta_diff.d_data[meta_key]
            if m_hash != d_hash:
                tempt1.append(meta_key)
        out['t1'] = tempt1

        # removal conflicts
        out['t21'] = self.am_meta_diff.removals.intersection(self.ad_meta_diff.mutations)
        out['t22'] = self.ad_meta_diff.removals.intersection(self.am_meta_diff.mutations)

        # mutation conflicts
        mutation_keys = self.am_meta_diff.mutations.intersection(self.ad_meta_diff.mutations)
        for meta_key in mutation_keys:
            m_hash = self.am_meta_diff.d_data[meta_key]
            d_hash = self.ad_meta_diff.d_data[meta_key]
            if m_hash != d_hash:
                tempt3.append(meta_key)
        out['t3'] = tempt3

        for k in list(out.keys()):
            out[k] = tuple(out[k])
        out['conflict'] = any([bool(len(x)) for x in out.values()])
        res = ConflictRecords(**out)
        return res

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
            ancestor_data=self.acont['datasets'], dev_data=self.mcont['datasets'])
        self.ad_dset_diff = DatasetDiffer(
            ancestor_data=self.acont['datasets'], dev_data=self.dcont['datasets'])

    def dataset_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out, tempt1, tempt3 = {}, [], []

        # addition conflicts
        addition_keys = self.am_dset_diff.additions.intersection(self.ad_dset_diff.additions)
        for dsetn in addition_keys:
            m_srec = self.am_dset_diff.schema_record_dict_to_nt(self.am_dset_diff.d_data[dsetn])
            d_srec = self.ad_dset_diff.schema_record_dict_to_nt(self.ad_dset_diff.d_data[dsetn])
            if m_srec != d_srec:
                tempt1.append(dsetn)
        out['t1'] = tempt1

        # removal conflicts
        out['t21'] = self.am_dset_diff.removals.intersection(self.ad_dset_diff.mutations)
        out['t22'] = self.ad_dset_diff.removals.intersection(self.am_dset_diff.mutations)

        # mutation conflicts
        mutation_keys = self.am_dset_diff.mutations.intersection(self.ad_dset_diff.mutations)
        for dsetn in mutation_keys:
            m_srec = self.am_dset_diff.schema_record_dict_to_nt(self.am_dset_diff.d_data[dsetn])
            d_srec = self.ad_dset_diff.schema_record_dict_to_nt(self.ad_dset_diff.d_data[dsetn])
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
            'master': self.am_dset_diff.kv_diff_out(),
            'dev': self.ad_dset_diff.kv_diff_out(),
        }
        return out

    # ----------------------------------------------------------------
    # Samples
    # ----------------------------------------------------------------

    def sample_diff(self):

        # ------------ ancestor -> master changes --------------------

        m_dsets = self.am_dset_diff.unchanged.union(
            self.am_dset_diff.additions).union(self.am_dset_diff.mutations)
        for dset_name in m_dsets:
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            m_dset_data = self.mcont['datasets'][dset_name]['data']
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        for dset_name in self.am_dset_diff.removals:
            m_dset_data = {}
            a_dset_data = self.acont['datasets'][dset_name]['data']
            self.am_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=m_dset_data)

        # ------------ ancestor -> dev changes --------------------

        d_dsets = self.ad_dset_diff.unchanged.union(
            self.ad_dset_diff.additions).union(self.ad_dset_diff.mutations)
        for dset_name in d_dsets:
            if dset_name in self.acont['datasets']:
                a_dset_data = self.acont['datasets'][dset_name]['data']
            else:
                a_dset_data = {}
            d_dset_data = self.dcont['datasets'][dset_name]['data']
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

        for dset_name in self.ad_dset_diff.removals:
            d_dset_data = {}
            a_dset_data = self.acont['datasets'][dset_name]['data']
            self.ad_samp_diff[dset_name] = SampleDiffer(
                dset_name=dset_name, ancestor_data=a_dset_data, dev_data=d_dset_data)

    def sample_conflicts(self):
        '''
        # t1: added in master & dev with different values
        # t21: removed in master, mutated in dev
        # t22: removed in dev, mutated in master
        # t3: mutated in master & dev to different values
        '''
        out = {}
        all_dset_names = set(self.ad_samp_diff.keys()).union(set(self.am_samp_diff.keys()))
        for dsetn in all_dset_names:
            samp_conflicts_t1, samp_conflicts_t3 = [], []

            if dsetn in self.am_samp_diff:
                mdiff = self.am_samp_diff[dsetn]
            else:
                mdiff = SampleDiffer(dsetn, {}, {})
            if dsetn in self.ad_samp_diff:
                ddiff = self.ad_samp_diff[dsetn]
            else:
                ddiff = SampleDiffer(dsetn, {}, {})

            # addition conflicts
            addition_keys = mdiff.additions.intersection(ddiff.additions)
            for samp in addition_keys:
                m_rec = mdiff.d_data[samp]
                d_rec = ddiff.d_data[samp]
                if m_rec != d_rec:
                    samp_conflicts_t1.append(samp)

            # removal conflicts
            samp_conflicts_t21 = mdiff.removals.intersection(ddiff.mutations)
            samp_conflicts_t22 = ddiff.removals.intersection(mdiff.mutations)

            # mutation conflicts
            mutation_keys = mdiff.mutations.intersection(ddiff.mutations)
            for samp in mutation_keys:
                m_rec = mdiff.d_data[samp]
                d_rec = ddiff.d_data[samp]
                if m_rec != d_rec:
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
