from functools import partial
from typing import (
    Callable, Iterable, List, MutableMapping, NamedTuple, Set, Union)

import lmdb

from .records import commiting, heads
from .records.parsing import (MetadataRecordKey, MetadataRecordVal,
                              RawDataRecordKey, RawDataRecordVal,
                              RawArraysetSchemaVal)
from .records.queries import RecordQuery

HistoryDiffStruct = NamedTuple('HistoryDiffStruct', [('masterHEAD', str),
                                                     ('devHEAD', str),
                                                     ('ancestorHEAD', str),
                                                     ('canFF', bool)])


class BaseUserDiff(object):

    def __init__(self, branchenv: lmdb.Environment, refenv: lmdb.Environment, *args, **kwargs):

        self._branchenv: lmdb.Environment = branchenv
        self._refenv: lmdb.Environment = refenv

    def _determine_ancestors(self, mHEAD: str, dHEAD: str) -> HistoryDiffStruct:
        """Search the commit history to determine the closest common ancestor.

        The closest common ancestor is important because it serves as the "merge
        base" in a 3-way merge strategy. This is a very naive implementation, but it
        works well enough right now for simple branch histories.

        Parameters
        ----------
        mHEAD : str
            full commit hash to use as the `master` branch head commit
        dHEAD : str
            full commit hash to use as the `dev` branch head commit

        Returns
        -------
        HistoryDiffStruct
            indicating the masterHEAD, devHEAD, ancestorHEAD, and canFF which
            tells if this is a fast-forward-able commit.
        """

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
        ancestorToDevDiffer = ThreeWayCommitDiffer(m_cont, m_cont, d_cont)
        res = ancestorToDevDiffer.all_changes(include_master=True)
        conflicts = ancestorToDevDiffer.determine_conflicts()
        return (res, conflicts)


class ReaderUserDiff(BaseUserDiff):

    def __init__(self, commit_hash, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._commit_hash = commit_hash

    def commit(self, dev_commit_hash: str) -> tuple:
        """Compute changes and conflicts for diff between HEAD and a commit hash.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        """
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
        """Compute changes and conflicts for diff between HEAD and a branch name.

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
        """
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

    def __init__(self, stageenv: lmdb.Environment, branch_name: str, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._stageenv: lmdb.Environment = stageenv
        self._branch_name: str = branch_name

    def status(self) -> str:
        """Determine if changes have been made in the staging area

        If the contents of the staging area and it's parent commit are the same,
        the status is said to be "CLEAN". If even one arrayset or metadata record
        has changed however, the status is "DIRTY".

        Returns
        -------
        str
            "CLEAN" if no changes have been made, otherwise "DIRTY"
        """
        head_commit = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        if head_commit == '':
            base_refs = ()
        else:
            base_refs = commiting.get_commit_ref(self._refenv, head_commit)

        stage_refs = tuple(RecordQuery(self._stageenv)._traverse_all_records())
        status = 'DIRTY' if (base_refs != stage_refs) else 'CLEAN'
        return status

    def commit(self, dev_commit_hash: str) -> tuple:
        """Compute changes and conflicts for diff between HEAD and a commit hash.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        tuple
            two-tuple of changes, conflicts (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        """
        if not commiting.check_commit_hash_in_history(self._refenv, dev_commit_hash):
            msg = f'HANGAR VALUE ERROR: dev_commit_hash: {dev_commit_hash} does not exist'
            raise ValueError(msg)

        commit_hash = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        hist = self._determine_ancestors(commit_hash, dev_commit_hash)
        stage_cont = RecordQuery(self._stageenv).all_records()
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(stage_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, stage_cont, d_cont)

    def branch(self, dev_branch_name: str) -> tuple:
        """Compute changes and conflicts for diff between HEAD and a branch name.

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
        """
        branchNames = heads.get_branch_names(self._branchenv)
        if dev_branch_name in branchNames:
            dHEAD = heads.get_branch_head_commit(self._branchenv, dev_branch_name)
        else:
            msg = f'HANGAR VALUE ERROR: dev_branch_name: {dev_branch_name} invalid branch name'
            raise ValueError(msg)

        commit_hash = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        hist = self._determine_ancestors(commit_hash, dHEAD)
        stage_cont = RecordQuery(self._stageenv).all_records()
        d_cont = commiting.get_commit_ref_contents(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(stage_cont, d_cont)
        else:
            a_cont = commiting.get_commit_ref_contents(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_cont, stage_cont, d_cont)

    def staged(self):
        """Return the diff of the staging area contents to the staging base HEAD

        Returns
        -------
        dict
            contains all `addition`, `mutation`, `removals` and `unchanged` arraysets
            schema, samples, and metadata references in the current staging area.
        """
        commit_hash = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        base_cont = commiting.get_commit_ref_contents(self._refenv, commit_hash)
        stage_cont = RecordQuery(self._stageenv).all_records()

        ancestorToDevDiffer = ThreeWayCommitDiffer(base_cont, stage_cont, base_cont)
        res = ancestorToDevDiffer.all_changes(include_dev=False)
        conflicts = ancestorToDevDiffer.determine_conflicts()
        return (res, conflicts)


# --------------------- Differ Primitives ------------------------------------


class DifferBase(object):
    """Low level class implementing methods common to all record differ objects

    Parameters
    ----------
    mut_func : func
        creates a NT descriptor to enable set operations on k/v record pairs.
    """

    def __init__(self, mut_func: Callable[[dict, dict], set]):

        self._mut_func = mut_func

        self._a_data: dict = {}
        self._d_data: dict = {}
        self._a_data_keys: Set[str] = set()
        self._d_data_keys: Set[str] = set()

        self.additions: set = set()
        self.removals: set = set()
        self.unchanged: set = set()
        self.mutations: set = set()

    @property
    def a_data(self):
        """getter for the ancestor data dict
        """
        return self._a_data

    @a_data.setter
    def a_data(self, value):
        """setter for the ancestor data dict
        """
        self._a_data = value
        self._a_data_keys = set(value.keys())

    @property
    def a_data_keys(self):
        """Getter for the ancestor data keys, set when a_data is set
        """
        return self._a_data_keys

    @property
    def d_data(self):
        """getter for the dev data dict
        """
        return self._d_data

    @d_data.setter
    def d_data(self, value):
        """setter for the dev data dict
        """
        self._d_data = value
        self._d_data_keys = set(value.keys())

    @property
    def d_data_keys(self):
        """Getter for the dev data keys, set when d_data is set
        """
        return self._d_data_keys

    def compute(self):
        """perform the computation
        """
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
        """summary of the changes between ancestor and dev data

        Returns
        -------
        dict
            dict listing all changes via the form: `additions`, `removals`,
            `mutations`, `unchanged`.
        """
        out = {
            'additions': {k: self.d_data[k] for k in self.additions},
            'removals': {k: self.a_data[k] for k in self.removals},
            'mutations': {k: self.d_data[k] for k in self.mutations},
            'unchanged': {k: self.a_data[k] for k in self.unchanged},
        }
        return out


# -------------------------- Metadata Differ ----------------------------------


MetaRecord = NamedTuple('MetaRecord', [
    ('meta_key', Union[str, int]),
    ('meta_hash', str)
])
MetaRecordKV = MutableMapping[MetadataRecordKey, MetadataRecordVal]


def _meta_mutation_finder(a_unchanged_kv: MetaRecordKV,
                          d_unchanged_kv: MetaRecordKV
                          ) -> Set[MetadataRecordKey]:
    """Determine mutated metadata records between an ancestor and dev commit

    Parameters
    ----------
    a_unchanged_kv : MetaRecordKV
        dict containing metadata names as keys and hash values as samples for
        the ancestor commit
    d_unchanged_kv : MetaRecordKV
        dict containing metadata names as keys and hash values as samples for
        the dev commit

    Returns
    -------
    Set[MetadataRecordKey]
        metadata names (keys in the input dicts) which changed hash value from
        ancestor to dev.
    """
    def meta_nt_func(record_dict: MetaRecordKV) -> Set[MetaRecord]:
        records = set()
        for k, v in record_dict.items():
            records.add(MetaRecord(meta_key=k.meta_name, meta_hash=v.meta_hash))
        return records

    arecs = meta_nt_func(a_unchanged_kv)
    drecs = meta_nt_func(d_unchanged_kv)
    mutations = set([MetadataRecordKey(m.meta_key) for m in arecs.difference(drecs)])
    return mutations

# -------------------- Arrayset Schemas Differ ---------------------------------


ArraysetSchemaRecord = NamedTuple('ArraysetSchemaRecord', [
    ('aset_name', str),
    ('schema_hash', str),
    ('schema_dtype', int),
    ('schema_is_var', bool),
    ('schema_max_shape', tuple),
    ('schema_is_named', bool),
])
ArraysetSchemaKV = MutableMapping[str, RawArraysetSchemaVal]


def _isolate_aset_schemas(arrayset_specs: MutableMapping[str, dict]) -> ArraysetSchemaKV:
    """Isolate only the schema specification from a full arrayset records dict.

    Parameters
    ----------
    arrayset_specs : MutableMapping[str, dict]
        dict containing both arrayset names and keys of `schema` and `data`
        record specification for any number of arraysets

    Returns
    -------
    ArraysetSchemaKV
        containing keys for arrayset names and values of the schema specification
    """
    schemas_dict = {}
    for k, v in arrayset_specs.items():
        schemas_dict[k] = v['schema']
    return schemas_dict


def _schema_dict_to_nt(record_dict: MutableMapping[str, ArraysetSchemaRecord]) -> Set[ArraysetSchemaRecord]:
    """Convert schema records specification dict into set of named tuples

    Parameters
    ----------
    record_dict : dict
        dict containing keys for arrayset names and values as nested dicts of the
        schema specification

    Returns
    -------
    Set[ArraysetSchemaRecord]
        of nametuples each recording a arrayset schema specification
    """
    records = set()
    for k, v in record_dict.items():
        rec = ArraysetSchemaRecord(
            aset_name=k,
            schema_hash=v.schema_hash,
            schema_dtype=v.schema_dtype,
            schema_is_var=v.schema_is_var,
            schema_max_shape=tuple(v.schema_max_shape),
            schema_is_named=v.schema_is_named)
        records.add(rec)
    return records


def _schema_mutation_finder(sch_nt_func, a_unchanged_kv: dict,
                            d_unchanged_kv: dict) -> Set[str]:
    """Determine mutated arrayset schemas between an ancestor and dev commit

    Parameters
    ----------
    sch_nt_func : function
        function to be used to convert the schema specification dict into set of
        named tuples
    a_unchanged_kv : dict
        containing arrayset names as keys and nested dictionary (specifying
        schema parameters) as the value for the ancestor commit
    d_unchanged_kv : dict
        containing arrayset names as keys and nested dictionary (specifying
        schema parameters) as the value for the dev commit

    Returns
    -------
    Set[str]
        of mutated arrayset names whose schemas mutated
    """
    arecords = sch_nt_func(a_unchanged_kv)
    drecords = sch_nt_func(d_unchanged_kv)
    mutations = set([m.aset_name for m in arecords.difference(drecords)])
    return mutations

# ---------------------- Sample Differ ----------------------------------------


SamplesDataRecord = NamedTuple('SamplesDataRecord', [
    ('aset_name', str),
    ('data_name', Union[str, int]),
    ('data_hash', str),
])
SamplesDataKV = MutableMapping[RawDataRecordKey, RawDataRecordVal]


def _samples_mutation_finder(a_unchanged_kv: SamplesDataKV,
                             d_unchanged_kv: SamplesDataKV
                             ) -> Set[RawDataRecordKey]:
    """Determine mutated sample records between an ancestor and dev commit

    Parameters
    ----------
    a_unchanged_kv : SamplesDataKV
        of aset & sample names / hash values for ancestor commit
    d_unchanged_kv : SamplesDataKV
        of aset & sample names / hash values for dev commit

    Returns
    -------
    Set[RawDataRecordKey]
        of named tuples each specifying aset & sample name for mutated sample
        records
    """

    def samp_nt_func(record_dict: SamplesDataKV) -> Set[SamplesDataRecord]:
        records = set()
        for k, v in record_dict.items():
            rec = SamplesDataRecord(
                aset_name=k.aset_name, data_name=k.data_name, data_hash=v.data_hash)
            records.add(rec)
        return records

    mutations = set()
    arecords = samp_nt_func(a_unchanged_kv)
    drecords = samp_nt_func(d_unchanged_kv)
    for m in arecords.difference(drecords):
        rec = RawDataRecordKey(aset_name=m.aset_name, data_name=m.data_name)
        mutations.add(rec)
    return mutations

# ------------------------- Commit Differ -------------------------------------


ConflictKeys = Union[str, RawDataRecordKey, MetadataRecordKey]

ConflictRecords = NamedTuple('ConflictRecords',
                             [('t1', Iterable[ConflictKeys]),
                              ('t21', Iterable[ConflictKeys]),
                              ('t22', Iterable[ConflictKeys]),
                              ('t3', Iterable[ConflictKeys]),
                              ('conflict', bool)])
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

        self.am_asetD: DifferBase  # ancestor -> master aset diff
        self.ad_asetD: DifferBase  # ancestor -> dev aset diff
        self.am_metaD: DifferBase  # ancestor -> master metadata diff
        self.ad_metaD: DifferBase  # ancestor -> dev metadata diff

        self.am_sampD: MutableMapping[str, DifferBase] = {}
        self.ad_sampD: MutableMapping[str, DifferBase] = {}

        self._meta_diff()
        self._arrayset_diff()
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

    def meta_conflicts(self) -> ConflictRecords:
        """
        t1: added in master & dev with different values
        t21: removed in master, mutated in dev
        t22: removed in dev, mutated in master
        t3: mutated in master & dev to different values
        """
        out: MutableMapping[str, Union[bool, Iterable[ConflictKeys]]] = {}
        tempt1: List[ConflictKeys] = []
        tempt3: List[ConflictKeys] = []

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

    # -------------------- Arrayset Diff / Conflicts ---------------------------

    def _arrayset_diff(self):

        mutFunc = partial(_schema_mutation_finder, sch_nt_func=_schema_dict_to_nt)
        self.am_asetD = DifferBase(mut_func=mutFunc)
        self.am_asetD.a_data = _isolate_aset_schemas(self.acont['arraysets'])
        self.am_asetD.d_data = _isolate_aset_schemas(self.mcont['arraysets'])
        self.am_asetD.compute()

        self.ad_asetD = DifferBase(mut_func=mutFunc)
        self.ad_asetD.a_data = _isolate_aset_schemas(self.acont['arraysets'])
        self.ad_asetD.d_data = _isolate_aset_schemas(self.dcont['arraysets'])
        self.ad_asetD.compute()

    def arrayset_conflicts(self) -> ConflictRecords:
        """
        t1: added in master & dev with different values
        t21: removed in master, mutated in dev
        t22: removed in dev, mutated in master
        t3: mutated in master & dev to different values
        """
        out: MutableMapping[str, Union[bool, Iterable[ConflictKeys]]] = {}
        tempt1: List[ConflictKeys] = []
        tempt3: List[ConflictKeys] = []

        # addition conflicts
        addition_keys = self.am_asetD.additions.intersection(self.ad_asetD.additions)
        for asetn in addition_keys:
            m_srec = _schema_dict_to_nt({asetn: self.am_asetD.d_data[asetn]})
            d_srec = _schema_dict_to_nt({asetn: self.ad_asetD.d_data[asetn]})
            if m_srec != d_srec:
                tempt1.append(asetn)
        out['t1'] = tempt1

        # removal conflicts
        out['t21'] = self.am_asetD.removals.intersection(self.ad_asetD.mutations)
        out['t22'] = self.ad_asetD.removals.intersection(self.am_asetD.mutations)

        # mutation conflicts
        mutation_keys = self.am_asetD.mutations.intersection(self.ad_asetD.mutations)
        for asetn in mutation_keys:
            m_srec = _schema_dict_to_nt({asetn: self.am_asetD.d_data[asetn]})
            d_srec = _schema_dict_to_nt({asetn: self.ad_asetD.d_data[asetn]})
            if m_srec != d_srec:
                tempt3.append(asetn)
        out['t3'] = tempt3

        for k in list(out.keys()):
            out[k] = tuple(out[k])
        out['conflict'] = any([bool(len(x)) for x in out.values()])
        res = ConflictRecords(**out)
        return res

    def arrayset_changes(self):
        out = {
            'master': self.am_asetD.kv_diff_out(),
            'dev': self.ad_asetD.kv_diff_out(),
        }
        return out

    # -------------------- Sample Diff / Conflicts ----------------------------

    def _sample_diff(self):

        # ---------------- ancestor -> master changes -------------------------

        m_asets = self.am_asetD.unchanged.union(
            self.am_asetD.additions).union(self.am_asetD.mutations)
        for aset_name in m_asets:
            if aset_name in self.acont['arraysets']:
                a_aset_data = self.acont['arraysets'][aset_name]['data']
            else:
                a_aset_data = {}
            self.am_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.am_sampD[aset_name].a_data = a_aset_data
            self.am_sampD[aset_name].d_data = self.mcont['arraysets'][aset_name]['data']
            self.am_sampD[aset_name].compute()

        for aset_name in self.am_asetD.removals:
            self.am_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.am_sampD[aset_name].a_data = self.acont['arraysets'][aset_name]['data']
            self.am_sampD[aset_name].d_data = {}
            self.am_sampD[aset_name].compute()

        # ---------------- ancestor -> dev changes ----------------------------

        d_asets = self.ad_asetD.unchanged.union(
            self.ad_asetD.additions).union(self.ad_asetD.mutations)
        for aset_name in d_asets:
            if aset_name in self.acont['arraysets']:
                a_aset_data = self.acont['arraysets'][aset_name]['data']
            else:
                a_aset_data = {}
            self.ad_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.ad_sampD[aset_name].a_data = a_aset_data
            self.ad_sampD[aset_name].d_data = self.dcont['arraysets'][aset_name]['data']
            self.ad_sampD[aset_name].compute()

        for aset_name in self.ad_asetD.removals:
            self.ad_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
            self.ad_sampD[aset_name].a_data = self.acont['arraysets'][aset_name]['data']
            self.ad_sampD[aset_name].d_data = {}
            self.ad_sampD[aset_name].compute()

    def sample_conflicts(self) -> ConflictRecords:
        """
        t1: added in master & dev with different values
        t21: removed in master, mutated in dev
        t22: removed in dev, mutated in master
        t3: mutated in master & dev to different values
        """
        out = {}
        all_aset_names = set(self.ad_sampD.keys()).union(set(self.am_sampD.keys()))
        for asetn in all_aset_names:
            # When arrayset IN `dev` OR `master` AND NOT IN ancestor OR `other`.
            try:
                mdiff = self.am_sampD[asetn]
            except KeyError:
                mdiff = DifferBase(mut_func=_samples_mutation_finder)
            try:
                ddiff = self.ad_sampD[asetn]
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
            out[asetn] = ConflictRecords(**tempOut)

        return out

    def sample_changes(self):
        out = {
            'master': {asetn: self.am_sampD[asetn].kv_diff_out() for asetn in self.am_sampD},
            'dev': {asetn: self.ad_sampD[asetn].kv_diff_out() for asetn in self.ad_sampD},
        }
        return out

    # ----------------------------- Summary Methods ---------------------------

    def all_changes(self, include_master: bool = True, include_dev: bool = True) -> dict:

        meta = self.meta_changes()
        asets = self.arrayset_changes()
        samples = self.sample_changes()

        if not include_master:
            meta.__delitem__('master')
            asets.__delitem__('master')
            samples.__delitem__('master')
        elif not include_dev:
            meta.__delitem__('dev')
            asets.__delitem__('dev')
            samples.__delitem__('dev')

        res = {
            'metadata': meta,
            'arraysets': asets,
            'samples': samples,
        }
        return res

    def determine_conflicts(self):
        """Evaluate and collect all possible conflicts in a repo differ instance

        Parameters
        ----------
        differ : CommitDiffer
            instance initialized with branch commit contents.

        Returns
        -------
        dict
            containing conflict info in `aset`, `meta`, `sample` and
            `conflict_found` boolean field.
        """
        aset_confs = self.arrayset_conflicts()
        meta_confs = self.meta_conflicts()
        sample_confs = self.sample_conflicts()

        conflictFound = any(
            [aset_confs.conflict, meta_confs.conflict,
             *[confval.conflict for confval in sample_confs.values()]])

        confs = {
            'aset': aset_confs,
            'meta': meta_confs,
            'sample': sample_confs,
            'conflict_found': conflictFound,
        }
        return confs
