from functools import partial
from typing import (
    Callable, Iterable, List, MutableMapping, NamedTuple, Set, Union, Tuple)

import lmdb

from .records import commiting, heads, parsing
from .records.parsing import (MetadataRecordKey, MetadataRecordVal,
                              RawDataRecordKey, RawDataRecordVal,
                              RawArraysetSchemaVal)
from .records.queries import RecordQuery
from .context import TxnRegister
from . import constants as c

HistoryDiffStruct = NamedTuple('HistoryDiffStruct', [('masterHEAD', str),
                                                     ('devHEAD', str),
                                                     ('ancestorHEAD', str),
                                                     ('canFF', bool)])

DiffOut = NamedTuple('DiffOut', [
    ('added', Set[Tuple[bytes, bytes]]),
    ('deleted', Set[Tuple[bytes, bytes]]),
    ('mutated', Set[Tuple[bytes, bytes]]),
])

RawChanges = NamedTuple('RawChanges', [
    ('schema', dict),
    ('samples', dict),
    ('metadata', dict),
])

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

MasterDevDiff = NamedTuple('MasterDevDiff', [
    ('master', DiffOut),
    ('dev', DiffOut),
    ('conflicts', ConflictRecords)
])


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

    def _diff_three_way_cmts(self, a_env: lmdb.Environment,
                             m_env: lmdb.Environment,
                             d_env: lmdb.Environment) -> MasterDevDiff:
        m_diff = diff_envs(a_env, m_env)
        d_diff = diff_envs(a_env, d_env)
        conflicts = find_conflicts(m_diff, d_diff)
        return MasterDevDiff(master=m_diff, dev=d_diff, conflicts=conflicts)

    def _diff_ff_cmts(self, a_env: lmdb.Environment,
                      m_env: lmdb.Environment) -> MasterDevDiff:
        m_diff = diff_envs(a_env, m_env)
        d_diff = DiffOut(set(), set(), set())
        conflicts = ConflictRecords((), (), (), (), False)
        return MasterDevDiff(master=m_diff, dev=d_diff, conflicts=conflicts)


# ----------------------------------------------------------


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
        m_env = commiting.get_commit_ref_env(self._refenv, hist.masterHEAD)
        d_env = commiting.get_commit_ref_env(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(m_env, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_env, m_env, d_env)

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
        m_env = commiting.get_commit_ref_env(self._refenv, hist.masterHEAD)
        d_env = commiting.get_commit_ref_env(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(m_env, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_env, m_env, d_env)


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

        d_env = commiting.get_commit_ref_env(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(self._stageenv, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_env, self._stageenv, d_env)

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

        d_env = commiting.get_commit_ref_env(self._refenv, hist.devHEAD)
        if hist.canFF is True:
            return self._diff_ff_cmts(self._stageenv, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff_three_way_cmts(a_env, self._stageenv, d_env)

    def staged(self):
        """Return the diff of the staging area contents to the staging base HEAD

        Returns
        -------
        dict
            contains all `addition`, `mutation`, `removals` and `unchanged` arraysets
            schema, samples, and metadata references in the current staging area.
        """
        commit_hash = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        base_env = commiting.get_commit_ref_env(self._refenv, commit_hash)
        return self._diff_ff_cmts(base_env, self._stageenv)


# --------------------------------------------------------------------


def diff_envs(base_env: lmdb.Environment, head_env: lmdb.Environment) -> DiffOut:
    added, deleted, mutated = [], [], []
    cont, moreBase, moreHead = True, True, True

    try:
        baseTxn = TxnRegister().begin_reader_txn(base_env)
        headTxn = TxnRegister().begin_reader_txn(head_env)
        baseCur = baseTxn.cursor()
        headCur = headTxn.cursor()
        baseCur.first()
        headCur.first()
        while cont is True:
            if (moreBase is False) and (moreHead is False):
                cont = False
                continue
            bKey, bVal = baseCur.item()
            hKey, hVal = headCur.item()
            # inserted
            if bKey > hKey:
                added.append((hKey, hVal))
                moreHead = headCur.next()
                continue
            # deleted
            elif bKey < hKey:
                deleted.append((bKey, bVal))
                moreBase = baseCur.next()
                continue
            # no change
            elif (bKey == hKey) and (bVal == hVal):
                moreBase = baseCur.next()
                moreHead = headCur.next()
                continue
            # mutated
            elif (bKey == hKey) and (bVal != hVal):
                mutated.append((hKey, hVal))
                moreBase = baseCur.next()
                moreHead = headCur.next()
                continue
            # catch all - TODO: remove!
            else:
                raise RuntimeError('Should not reach here!')
                moreBase = baseCur.next()
                moreHead = headCur.next()
                continue
    finally:
        baseCur.close()
        headCur.close()
        base_env = TxnRegister().abort_reader_txn(base_env)
        head_env = TxnRegister().abort_reader_txn(head_env)

    return DiffOut(set(added), set(deleted), set(mutated))


def raw_from_db_change(changes: Set[Tuple[bytes, bytes]]) -> RawChanges:
    arraysets, metadata, schema = {}, {}, {}
    for k, v in changes:
        if k.startswith(b'a:'):
            if k.endswith(b':'):
                continue
            rk = parsing.data_record_raw_key_from_db_key(k)
            rv = parsing.data_record_raw_val_from_db_val(v)
            arraysets[rk] = rv
        elif k.startswith(b'l:'):
            if k.endswith(b':'):
                continue
            rk = parsing.metadata_record_raw_key_from_db_key(k)
            rv = parsing.metadata_record_raw_val_from_db_val(v)
            metadata[rk] = rv
        elif k.startswith(b's:'):
            if k.endswith(b':'):
                continue
            rk = parsing.arrayset_record_schema_raw_key_from_db_key(k)
            rv = parsing.arrayset_record_schema_raw_val_from_db_val(v)
            schema[rk] = rv
    return RawChanges(schema=schema, samples=arraysets, metadata=metadata)


# ------------------------- Commit Differ -------------------------------------


def _symmetric_difference_keys(pair1: Set[Tuple[bytes, bytes]],
                               pair2: Set[Tuple[bytes, bytes]]
                               ) -> List[Tuple[bytes]]:
    """Find all keys common to both input pairs AND which have different values.

    Essentially a moddified `symmetric_difference` set operation, which keeps
    track of all seen items. Note: This ignores any `count` tracking values in
    the input tuples (ie. lmdb keys ending in ":")

    Parameters
    ----------
    pair1 : Set[Tuple[bytes, bytes]]
        key/value pairs making up the first set
    pair2 : Set[Tuple[bytes, bytes]]
        key/value pairs making up the second set

    Returns
    -------
    List[Tuple[bytes]]
        keys which appear in both input pair sets but which have different values.
    """
    seen = set()
    conflict = []
    for k, _ in pair1.symmetric_difference(pair2):
        if k in seen:
            if k.endswith(c.SEP_KEY.encode()):
                continue
            else:
                conflict.append(k)
        else:
            seen.add(k)
    return conflict


def find_conflicts(master_diff: DiffOut, dev_diff: DiffOut) -> ConflictRecords:
    """Determine if/which type of conflicting changes occur in diverged commits.

    This function expects the output of :func:`diff_envs` for two commits
    between a base commit.

    Parameters
    ----------
    master_diff : DiffOut
        changes (adds, dels, mutations) between base and master HEAD
    dev_diff : DiffOut
        changes (adds, dels, mutations) between base and dev HEAD

    Returns
    -------
    ConflictRecords
        Tuple containing fields for `t1`, `t21`, `t22`, `t3`, and (bool)
        `conflicts` recording output info for if and what type of conflict has
        occured
    """
    t1 = _symmetric_difference_keys(master_diff.added, dev_diff.added)
    t21 = _symmetric_difference_keys(master_diff.deleted, dev_diff.mutated)
    t22 = _symmetric_difference_keys(master_diff.mutated, dev_diff.deleted)
    t3 = _symmetric_difference_keys(master_diff.mutated, dev_diff.mutated)
    isConflict = any([t1, t21, t22, t3])

    res = ConflictRecords(t1=t1, t21=t21, t22=t22, t3=t3, conflict=isConflict)
    return res


# class ThreeWayCommitDiffer(object):

#     def __init__(self, ancestor_env, master_env, dev_env: dict):

#         self.aenv = ancestor_env  # ancestor contents
#         self.menv = master_env    # master contents
#         self.denv = dev_env       # dev contents

#         self.mdiff = diff_envs(self.aenv, self.menv)
#         self.ddiff = diff_envs(self.aenv, self.denv)

#         t1 = find_conflicts(self.mdiff.added, self.ddiff.added)
#         t21 = find_conflicts(self.mdiff.mutated, self.ddiff.deleted)
#         t22 = find_conflicts(self.mdiff.deleted, self.ddiff.mutated)
#         t3 = find_conflicts(self.mdiff.mutated, self.ddiff.mutated)

#         confs = ConflictRecords(t1=t1, t21 =t21, t22=t22, t3=t3)

#     # -------------------- Metadata Diff / Conflicts --------------------------

#     def _meta_diff(self):

#         self.am_metaD = DifferBase(mut_func=_meta_mutation_finder)
#         self.am_metaD.a_data = self.acont['metadata']
#         self.am_metaD.d_data = self.mcont['metadata']
#         self.am_metaD.compute()

#         self.ad_metaD = DifferBase(mut_func=_meta_mutation_finder)
#         self.ad_metaD.a_data = self.acont['metadata']
#         self.ad_metaD.d_data = self.dcont['metadata']
#         self.ad_metaD.compute()

#     def meta_conflicts(self) -> ConflictRecords:
#         """
#         t1: added in master & dev with different values
#         t21: removed in master, mutated in dev
#         t22: removed in dev, mutated in master
#         t3: mutated in master & dev to different values
#         """
#         out: MutableMapping[str, Union[bool, Iterable[ConflictKeys]]] = {}
#         tempt1: List[ConflictKeys] = []
#         tempt3: List[ConflictKeys] = []

#         # addition conflicts
#         addition_keys = self.am_metaD.additions.intersection(self.ad_metaD.additions)
#         for meta_key in addition_keys:
#             if self.am_metaD.d_data[meta_key] != self.ad_metaD.d_data[meta_key]:
#                 tempt1.append(meta_key)
#         out['t1'] = tempt1

#         # removal conflicts
#         out['t21'] = self.am_metaD.removals.intersection(self.ad_metaD.mutations)
#         out['t22'] = self.ad_metaD.removals.intersection(self.am_metaD.mutations)

#         # mutation conflicts
#         mutation_keys = self.am_metaD.mutations.intersection(self.ad_metaD.mutations)
#         for meta_key in mutation_keys:
#             if self.am_metaD.d_data[meta_key] != self.ad_metaD.d_data[meta_key]:
#                 tempt3.append(meta_key)
#         out['t3'] = tempt3

#         for k in list(out.keys()):
#             out[k] = tuple(out[k])
#         out['conflict'] = any([bool(len(x)) for x in out.values()])
#         res = ConflictRecords(**out)
#         return res

#     def meta_changes(self):
#         out = {
#             'master': self.am_metaD.kv_diff_out(),
#             'dev': self.ad_metaD.kv_diff_out(),
#         }
#         return out

#     # -------------------- Arrayset Diff / Conflicts ---------------------------

#     def _arrayset_diff(self):

#         mutFunc = partial(_schema_mutation_finder, sch_nt_func=_schema_dict_to_nt)
#         self.am_asetD = DifferBase(mut_func=mutFunc)
#         self.am_asetD.a_data = _isolate_aset_schemas(self.acont['arraysets'])
#         self.am_asetD.d_data = _isolate_aset_schemas(self.mcont['arraysets'])
#         self.am_asetD.compute()

#         self.ad_asetD = DifferBase(mut_func=mutFunc)
#         self.ad_asetD.a_data = _isolate_aset_schemas(self.acont['arraysets'])
#         self.ad_asetD.d_data = _isolate_aset_schemas(self.dcont['arraysets'])
#         self.ad_asetD.compute()

#     def arrayset_conflicts(self) -> ConflictRecords:
#         """
#         t1: added in master & dev with different values
#         t21: removed in master, mutated in dev
#         t22: removed in dev, mutated in master
#         t3: mutated in master & dev to different values
#         """
#         out: MutableMapping[str, Union[bool, Iterable[ConflictKeys]]] = {}
#         tempt1: List[ConflictKeys] = []
#         tempt3: List[ConflictKeys] = []

#         # addition conflicts
#         addition_keys = self.am_asetD.additions.intersection(self.ad_asetD.additions)
#         for asetn in addition_keys:
#             m_srec = _schema_dict_to_nt({asetn: self.am_asetD.d_data[asetn]})
#             d_srec = _schema_dict_to_nt({asetn: self.ad_asetD.d_data[asetn]})
#             if m_srec != d_srec:
#                 tempt1.append(asetn)
#         out['t1'] = tempt1

#         # removal conflicts
#         out['t21'] = self.am_asetD.removals.intersection(self.ad_asetD.mutations)
#         out['t22'] = self.ad_asetD.removals.intersection(self.am_asetD.mutations)

#         # mutation conflicts
#         mutation_keys = self.am_asetD.mutations.intersection(self.ad_asetD.mutations)
#         for asetn in mutation_keys:
#             m_srec = _schema_dict_to_nt({asetn: self.am_asetD.d_data[asetn]})
#             d_srec = _schema_dict_to_nt({asetn: self.ad_asetD.d_data[asetn]})
#             if m_srec != d_srec:
#                 tempt3.append(asetn)
#         out['t3'] = tempt3

#         for k in list(out.keys()):
#             out[k] = tuple(out[k])
#         out['conflict'] = any([bool(len(x)) for x in out.values()])
#         res = ConflictRecords(**out)
#         return res

#     def arrayset_changes(self):
#         out = {
#             'master': self.am_asetD.kv_diff_out(),
#             'dev': self.ad_asetD.kv_diff_out(),
#         }
#         return out

#     # -------------------- Sample Diff / Conflicts ----------------------------

#     def _sample_diff(self):

#         # ---------------- ancestor -> master changes -------------------------

#         m_asets = self.am_asetD.unchanged.union(
#             self.am_asetD.additions).union(self.am_asetD.mutations)
#         for aset_name in m_asets:
#             if aset_name in self.acont['arraysets']:
#                 a_aset_data = self.acont['arraysets'][aset_name]['data']
#             else:
#                 a_aset_data = {}
#             self.am_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
#             self.am_sampD[aset_name].a_data = a_aset_data
#             self.am_sampD[aset_name].d_data = self.mcont['arraysets'][aset_name]['data']
#             self.am_sampD[aset_name].compute()

#         for aset_name in self.am_asetD.removals:
#             self.am_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
#             self.am_sampD[aset_name].a_data = self.acont['arraysets'][aset_name]['data']
#             self.am_sampD[aset_name].d_data = {}
#             self.am_sampD[aset_name].compute()

#         # ---------------- ancestor -> dev changes ----------------------------

#         d_asets = self.ad_asetD.unchanged.union(
#             self.ad_asetD.additions).union(self.ad_asetD.mutations)
#         for aset_name in d_asets:
#             if aset_name in self.acont['arraysets']:
#                 a_aset_data = self.acont['arraysets'][aset_name]['data']
#             else:
#                 a_aset_data = {}
#             self.ad_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
#             self.ad_sampD[aset_name].a_data = a_aset_data
#             self.ad_sampD[aset_name].d_data = self.dcont['arraysets'][aset_name]['data']
#             self.ad_sampD[aset_name].compute()

#         for aset_name in self.ad_asetD.removals:
#             self.ad_sampD[aset_name] = DifferBase(mut_func=_samples_mutation_finder)
#             self.ad_sampD[aset_name].a_data = self.acont['arraysets'][aset_name]['data']
#             self.ad_sampD[aset_name].d_data = {}
#             self.ad_sampD[aset_name].compute()

#     def sample_conflicts(self) -> ConflictRecords:
#         """
#         t1: added in master & dev with different values
#         t21: removed in master, mutated in dev
#         t22: removed in dev, mutated in master
#         t3: mutated in master & dev to different values
#         """
#         out = {}
#         all_aset_names = set(self.ad_sampD.keys()).union(set(self.am_sampD.keys()))
#         for asetn in all_aset_names:
#             # When arrayset IN `dev` OR `master` AND NOT IN ancestor OR `other`.
#             try:
#                 mdiff = self.am_sampD[asetn]
#             except KeyError:
#                 mdiff = DifferBase(mut_func=_samples_mutation_finder)
#             try:
#                 ddiff = self.ad_sampD[asetn]
#             except KeyError:
#                 ddiff = DifferBase(mut_func=_samples_mutation_finder)

#             samp_conflicts_t1, samp_conflicts_t3 = [], []

#             # addition conflicts
#             addition_keys = mdiff.additions.intersection(ddiff.additions)
#             for samp in addition_keys:
#                 if mdiff.d_data[samp] != ddiff.d_data[samp]:
#                     samp_conflicts_t1.append(samp)

#             # removal conflicts
#             samp_conflicts_t21 = mdiff.removals.intersection(ddiff.mutations)
#             samp_conflicts_t22 = ddiff.removals.intersection(mdiff.mutations)

#             # mutation conflicts
#             mutation_keys = mdiff.mutations.intersection(ddiff.mutations)
#             for samp in mutation_keys:
#                 if mdiff.d_data[samp] != ddiff.d_data[samp]:
#                     samp_conflicts_t3.append(samp)

#             tempOut = {
#                 't1': tuple(samp_conflicts_t1),
#                 't21': tuple(samp_conflicts_t21),
#                 't22': tuple(samp_conflicts_t22),
#                 't3': tuple(samp_conflicts_t3)
#             }
#             for k in list(tempOut.keys()):
#                 tempOut[k] = tuple(tempOut[k])
#             tempOut['conflict'] = any([bool(len(x)) for x in tempOut.values()])
#             out[asetn] = ConflictRecords(**tempOut)

#         return out

#     def sample_changes(self):
#         out = {
#             'master': {asetn: self.am_sampD[asetn].kv_diff_out() for asetn in self.am_sampD},
#             'dev': {asetn: self.ad_sampD[asetn].kv_diff_out() for asetn in self.ad_sampD},
#         }
#         return out

#     # ----------------------------- Summary Methods ---------------------------

#     def all_changes(self, include_master: bool = True, include_dev: bool = True) -> dict:

#         meta = self.meta_changes()
#         asets = self.arrayset_changes()
#         samples = self.sample_changes()

#         if not include_master:
#             meta.__delitem__('master')
#             asets.__delitem__('master')
#             samples.__delitem__('master')
#         elif not include_dev:
#             meta.__delitem__('dev')
#             asets.__delitem__('dev')
#             samples.__delitem__('dev')

#         res = {
#             'metadata': meta,
#             'arraysets': asets,
#             'samples': samples,
#         }
#         return res

#     def determine_conflicts(self):
#         """Evaluate and collect all possible conflicts in a repo differ instance

#         Parameters
#         ----------
#         differ : CommitDiffer
#             instance initialized with branch commit contents.

#         Returns
#         -------
#         dict
#             containing conflict info in `aset`, `meta`, `sample` and
#             `conflict_found` boolean field.
#         """
#         aset_confs = self.arrayset_conflicts()
#         meta_confs = self.meta_conflicts()
#         sample_confs = self.sample_conflicts()

#         conflictFound = any(
#             [aset_confs.conflict, meta_confs.conflict,
#              *[confval.conflict for confval in sample_confs.values()]])

#         confs = {
#             'aset': aset_confs,
#             'meta': meta_confs,
#             'sample': sample_confs,
#             'conflict_found': conflictFound,
#         }
#         return confs