from itertools import starmap
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Set
from typing import Tuple
from typing import Union

import lmdb

from . import constants as c
from .context import TxnRegister
from .records import commiting
from .records import heads
from .records import parsing
from .records.parsing import MetadataRecordKey
from .records.parsing import RawDataRecordKey
from .records.parsing import arrayset_record_schema_raw_key_from_db_key
from .records.parsing import arrayset_record_schema_raw_val_from_db_val
from .records.parsing import data_record_raw_key_from_db_key
from .records.parsing import data_record_raw_val_from_db_val
from .records.parsing import metadata_record_raw_key_from_db_key
from .records.parsing import metadata_record_raw_val_from_db_val
from .records.queries import RecordQuery

# ------------------------- Differ Types --------------------------------------


HistoryDiffStruct = NamedTuple('HistoryDiffStruct', [('masterHEAD', str),
                                                     ('devHEAD', str),
                                                     ('ancestorHEAD', str),
                                                     ('canFF', bool)])

RawChanges = NamedTuple('RawChanges', [
    ('schema', dict),
    ('samples', dict),
    ('metadata', dict),
])

DiffOutDB = NamedTuple('DiffOutDB', [
    ('added', Set[Tuple[bytes, bytes]]),
    ('deleted', Set[Tuple[bytes, bytes]]),
    ('mutated', Set[Tuple[bytes, bytes]]),
])

DiffOutRaw = NamedTuple('DiffOutRaw', [
    ('added', RawChanges),
    ('deleted', RawChanges),
    ('mutated', RawChanges),
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

DiffConflictsOutDB = NamedTuple('DiffConflictsOutDB', [
    ('diff', DiffOutDB),
    ('conflict', ConflictRecords),
])

DiffConflictsOutRaw = NamedTuple('DiffConflictsOutRaw', [
    ('diff', DiffOutRaw),
    ('conflict', ConflictRecords),
])

# ------------------------------- Differ Methods ------------------------------


def diff_envs(base_env: lmdb.Environment, head_env: lmdb.Environment) -> DiffOutDB:
    """Main diff algorithm to determine changes between unpacked lmdb environments.

    Parameters
    ----------
    base_env : lmdb.Environment
        starting point to calculate changes from
    head_env : lmdb.Environment
        some commit which should be compared to BASE

    Returns
    -------
    DiffOutDB
        iterable of db formatted key/value pairs for `added`, `deleted`,
        `mutated` fields

    Raises
    ------
    RuntimeError
        Should never occur.
        TODO: REMOVE!
    """
    added, deleted, mutated = [], [], []
    cont, moreBase, moreHead = True, True, True

    try:
        baseTxn = TxnRegister().begin_reader_txn(base_env)
        headTxn = TxnRegister().begin_reader_txn(head_env)
        baseCur = baseTxn.cursor()
        headCur = headTxn.cursor()
        baseCur.first()
        headCur.first()
        while True:
            if moreBase and moreHead:
                bKey, bVal = baseCur.item()
                hKey, hVal = headCur.item()
            elif (not moreBase) and (not moreHead):
                break
            # necessary to avoid deadlock at last items
            elif not moreBase:
                bKey = b'x'
                bVal = b''
                hKey, hVal = headCur.item()
            elif not moreHead:
                hKey = b'x'
                hVal = b''
                bKey, bVal = baseCur.item()
            else:
                raise RuntimeError('should not reach here!')

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
    finally:
        baseCur.close()
        headCur.close()
        base_env = TxnRegister().abort_reader_txn(base_env)
        head_env = TxnRegister().abort_reader_txn(head_env)

    return DiffOutDB(set(added), set(deleted), set(mutated))


def _raw_from_db_change(changes: Set[Tuple[bytes, bytes]]) -> RawChanges:
    """Perform conversion for records from db -> raw

    Parameters
    ----------
    changes : Set[Tuple[bytes, bytes]]
        iterable of db formatted key/value pairs

    Returns
    -------
    RawChanges
        human readable formatted dict of key/value pairs.
    """
    arraysets, metadata, schema = {}, {}, {}
    for k, v in changes:
        if k.startswith(b'a:'):
            rk = data_record_raw_key_from_db_key(k)
            rv = data_record_raw_val_from_db_val(v)
            arraysets[rk] = rv
        elif k.startswith(b'l:'):
            rk = metadata_record_raw_key_from_db_key(k)
            rv = metadata_record_raw_val_from_db_val(v)
            metadata[rk] = rv
        elif k.startswith(b's:'):
            rk = arrayset_record_schema_raw_key_from_db_key(k)
            rv = arrayset_record_schema_raw_val_from_db_val(v)
            schema[rk] = rv
    return RawChanges(schema=schema, samples=arraysets, metadata=metadata)


def _all_raw_from_db_changes(outDb: DiffConflictsOutDB) -> DiffConflictsOutRaw:
    """Convert db formatted db diff/conflict results to human readable

    Parameters
    ----------
    outDb : DiffConflictsOutDB
        raw formatted structure containg `diff` and `conflict` fields

    Returns
    -------
    DiffConflictsOutRaw
        Human readable struct containing `diff` and `conflict` fields.
    """
    it = (outDb.diff.added, outDb.diff.deleted, outDb.diff.mutated)
    out = map(_raw_from_db_change, it)  # significant perf improvement for large commits
    outRawDiff = DiffOutRaw(*out)

    t1 = _raw_from_db_change(outDb.conflict.t1)
    t21 = _raw_from_db_change(outDb.conflict.t21)
    t22 = _raw_from_db_change(outDb.conflict.t22)
    t3 = _raw_from_db_change(outDb.conflict.t3)
    outRawConf = ConflictRecords(t1=t1, t21=t21, t22=t22, t3=t3, conflict=outDb.conflict.conflict)
    res = DiffConflictsOutRaw(diff=outRawDiff, conflict=outRawConf)
    return res

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
    for k, v in pair1.symmetric_difference(pair2):
        if k in seen:
            conflict.append((k, v))
        else:
            seen.add(k)
    return conflict


def find_conflicts(master_diff: DiffOutDB, dev_diff: DiffOutDB) -> ConflictRecords:
    """Determine if/which type of conflicting changes occur in diverged commits.

    This function expects the output of :func:`diff_envs` for two commits
    between a base commit.

    Parameters
    ----------
    master_diff : DiffOutDB
        changes (adds, dels, mutations) between base and master HEAD
    dev_diff : DiffOutDB
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
    isConflict = bool(any([t1, t21, t22, t3]))

    res = ConflictRecords(t1=t1, t21=t21, t22=t22, t3=t3, conflict=isConflict)
    return res


# ---------------------------- Differ Base  -----------------------------------


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

    def _diff3(self,
               a_env: lmdb.Environment,
               m_env: lmdb.Environment,
               d_env: lmdb.Environment) -> DiffConflictsOutDB:
        """Three way diff and conflict finder from ancestor, master, and dev commits.

        Parameters
        ----------
        a_env : lmdb.Environment
            unpacked lmdb environment for the ancestor commit
        m_env : lmdb.Environment
            unpacked lmdb environment for the master commit, current HEAD
        d_env : lmdb.Environment
            unpacked lmdb environment for the dev commit, compare to HEAD

        Returns
        -------
        DiffConflictsOutDB
            structure containing (`additions`, `deletions`, `mutations`) for
            diff, as well as the ConflictRecord struct.
        """
        # it = (m_diff, d_diff, md_diff)
        it = ((a_env, m_env), (a_env, d_env), (d_env, m_env))
        diffs = list(starmap(diff_envs, it))  # significant perf improvement by map.
        conflict = find_conflicts(diffs[0], diffs[1])
        return DiffConflictsOutDB(diff=diffs[2], conflict=conflict)

    def _diff(self,
              a_env: lmdb.Environment,
              m_env: lmdb.Environment) -> DiffConflictsOutDB:
        """Fast Forward differ from ancestor to master commit.

        Note: this method returns the same MasterDevDiff struct as the three
        way commit diff method, but the `dev` and `conflicts` fields will be
        empty

        Parameters
        ----------
        a_env : lmdb.Environment
            unpacked lmdb environment for the ancestor commit
        m_env : lmdb.Environment
            unpacked lmdb environment for the master commit

        Returns
        -------
        DiffConflictsOutDB
            structure containing (`additions`, `deletions`, `mutations`) for
            the ancestor -> master (head) env diff
        """
        m_diff = diff_envs(a_env, m_env)
        conflict = ConflictRecords(t1=[], t21=[], t22=[], t3=[], conflict=False)
        return DiffConflictsOutDB(diff=m_diff, conflict=conflict)


# ------------------------ Read-Only Checkouts Only ---------------------------


class ReaderUserDiff(BaseUserDiff):

    def __init__(self, commit_hash, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._commit_hash = commit_hash

    def _commit(self, dev_commit_hash: str) -> DiffConflictsOutDB:
        """Compute diff between head and commit hash, returning DB formatted results

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        DiffConflictsOutDB
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

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
            outDb = self._diff(m_env, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            outDb = self._diff3(a_env, m_env, d_env)
        return outDb

    def _branch(self, dev_branch_name: str) -> DiffConflictsOutDB:
        """Compute diff between head and branch name, returning DB formatted results.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        DiffConflictsOutDB
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

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
            outDb = self._diff(m_env, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            outDb = self._diff3(a_env, m_env, d_env)
        return outDb

    def commit(self, dev_commit_hash: str) -> DiffConflictsOutRaw:
        """Compute diff between HEAD and commit hash, returning user-facing results format.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        DiffConflictsOutRaw
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        """
        outDb = self._commit(dev_commit_hash=dev_commit_hash)
        outRaw = _all_raw_from_db_changes(outDb)
        return outRaw

    def branch(self, dev_branch_name: str) -> DiffConflictsOutRaw:
        """Compute diff between HEAD and branch name, returning user-facing results format.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        DiffConflictsOutRaw
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            If the specified `dev_branch_name` does not exist.
        """
        outDb = self._branch(dev_branch_name=dev_branch_name)
        outRaw = _all_raw_from_db_changes(outDb)
        return outRaw


# ---------------------- Write Enabled Checkouts Only -------------------------


class WriterUserDiff(BaseUserDiff):

    def __init__(self, stageenv: lmdb.Environment, branch_name: str, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self._stageenv: lmdb.Environment = stageenv
        self._branch_name: str = branch_name

    def _commit(self, dev_commit_hash: str) -> DiffConflictsOutDB:
        """Compute diff between head and commit hash, returning DB formatted results.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        DiffConflictsOutDB
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

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
            return self._diff(self._stageenv, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff3(a_env, self._stageenv, d_env)

    def _branch(self, dev_branch_name: str) -> DiffConflictsOutDB:
        """Compute diff between head and branch name, returning DB formatted results.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        DiffConflictsOutDB
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

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
            return self._diff(self._stageenv, d_env)
        else:
            a_env = commiting.get_commit_ref_env(self._refenv, hist.ancestorHEAD)
            return self._diff3(a_env, self._stageenv, d_env)

    def _staged(self) -> DiffConflictsOutDB:
        """Return the diff of the staging area to last HEAD, returning db formatted results.

        Returns
        -------
        DiffConflictsOutDB
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.
        """
        commit_hash = heads.get_branch_head_commit(self._branchenv, self._branch_name)
        base_env = commiting.get_commit_ref_env(self._refenv, commit_hash)
        return self._diff(base_env, self._stageenv)

    def commit(self, dev_commit_hash: str) -> DiffConflictsOutRaw:
        """Compute diff between HEAD and commit hash, returning user-facing results format.

        Parameters
        ----------
        dev_commit_hash : str
            hash of the commit to be used as the comparison.

        Returns
        -------
        DiffConflictsOutRaw
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            if the specified `dev_commit_hash` is not a valid commit reference.
        """
        outDb = self._commit(dev_commit_hash=dev_commit_hash)
        outRaw = _all_raw_from_db_changes(outDb)
        return outRaw

    def branch(self, dev_branch_name: str) -> DiffConflictsOutRaw:
        """Compute diff between HEAD and branch name, returning user-facing results format.

        Parameters
        ----------
        dev_branch_name : str
            name of the branch whose HEAD will be used to calculate the diff of.

        Returns
        -------
        DiffConflictsOutRaw
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.

        Raises
        ------
        ValueError
            If the specified `dev_branch_name` does not exist.
        """
        outDb = self._branch(dev_branch_name=dev_branch_name)
        outRaw = _all_raw_from_db_changes(outDb)
        return outRaw

    def staged(self) -> DiffConflictsOutRaw:
        """Return the diff of the staging to last HEAD, returning user-facing results format.

        Returns
        -------
        DiffConflictsOutRaw
            two-tuple of `diff`, `conflict` (if any) calculated in the diff algorithm.
        """
        outDb = self._staged()
        outRaw = _all_raw_from_db_changes(outDb)
        return outRaw

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
