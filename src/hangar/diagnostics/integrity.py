from pathlib import Path
import warnings

import lmdb
from tqdm import tqdm

from ..records import (
    hash_data_db_key_from_raw_key,
    hash_schema_db_key_from_raw_key,
)
from ..backends import BACKEND_ACCESSOR_MAP
from ..txnctx import TxnRegister
from ..records import commiting, hashmachine, hashs, parsing, queries, heads
from ..op_state import report_corruption_risk_on_parsing_error


@report_corruption_risk_on_parsing_error
def _verify_column_integrity(hashenv: lmdb.Environment, repo_path: Path):

    hq = hashs.HashQuery(hashenv)
    narrays, nremote = hq.num_data_records(), 0
    array_kvs = hq.gen_all_data_digests_and_parsed_backend_specs()
    try:
        bes = {}
        for digest, spec in tqdm(array_kvs, total=narrays, desc='verifying column data'):
            if spec.backend not in bes:
                bes[spec.backend] = BACKEND_ACCESSOR_MAP[spec.backend](repo_path, None, None)
                bes[spec.backend].open(mode='r')
            if spec.islocal is False:
                nremote += 1
                continue
            data = bes[spec.backend].read_data(spec)
            tcode = hashmachine.hash_type_code_from_digest(digest)

            hash_func = hashmachine.hash_func_from_tcode(tcode)
            calc_digest = hash_func(data)
            if calc_digest != digest:
                raise RuntimeError(
                    f'Data corruption detected for array. Expected digest `{digest}` '
                    f'currently mapped to spec `{spec}`. Found digest `{calc_digest}`')
        if nremote > 0:
            warnings.warn(
                'Can not verify integrity of partially fetched array data references. '
                f'For complete proof, fetch all remote data locally. Did not verify '
                f'{nremote}/{narrays} arrays', RuntimeWarning)
    finally:
        for be in bes.keys():
            bes[be].close()


@report_corruption_risk_on_parsing_error
def _verify_schema_integrity(hashenv: lmdb.Environment):

    hq = hashs.HashQuery(hashenv)
    schema_kvs = hq.gen_all_schema_digests_and_parsed_specs()
    nschemas = hq.num_schema_records()
    for digest, val in tqdm(schema_kvs, total=nschemas, desc='verifying schemas'):
        tcode = hashmachine.hash_type_code_from_digest(digest)
        hash_func = hashmachine.hash_func_from_tcode(tcode)
        calc_digest = hash_func(val)
        if calc_digest != digest:
            raise RuntimeError(
                f'Data corruption detected for schema. Expected digest `{digest}` '
                f'currently mapped to spec `{val}`. Found digest `{calc_digest}`')


@report_corruption_risk_on_parsing_error
def _verify_commit_tree_integrity(refenv: lmdb.Environment):

    initialCmt = None
    all_commits = set(commiting.list_all_commits(refenv))
    reftxn = TxnRegister().begin_reader_txn(refenv)
    try:
        for cmt in tqdm(all_commits, desc='verifying commit trees'):
            pKey = parsing.commit_parent_db_key_from_raw_key(cmt)
            pVal = reftxn.get(pKey, default=False)
            if pVal is False:
                raise RuntimeError(
                    f'Data corruption detected for parent ref of commit `{cmt}`. '
                    f'Parent ref not recorded in refs db.')

            p_val = parsing.commit_parent_raw_val_from_db_val(pVal)
            parents = p_val.ancestor_spec
            if parents.master_ancestor != '':
                if parents.master_ancestor not in all_commits:
                    raise RuntimeError(
                        f'Data corruption detected in commit tree. Commit `{cmt}` '
                        f'with ancestors val `{parents}` references non-existing '
                        f'master ancestor `{parents.master_ancestor}`.')
            if parents.dev_ancestor != '':
                if parents.dev_ancestor not in all_commits:
                    raise RuntimeError(
                        f'Data corruption detected in commit tree. Commit `{cmt}` '
                        f'with ancestors val `{parents}` references non-existing '
                        f'dev ancestor `{parents.dev_ancestor}`.')
            if (parents.master_ancestor == '') and (parents.dev_ancestor == ''):
                if initialCmt is not None:
                    raise RuntimeError(
                        f'Commit tree integrity compromised. Multiple "initial" (commits '
                        f'with no parents) found. First `{initialCmt}`, second `{cmt}`')
                else:
                    initialCmt = cmt
    finally:
        TxnRegister().abort_reader_txn(refenv)


@report_corruption_risk_on_parsing_error
def _verify_commit_ref_digests_exist(hashenv: lmdb.Environment, refenv: lmdb.Environment):

    all_commits = commiting.list_all_commits(refenv)
    datatxn = TxnRegister().begin_reader_txn(hashenv, buffer=True)
    try:
        with datatxn.cursor() as cur:
            for cmt in tqdm(all_commits, desc='verifying commit ref digests'):
                with commiting.tmp_cmt_env(refenv, cmt) as tmpDB:
                    rq = queries.RecordQuery(tmpDB)
                    array_data_digests = set(rq.data_hashes())
                    schema_digests = set(rq.schema_hashes())

                    for datadigest in array_data_digests:
                        dbk = hash_data_db_key_from_raw_key(datadigest)
                        exists = cur.set_key(dbk)
                        if exists is False:
                            raise RuntimeError(
                                f'Data corruption detected in commit refs. Commit `{cmt}` '
                                f'references array data digest `{datadigest}` which does not '
                                f'exist in data hash db.')

                    for schemadigest in schema_digests:
                        dbk = hash_schema_db_key_from_raw_key(schemadigest)
                        exists = cur.set_key(dbk)
                        if exists is False:
                            raise RuntimeError(
                                f'Data corruption detected in commit refs. Commit `{cmt}` '
                                f'references schema digest `{schemadigest}` which does not '
                                f'exist in data hash db.')

    finally:
        TxnRegister().abort_reader_txn(hashenv)


@report_corruption_risk_on_parsing_error
def _verify_branch_integrity(branchenv: lmdb.Environment, refenv: lmdb.Environment):

    branch_names = heads.get_branch_names(branchenv)
    if len(branch_names) < 1:
        raise RuntimeError(
            f'Branch map compromised. Repo must contain atleast one branch. '
            f'Found {len(branch_names)} branches.')

    for bname in tqdm(branch_names, desc='verifying branches'):
        bhead = heads.get_branch_head_commit(branchenv=branchenv, branch_name=bname)
        exists = commiting.check_commit_hash_in_history(refenv=refenv, commit_hash=bhead)
        if exists is False:
            raise RuntimeError(
                f'Branch commit map compromised. Branch name `{bname}` references '
                f'commit digest `{bhead}` which does not exist in refs db.')

    staging_bname = heads.get_staging_branch_head(branchenv)
    if staging_bname not in branch_names:
        raise RuntimeError(
            f'Brach commit map compromised. Staging head refers to branch name '
            f'`{staging_bname}` which does not exist in the branch db.')


def run_verification(branchenv: lmdb.Environment,
                     hashenv: lmdb.Environment,
                     refenv: lmdb.Environment,
                     repo_path: Path):

    _verify_branch_integrity(branchenv, refenv)
    _verify_commit_tree_integrity(refenv)
    _verify_commit_ref_digests_exist(hashenv, refenv)
    _verify_schema_integrity(hashenv)
    _verify_column_integrity(hashenv, repo_path)
