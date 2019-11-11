import os
import time
from io import StringIO

import numpy as np
import lmdb

from . import heads, commiting, queries
from ..context import TxnRegister
from ..diff import DiffOut, Changes
from ..utils import format_bytes, file_size, folder_size, unique_everseen


def list_history(refenv, branchenv, branch_name=None, commit_hash=None):
    """Traverse commit history to specifying ancestor DAG and all ancestor specs.

    Parameters
    ----------
    refenv : lmdb.Environment
        environment containing all repository commit data.
    branchenv : lmdb.Environment
        environment containing the current staging head branch and branch head
        commit hashes
    branch_name : string, optional
        if specified, get the history starting at the head commit of this named
        branch (the default is None, which will use the `commit_hash` arg if
        available, or staging area head)
    commit_hash : string, optional
        if specified, get the history starting at this specific commit,
        overrides branch name if both are specified (the default is `None`,
        which will use the branch_name arg if available, or staging area head)

    Returns
    -------
    dict
        dict containing information about the repo history. specifies fields for
        `head`, `ancestors` (DAG of commit), and `specs` of each commit, also `order`
        encountered.
    """

    if commit_hash is not None:
        head_commit = commit_hash
    elif branch_name is not None:
        head_commit = heads.get_branch_head_commit(branchenv=branchenv, branch_name=branch_name)
    else:
        head_branch = heads.get_staging_branch_head(branchenv)
        head_commit = heads.get_branch_head_commit(branchenv, head_branch)

    ancestors = commiting.get_commit_ancestors_graph(
        refenv=refenv, starting_commit=head_commit)

    commitSpecs = {}
    for commit in ancestors.keys():
        commitSpecs[commit] = dict(commiting.get_commit_spec(refenv, commit_hash=commit)._asdict())

    cmtTimeSorter = [(k, v['commit_time']) for k, v in commitSpecs.items()]
    cmtTimeSorter.sort(key=lambda t: t[1], reverse=True)
    showparentsOrder = [x[0] for x in cmtTimeSorter]

    res = {
        'head': head_commit,
        'ancestors': ancestors,
        'specs': commitSpecs,
        'order': showparentsOrder,
    }
    return res


def details(env: lmdb.Environment, line_limit=100, line_length=100) -> StringIO:  # pragma: no cover
    """Print the details of an lmdb environment to stdout

    Parameters
    ----------
    env : lmdb.Environment
        environment handle to print records of
    line_limit : int, optional
        limit to the amount of record lines printed, by default 100
    line_length : int, optional
        limit the amount of text printed per line, by default 100

    Returns
    -------
    StringIO
        buffer containing detail data.
    """
    buf = StringIO()
    buf.write('\n======================\n')
    buf.write(f'{os.path.basename(env.path())}')
    try:
        buf.write(f'File Size: {format_bytes(file_size(env.path()))}\n')
    except FileNotFoundError:
        pass
    buf.write('======================\n\n')
    txn = TxnRegister().begin_reader_txn(env)
    entries = txn.stat()['entries'] - 10
    with txn.cursor() as cursor:
        count, once = 0, False
        for key, value in cursor:
            if (count >= line_limit) and (count < entries):
                count += 1
                if (once is False) and (count < entries):
                    once = True
                    buf.write('...\n...\n...\n')
                continue
            else:
                if len(value) >= line_length:
                    buf.write(f'{key} long binary\n')
                else:
                    buf.write(f'{key} {value}\n')
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env)
    return buf


def summary(env, *, branch='', commit='') -> StringIO:
    """Summary of data set stored in repository.

    Parameters
    ----------
    env : :class:`Environments`
        class which contains all of the lmdb environments pre-initialized for use.
    commit : str
        commit hash to query. if left empty, HEAD commit is used (Default value = '')
    branch : str
        branch name to query, if left empty, HEAD will be used. (Default value = '')

    Returns
    -------
    StringIO:
        buffer formatting the contents of the commit ref at the queried commit.
    """
    if commit != '':
        cmt = commit
    elif branch != '':
        cmt = heads.get_branch_head_commit(env.branchenv, branch)
    else:
        headBranch = heads.get_staging_branch_head(env.branchenv)
        cmt = heads.get_branch_head_commit(env.branchenv, headBranch)

    spec = commiting.get_commit_spec(env.refenv, cmt)._asdict()
    if cmt == '':
        print('No commits made')
        return {}

    with commiting.tmp_cmt_env(env.refenv, cmt) as cmtrefenv:
        query = queries.RecordQuery(cmtrefenv)

        nbytes = folder_size(env.repo_path, recurse=True)
        humanBytes = format_bytes(nbytes)
        buf = StringIO()
        buf.write(f'Summary of Contents Contained in Data Repository \n')
        buf.write(f' \n')
        buf.write(f'================== \n')
        buf.write(f'| Repository Info \n')
        buf.write(f'|----------------- \n')
        buf.write(f'|  Base Directory: {os.path.dirname(env.repo_path)} \n')
        buf.write(f'|  Disk Usage: {humanBytes} \n')
        buf.write(f' \n')

        buf.write(f'=================== \n')
        buf.write(f'| Commit Details \n')
        buf.write(f'------------------- \n')
        buf.write(f'|  Commit: {cmt} \n')
        buf.write(f'|  Created: {time.asctime(time.gmtime(spec["commit_time"]))} \n')
        buf.write(f'|  By: {spec["commit_user"]} \n')
        buf.write(f'|  Email: {spec["commit_email"]} \n')
        buf.write(f'|  Message: {spec["commit_message"]} \n')
        buf.write(f' \n')
        buf.write(f'================== \n')
        buf.write(f'| DataSets \n')
        buf.write(f'|----------------- \n')

        buf.write(f'|  Number of Named Arraysets: {query.arrayset_count()} \n')
        for asetn, asetnSchema in query.schema_specs().items():
            buf.write(f'|\n')
            buf.write(f'|  * Arrayset Name: {asetn} \n')
            buf.write(f'|    Num Arrays: {query.arrayset_data_count(asetn)} \n')

            buf.write(f'|    Details: \n')
            for k, v in asetnSchema._asdict().items():
                buf.write(f'|    - {k}: {v} \n')

        buf.write(f' \n')
        buf.write(f'================== \n')
        buf.write(f'| Metadata: \n')
        buf.write(f'|----------------- \n')
        buf.write(f'|  Number of Keys: {query.metadata_count()} \n')

    return buf


def status(branch_name: str, diff: DiffOut) -> StringIO:
    """Format human readable string buffer of changes in a staging area

    Parameters
    ----------
    branch_name : str
        Name of the branch the diff is from.
    diff : DiffOut
        diff struct tuple returned from standard diff tool.

    Returns
    -------
    StringIO
        Buffer containing human readable printable string of change summary
    """
    def _diff_info(df: Changes) -> StringIO:
        """Format buffer for each of `ADDED`, `DELETED`, `MUTATED` changes
        """
        buf = StringIO()
        buf.write(f'|---------- \n')
        buf.write(f'| Schema: {len(df.schema)} \n')
        for k, v in df.schema.items():
            buf.write(f'|  - "{k}": \n')
            buf.write(f'|       named: {v.schema_is_named} \n')
            buf.write(f'|       dtype: {np.dtype(np.typeDict[v.schema_dtype])} \n')
            buf.write(f'|       (max) shape: {v.schema_max_shape} \n')
            buf.write(f'|       variable shape: {v.schema_is_var} \n')
            buf.write(f'|       backend: {v.schema_default_backend} \n')
            buf.write(f'|       backend opts: {v.schema_default_backend_opts} \n')

        buf.write('|---------- \n')
        buf.write(f'| Samples: {len(df.samples)} \n')
        unique = unique_everseen(df.samples.keys(), lambda x: x.aset_name)
        for u in unique:
            un = u.aset_name
            count = sum((1 for k in df.samples.keys() if k.aset_name == un))
            buf.write(f'|  - "{un}": {count} \n')

        buf.write('|---------- \n')
        buf.write(f'| Metadata: {len(df.metadata)} \n')
        buf.write(' \n')
        return buf

    buf = StringIO()
    buf.write('============ \n')
    buf.write(f'| Branch: {branch_name} \n')
    buf.write(' \n')
    for changes, changeType in zip(diff, diff.__annotations__.keys()):
        buf.write('============ \n')
        buf.write(f'| {changeType.upper()} \n')
        change_buf = _diff_info(changes)
        buf.write(change_buf.getvalue())
    return buf