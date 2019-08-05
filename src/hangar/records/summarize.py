import os
import time
from io import StringIO

import lmdb

from . import heads, commiting
from ..context import TxnRegister
from ..utils import format_bytes, file_size, folder_size


def list_history(refenv, branchenv, branch_name=None, commit_hash=None):
    """Traverse commit history to specifying ancestor DAG and all ancestor specs.

    Parameters
    ----------
    refenv : lmdb.Environment
        environment containing all repository commit data.
    branchenv : lmdb.Environment
        environment contianing the current staging head branch and branch head
        commit hashes
    branch_name : string, optional
        if specified, get the history starting at the head commit of this named
        branch (the default is None, which will use the `commit_hash` arg if
        available, or staging area head)
    commit_hash : string, optional
        if specified, get the history starting at this specific commit,
        overrides branch name if both are specified (the default is `None`, which
        will use the branch_name arg if available, or staging area head)

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


def details(env: lmdb.Environment, line_limit=100) -> StringIO:  # pragma: no cover
    """Print the details of an lmdb environment to stdout

    Parameters
    ----------
    env : lmdb.Environment
        environment handle to print records of
    line_limit : int, optional
        limit to the amount of record lines printed, by default 100

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
                if len(value) >= 100:
                    buf.write(f'{key} long binary\n')
                else:
                    buf.write(f'{key} {value}\n')
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env)
    return buf


def summary(env, *, branch='', commit=''):
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
    dict:
        the contents of the commit ref at the queried commit.
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

    res = commiting.get_commit_ref_contents(env.refenv, cmt)
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

    buf.write(f'|  Number of Named Arraysets: {len(res["arraysets"])} \n')
    for asetn in res['arraysets']:
        buf.write(f'|\n')
        buf.write(f'|  * Arrayset Name: {asetn} \n')
        buf.write(f'|    Num Arrays: {len(res["arraysets"][asetn]["data"])} \n')

        buf.write(f'|    Details: \n')
        for k, v in res["arraysets"][asetn]["schema"]._asdict().items():
            buf.write(f'|    - {k}: {v} \n')

    buf.write(f' \n')
    buf.write(f'================== \n')
    buf.write(f'| Metadata: \n')
    buf.write(f'|----------------- \n')
    buf.write(f'|  Number of Keys: {len(res["metadata"])} \n')

    return buf, res
