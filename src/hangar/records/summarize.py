from pathlib import Path
import time
from io import StringIO

import lmdb

from .commiting import (
    get_commit_ancestors_graph,
    get_commit_spec,
    tmp_cmt_env,
)
from .heads import (
    get_staging_branch_head,
    get_branch_head_commit,
    commit_hash_to_branch_name_map,
)
from .queries import RecordQuery
from .hashs import HashQuery
from ..diff import DiffOut, Changes
from ..txnctx import TxnRegister
from ..utils import format_bytes, file_size, folder_size, unique_everseen
from ..diagnostics import graphing


def log(branchenv: lmdb.Environment,
        refenv: lmdb.Environment,
        branch: str = None,
        commit: str = None,
        *,
        return_contents: bool = False,
        show_time: bool = False,
        show_user: bool = False):
    """Displays a pretty printed commit log graph to the terminal.

    .. note::

        For programatic access, the return_contents value can be set to true
        which will retrieve relevant commit specifications as dictionary
        elements.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db storing information on named branch HEADS
    refenv : lmdb.Environment
        db storing full commit history refs (compressed).
    branch : str, optional
        The name of the branch to start the log process from. (Default value
        = None)
    commit : str, optional
        The commit hash to start the log process from. (Default value = None)
    return_contents : bool, optional, kwarg only
        If true, return the commit graph specifications in a dictionary
        suitable for programatic access/evaluation.
    show_time : bool, optional, kwarg only
        If true and return_contents is False, show the time of each commit
        on the printed log graph
    show_user : bool, optional, kwarg only
        If true and return_contents is False, show the committer of each
        commit on the printed log graph
    Returns
    -------
    Optional[dict]
        Dict containing the commit ancestor graph, and all specifications.
    """
    res = list_history(
        refenv=refenv,
        branchenv=branchenv,
        branch_name=branch,
        commit_hash=commit)
    branchMap = dict(commit_hash_to_branch_name_map(branchenv=branchenv))

    if return_contents:
        for digest in list(branchMap.keys()):
            if digest not in res['order']:
                del branchMap[digest]
        res['branch_heads'] = branchMap
        return res
    else:
        g = graphing.Graph()
        g.show_nodes(dag=res['ancestors'],
                     spec=res['specs'],
                     branch=branchMap,
                     start=res['head'],
                     order=res['order'],
                     show_time=show_time,
                     show_user=show_user)


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
        head_commit = get_branch_head_commit(branchenv=branchenv, branch_name=branch_name)
    else:
        head_branch = get_staging_branch_head(branchenv)
        head_commit = get_branch_head_commit(branchenv, head_branch)

    ancestors = get_commit_ancestors_graph(
        refenv=refenv, starting_commit=head_commit)

    commitSpecs = {}
    for commit in ancestors.keys():
        commitSpecs[commit] = dict(get_commit_spec(refenv, commit_hash=commit)._asdict())

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
    buf.write(f'{Path(env.path()).name}\n')
    try:
        buf.write(f'File Size: {format_bytes(file_size(Path(env.path())))}\n')
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
    env : :class:`..context.Environments`
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
        cmt = get_branch_head_commit(env.branchenv, branch)
    else:
        headBranch = get_staging_branch_head(env.branchenv)
        cmt = get_branch_head_commit(env.branchenv, headBranch)

    spec = get_commit_spec(env.refenv, cmt)._asdict()
    if cmt == '':
        buf = StringIO()
        buf.write('No commits made')
        return buf

    def _schema_digest_spec_dict(hashenv, digest):
        hq = HashQuery(hashenv)
        res = hq.get_schema_digest_spec(digest)
        return res

    with tmp_cmt_env(env.refenv, cmt) as cmtrefenv:
        query = RecordQuery(cmtrefenv)

        nbytes = folder_size(env.repo_path, recurse=True)
        humanBytes = format_bytes(nbytes)
        buf = StringIO()
        buf.write(f'Summary of Contents Contained in Data Repository \n')
        buf.write(f' \n')
        buf.write(f'================== \n')
        buf.write(f'| Repository Info \n')
        buf.write(f'|----------------- \n')
        buf.write(f'|  Base Directory: {str(env.repo_path.parent)} \n')
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

        buf.write(f'|  Number of Named Columns: {query.column_count()} \n')
        for asetn, asetnSchema in query.schema_specs().items():
            buf.write(f'|\n')
            buf.write(f'|  * Column Name: {asetn} \n')
            buf.write(f'|    Num Data Pieces: {query.column_data_count(asetn.column)} \n')

            buf.write(f'|    Details: \n')
            schema_dict = _schema_digest_spec_dict(env.hashenv, asetnSchema.digest)
            for k, v in schema_dict.items():
                buf.write(f'|    - {k}: {v} \n')

    return buf


def status(hashenv: lmdb.Environment, branch_name: str, diff: DiffOut) -> StringIO:
    """Format human readable string buffer of changes in a staging area

    Parameters
    ----------
    hashenv : lmdb.Environment
        hashenv to pull usefull schema spec info from.
    branch_name : str
        Name of the branch the diff is from.
    diff : DiffOut
        diff struct tuple returned from standard diff tool.

    Returns
    -------
    StringIO
        Buffer containing human readable printable string of change summary
    """
    def _schema_digest_spec_dict(digest):
        hq = HashQuery(hashenv)
        res = hq.get_schema_digest_spec(digest)
        return res

    def _diff_info(df: Changes) -> StringIO:
        """Format buffer for each of `ADDED`, `DELETED`, `MUTATED` changes
        """
        buf = StringIO()
        buf.write(f'|---------- \n')
        buf.write(f'| Schema: {len(df.schema)} \n')
        for k, v in df.schema.items():
            digest = v.digest
            buf.write(f'|  - "{k.column}": \n')
            buf.write(f'|       digest="{digest}" \n')
            schema_spec = _schema_digest_spec_dict(digest)
            for schema_key, schema_val in schema_spec.items():
                buf.write(f'|       {schema_key}: {schema_val} \n')

        buf.write('|---------- \n')
        buf.write(f'| Samples: {len(df.samples)} \n')
        unique = unique_everseen(df.samples, lambda x: x.column)
        for u in unique:
            un = u.column
            count = sum((1 for k in df.samples if k.column == un))
            buf.write(f'|  - "{un}": {count} \n')
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
