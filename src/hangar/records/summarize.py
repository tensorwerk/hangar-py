import time
from io import StringIO

from . import heads
from . import commiting
from ..context import TxnRegister
from ..utils import format_bytes, file_size, folder_size


def list_history(refenv, branchenv, branch_name=None, commit_hash=None):
    '''Traverse commit history to specifying ancestor DAG and all ancestor specs.

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
    '''

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


'''
interaction
--------------
'''


def details(env):

    print('')
    print('======================')
    print('Branch')
    print(f'File Size: {format_bytes(file_size(env.branchenv.path()))}')
    print('======================')
    print('')
    branchtxn = TxnRegister().begin_reader_txn(env.branchenv)
    with branchtxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 100:
                break
            print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.branchenv)

    print('')
    print('======================')
    print('Label')
    print(f'File Size: {format_bytes(file_size(env.labelenv.path()))}')
    print('======================')
    print('')
    labeltxn = TxnRegister().begin_reader_txn(env.labelenv)
    with labeltxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 200:
                break
            if len(value) >= 100:
                print(key, 'long binary')
            else:
                print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.labelenv)

    print('')
    print('======================')
    print('HASH')
    print(f'File Size: {format_bytes(file_size(env.hashenv.path()))}')
    print('======================')
    print('')
    hashtxn = TxnRegister().begin_reader_txn(env.hashenv)
    with hashtxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 100:
                break
            print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.hashenv)

    print('')
    print('======================')
    print('STAGE HASH')
    print(f'File Size: {format_bytes(file_size(env.stagehashenv.path()))}')
    print('======================')
    print('')
    stagehashtxn = TxnRegister().begin_reader_txn(env.stagehashenv)
    with stagehashtxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 100:
                break
            print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.stagehashenv)

    print('')
    print('======================')
    print('Commit')
    print(f'File Size: {format_bytes(file_size(env.refenv.path()))}')
    print('======================')
    print('')
    reftxn = TxnRegister().begin_reader_txn(env.refenv)
    with reftxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 200:
                break
            if len(value) >= 100:
                print(key, 'long binary')
            else:
                print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.refenv)

    print('')
    print('======================')
    print('Stage')
    print(f'File Size: {format_bytes(file_size(env.stageenv.path()))}')
    print('======================')
    print('')
    stagetxn = TxnRegister().begin_reader_txn(env.stageenv)
    with stagetxn.cursor() as cursor:
        count = 0
        for key, value in cursor:
            if count >= 100:
                break
            print(key, value)
            count += 1
    cursor.close()
    TxnRegister().abort_reader_txn(env.stageenv)

    for commit, commitenv in env.cmtenv.items():
        print('')
        print('======================')
        print(f'Commit: {commit}')
        print(f'File Size: {format_bytes(file_size(commitenv.path()))}')
        print('======================')
        print('')
        cmttxn = TxnRegister().begin_reader_txn(commitenv)
        with cmttxn.cursor() as cursor:
            count = 0
            for key, value in cursor:
                if count >= 100:
                    break
                print(key, value)
                count += 1
        cursor.close()
        TxnRegister().abort_reader_txn(commitenv)

    return


def summary(env, *, branch_name='', commit=''):
    '''Summary of data set stored in repository.

    Parameters
    ----------
    env : :class:`Environments`
        class which contains all of the lmdb environments pre-initialized for use.
    commit : str
        commit hash to query. if left empty, HEAD commit is used (Default value = '')
    branch_name : str
        branch name to query, if left empty, HEAD will be used. (Default value = '')

    Returns
    -------
    dict:
        the contents of the commit ref at the queried commit.
    '''
    if commit != '':
        cmt = commit
    elif branch_name != '':
        cmt = heads.get_branch_head_commit(env.branchenv, branch_name)
    else:
        headBranch = heads.get_staging_branch_head(env.branchenv)
        cmt = heads.get_branch_head_commit(env.branchenv, headBranch)

    spec = commiting.get_commit_spec(env.refenv, cmt)._asdict()
    if cmt == '':
        print('No commits made')
        return {}

    res = commiting.get_commit_ref_contents(env.refenv, cmt)
    nbytes = folder_size(env.repo_path, recurse_directories=True)
    humanBytes = format_bytes(nbytes)
    buf = StringIO()
    buf.write(f'Summary of Contents Contained in Data Repository \n')
    buf.write(f' \n')
    buf.write(f'================== \n')
    buf.write(f'| Repository Info \n')
    buf.write(f'|----------------- \n')
    buf.write(f'|  Directory: {env.repo_path} \n')
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

    buf.write(f'|  Number of Named Datasets: {len(res["datasets"])} \n')
    for dsetn in res['datasets']:
        buf.write(f'|\n')
        buf.write(f'|  * Dataset Name: {dsetn} \n')
        buf.write(f'|    Num Arrays: {len(res["datasets"][dsetn]["data"])} \n')

        buf.write(f'|    Details: \n')
        for k, v in res["datasets"][dsetn]["schema"]._asdict().items():
            buf.write(f'|    - {k}: {v} \n')

    buf.write(f' \n')
    buf.write(f'================== \n')
    buf.write(f'| Metadata: \n')
    buf.write(f'|----------------- \n')
    buf.write(f'|  Number of Keys: {len(res["metadata"])} \n')

    return buf, res