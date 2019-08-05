"""Module that contains the command line app.

Why does this file exist, and why not put this in __main__?

    You might be tempted to import things from __main__ later, but that will cause
    problems: the code will get executed twice:

    - When you run `python -mhangar` python will execute
      ``__main__.py`` as a script. That means there won't be any
      ``hangar.__main__`` in ``sys.modules``.
    - When you import __main__ it will get executed again (as a module) because
      there's no ``hangar.__main__`` in ``sys.modules``.

    Also see (1) from http://click.pocoo.org/7/setuptools/#setuptools-integration
"""
import os
import time

import click

from hangar import Repository
from hangar import serve
from hangar.records.commiting import expand_short_commit_digest


@click.group(no_args_is_help=True, add_help_option=True, invoke_without_command=True)
@click.option('--version', '-v', is_flag=True, default=False, required=False,
              help='display the Hangar version currently installed')
@click.pass_context
def main(ctx, version):
    if version:
        import hangar
        click.echo(hangar.__version__)
    pass


@main.command()
@click.option('--name', prompt='User Name', help='first and last name of user')
@click.option('--email', prompt='User Email', help='email address of the user')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite a repository if it exists at the current path')
@click.pass_context
def init(ctx, name, email, overwrite):
    """Initialize an empty repository at the current path
    """
    P = os.getcwd()
    repo = Repository(path=P, exists=False)
    try:
        repo.init(user_name=name, user_email=email, remove_old=overwrite)
        click.echo(f'Hangar repository initialized at {P}')
    except OSError as e:
        click.echo(e)


@main.command()
@click.option('--name', prompt='User Name', help='first and last name of user')
@click.option('--email', prompt='User Email', help='email address of the user')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite a repository if it exists at the current path')
@click.argument('remote', nargs=1, required=True)
@click.pass_context
def clone(ctx, remote, name, email, overwrite):
    """Initialize a repository at the current path and fetch updated records from REMOTE.

    Note: This method does not actually download the data to disk. Please look
    into the ``fetch-data`` command.
    """
    P = os.getcwd()
    repo = Repository(path=P, exists=False)
    repo.clone(user_name=name,
               user_email=email,
               remote_address=remote,
               remove_old=overwrite)
    click.echo(f'Hangar repository initialized at {P}')


@main.command(name='fetch')
@click.argument('remote', nargs=1, required=True)
@click.argument('branch', nargs=1, required=True)
@click.pass_context
def fetch_records(ctx, remote, branch):
    """Retrieve the commit history from REMOTE for BRANCH.

    This method does not fetch the data associated with the commits. See
    `fetch-data` to download the tensor data corresponding to a commit.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    bName = repo.remote.fetch(remote=remote, branch=branch)
    click.echo(f'Fetch to Branch Name: {bName}')


@main.command(name='fetch-data')
@click.argument('remote', nargs=1, required=True)
@click.argument('startpoint', nargs=1, required=True)
@click.option('--aset', '-d', multiple=True, required=False, default=None,
              help='specify any number of aset keys to fetch data for.')
@click.option('--nbytes', '-n', default=None, required=False,
              help='total amount of data to retrieve in MB/GB.')
@click.option('--all-history', '-a', 'all_', is_flag=True, default=False, required=False,
              help='Retrieve data referenced in every parent commit accessible to the STARTPOINT')
@click.pass_context
def fetch_data(ctx, remote, startpoint, aset, nbytes, all_):
    """Get data from REMOTE referenced by STARTPOINT (short-commit or branch).

    The default behavior is to only download a single commit's data or the HEAD
    commit of a branch. Please review optional arguments for other behaviors
    """
    from hangar.records.heads import get_branch_head_commit, get_staging_branch_head
    from hangar.utils import parse_bytes

    P = os.getcwd()
    repo = Repository(path=P)
    if startpoint is None:
        branch = get_staging_branch_head(repo._env.branchenv)
        commit = get_branch_head_commit(repo._env.branchenv, branch)
        click.echo(f'No startpoint supplied, fetching data of HEAD: {commit} for BRANCH: {branch}')
    elif startpoint in repo.list_branches():
        commit = get_branch_head_commit(repo._env.branchenv, startpoint)
        click.echo(f'Fetching data for HEAD: {commit} of STARTPOINT BRANCH: {startpoint}')
    else:
        commit = expand_short_commit_digest(repo._env.refenv, startpoint)
        click.echo(f'Fetching data for STARTPOINT HEAD: {commit}')

    click.echo(f'aset argument: {aset}')
    try:
        max_nbytes = parse_bytes(nbytes)
        click.echo(f'nbytes argument: {max_nbytes}')
    except AttributeError:
        max_nbytes = None

    if len(aset) == 0:
        aset = None

    commits = repo.remote.fetch_data(remote=remote,
                                     commit=commit,
                                     arrayset_names=aset,
                                     max_num_bytes=max_nbytes,
                                     retrieve_all_history=all_)
    click.echo(f'completed data for commits: {commits}')


@main.command()
@click.argument('remote', nargs=1, required=True)
@click.argument('branch', nargs=1, required=True)
@click.pass_context
def push(ctx, remote, branch):
    """Upload local BRANCH commit history / data to REMOTE server.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    commit_hash = repo.remote.push(remote=remote, branch=branch)
    click.echo(f'Push data for commit hash: {commit_hash}')


@main.group(no_args_is_help=True, add_help_option=True)
@click.pass_context
def remote(ctx):
    """Operations for working with remote server references
    """
    pass


@remote.command(name='list')
@click.pass_context
def list_remotes(ctx):
    """List all remote repository records.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.list_all())


@remote.command(name='add')
@click.argument('name', nargs=1, required=True)
@click.argument('address', nargs=1, required=True)
@click.pass_context
def add_remote(ctx, name, address):
    """Add a new remote server NAME with url ADDRESS to the local client.

    This name must be unique. In order to update an old remote, please remove it
    and re-add the remote NAME / ADDRESS combination
    """
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.add(name=name, address=address))


@remote.command(name='remove')
@click.argument('name', nargs=1, required=True)
@click.pass_context
def remove_remote(ctx, name):
    """Remove the remote server NAME from the local client.

    This will not remove any tracked remote reference branches.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.remove(name=name))


@main.command()
@click.argument('startpoint', nargs=1, required=False)
@click.pass_context
def summary(ctx, startpoint):
    """Display content summary at STARTPOINT (short-digest or branch).

    If no argument is passed in, the staging area branch HEAD wil be used as the
    starting point. In order to recieve a machine readable, and more complete
    version of this information, please see the ``Repository.summary()`` method
    of the API.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    if startpoint is None:
        click.echo(repo.summary())
    elif startpoint in repo.list_branches():
        click.echo(repo.summary(branch=startpoint))
    else:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
        click.echo(repo.summary(commit=base_commit))


@main.command()
@click.argument('startpoint', required=False, default=None)
@click.pass_context
def log(ctx, startpoint):
    """Display commit graph starting at STARTPOINT (short-digest or name)

    If no argument is passed in, the staging area branch HEAD will be used as the
    starting point.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    if startpoint is None:
        click.echo(repo.log())
    elif startpoint in repo.list_branches():
        click.echo(repo.log(branch=startpoint))
    else:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
        click.echo(repo.log(commit=base_commit))


@main.group(no_args_is_help=True, add_help_option=True)
@click.pass_context
def branch(ctx):
    """operate on and list branch pointers.
    """
    pass


@branch.command(name='list')
@click.pass_context
def branch_list(ctx):
    """list all branch names

    Includes both remote branches as well as local branches.
    """
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.list_branches())


@branch.command(name='create')
@click.argument('name', nargs=1, required=True)
@click.argument('startpoint', nargs=1, default=None, required=False)
@click.pass_context
def branch_create(ctx, name, startpoint):
    """Create a branch with NAME at STARTPOINT (short-digest or branch)

    If no STARTPOINT is provided, the new branch is positioned at the HEAD of
    the staging area branch, automatically.
    """
    from hangar.records.heads import get_branch_head_commit, get_staging_branch_head

    P = os.getcwd()
    repo = Repository(path=P)
    branch_names = repo.list_branches()
    if name in branch_names:
        raise ValueError(f'branch name: {name} already exists')

    if startpoint is None:
        branch = get_staging_branch_head(repo._env.branchenv)
        base_commit = get_branch_head_commit(repo._env.branchenv, branch)
    elif startpoint in branch_names:
        base_commit = get_branch_head_commit(repo._env.branchenv, startpoint)
    else:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)

    click.echo(f'BRANCH: ' +
               repo.create_branch(name, base_commit=base_commit) +
               f' HEAD: {base_commit}')


@main.command()
@click.option('--overwrite', is_flag=True, default=False,
              help='overwrite the hangar server instance if it exists at the current path.')
@click.option('--ip', default='localhost', show_default=True,
              help='the ip to start the server on. default is `localhost`')
@click.option('--port', default='50051', show_default=True,
              help='port to start the server on. default in `50051`')
@click.option('--timeout', default=60*60*24, required=False, show_default=True,
              help='time (in seconds) before server is stopped automatically')
def server(overwrite, ip, port, timeout):
    """Start a hangar server, initializing one if does not exist.

    The server is configured to top working in 24 Hours from the time it was
    initially started. To modify this value, please see the ``--timeout``
    parameter.

    The hangar server directory layout, contents, and access conventions are
    similar, though significantly different enough to the regular user "client"
    implementation that it is not possible to fully access all information via
    regular API methods. These changes occur as a result of the uniformity of
    operations promised by both the RPC structure and negotiations between the
    client/server upon connection.

    More simply put, we know more, so we can optimize access more; similar, but
    not identical.
    """
    P = os.getcwd()
    ip_port = f'{ip}:{port}'
    server, hangserver, channel_address = serve(P, overwrite, channel_address=ip_port)
    server.start()
    click.echo(f'Hangar Server Started')
    click.echo(f'* Start Time: {time.asctime()}')
    click.echo(f'* Base Directory Path: {P}')
    click.echo(f'* Operating on `IP_ADDRESS:PORT`: {channel_address}')
    try:
        startTime = time.time()
        while True:
            time.sleep(0.1)
            if time.time() - startTime > timeout:
                raise SystemExit
    except (KeyboardInterrupt, SystemExit):
        click.echo(f'Server Stopped at Time: {time.asctime()}')
        hangserver.env._close_environments()
        server.stop(0)


@main.command(name='db-view', hidden=True)
@click.option('-a', is_flag=True, help='display all dbs in the repository')
@click.option('-b', is_flag=True, help='display the branch/heads db')
@click.option('-r', is_flag=True, help='display the references db')
@click.option('-d', is_flag=True, help='display the data hash db')
@click.option('-m', is_flag=True, help='display the metadata hash db')
@click.option('-s', is_flag=True, help='display the stage record db')
@click.option('-z', is_flag=True, help='display the staged hash record db')
@click.option('--limit', default=30, help='limit the amount of records displayed before truncation')
def lmdb_record_details(a, b, r, d, m, s, z, limit):  # pragma: no cover
    """DEVELOPER TOOL ONLY

    display key/value pairs making up the dbs
    """
    from hangar.context import Environments
    from hangar.records.summarize import details
    from hangar import constants as c
    P = os.getcwd()
    if os.path.isdir(os.path.join(P, c.DIR_HANGAR_SERVER)):
        repo_path = os.path.join(P, c.DIR_HANGAR_SERVER)
    elif os.path.isdir(os.path.join(P, c.DIR_HANGAR)):
        repo_path = os.path.join(P, c.DIR_HANGAR)
    else:
        click.echo(f'NO HANGAR INSTALLATION AT PATH: {P}')
        return

    envs = Environments(pth=repo_path)
    if a:
        b, r, d, m, s, z = True, True, True, True, True, True
    if b:
        click.echo(details(envs.branchenv, line_limit=limit).getvalue())
    if r:
        click.echo(details(envs.refenv, line_limit=limit).getvalue())
    if d:
        click.echo(details(envs.hashenv, line_limit=limit).getvalue())
    if m:
        click.echo(details(envs.labelenv, line_limit=limit).getvalue())
    if s:
        click.echo(details(envs.stageenv, line_limit=limit).getvalue())
    if z:
        click.echo(details(envs.stagehashenv, line_limit=limit).getvalue())
