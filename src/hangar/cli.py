"""
Module that contains the command line app.

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


@click.group()
def main():
    pass


@main.command()
@click.option('--name', prompt='User Name', help='first and last name of user')
@click.option('--email', prompt='User Email', help='email address of the user')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite a repository if it exists at the current path')
def init(name, email, overwrite):
    '''Initialize an empty repository at the current path
    '''
    P = os.getcwd()
    repo = Repository(path=P)
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
def clone(remote, name, email, overwrite):
    '''Initialize a repository at the current path and fetch data records from REMOTE server.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    repo.clone(
        user_name=name,
        user_email=email,
        remote_address=remote,
        remove_old=overwrite)
    click.echo(f'Hangar repository initialized at {P}')


@main.command(name='fetch')
@click.argument('remote', nargs=1, required=True)  # help='name of the remote server')
@click.argument('branch', nargs=1, required=True)  # help='branch name to fetch')
def fetch_records(remote, branch):
    '''Retrieve the commit history records for BRANCH from the REMOTE server.

    This method does not fetch the data associated with the commits. See
    `fetch-data` to download the tensor data corresponding to a commit.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    bName = repo.remote.fetch(remote=remote, branch=branch)
    click.echo(f'Fetch to Branch Name: {bName}')


@main.command(name='fetch-data')
@click.argument('remote', nargs=1, required=True)  # help='name of the remote server')
@click.argument('commit', nargs=1, required=True)  # help='commit hash for which data should be retrieved')
def fetch_data(remote, commit):
    '''Download the tensor data from the REMOTE server referenced by COMMIT.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    commit_hash = repo.remote.fetch_data(remote=remote, commit=commit)
    click.echo(f'Retrieved data for commit hash: {commit_hash}')


@main.command()
@click.argument('remote', nargs=1, required=True)  # help='name of the remote server')
@click.argument('branch', nargs=1, required=True)  # help='branch name to push')
def push(remote, branch):
    '''Upload local BRANCH commit history / data to REMOTE server.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    commit_hash = repo.remote.push(remote=remote, branch=branch)
    click.echo(f'Push data for commit hash: {commit_hash}')


@main.group()
def remote():
    pass

@remote.command(name='list')
def list_remotes():
    '''List all recorded remotes.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.list_all())


@remote.command(name='add')
@click.argument('name', nargs=1, required=True)  # help='name of the remote repository')
@click.argument('address', nargs=1, required=True)  # help='location where the remote can be accessed')
def add_remote(name, address):
    '''Add a new remote server NAME with url ADDRESS to the local client.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.add(name=name, address=address))


@remote.command(name='remove')
@click.argument('name', nargs=1, required=True)  # help='name of the remote repository')
def remove_remote(name):
    '''Remove the remote server NAME from the local client.
    '''
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.remote.remove(name=name))


@main.command(help='show a summary of the repository')
@click.option('-b', nargs=1, required=False, help='name of the branch to show the head commit details of')
@click.option('-c', nargs=1, required=False, help='commit hash to show the summary of')
def summary(b, c):
    P = os.getcwd()
    repo = Repository(path=P)
    if c:
        click.echo(repo.summary(commit=c))
    elif b:
        click.echo(repo.summary(branch_name=b))
    else:
        click.echo(repo.summary())


@main.command(help='show the commit log graph')
@click.option('-b', required=False, default=None, help='branch name')
def log(b):
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.log(branch_name=b))


@main.command(help='list or create branches')
@click.option('-l', is_flag=True, help='list the branches in the repository')
@click.option('-b', nargs=1, required=False, help='create branch from HEAD commit with provided name')
def branch(l, b):
    P = os.getcwd()
    repo = Repository(path=P)
    if l:
        click.echo(repo.list_branches())
    elif b:
        succ = repo.create_branch(b)
        click.echo(f'create branch operation success: {succ}')


@main.command(help='start a hangar server at the given location')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite the hangar server instance if it exists at the current path.')
@click.option('--ip', default='localhost', help='the ip to start the server on. default is `localhost`')
@click.option('--port', default='50051', help='port to start the server on. default in `50051`')
def server(overwrite, ip, port):
    P = os.getcwd()
    ip_port = f'{ip}:{port}'
    server, hangserver, channel_address = serve(P, overwrite, channel_address=ip_port)
    server.start()
    click.echo(f'Hangar Server Started')
    click.echo(f'* Start Time: {time.asctime()}')
    click.echo(f'* Base Directory Path: {P}')
    click.echo(f'* Operating on `IP_ADDRESS:PORT`: {channel_address}')
    try:
        while True:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        click.echo(f'Server Stopped at Time: {time.asctime()}')
        hangserver.env._close_environments()
        server.stop(0)


@main.command(
    name='db-view',
    help='display the key/value record pair details from the lmdb database')
@click.option('-a', is_flag=True, help='display all dbs in the repository')
@click.option('-b', is_flag=True, help='display the branch/heads db')
@click.option('-r', is_flag=True, help='display the references db')
@click.option('-d', is_flag=True, help='display the data hash db')
@click.option('-m', is_flag=True, help='display the metadata hash db')
@click.option('-s', is_flag=True, help='display the stage record db')
@click.option('-z', is_flag=True, help='display the staged hash record db')
@click.option('--limit', default=30, help='limit the amount of records displayed before truncation')
def lmdb_record_details(a, b, r, d, m, s, z, limit):
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

    envs = Environments(repo_path=repo_path)
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