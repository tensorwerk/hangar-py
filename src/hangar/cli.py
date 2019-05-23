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

    Also see (1) from http://click.pocoo.org/5/setuptools/#setuptools-integration
"""
import os
import logging

import click
from hangar import Repository
from hangar import serve

logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


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


@main.command(help='show the commit log')
@click.option('-b', required=False, default=None, help='branch name')
def log(b):
    P = os.getcwd()
    repo = Repository(path=P)
    click.echo(repo.log(branch_name=b))


@main.command(help='list or create branches')
@click.option('-l', is_flag=True, help='list the branches in the repository')
@click.option('-b', nargs=1, required=False, help='create branch from HEAD commit with provided name')
def branch(l, b):
    if l:
        P = os.getcwd()
        repo = Repository(path=P)
        click.echo(repo.list_branch_names())
    if b:
        P = os.getcwd()
        repo = Repository(path=P)
        succ = repo.create_branch(b)
        click.echo(f'create branch operation success: {succ}')


@main.command(help='initialize environment')
@click.option('-uname', nargs=2, required=False, default='', help='first and last name of user')
@click.option('-email', nargs=1, required=False, default='', help='email of the user')
@click.option('--overwrite', is_flag=True, help='overwrite a repository if it exists at the current path')
def init(uname, email, overwrite):
    P = os.getcwd()
    if isinstance(uname, (list, tuple)):
        uname = ' '.join(uname)
    repo = Repository(path=P)
    if overwrite:
        repoDir = repo.init(user_name=uname, user_email=email, remove_old=True)
    else:
        try:
            repoDir = repo.init(user_name=uname, user_email=email, remove_old=False)
        except OSError as e:
            click.echo(e)


@main.command(help='clone an environment into the given path')
@click.option('-uname', nargs=2, required=False, default='', help='first and last name of user')
@click.option('-email', nargs=1, required=False, default='', help='email of the user')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite a repository if it exists at the current path')
@click.argument('remote', nargs=1, required=True)
def clone(remote, uname, email, overwrite):
    if isinstance(uname, (list, tuple)):
        uname = ' '.join(uname)
    P = os.getcwd()
    repo = Repository(path=P)
    repo.clone(
        user_name=uname,
        user_email=email,
        remote_address=remote,
        remove_old=overwrite)
    click.echo(f'Hangar repository initialized at {P}')


@main.command(help='start a hangar server at the given location')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite the hangar server instance if it exists at the current path.')
def server(overwrite):
    P = os.getcwd()
    if overwrite:
        serve(P, True)
    else:
        serve(P, False)