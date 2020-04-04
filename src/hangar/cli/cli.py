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
from pathlib import Path

import click
import numpy as np

from hangar import Repository, __version__

from .utils import parse_custom_arguments, StrOrIntType


pass_repo = click.make_pass_decorator(Repository, ensure=True)


@click.group(no_args_is_help=True, add_help_option=True, invoke_without_command=True)
@click.version_option(version=__version__, help='display current Hangar Version')
@click.pass_context
def main(ctx):  # pragma: no cover
    P = os.getcwd()
    ctx.obj = Repository(path=P, exists=False)


# -------------------------------- Init ---------------------------------------


@main.command()
@click.option('--name', prompt='User Name', help='first and last name of user')
@click.option('--email', prompt='User Email', help='email address of the user')
@click.option('--overwrite', is_flag=True, default=False,
              help='overwrite a repository if it exists at the current path')
@pass_repo
def init(repo: Repository, name, email, overwrite):
    """Initialize an empty repository at the current path
    """
    if repo.initialized and (not overwrite):
        click.echo(f'Repo already exists at: {repo.path}')
    else:
        repo.init(user_name=name, user_email=email, remove_old=overwrite)


# -------------------------- Writer Lock -------------------------------------


@main.command(name='writer-lock')
@click.option('--force-release', 'force_release_', is_flag=True, default=False,
              help='force release writer lock from the CLI.')
@pass_repo
def writer_lock_held(repo: Repository, force_release_):
    """Determine if the writer lock is held for a repository.

    Passing the --force-release flag will instantly release the writer lock,
    invalidating any process which currently holds it.
    """
    if force_release_:
        repo.force_release_writer_lock()
        click.echo(f'Success force release of writer lock.')
    else:
        if repo.writer_lock_held:
            click.echo(f'Writer lock is held.')
        else:
            click.echo(f'Writer lock is available.')



# -------------------------- Checkout Writer ----------------------------------


@main.command()
@click.argument('branchname', nargs=1, required=True)
@pass_repo
def checkout(repo: Repository, branchname):
    """Checkout writer head branch at BRANCHNAME.

    This method requires that no process currently holds the writer lock.
    In addition, it requires that the contents of the staging area are
    'CLEAN' (no changes have been staged).
    """
    try:
        co = repo.checkout(write=True, branch=branchname)
        co.close()
        click.echo(f'Writer checkout head set to branch: {branchname}')
    except (ValueError, PermissionError) as e:
        raise click.ClickException(e)


@main.command()
@click.option('--message', '-m', multiple=True,
              help=('The commit message. If provided multiple times '
                    'each argument gets converted into a new line.'))
@pass_repo
def commit(repo: Repository, message):
    """Commits outstanding changes.

    Commit changes to the given files into the repository. You will need to
    'push' to push up your changes to other repositories.
    """
    from hangar.records.summarize import status

    co = repo.checkout(write=True)
    try:
        if not message:
            diff = co.diff.staged()
            status_txt = status(co._hashenv, co.branch_name, diff.diff)
            status_txt.seek(0)
            marker = '# Changes To Be committed: \n'
            hint = ['\n', '\n', marker, '# \n']
            for line in status_txt.readlines():
                hint.append(f'# {line}')
            # open default system editor
            message = click.edit(''.join(hint))
            if message is None:
                click.echo('Aborted!')
                return
            msg = message.split(marker)[0].rstrip()
            if not msg:
                click.echo('Aborted! Empty commit message')
                return
        else:
            msg = '\n'.join(message)

        click.echo('Commit message:\n' + msg)
        try:
            digest = co.commit(msg)
            click.echo(f'Commit Successful. Digest: {digest}')
        except RuntimeError as e:
            raise click.ClickException(e)
    finally:
        co.close()


# -------------------------- Column Interactor ------------------------------


@main.group(no_args_is_help=True, add_help_option=True)
@click.pass_context
def column(ctx):  # pragma: no cover
    """Operations for working with columns in the writer checkout.
    """
    pass


@column.command(name='create')
@click.option('--variable-shape', 'variable_', is_flag=True, default=False,
              help='flag indicating sample dimensions can be any size up to max shape.')
@click.option('--contains-subsamples', 'subsamples_', is_flag=True, default=False,
              help=('flag indicating if this is a column which nests multiple '
                    'subsamples under a common sample key.'))
@click.argument('name', nargs=1, type=click.STRING, required=True)
@click.argument('dtype', nargs=1, type=click.Choice([
    'UINT8', 'INT8', 'UINT16', 'INT16', 'UINT32', 'INT32',
    'UINT64', 'INT64', 'FLOAT16', 'FLOAT32', 'FLOAT64', 'STR']), required=True)
@click.argument('shape', nargs=-1, type=click.INT, required=False)
@pass_repo
def create_column(repo: Repository, name, dtype, shape, variable_, subsamples_):
    """Create an column with NAME and DTYPE of SHAPE.

    The column will be created in the staging area / branch last used by a
    writer-checkout. Valid NAMEs contain only ascii letters and [``'.'``,
    ``'_'``, ``'-'``] (no whitespace). The DTYPE must be one of [``'UINT8'``,
    ``'INT8'``, ``'UINT16'``, ``'INT16'``, ``'UINT32'``, ``'INT32'``,
    ``'UINT64'``, ``'INT64'``, ``'FLOAT16'``, ``'FLOAT32'``, ``'FLOAT64'``,
    ``'STR'``].

    If a ndarray dtype is specified (not 'STR'), then the SHAPE must be the
    last argument(s) specified, where each dimension size is identified by
    a (space seperated) list of numbers.

    Examples:

    To specify, an column for some training images of dtype uint8 and shape
    (256, 256, 3) we should say:

       .. code-block:: console

          $ hangar column create train_images UINT8 256 256 3

    To specify that the samples can be variably shaped (have any dimension size
    up to the maximum SHAPE specified) we would say:

       .. code-block:: console

          $ hangar column create train_images UINT8 256 256 3 --variable-shape

    or equivalently:

       .. code-block:: console

          $ hangar column create --variable-shape train_images UINT8 256 256 3

    To specify that the column contains a nested set of subsample data under a
    common sample key, the ``--contains-subsamples`` flag can be used.

       .. code-block:: console

          $ hangar column create --contains-subsamples train_images UINT8 256 256 3

    """
    try:
        co = repo.checkout(write=True)
        if dtype == 'STR':
            col = co.add_str_column(name=name, contains_subsamples=subsamples_)
        else:
            col = co.add_ndarray_column(name=name,
                                        shape=shape,
                                        dtype=np.typeDict[dtype.lower()],
                                        variable_shape=variable_,
                                        contains_subsamples=subsamples_)
        click.echo(f'Initialized Column: {col.column}')
    except (ValueError, LookupError, PermissionError) as e:
        raise click.ClickException(e)
    finally:
        try:
            co.close()
        except NameError:
            pass


@column.command(name='remove')
@click.argument('name', nargs=1, type=click.STRING, required=True)
@pass_repo
def remove_column(repo: Repository, name):
    """Delete the column NAME (and all samples) from staging area.

    The column will be removed from the staging area / branch last used by a
    writer-checkout.
    """
    try:
        co = repo.checkout(write=True)
        removed = co.columns.delete(name)
        click.echo(f'Successfully removed column: {removed}')
    except (ValueError, KeyError, PermissionError) as e:
        raise click.ClickException(e)
    finally:
        try:
            co.close()
        except NameError:
            pass


# ---------------------------- Remote Interaction -----------------------------


@main.command()
@click.argument('remote', nargs=1, required=True)
@click.option('--name', prompt='User Name', help='first and last name of user')
@click.option('--email', prompt='User Email', help='email address of the user')
@click.option('--overwrite', is_flag=True, default=False,
              help='overwrite a repository if it exists at the current path')
@pass_repo
def clone(repo: Repository, remote, name, email, overwrite):
    """Initialize a repository at the current path and fetch updated records from REMOTE.

    Note: This method does not actually download the data to disk. Please look
    into the ``fetch-data`` command.
    """
    if repo.initialized and (not overwrite):
        click.echo(f'Repo already exists at: {repo.path}')
    else:
        repo.clone(name, email, remote, remove_old=overwrite)


@main.command(name='fetch')
@click.argument('remote', nargs=1, required=True)
@click.argument('branch', nargs=1, required=True)
@pass_repo
def fetch_records(repo: Repository, remote, branch):
    """Retrieve the commit history from REMOTE for BRANCH.

    This method does not fetch the data associated with the commits. See
    ``fetch-data`` to download the tensor data corresponding to a commit.
    """
    bName = repo.remote.fetch(remote=remote, branch=branch)
    click.echo(f'Fetched branch Name: {bName}')


@main.command(name='fetch-data')
@click.argument('remote', nargs=1, required=True)
@click.argument('startpoint', nargs=1, required=True)
@click.option('--column', '-d', multiple=True, required=False, default=None,
              help='specify any number of column keys to fetch data for.')
@click.option('--nbytes', '-n', default=None, required=False,
              help='total amount of data to retrieve in MB/GB.')
@click.option('--all-history', '-a', 'all_', is_flag=True, default=False, required=False,
              help='Retrieve data referenced in every parent commit accessible to the STARTPOINT')
@pass_repo
def fetch_data(repo: Repository, remote, startpoint, column, nbytes, all_):
    """Get data from REMOTE referenced by STARTPOINT (short-commit or branch).

    The default behavior is to only download a single commit's data or the HEAD
    commit of a branch. Please review optional arguments for other behaviors
    """
    from hangar.records.commiting import expand_short_commit_digest
    from hangar.records.heads import get_branch_head_commit
    from hangar.records.heads import get_staging_branch_head
    from hangar.utils import parse_bytes

    if startpoint is None:
        branch = get_staging_branch_head(repo._env.branchenv)
        commit = get_branch_head_commit(repo._env.branchenv, branch)
    elif startpoint in repo.list_branches():
        commit = get_branch_head_commit(repo._env.branchenv, startpoint)
    else:
        commit = expand_short_commit_digest(repo._env.refenv, startpoint)
    click.echo(f'Fetching data for commit: {commit}')

    try:
        max_nbytes = parse_bytes(nbytes)
    except AttributeError:
        max_nbytes = None
    if len(column) == 0:
        column = None

    commits = repo.remote.fetch_data(remote=remote,
                                     commit=commit,
                                     column_names=column,
                                     max_num_bytes=max_nbytes,
                                     retrieve_all_history=all_)
    click.echo(f'completed data for commits: {commits}')


@main.command()
@click.argument('remote', nargs=1, required=True)
@click.argument('branch', nargs=1, required=True)
@pass_repo
def push(repo: Repository, remote, branch):
    """Upload local BRANCH commit history / data to REMOTE server.
    """
    commit_hash = repo.remote.push(remote=remote, branch=branch)
    click.echo(f'Push data for commit hash: {commit_hash}')


# ----------------------- Remote Server References ----------------------------


@main.group(no_args_is_help=True, add_help_option=True)
@click.pass_context
def remote(ctx):  # pragma: no cover
    """Operations for working with remote server references
    """
    pass


@remote.command(name='list')
@pass_repo
def list_remotes(repo: Repository):
    """List all remote repository records.
    """
    click.echo(repo.remote.list_all())


@remote.command(name='add')
@click.argument('name', nargs=1, required=True)
@click.argument('address', nargs=1, required=True)
@pass_repo
def add_remote(repo: Repository, name, address):
    """Add a new remote server NAME with url ADDRESS to the local client.

    This name must be unique. In order to update an old remote, please remove it
    and re-add the remote NAME / ADDRESS combination
    """
    click.echo(repo.remote.add(name=name, address=address))


@remote.command(name='remove')
@click.argument('name', nargs=1, required=True)
@pass_repo
def remove_remote(repo: Repository, name):
    """Remove the remote server NAME from the local client.

    This will not remove any tracked remote reference branches.
    """
    click.echo(repo.remote.remove(name=name))


# ---------------------------- User Visualizations ----------------------------


@main.command()
@click.argument('dev', nargs=1, required=True)
@click.argument('master', nargs=1, required=False, default=None)
@pass_repo
def diff(repo: Repository, dev, master):
    """Display diff of DEV commit/branch to MASTER commit/branch.

    If no MASTER is specified, then the staging area branch HEAD will
    will be used as the commit digest for MASTER. This operation will
    return a diff which could be interpreted as if you were merging
    the changes in DEV into MASTER.

    TODO: VERIFY ORDER OF OUTPUT IS CORRECT.
    """
    from hangar.records.commiting import expand_short_commit_digest
    from hangar.records.commiting import get_staging_branch_head
    from hangar.records.summarize import status

    if dev not in repo.list_branches():
        dev = expand_short_commit_digest(repo._env.refenv, dev)

    if master is None:
        master = get_staging_branch_head(repo._env.branchenv)
    elif master not in repo.list_branches():
        master = expand_short_commit_digest(repo._env.refenv, master)

    diff_spec = repo.diff(master, dev)
    buf = status(hashenv=repo._env.hashenv, branch_name=dev, diff=diff_spec.diff)
    click.echo(buf.getvalue())

@main.command()
@click.argument('startpoint', nargs=1, required=False)
@pass_repo
def summary(repo: Repository, startpoint):
    """Display content summary at STARTPOINT (short-digest or branch).

    If no argument is passed in, the staging area branch HEAD wil be used as the
    starting point. In order to recieve a machine readable, and more complete
    version of this information, please see the ``Repository.summary()`` method
    of the API.
    """
    from hangar.records.commiting import expand_short_commit_digest

    if startpoint is None:
        click.echo(repo.summary())
    elif startpoint in repo.list_branches():
        click.echo(repo.summary(branch=startpoint))
    else:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
        click.echo(repo.summary(commit=base_commit))


@main.command()
@click.argument('startpoint', required=False, default=None)
@pass_repo
def log(repo: Repository, startpoint):
    """Display commit graph starting at STARTPOINT (short-digest or name)

    If no argument is passed in, the staging area branch HEAD will be used as the
    starting point.
    """
    from hangar.records.commiting import expand_short_commit_digest

    if startpoint is None:
        click.echo(repo.log())
    elif startpoint in repo.list_branches():
        click.echo(repo.log(branch=startpoint))
    else:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
        click.echo(repo.log(commit=base_commit))


@main.command()
@pass_repo
def status(repo: Repository):
    """Display changes made in the staging area compared to it's base commit
    """
    from hangar.records.summarize import status
    co = repo.checkout(write=True)
    try:
        diff = co.diff.staged()
        click.echo(status(co._hashenv, co.branch_name, diff.diff).getvalue(), nl=False)
    finally:
        co.close()


# ------------------------------- Branching -----------------------------------


@main.group(no_args_is_help=True, add_help_option=True)
@click.pass_context
def branch(ctx):  # pragma: no cover
    """operate on and list branch pointers.
    """
    pass


@branch.command(name='list')
@pass_repo
def branch_list(repo: Repository):
    """list all branch names

    Includes both remote branches as well as local branches.
    """
    click.echo(repo.list_branches())


@branch.command(name='create')
@click.argument('name', nargs=1, required=True)
@click.argument('startpoint', nargs=1, default=None, required=False)
@pass_repo
def branch_create(repo: Repository, name, startpoint):
    """Create a branch with NAME at STARTPOINT (short-digest or branch)

    If no STARTPOINT is provided, the new branch is positioned at the HEAD of
    the staging area branch, automatically.
    """
    from hangar.records.commiting import expand_short_commit_digest
    from hangar.records.heads import get_branch_head_commit
    from hangar.records.heads import get_staging_branch_head

    branch_names = repo.list_branches()
    if name in branch_names:
        e = ValueError(f'branch name: {name} already exists')
        raise click.ClickException(e)

    try:
        if startpoint is None:
            branch = get_staging_branch_head(repo._env.branchenv)
            base_commit = get_branch_head_commit(repo._env.branchenv, branch)
        elif startpoint in branch_names:
            base_commit = get_branch_head_commit(repo._env.branchenv, startpoint)
        else:
            base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)

        res = repo.create_branch(name, base_commit=base_commit)
    except (KeyError, ValueError, RuntimeError) as e:
        raise click.ClickException(e)

    click.echo(f'Created BRANCH: {res.name} HEAD: {res.digest}')


@branch.command(name='delete')
@click.argument('name', nargs=1, required=True)
@click.option('--force', '-f', is_flag=True, default=False,
              help='flag to force delete branch which has un-merged history.')
@pass_repo
def branch_remove(repo: Repository, name, force):
    """Remove a branch pointer with the provided NAME

    The NAME must be a branch present on the local machine.
    """
    try:
        res = repo.remove_branch(name, force_delete=force)
    except (ValueError, PermissionError, RuntimeError) as e:
        raise click.ClickException(e)

    click.echo(f'Deleted BRANCH: {res.name} HEAD: {res.digest}')


# ---------------------------- Server Commands --------------------------------


@main.command()
@click.option('--overwrite', is_flag=True, default=False,
              help='overwrite the hangar server instance if it exists at the current path.')
@click.option('--ip', default='localhost', show_default=True,
              help='the ip to start the server on. default is `localhost`')
@click.option('--port', default='50051', show_default=True,
              help='port to start the server on. default in `50051`')
@click.option('--timeout', default=60 * 60 * 24, required=False, show_default=True,
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
    from hangar.remote.server import serve

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
        hangserver.close()
        server.stop(0)


# ---------------------------- Import Exporters -------------------------------


@main.command(name='import',
              context_settings=dict(allow_extra_args=True, ignore_unknown_options=True, ))
@click.argument('column', required=True)
@click.argument('path',
                required=True,
                type=click.Path(exists=True, dir_okay=True, file_okay=True, readable=True,
                                resolve_path=True))
@click.option('--branch', default=None, help='branch to import data')
@click.option('--plugin', default=None, help='override auto-infered plugin')
@click.option('--overwrite', is_flag=True,
              help='overwrite data samples with the same name as the imported data file ')
@pass_repo
@click.pass_context
def import_data(ctx, repo: Repository, column, path, branch, plugin, overwrite):
    """Import file or directory of files at PATH to COLUMN in the staging area.

    If passing in a directory, all files in the directory will be imported, if
    passing in a file, just that files specified will be
    imported
    """
    # TODO: ignore warning through env variable
    from types import GeneratorType
    from hangar import external
    from hangar.records.heads import get_staging_branch_head

    kwargs = parse_custom_arguments(ctx.args)
    if branch is None:
        branch = get_staging_branch_head(repo._env.branchenv)
    elif branch not in repo.list_branches():
        raise click.ClickException(f'Branch name: {branch} does not exist, Exiting.')
    click.echo(f'Writing to branch: {branch}')

    co = repo.checkout(write=True, branch=branch)
    try:
        active_aset = co.columns.get(column)
        p = Path(path)
        files = [f.resolve() for f in p.iterdir()] if p.is_dir() else [p.resolve()]
        with active_aset as aset, click.progressbar(files) as filesBar:
            for f in filesBar:
                ext = ''.join(f.suffixes).strip('.')  # multi-suffix files (tar.bz2)
                loaded = external.load(f, plugin=plugin, extension=ext, **kwargs)
                if not isinstance(loaded, GeneratorType):
                    loaded = [loaded]
                for arr, fname in loaded:
                    if (not overwrite) and (fname in aset):
                        continue
                    try:
                        aset[fname] = arr
                    except ValueError as e:
                        click.echo(e)
    except (ValueError, KeyError) as e:
        raise click.ClickException(e)
    finally:
        co.close()


@main.command(name='export',
              context_settings=dict(allow_extra_args=True, ignore_unknown_options=True, ))
@click.argument('column', nargs=1, required=True)
@click.argument('startpoint', nargs=1, default=None, required=False)
@click.option('-o', '--out', 'outdir',
              nargs=1,
              required=False,
              default=os.getcwd(),
              type=click.Path(exists=True, dir_okay=True, file_okay=False, readable=True,
                              resolve_path=True),
              help="Directory to export data")
@click.option('-s', '--sample',
              nargs=1,
              default=None,
              type=StrOrIntType(),
              help=('Sample name to export. Default implementation is to interpret all input '
                    'names as string type. As an column can contain samples with both ``str`` '
                    'and ``int`` types, we allow you to specify ``name type`` of the sample. To '
                    'identify a potentially ambiguous name, we allow you to prepend the type of '
                    'sample name followed by a colon and then the sample name (ex. ``str:54`` '
                    'or ``int:54``). this can be done for any sample key.'))
@click.option('-f', '--format', 'format_',
              nargs=1,
              required=False,
              help='File format of output file')
@click.option('--plugin', required=False, help='override auto-inferred plugin')
@pass_repo
@click.pass_context
def export_data(ctx, repo: Repository, column, outdir, startpoint, sample, format_, plugin):
    """Export COLUMN sample data as it existed a STARTPOINT to some format and path.

    Specifying which sample to be exported is possible by using the switch
    ``--sample`` (without this, all the samples in the given column will be
    exported). Since hangar supports both int and str datatype for the sample
    name, specifying that while mentioning the sample name might be necessary
    at times. It is possible to do that by separating the name and type by a
    colon.

    Example:

       1. if the sample name is string of numeric 10 - ``str:10`` or ``10``

       2. if the sample name is ``sample1`` - ``str:sample1`` or ``sample1``

       3. if the sample name is an int, let say 10 - ``int:10``
    """
    from hangar.records.commiting import expand_short_commit_digest
    from hangar.records.heads import get_branch_head_commit, get_staging_branch_head
    from hangar import external
    kwargs = parse_custom_arguments(ctx.args)

    if startpoint in repo.list_branches():
        base_commit = get_branch_head_commit(repo._env.branchenv, startpoint)
    elif startpoint:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
    else:
        branch_name = get_staging_branch_head(repo._env.branchenv)
        base_commit = get_branch_head_commit(repo._env.branchenv, branch_name)

    co = repo.checkout(commit=base_commit)
    try:
        aset = co.columns.get(column)
        sampleNames = [sample] if sample is not None else list(aset.keys())
        extension = format_.lstrip('.') if format_ else None
        with aset, click.progressbar(sampleNames) as sNamesBar:
            for sampleN in sNamesBar:
                data = aset[sampleN]
                formated_sampleN = f'{type(sampleN).__name__}:{sampleN}'
                try:
                    external.save(data, outdir, formated_sampleN, extension, plugin, **kwargs)
                except Exception as e:
                    raise click.ClickException(e)
    except KeyError as e:
        raise click.ClickException(e)
    finally:
        co.close()


@main.command(name='view',
              context_settings=dict(allow_extra_args=True, ignore_unknown_options=True, ))
@click.argument('column', nargs=1, type=str, required=True)
@click.argument('sample', nargs=1, type=StrOrIntType(), required=True)
@click.argument('startpoint', nargs=1, default=None, required=False)
@click.option('-f', '--format', 'format_', required=False, help='File format of output file')
@click.option('--plugin', default=None, help='Plugin name to use instead of auto-inferred plugin')
@pass_repo
@click.pass_context
def view_data(ctx, repo: Repository, column, sample, startpoint, format_, plugin):
    """Use a plugin to view the data of some SAMPLE in COLUMN at STARTPOINT.
    """
    from hangar.records.commiting import expand_short_commit_digest
    from hangar.records.heads import get_branch_head_commit, get_staging_branch_head
    from hangar import external

    kwargs = parse_custom_arguments(ctx.args)
    if startpoint in repo.list_branches():
        base_commit = get_branch_head_commit(repo._env.branchenv, startpoint)
    elif startpoint:
        base_commit = expand_short_commit_digest(repo._env.refenv, startpoint)
    else:
        branch_name = get_staging_branch_head(repo._env.branchenv)
        base_commit = get_branch_head_commit(repo._env.branchenv, branch_name)

    co = repo.checkout(commit=base_commit)
    try:
        aset = co.columns.get(column)
        extension = format_.lstrip('.') if format_ else None
        data = aset[sample]
        try:
            external.show(data, plugin=plugin, extension=extension, **kwargs)
        except Exception as e:
            raise click.ClickException(e)
    except KeyError as e:
        raise click.ClickException(e)
    finally:
        co.close()


# ---------------------------- Developer Utils --------------------------------


@main.command(name='db-view', hidden=True)
@click.option('-a', is_flag=True, help='display all dbs in the repository')
@click.option('-b', is_flag=True, help='display the branch/heads db')
@click.option('-r', is_flag=True, help='display the references db')
@click.option('-d', is_flag=True, help='display the data hash db')
@click.option('-s', is_flag=True, help='display the stage record db')
@click.option('-z', is_flag=True, help='display the staged hash record db')
@click.option('--limit', default=30, help='limit the amount of records displayed before truncation')
@pass_repo
def lmdb_record_details(repo: Repository, a, b, r, d, s, z, limit):
    """DEVELOPER TOOL ONLY

    display key/value pairs making up the dbs
    """
    from hangar.context import Environments
    from hangar.records.summarize import details
    from hangar import constants as c

    if repo._repo_path.is_dir():
        repo_path = repo._repo_path
    elif repo._repo_path.parent.joinpath(c.DIR_HANGAR_SERVER).is_dir():
        repo_path = repo._repo_path.parent.joinpath(c.DIR_HANGAR_SERVER)
    else:
        click.echo(f'NO HANGAR INSTALLATION AT PATH: {repo._repo_path.parent}')
        return

    envs = Environments(pth=repo_path)
    try:
        if a:
            b, r, d, s, z = True, True, True, True, True
        if b:
            click.echo(details(envs.branchenv, line_limit=limit).getvalue())
        if r:
            click.echo(details(envs.refenv, line_limit=limit).getvalue())
        if d:
            click.echo(details(envs.hashenv, line_limit=limit).getvalue())
        if s:
            click.echo(details(envs.stageenv, line_limit=limit).getvalue())
        if z:
            click.echo(details(envs.stagehashenv, line_limit=limit).getvalue())
    finally:
        envs._close_environments()
