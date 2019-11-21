from os import getcwd
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from hangar import Repository
from hangar.cli import cli
from hangar.external import PluginManager
from conftest import fixed_shape_backend_params

# -------------------------------- test data ----------------------------------


help_res = 'Usage: main [OPTIONS] COMMAND [ARGS]...\n'\
           '\n'\
           'Options:\n'\
           '  --version  display current Hangar Version\n'\
           '  --help     Show this message and exit.\n'\
           '\n'\
           'Commands:\n'\
           '  arrayset    Operations for working with arraysets in the writer checkout.\n'\
           '  branch      operate on and list branch pointers.\n'\
           '  checkout    Checkout writer head branch at BRANCHNAME.\n'\
           '  clone       Initialize a repository at the current path and fetch updated...\n'\
           '  commit      Commits outstanding changes.\n'\
           '  export      Export ARRAYSET sample data as it existed a STARTPOINT to some...\n'\
           '  fetch       Retrieve the commit history from REMOTE for BRANCH.\n'\
           '  fetch-data  Get data from REMOTE referenced by STARTPOINT (short-commit or...\n'\
           '  import      Import file or directory of files at PATH to ARRAYSET in the...\n'\
           '  init        Initialize an empty repository at the current path\n'\
           '  log         Display commit graph starting at STARTPOINT (short-digest or...\n'\
           '  push        Upload local BRANCH commit history / data to REMOTE server.\n'\
           '  remote      Operations for working with remote server references\n'\
           '  server      Start a hangar server, initializing one if does not exist.\n'\
           '  status      Display changes made in the staging area compared to it\'s base...\n'\
           '  summary     Display content summary at STARTPOINT (short-digest or branch).\n'\
           '  view        Use a plugin to view the data of some SAMPLE in ARRAYSET at...\n'


# ------------------------------- begin tests ---------------------------------


def test_help_option():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.main, ['--help'])
        assert res.exit_code == 0
        assert res.stdout == help_res


def test_help_no_args_option():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.main)
        assert res.exit_code == 0
        assert res.stdout == help_res


def test_version_long_option():
    import hangar
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.main, ['--version'])
        assert res.exit_code == 0
        assert res.stdout == f'main, version {hangar.__version__}\n'


def test_init_repo(managed_tmpdir):
    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'], obj=repo)
        assert res.exit_code == 0
        assert repo._Repository__verify_repo_initialized() is None


def test_checkout_writer_branch_works(dummy_repo: Repository):
    from hangar.records.heads import get_staging_branch_head
    dummy_repo.create_branch('dev')
    runner = CliRunner()
    res = runner.invoke(cli.checkout, ['dev'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Writer checkout head set to branch: dev\n'
    recorded_branch = get_staging_branch_head(dummy_repo._env.branchenv)
    assert recorded_branch == 'dev'
    assert dummy_repo.writer_lock_held is False


def test_checkout_writer_branch_nonexistant_branch_errors(dummy_repo: Repository):
    from hangar.records.heads import get_staging_branch_head
    runner = CliRunner()
    res = runner.invoke(cli.checkout, ['doesnotexist'], obj=dummy_repo)
    assert res.exit_code == 1
    assert res.stdout == 'Error: branch with name: doesnotexist does not exist. cannot get head.\n'
    recorded_branch = get_staging_branch_head(dummy_repo._env.branchenv)
    assert recorded_branch == 'master'
    assert dummy_repo.writer_lock_held is False


def test_checkout_writer_branch_lock_held_errors(dummy_repo: Repository):
    from hangar.records.heads import get_staging_branch_head
    dummy_repo.create_branch('testbranch')
    co = dummy_repo.checkout(write=True, branch='master')
    try:
        runner = CliRunner()
        res = runner.invoke(cli.checkout, ['testbranch'], obj=dummy_repo)
        assert res.exit_code == 1
        msg = res.stdout
        assert msg.startswith('Error: Cannot acquire the writer lock.') is True
        recorded_branch = get_staging_branch_head(dummy_repo._env.branchenv)
        assert recorded_branch == 'master'
        assert dummy_repo.writer_lock_held is True
        assert co.branch_name == 'master'
    finally:
        co.close()
    assert dummy_repo.writer_lock_held is False


def test_commit_cli_message(dummy_repo: Repository):
    co = dummy_repo.checkout(write=True)
    co.metadata['newkeyhere'] = 'somevaluehere'
    base_digest = co.commit_hash
    base_branch = co.branch_name
    co.close()
    assert base_branch == 'master'

    runner = CliRunner()
    res = runner.invoke(cli.commit, ['-m', 'this is my commit message'], obj=dummy_repo)
    assert res.exit_code == 0
    out = res.stdout
    assert out.startswith('Commit message:\nthis is my commit message\nCommit Successful') is True
    new_digest = out.split(' ')[-1].rstrip('\n')
    assert new_digest != base_digest

    nco = dummy_repo.checkout(write=True)
    try:
        assert nco.commit_hash == new_digest
        assert nco.branch_name == base_branch
    finally:
        nco.close()


def test_commit_cli_message_with_no_changes(dummy_repo: Repository):
    co = dummy_repo.checkout(write=True)
    base_digest = co.commit_hash
    base_branch = co.branch_name
    co.close()
    assert base_branch == 'master'

    runner = CliRunner()
    res = runner.invoke(cli.commit, ['-m', 'this is my commit message'], obj=dummy_repo)
    assert res.exit_code == 1
    assert res.stdout.endswith('Error: No changes made in staging area. Cannot commit.\n')

    co = dummy_repo.checkout(write=True)
    try:
        assert co.branch_name == base_branch
        assert co.commit_hash == base_digest
    finally:
        co.close()


def substitute_editor_commit_message(hint):
    return 'this is my commit message\n' + hint


def test_commit_editor_message(monkeypatch, dummy_repo: Repository):
    import click
    monkeypatch.setattr(click, 'edit', substitute_editor_commit_message)

    co = dummy_repo.checkout(write=True)
    co.metadata['newkeyhere'] = 'somevaluehere'
    base_digest = co.commit_hash
    base_branch = co.branch_name
    co.close()
    assert base_branch == 'master'

    runner = CliRunner()
    res = runner.invoke(cli.commit, obj=dummy_repo)
    assert res.exit_code == 0
    out = res.stdout
    assert out.startswith('Commit message:\nthis is my commit message\nCommit Successful') is True
    new_digest = out.split(' ')[-1].rstrip('\n')
    assert new_digest != base_digest

    nco = dummy_repo.checkout(write=True)
    try:
        assert nco.commit_hash == new_digest
        assert nco.branch_name == base_branch
    finally:
        nco.close()


def substitute_editor_empty_commit_message(hint):
    return hint


def test_commit_editor_empty_message(monkeypatch, dummy_repo: Repository):
    import click
    monkeypatch.setattr(click, 'edit', substitute_editor_empty_commit_message)

    co = dummy_repo.checkout(write=True)
    co.metadata['newkeyhere'] = 'somevaluehere'
    base_digest = co.commit_hash
    base_branch = co.branch_name
    co.close()
    assert base_branch == 'master'

    runner = CliRunner()
    res = runner.invoke(cli.commit, obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Aborted! Empty commit message\n'
    nco = dummy_repo.checkout(write=True)
    try:
        assert nco.commit_hash == base_digest
        assert nco.branch_name == base_branch
    finally:
        nco.close()


def test_clone(written_two_cmt_server_repo):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        new_repo = Repository(P, exists=False)
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'], obj=new_repo)

        assert res.exit_code == 0

        newLog = new_repo.log(return_contents=True)
        baseLog = base_repo.log(return_contents=True)
        assert newLog == baseLog
        assert new_repo.summary() == base_repo.summary()


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_push_fetch_records(server_instance, backend):

    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Repository(getcwd(), exists=False)
        repo.init('foo', 'bar')
        dummyData = np.arange(50)
        co1 = repo.checkout(write=True, branch='master')
        co1.arraysets.init_arrayset(
            name='dummy', prototype=dummyData, named_samples=True, backend_opts=backend)
        for idx in range(10):
            dummyData[:] = idx
            co1.arraysets['dummy'][str(idx)] = dummyData
        co1.metadata['hello'] = 'world'
        co1.metadata['somemetadatakey'] = 'somemetadatavalue'
        cmt1 = co1.commit('first commit adding dummy data and hello meta')
        co1.close()

        repo.create_branch('testbranch')
        co2 = repo.checkout(write=True, branch='testbranch')
        for idx in range(10, 20):
            dummyData[:] = idx
            co2.arraysets['dummy'][str(idx)] = dummyData
        co2.metadata['foo'] = 'bar'
        cmt2 = co2.commit('first commit on test branch adding non-conflict data and meta')
        co2.close()

        repo.remote.add('origin', server_instance)

        res = runner.invoke(cli.push, ['origin', 'master'], obj=repo)
        assert res.exit_code == 0
        res = runner.invoke(cli.push, ['origin', 'testbranch'], obj=repo)
        assert res.exit_code == 0



@pytest.mark.parametrize('backend', fixed_shape_backend_params)
@pytest.mark.parametrize('options', [
    ['origin', 'testbranch'],
    ['origin', 'master'],
    ['origin', 'testbranch', '--all-history'],
    ['origin', 'master', '--all-history'],
    ['origin', 'testbranch', '--aset', 'data'],
    ['origin', 'master', '--aset', 'data'],
    ['origin', 'testbranch', '--aset', 'data', '--all-history'],
    ['origin', 'master', '--aset', 'data', '--all-history'],
    ['origin', 'testbranch', '--aset', 'data', '--all-history'],
    ['origin', 'master', '--nbytes', '3Kb'],
])
def test_fetch_records_and_data(server_instance, backend, options):
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Repository(getcwd(), exists=False)
        repo.init('foo', 'bar')
        dummyData = np.arange(50)
        co1 = repo.checkout(write=True, branch='master')
        co1.arraysets.init_arrayset(
            name='dummy', prototype=dummyData, named_samples=True, backend_opts=backend)
        for idx in range(10):
            dummyData[:] = idx
            co1.arraysets['dummy'][str(idx)] = dummyData
        co1.metadata['hello'] = 'world'
        co1.metadata['somemetadatakey'] = 'somemetadatavalue'
        cmt1 = co1.commit('first commit adding dummy data and hello meta')
        co1.close()

        repo.create_branch('testbranch')
        co2 = repo.checkout(write=True, branch='testbranch')
        for idx in range(10, 20):
            dummyData[:] = idx
            co2.arraysets['dummy'][str(idx)] = dummyData
        co2.metadata['foo'] = 'bar'
        cmt2 = co2.commit('first commit on test branch adding non-conflict data and meta')
        co2.close()

        repo.remote.add('origin', server_instance)

        res = runner.invoke(cli.push, ['origin', 'master'], obj=repo)
        assert res.exit_code == 0
        res = runner.invoke(cli.push, ['origin', 'testbranch'], obj=repo)
        assert res.exit_code == 0

    with runner.isolated_filesystem():
        repo = Repository(getcwd(), exists=False)
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server_instance}'], obj=repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.fetch_records, ['origin', 'testbranch'], obj=repo)
        assert res.exit_code == 0
        res = runner.invoke(cli.branch_create, ['testbranch', 'origin/testbranch'], obj=repo)
        assert res.exit_code == 0
        res = runner.invoke(cli.fetch_data, options, obj=repo)
        assert res.exit_code == 0


def test_add_remote(managed_tmpdir):
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'], obj=repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'], obj=repo)
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"

        remote_list = repo.remote.list_all()
        assert remote_list == [RemoteInfo(name='origin', address='localhost:50051')]


def test_remove_remote(managed_tmpdir):
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'], obj=repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'], obj=repo)
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"

        remote_list = repo.remote.list_all()
        assert remote_list == [RemoteInfo(name='origin', address='localhost:50051')]

        res = runner.invoke(cli.remove_remote, ['origin'], obj=repo)
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"
        assert repo.remote.list_all() == []


def test_list_all_remotes(managed_tmpdir):
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        repo = Repository(P, exists=False)
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'], obj=repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'], obj=repo)
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"
        res = runner.invoke(cli.add_remote, ['upstream', 'foo:ip'], obj=repo)
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='upstream', address='foo:ip')\n"

        remote_list = repo.remote.list_all()
        assert remote_list == [
            RemoteInfo(name='origin', address='localhost:50051'),
            RemoteInfo(name='upstream', address='foo:ip')
        ]

        res = runner.invoke(cli.list_remotes, obj=repo)
        assert res.exit_code == 0
        expected_stdout = "[RemoteInfo(name='origin', address='localhost:50051'), "\
                          "RemoteInfo(name='upstream', address='foo:ip')]\n"
        assert res.stdout == expected_stdout


def test_summary(written_two_cmt_server_repo, capsys):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        with capsys.disabled():
            P = getcwd()
            new_repo = Repository(P, exists=False)
            res = runner.invoke(
                cli.clone,
                ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'], obj=new_repo)

            assert res.exit_code == 0
            assert new_repo.summary() == base_repo.summary()

        new_repo.summary()

        with capsys.disabled():
            res = runner.invoke(cli.summary, obj=new_repo)
            assert res.stdout == f"{capsys.readouterr().out}\n"


def test_log(written_two_cmt_server_repo, capsys):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        with capsys.disabled():
            P = getcwd()
            new_repo = Repository(P, exists=False)
            res = runner.invoke(
                cli.clone,
                ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'], obj=new_repo)

            assert res.exit_code == 0
            assert new_repo.log() == base_repo.log()

        new_repo.log()

        with capsys.disabled():
            res = runner.invoke(cli.log, ['master'], obj=new_repo)
            assert res.stdout == f"{capsys.readouterr().out}\n"


def test_status(dummy_repo):
    from hangar.records.summarize import status

    dummyData = np.arange(50).astype(np.int64)
    co2 = dummy_repo.checkout(write=True)
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.arraysets['dummy'][str(idx)] = dummyData
        co2.arraysets['dummy'][idx] = dummyData
    co2.metadata['foo'] = 'bar'
    df = co2.diff.staged()
    co2.close()
    expected = status('master', df.diff).getvalue()

    runner = CliRunner()
    res = runner.invoke(cli.status, obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == expected


def test_arrayset_create_uint8(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(
        cli.create_arrayset,
        ['train_images', 'UINT8', '256', '256', '3'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Initialized Arrayset: train_images\n'
    co = dummy_repo.checkout(write=True)
    try:
        assert 'train_images' in co.arraysets
        assert co.arraysets['train_images'].shape == (256, 256, 3)
        assert co.arraysets['train_images'].dtype == np.uint8
        assert co.arraysets['train_images'].named_samples is True
        assert co.arraysets['train_images'].variable_shape is False
        assert len(co.arraysets['train_images']) == 0
    finally:
        co.close()


def test_arrayset_create_float32(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(
        cli.create_arrayset,
        ['train_images', 'FLOAT32', '256'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Initialized Arrayset: train_images\n'
    co = dummy_repo.checkout(write=True)
    try:
        assert 'train_images' in co.arraysets
        assert co.arraysets['train_images'].shape == (256,)
        assert co.arraysets['train_images'].dtype == np.float32
        assert co.arraysets['train_images'].named_samples is True
        assert co.arraysets['train_images'].variable_shape is False
        assert len(co.arraysets['train_images']) == 0
    finally:
        co.close()


def test_arrayset_create_invalid_dtype_fails(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(
        cli.create_arrayset,
        ['train_images', 'FLOAT7', '256'], obj=dummy_repo)
    assert res.exit_code == 2
    expected = ('Error: Invalid value for "[UINT8|INT8|UINT16|INT16|UINT32|INT32|'
                'UINT64|INT64|FLOAT16|FLOAT32|FLOAT64]": invalid choice: FLOAT7. '
                '(choose from UINT8, INT8, UINT16, INT16, UINT32, INT32, UINT64, '
                'INT64, FLOAT16, FLOAT32, FLOAT64)\n')
    assert res.stdout.endswith(expected) is True
    co = dummy_repo.checkout(write=True)
    try:
        assert 'train_images' not in co.arraysets
    finally:
        co.close()


def test_arrayset_create_invalid_name_fails(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(cli.create_arrayset, ['tra#in', 'FLOAT32', '256'], obj=dummy_repo)
    assert res.exit_code == 1
    msg = res.stdout
    assert msg.startswith('Error: Arrayset name provided: `tra#in` is invalid.') is True
    co = dummy_repo.checkout(write=True)
    try:
        assert 'tra#in' not in co.arraysets
        assert 'dummy' in co.arraysets
        assert len(co.arraysets) == 1
    finally:
        co.close()


def test_arrayset_create_no_named_samples(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(
        cli.create_arrayset,
        ['train_images', 'FLOAT32', '256', '--not-named'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Initialized Arrayset: train_images\n'
    co = dummy_repo.checkout(write=True)
    try:
        assert 'train_images' in co.arraysets
        assert co.arraysets['train_images'].shape == (256,)
        assert co.arraysets['train_images'].dtype == np.float32
        assert co.arraysets['train_images'].named_samples is False
        assert co.arraysets['train_images'].variable_shape is False
        assert len(co.arraysets['train_images']) == 0
    finally:
        co.close()


def test_arrayset_create_variable_shape(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(
        cli.create_arrayset,
        ['train_images', 'FLOAT32', '256', '--variable-shape'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Initialized Arrayset: train_images\n'
    co = dummy_repo.checkout(write=True)
    try:
        assert 'train_images' in co.arraysets
        assert co.arraysets['train_images'].shape == (256,)
        assert co.arraysets['train_images'].dtype == np.float32
        assert co.arraysets['train_images'].named_samples is True
        assert co.arraysets['train_images'].variable_shape is True
        assert len(co.arraysets['train_images']) == 0
    finally:
        co.close()


def test_remove_arrayset(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(cli.remove_arrayset, ['dummy'], obj=dummy_repo)
    assert res.exit_code == 0
    assert res.stdout == 'Successfully removed arrayset: dummy\n'
    co = dummy_repo.checkout(write=True)
    try:
        assert 'dummy_repo' not in co.arraysets
        assert len(co.arraysets) == 0
        assert len(co.metadata) == 2
    finally:
        co.close()


def test_remove_non_existing_arrayset(dummy_repo):
    runner = CliRunner()
    res = runner.invoke(cli.remove_arrayset, ['doesnotexist'], obj=dummy_repo)
    assert res.exit_code == 1
    assert res.stdout == "Error: 'Cannot remove: doesnotexist. Key does not exist.'\n"
    co = dummy_repo.checkout(write=True)
    try:
        assert 'doesnotexist' not in co.arraysets
        assert 'dummy' in co.arraysets
        assert len(co.arraysets) == 1
        assert len(co.arraysets['dummy']) == 10
        assert len(co.metadata) == 2
    finally:
        co.close()


def test_branch_create_and_list(written_two_cmt_server_repo):
    server, base_repo = written_two_cmt_server_repo

    co = base_repo.checkout(write=True)
    cmt = co.commit_hash
    co.close()

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        new_repo = Repository(P, exists=False)
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'], obj=new_repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.branch_create, ['testbranch'], obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == f"Created BRANCH: testbranch HEAD: {cmt}\n"

        branches = new_repo.list_branches()
        assert branches == ['master', 'origin/master', 'testbranch']

        res = runner.invoke(cli.branch_list, obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == "['master', 'origin/master', 'testbranch']\n"


@pytest.mark.filterwarnings("ignore:Arrayset.* contains `reference-only` samples")
def test_branch_create_and_delete(written_two_cmt_server_repo):
    server, base_repo = written_two_cmt_server_repo

    co = base_repo.checkout(write=True)
    cmt = co.commit_hash
    co.close()

    runner = CliRunner()
    with runner.isolated_filesystem():
        P = getcwd()
        new_repo = Repository(P, exists=False)
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'], obj=new_repo)
        assert res.exit_code == 0

        res = runner.invoke(cli.branch_create, ['testbranch'], obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == f"Created BRANCH: testbranch HEAD: {cmt}\n"

        branches = new_repo.list_branches()
        assert branches == ['master', 'origin/master', 'testbranch']

        res = runner.invoke(cli.branch_remove, ['testbranch'], obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == f"Deleted BRANCH: testbranch HEAD: {cmt}\n"

        branches = new_repo.list_branches()
        assert branches == ['master', 'origin/master']

        new_repo.create_branch('secondtest')
        co = new_repo.checkout(write=True, branch='secondtest')
        co.metadata['foodadaa'] = '34a345'
        newDigest = co.commit('dummy commit')
        co.close()

        # re-open with staging set to master so we can try to delete secondtest
        co = new_repo.checkout(write=True, branch='master')
        co.close()

        res = runner.invoke(cli.branch_remove, ['secondtest'], obj=new_repo)
        assert res.exit_code == 1

        res = runner.invoke(cli.branch_remove, ['secondtest', '-f'], obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == f"Deleted BRANCH: secondtest HEAD: {newDigest}\n"

        res = runner.invoke(cli.branch_list, obj=new_repo)
        assert res.exit_code == 0
        assert res.stdout == "['master', 'origin/master']\n"


def test_start_server(managed_tmpdir):
    import time
    runner = CliRunner()
    with runner.isolated_filesystem():
        startTime = time.time()
        res = runner.invoke(cli.server, ['--ip', 'localhost', '--port', '50111', '--timeout', '1'])
        assert time.time() - startTime <= 1.8  # buffer to give it time to stop
        assert res.exit_code == 0
        assert 'Hangar Server Started' in res.stdout


# =========================== External Plugin =================================


def monkeypatch_scan(provides, accepts, attribute, func):
    def wrapper(self):
        from hangar.external import BasePlugin

        plugin = BasePlugin(provides, accepts)
        plugin.__dict__[attribute] = func

        self._plugin_store['myplugin'] = plugin
    return wrapper


@pytest.fixture()
def written_repo_with_1_sample(written_repo):
    aset_name = 'writtenaset'
    shape = (5, 7)
    co = written_repo.checkout(write=True)
    aset = co.arraysets[aset_name]
    aset['data'] = np.random.random(shape)
    aset['123'] = np.random.random(shape)
    aset[123] = np.random.random(shape)
    co.commit('added')
    co.close()
    yield written_repo


class TestImport(object):

    @staticmethod
    def load(fpath, *args, **kwargs):
        data = np.random.random((5, 7)).astype(np.float64)
        if isinstance(fpath, Path):
            fpath = fpath.name
        return data, fpath

    def test_import(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        shape = (5, 7)
        fpath = 'data.ext'
        aset_name = 'writtenaset'

        with monkeypatch.context() as m, runner.isolated_filesystem():
            with open('data.ext', 'w') as f:
                f.write('test')

            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['load'], ['ext'], 'load', self.load))
            # adding data
            res = runner.invoke(cli.import_data, [aset_name, fpath], obj=repo)
            assert res.exit_code == 0
            co = repo.checkout(write=True)
            co.commit('added data')
            d1 = co.arraysets[aset_name][fpath]
            co.close()

            # without overwrite
            res = runner.invoke(cli.import_data, [aset_name, fpath], obj=repo)
            assert res.exit_code == 0
            co = repo.checkout()
            d2 = co.arraysets[aset_name][fpath]
            co.close()
            assert np.allclose(d1, d2)

            # with overwrite
            res = runner.invoke(cli.import_data, [aset_name, fpath, '--overwrite'], obj=repo)
            assert res.exit_code == 0
            co = repo.checkout(write=True)
            co.commit('added data')
            d3 = co.arraysets[aset_name][fpath]
            co.close()
            assert not np.allclose(d1, d3)
        assert d1.shape == d2.shape == d3.shape == shape

    def test_import_wrong_args(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()

        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['load'], ['ext'], 'load', self.load))

            with runner.isolated_filesystem():

                # invalid file
                res = runner.invoke(cli.import_data, [aset_name, 'valid.ext'], obj=repo)
                assert res.exit_code == 2
                assert res.stdout.endswith('Invalid value for "PATH": Path "valid.ext" does not exist.\n')

                with open('valid.ext', 'w') as f:
                    f.write('empty')

                with open('valid.ext.bz2', 'w') as f:
                    f.write('empty')

                res = runner.invoke(cli.import_data, [aset_name, 'valid.ext.bz2'], obj=repo)
                assert res.exit_code == 1
                assert res.stdout.endswith('No plugins found for the file extension ext.bz2 that could do load\n')

                # invalid branch
                res = runner.invoke(cli.import_data, [aset_name, 'valid.ext', '--branch', 'invalid'], obj=repo)
                assert res.exit_code == 1
                assert res.stdout.endswith('Branch name: invalid does not exist, Exiting.\n')

                # invalid plugin
                res = runner.invoke(cli.import_data, [aset_name, 'valid.ext', '--plugin', 'invalid'], obj=repo)
                assert res.exit_code == 1
                assert res.stdout.endswith('Plugin invalid not found\n')

    def test_import_generator_on_load(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        fpath = 'data.ext'
        aset_name = 'writtenaset'

        def load(fpath, *args, **kwargs):
            for i in range(10):
                data, name = self.load(fpath, *args, **kwargs)
                if isinstance(name, Path):
                    name = name.name
                yield data, f"{i}_{name}"

        with monkeypatch.context() as m, runner.isolated_filesystem():
            with open('data.ext', 'w') as f:
                f.write('test')
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['load'], ['ext'], 'load', load))
            res = runner.invoke(cli.import_data, [aset_name, fpath], obj=repo)
            assert res.exit_code == 0
            co = repo.checkout(write=True)
            co.commit('added data')
            aset = co.arraysets[aset_name]
            for i in range(10):
                assert f"{i}_{fpath}" in aset.keys()
            co.close()


class TestExport(object):
    save_msg = "Data saved from custom save function"

    @classmethod
    def save(cls, data, outdir, sampleN, extension, *args, **kwargs):
        print(cls.save_msg)
        fpath = os.path.join(outdir, f"{sampleN}.{extension}")
        print(fpath)

    def test_export_success(self, monkeypatch, written_repo_with_1_sample, tmp_path):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))

            # single sample
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--sample', 'data', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert self.save_msg in res.output

            # with sample name and sample type
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--sample', 'int:123', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert os.path.join(tmp_path, 'int:123.ext') in res.output
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--sample', 'str:123', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert os.path.join(tmp_path, 'str:123.ext') in res.output
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--sample', '123', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert os.path.join(tmp_path, 'str:123.ext') in res.output

            # whole arrayset
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert os.path.join(tmp_path, 'str:data.ext') in res.output
            assert os.path.join(tmp_path, 'str:123.ext') in res.output
            assert os.path.join(tmp_path, 'int:123.ext') in res.output

    def test_export_wrong_out_location(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))

            # single sample
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', 'wrongpath', '--sample', 'data', '--format', 'ext'], obj=repo)
            assert res.exit_code == 2
            assert 'Invalid value for "-o"' in res.stdout

    def test_export_wrong_arg(self, monkeypatch, written_repo_with_1_sample, tmp_path):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))
            res = runner.invoke(
                cli.export_data, [aset_name, '-o', str(tmp_path), '--plugin', 'invalid'], obj=repo)
            assert res.exit_code == 1
            assert 'Plugin invalid not found' in res.stdout

    def test_export_without_specifying_out(self, monkeypatch, written_repo_with_1_sample):
        import os
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))
            res = runner.invoke(
                cli.export_data, [aset_name, '--sample', 'data', '--format', 'ext'], obj=repo)
            assert os.getcwd() in res.output

    def test_export_for_non_existent_sample(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))
            res = runner.invoke(
                cli.export_data, [aset_name, '--sample', 'wrongname', '--format', 'ext'], obj=repo)
            assert 'KEY ERROR' in res.output

    def test_export_for_specified_branch(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['save'], ['ext'], 'save', self.save))
            res = runner.invoke(
                cli.export_data, [aset_name, 'master', '--sample', 'data', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0


class TestShow(object):
    show_msg = "Data is displayed from custom show function"

    @classmethod
    def show(cls, fpath, *args, **kwargs):
        print(cls.show_msg)

    def test_show_success(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['show'], ['ext'], 'show', self.show))
            res = runner.invoke(
                cli.view_data, [aset_name, 'data', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            assert self.show_msg in res.output

    def test_show_on_startpoint(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['show'], ['ext'], 'show', self.show))
            res = runner.invoke(
                cli.view_data, [aset_name, 'data', 'master', '--format', 'ext'], obj=repo)
            assert res.exit_code == 0
            res = runner.invoke(
                cli.view_data, [aset_name, 'data', 'wrongstartpoint', '--format', 'ext'], obj=repo)
            assert 'No expanded commit hash found for short: wrongstartpoint' in str(res.exception)

    def test_show_with_wrong_arg(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'

        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['show'], ['ext'], 'show', self.show))
            res = runner.invoke(
                cli.view_data, [aset_name, 'data', '--format', 'wrong'], obj=repo)
            assert res.exit_code == 1
            assert 'No plugins found' in res.stdout

    def test_wrong_sample_name(self, monkeypatch, written_repo_with_1_sample):
        repo = written_repo_with_1_sample
        runner = CliRunner()
        aset_name = 'writtenaset'
        with monkeypatch.context() as m:
            m.setattr(PluginManager, "_scan_plugins", monkeypatch_scan(['show'], ['ext'], 'show', self.show))
            res = runner.invoke(
                cli.view_data, [aset_name, 'wrongsample', '--format', 'ext'], obj=repo)
            assert "wrongsample not in aset" in res.stdout
