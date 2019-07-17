import platform
import time
import os

from os import mkdir, getcwd, chdir
from os.path import join as pjoin
from random import randint

import numpy as np
import pytest
from click.testing import CliRunner

from hangar import Repository, cli

help_res = '''Usage: main [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  branch      list or create branches
  clone       Initialize a repository at the current path and fetch data...
  db-view     display the key/value record pair details from the lmdb database
  fetch       Retrieve the commit history records for BRANCH from the REMOTE...
  fetch-data  Download the tensor data from the REMOTE server referenced by...
  init        Initialize an empty repository at the current path
  log         show the commit log graph
  push        Upload local BRANCH commit history / data to REMOTE server.
  remote
  server      start a hangar server at the given location
  summary     show a summary of the repository
'''


def test_help():
    runner = CliRunner()
    res = runner.invoke(cli.main, ['--help'])
    assert res.exit_code == 0
    assert res.stdout == help_res


def test_init_repo():
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'])
        assert res.exit_code == 0
        P = getcwd()
        repo = Repository(P)
        assert repo._Repository__verify_repo_initialized() is None


def test_clone(written_two_cmt_server_repo):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'])

        assert res.exit_code == 0
        P = getcwd()
        new_repo = Repository(P)

        newLog = new_repo.log(return_contents=True)
        baseLog = base_repo.log(return_contents=True)
        assert newLog == baseLog
        assert new_repo.summary() == base_repo.summary()


@pytest.mark.parametrize('backend', ['00', '10'])
def test_push_fetch_records(server_instance, backend):

    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Repository(getcwd())
        repo.init('foo', 'bar')
        dummyData = np.arange(50)
        co1 = repo.checkout(write=True, branch_name='master')
        co1.datasets.init_dataset(
            name='dummy', prototype=dummyData, named_samples=True, backend=backend)
        for idx in range(10):
            dummyData[:] = idx
            co1.datasets['dummy'][str(idx)] = dummyData
        co1.metadata['hello'] = 'world'
        co1.metadata['somemetadatakey'] = 'somemetadatavalue'
        cmt1 = co1.commit('first commit adding dummy data and hello meta')
        co1.close()

        repo.create_branch('testbranch')
        co2 = repo.checkout(write=True, branch_name='testbranch')
        for idx in range(10, 20):
            dummyData[:] = idx
            co2.datasets['dummy'][str(idx)] = dummyData
        co2.metadata['foo'] = 'bar'
        cmt2 = co2.commit('first commit on test branch adding non-conflict data and meta')
        co2.close()

        repo.remote.add('origin', server_instance)

        res = runner.invoke(cli.push, ['origin', 'master'])
        assert res.exit_code == 0
        res = runner.invoke(cli.push, ['origin', 'testbranch'])
        assert res.exit_code == 0



@pytest.mark.parametrize('backend', ['00', '10'])
def test_fetch_records_and_data(server_instance, backend):

    runner = CliRunner()
    with runner.isolated_filesystem():
        repo = Repository(getcwd())
        repo.init('foo', 'bar')
        dummyData = np.arange(50)
        co1 = repo.checkout(write=True, branch_name='master')
        co1.datasets.init_dataset(
            name='dummy', prototype=dummyData, named_samples=True, backend=backend)
        for idx in range(10):
            dummyData[:] = idx
            co1.datasets['dummy'][str(idx)] = dummyData
        co1.metadata['hello'] = 'world'
        co1.metadata['somemetadatakey'] = 'somemetadatavalue'
        cmt1 = co1.commit('first commit adding dummy data and hello meta')
        co1.close()

        repo.create_branch('testbranch')
        co2 = repo.checkout(write=True, branch_name='testbranch')
        for idx in range(10, 20):
            dummyData[:] = idx
            co2.datasets['dummy'][str(idx)] = dummyData
        co2.metadata['foo'] = 'bar'
        cmt2 = co2.commit('first commit on test branch adding non-conflict data and meta')
        co2.close()

        repo.remote.add('origin', server_instance)

        res = runner.invoke(cli.push, ['origin', 'master'])
        assert res.exit_code == 0
        res = runner.invoke(cli.push, ['origin', 'testbranch'])
        assert res.exit_code == 0

    with runner.isolated_filesystem():
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server_instance}'])
        assert res.exit_code == 0

        res = runner.invoke(cli.fetch_records, ['origin', 'testbranch'])
        assert res.exit_code == 0
        res = runner.invoke(cli.fetch_data, ['origin', cmt2])
        assert res.exit_code == 0


def test_add_remote():
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'])
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'])
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"

        P = getcwd()
        repo = Repository(P)
        remote_list = repo.remote.list_all()
        assert remote_list == [RemoteInfo(name='origin', address='localhost:50051')]


def test_remove_remote():
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'])
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'])
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"

        P = getcwd()
        repo = Repository(P)
        remote_list = repo.remote.list_all()
        assert remote_list == [RemoteInfo(name='origin', address='localhost:50051')]

        res = runner.invoke(cli.remove_remote, ['origin'])
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"
        assert repo.remote.list_all() == []


def test_list_all_remotes():
    from hangar.remotes import RemoteInfo

    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(cli.init, ['--name', 'test', '--email', 'test@foo.com'])
        assert res.exit_code == 0

        res = runner.invoke(cli.add_remote, ['origin', 'localhost:50051'])
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='origin', address='localhost:50051')\n"
        res = runner.invoke(cli.add_remote, ['upstream', 'foo:ip'])
        assert res.exit_code == 0
        assert res.stdout == "RemoteInfo(name='upstream', address='foo:ip')\n"

        P = getcwd()
        repo = Repository(P)
        remote_list = repo.remote.list_all()
        assert remote_list == [
            RemoteInfo(name='origin', address='localhost:50051'),
            RemoteInfo(name='upstream', address='foo:ip')
        ]

        res = runner.invoke(cli.list_remotes)
        assert res.exit_code == 0
        expected_stdout = "[RemoteInfo(name='origin', address='localhost:50051'), "\
                          "RemoteInfo(name='upstream', address='foo:ip')]\n"
        assert res.stdout == expected_stdout


def test_summary(written_two_cmt_server_repo, capsys):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        with capsys.disabled():
            res = runner.invoke(
                cli.clone,
                ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'])

            assert res.exit_code == 0
            P = getcwd()
            new_repo = Repository(P)
            assert new_repo.summary() == base_repo.summary()

        new_repo.summary()

        with capsys.disabled():
            res = runner.invoke(cli.summary)
            assert res.stdout == f"{capsys.readouterr().out}\n"


def test_log(written_two_cmt_server_repo, capsys):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        with capsys.disabled():
            res = runner.invoke(
                cli.clone,
                ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'])

            assert res.exit_code == 0
            P = getcwd()
            new_repo = Repository(P)
            assert new_repo.log() == base_repo.log()

        new_repo.log()

        with capsys.disabled():
            res = runner.invoke(cli.log, ['-b', 'master'])
            assert res.stdout == f"{capsys.readouterr().out}\n"


def test_branch_create_and_list(written_two_cmt_server_repo):
    server, base_repo = written_two_cmt_server_repo
    runner = CliRunner()
    with runner.isolated_filesystem():
        res = runner.invoke(
            cli.clone,
            ['--name', 'Foo Tester', '--email', 'foo@email.com', f'{server}'])
        assert res.exit_code == 0

        res = runner.invoke(cli.branch, ['-b', 'testbranch'])
        assert res.exit_code == 0
        assert res.stdout == "create branch operation success: testbranch\n"

        P = getcwd()
        new_repo = Repository(P)
        branches = new_repo.list_branches()
        assert branches == ['master', 'origin/master', 'testbranch']

        res = runner.invoke(cli.branch, ['-l'])
        assert res.exit_code == 0
        assert res.stdout == "['master', 'origin/master', 'testbranch']\n"


def test_start_server():
    import time
    runner = CliRunner()
    with runner.isolated_filesystem():
        startTime = time.time()
        res = runner.invoke(cli.server, ['--ip', 'localhost', '--port', '50111', '--timeout', '3'])
        assert time.time() - startTime >= 3
        assert res.exit_code == 0
        assert res.stdout.startswith('Hangar Server Started')