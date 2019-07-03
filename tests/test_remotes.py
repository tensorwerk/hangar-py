import pytest
import numpy as np
import time
from os.path import join as pjoin
from os import mkdir
from random import randint
import socket


@pytest.fixture()
def server_instance(managed_tmpdir, worker_id):
    from hangar import serve

    tcp_socket = socket.socket(socket.AF_INET)
    tcp_socket.bind(('', 0))
    address_tuple = tcp_socket.getsockname()
    address = f'localhost:{address_tuple[1]}'
    base_tmpdir = pjoin(managed_tmpdir, f'{worker_id[-1]}')
    mkdir(base_tmpdir)

    server, hangserver = serve(base_tmpdir, overwrite=True, channel_address=address)
    server.start()
    time.sleep(randint(100, 200) * 0.01)
    yield address

    hangserver.env._close_environments()
    tcp_socket.close()
    server.stop(0.0)
    time.sleep(randint(100, 200) * 0.01)


@pytest.fixture()
def written_two_cmt_server_repo(server_instance, written_two_cmt_repo) -> tuple:
    written_two_cmt_repo.add_remote('origin', server_instance)
    success = written_two_cmt_repo.push('origin', 'master')
    assert success is True
    yield (server_instance, written_two_cmt_repo)


def test_server_is_started_multiple_times_via_ping_pong(server_instance, written_repo):
    # start multiple times and test that pings go through multiple times
    written_repo.add_remote('origin', server_instance)
    assert written_repo._ping_server('origin') == 'PONG'


@pytest.mark.parametrize('nCommits,nSamples', [[1, 10], [10, 10]])
def test_push_and_clone_master_linear_history_multiple_commits(
        server_instance, repo, managed_tmpdir, array5by7, nCommits, nSamples):
    from hangar import Repository

    cmtList = []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        sampList = []
        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                sampList.append(arr)
        cmt = co.commit(f'commit number: {cIdx}')
        cmtList.append((cmt, sampList))
        co.close()

    repo.add_remote('origin', server_instance)
    push1 = repo.push('origin', 'master')
    assert push1 is True

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branch_names() == ['master']
    for cmt, sampList in cmtList:
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert '_dset' in nco.datasets
        assert len(nco.datasets['_dset']) == nSamples
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['_dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 10], [10, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 5], [5, 5]])
def test_server_push_second_branch_with_new_commit(server_instance, repo,
                                                   array5by7, nMasterCommits,
                                                   nMasterSamples, nDevCommits,
                                                   nDevSamples):

    masterCmtList, devCmtList = [], []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nMasterSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                masterSampList.append(arr)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, masterSampList))
        co.close()

    repo.add_remote('origin', server_instance)
    push1 = repo.push('origin', 'master')
    assert push1 is True

    branch = repo.create_branch('testbranch')
    for cIdx in range(nDevCommits):
        co = repo.checkout(write=True, branch_name=branch)
        devSampList = []
        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nDevSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                devSampList.append(arr)
        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmtList.append((cmt, devSampList))
        co.close()

    push2 = repo.push('origin', branch)
    assert push2 is True


@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 10], [10, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 5], [5, 5]])
def test_server_push_clone_second_branch_with_new_commit(
        server_instance, repo, managed_tmpdir, array5by7, nMasterCommits,
        nMasterSamples, nDevCommits, nDevSamples):
    from hangar import Repository

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nMasterSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                masterSampList.append(arr)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, masterSampList))
        co.close()

    repo.add_remote('origin', server_instance)
    push1 = repo.push('origin', 'master')
    assert push1 is True

    # Plush dev branch test
    devCmtList = []
    branch = repo.create_branch('testbranch')
    for cIdx in range(nDevCommits):
        co = repo.checkout(write=True, branch_name=branch)
        devSampList = []
        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nDevSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                devSampList.append(arr)
        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmtList.append((cmt, devSampList))
        co.close()

    push2 = repo.push('origin', branch)
    assert push2 is True

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branch_names() == ['master']
    for cmt, sampList in masterCmtList:
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert '_dset' in nco.datasets
        assert len(nco.datasets['_dset']) == nMasterSamples
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['_dset'][str(sIdx)], samp)
        nco.close()

    # Fetch test
    fetch = newRepo.fetch('origin', branch)
    assert fetch == f'origin/{branch}'
    assert newRepo.list_branch_names() == ['master', f'origin/{branch}']
    for cmt, sampList in devCmtList:
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert '_dset' in nco.datasets
        assert len(nco.datasets['_dset']) == nDevSamples
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['_dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_unchanged_repo_makes_no_modifications(written_two_cmt_server_repo):
    _, repo = written_two_cmt_server_repo
    success = repo.push('origin', 'master')
    assert not success


def test_fetch_unchanged_repo_makes_no_modifications(written_two_cmt_server_repo):
    _, repo = written_two_cmt_server_repo
    success = repo.fetch('origin', 'master')
    assert not success


def test_fetch_newer_disk_repo_makes_no_modifications(written_two_cmt_server_repo):
    _, repo = written_two_cmt_server_repo
    co = repo.checkout(write=True)
    co.metadata['new_foo_abc'] = 'bar'
    co.commit('newer commit')
    co.close()
    success = repo.fetch('origin', 'master')
    assert not success