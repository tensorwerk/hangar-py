import pytest
import numpy as np
import time
from os.path import join as pjoin
from os import mkdir
from random import randint
import socket


@pytest.fixture(scope='function')
def server_instance(managed_tmpdir, worker_id):
    from hangar import serve

    if socket.has_ipv6:
        tcp_socket = socket.socket(socket.AF_INET6)
    else:
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
    server.stop(0.0)
    tcp_socket.close()


@pytest.mark.parametrize('it', range(5))
def test_server_is_started_multiple_times_via_ping_pong(server_instance, written_repo, it):
    # start multiple times and test that pings go through multiple times
    written_repo.add_remote('origin', server_instance)
    for i in range(it):
        assert written_repo._ping_server('origin') == 'PONG'


@pytest.mark.parametrize('nCommits', [1, 3, 10])
@pytest.mark.parametrize('nSamples', [20, 50])
def test_push_master_linear_history_multiple_commits(server_instance, repo, array5by7, nCommits, nSamples):

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


@pytest.mark.parametrize('nCommits', [1, 3, 10])
@pytest.mark.parametrize('nSamples', [20, 50])
def test_push_clone_master_linear_history_multiple_commits(
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


@pytest.mark.parametrize('nMasterCommits', [1, 7])
@pytest.mark.parametrize('nMasterSamples', [20, 50])
@pytest.mark.parametrize('nDevCommits', [1, 5])
@pytest.mark.parametrize('nDevSamples', [15, 45])
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


@pytest.mark.parametrize('nMasterCommits', [1, 7])
@pytest.mark.parametrize('nMasterSamples', [20, 50])
@pytest.mark.parametrize('nDevCommits', [1, 5])
@pytest.mark.parametrize('nDevSamples', [15, 45])
def test_server_push_clone_second_branch_with_new_commit(
        server_instance, repo, managed_tmpdir, array5by7, nMasterCommits,
        nMasterSamples, nDevCommits, nDevSamples):
    from hangar import Repository

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


def test_client_only_clone_master_branch_when_two_branch_two_commit(server_instance, managed_tmpdir, array5by7):
    from hangar import Repository

    base_tmpdir = pjoin(managed_tmpdir, 'base')
    mkdir(base_tmpdir)
    repo = Repository(path=base_tmpdir)
    repo.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    co.datasets['_dset']['0'] = array5by7.astype(np.float32)
    co.metadata['hello'] = 'world'
    co.commit('first')
    co.close()

    repo.add_remote('origin', server_instance)
    push1 = repo.push('origin', 'master')
    assert push1 is True

    branch = repo.create_branch('testbranch')
    co = repo.checkout(write=True, branch_name=branch)
    co.datasets['_dset']['1'] = array5by7.astype(np.float32)
    co.metadata.add('a', 'b')
    co.commit('this is a commit message')
    co.close()
    push2 = repo.push('origin', branch)
    assert push2 is True
    repo._env._close_environments()

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branch_names() == ['master']

    nco = newRepo.checkout()
    assert len(nco.datasets) == 1
    assert '_dset' in nco.datasets
    assert np.allclose(nco.datasets['_dset']['0'], array5by7.astype(np.float32))
    assert len(nco.metadata) == 1
    assert nco.metadata['hello'] == 'world'
    nco.close()
    newRepo._env._close_environments()