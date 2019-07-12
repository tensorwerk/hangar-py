import pytest
import numpy as np
import time
from os.path import join as pjoin
from os import mkdir
from random import randint
import platform


def test_cannot_add_remote_twice_with_same_name(repo):
    remote_spec = repo.remote.add('origin', 'test')
    assert remote_spec.name == 'origin'
    assert remote_spec.address == 'test'
    with pytest.raises(ValueError):
        repo.remote.add('origin', 'new')


def test_remote_remote_which_does_not_exist_fails(repo):
    with pytest.raises(ValueError):
        repo.remote.remove('origin')


def test_can_update_remote_after_removal(repo):
    remote_spec = repo.remote.add('origin', 'test')
    assert remote_spec.name == 'origin'
    assert remote_spec.address == 'test'
    channel_address_removed = repo.remote.remove('origin')
    assert channel_address_removed.name == 'origin'
    assert channel_address_removed.address == 'test'
    new_name = repo.remote.add('origin', 'test2')
    assert new_name.name == 'origin'
    assert new_name.address == 'test2'


def test_server_is_started_multiple_times_via_ping_pong(server_instance, written_repo):
    # start multiple times and test that pings go through multiple times
    written_repo.remote.add('origin', server_instance)
    roundTripTime = written_repo.remote.ping('origin')
    assert isinstance(roundTripTime, float)


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

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in cmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
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

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'

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

    push2 = repo.remote.push('origin', branch)
    assert push2 == branch


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

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'

    # Push dev branch test
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

    push2 = repo.remote.push('origin', branch)
    assert push2 == branch

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert '_dset' in nco.datasets
        assert len(nco.datasets['_dset']) == nMasterSamples
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['_dset'][str(sIdx)], samp)
        nco.close()

    # Fetch test
    fetch = newRepo.remote.fetch('origin', branch=branch)
    assert fetch == f'origin/{branch}'
    assert newRepo.list_branches() == ['master', 'origin/master', f'origin/{branch}']
    for cmt, sampList in devCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
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
    with pytest.warns(UserWarning):
        branchName = repo.remote.push('origin', 'master')
    assert branchName == 'master'


def test_fetch_unchanged_repo_makes_no_modifications(written_two_cmt_server_repo):
    _, repo = written_two_cmt_server_repo
    with pytest.warns(UserWarning):
        branchName = repo.remote.fetch('origin', 'master')
    assert branchName == 'master'


def test_fetch_newer_disk_repo_makes_no_modifications(written_two_cmt_server_repo):
    _, repo = written_two_cmt_server_repo
    co = repo.checkout(write=True)
    co.metadata['new_foo_abc'] = 'bar'
    co.commit('newer commit')
    co.close()
    with pytest.warns(UserWarning):
        branchName = repo.remote.fetch('origin', 'master')
    assert branchName == 'master'


def test_fetch_branch_which_does_not_exist_client_server_raises_rpc_error(written_two_cmt_server_repo):
    import grpc
    _, repo = written_two_cmt_server_repo
    with pytest.raises(grpc.RpcError) as rpc_error:
        repo.remote.fetch('origin', 'not-a-branch')
    assert rpc_error.value._state.code == grpc.StatusCode.NOT_FOUND


def test_fetch_branch_on_client_which_does_not_existserver_raises_rpc_error(written_two_cmt_server_repo):
    import grpc
    _, repo = written_two_cmt_server_repo
    repo.create_branch('new-branch')
    with pytest.raises(grpc.RpcError) as exc_info:
        repo.remote.fetch('origin', 'new-branch')
    assert exc_info.value._state.code == grpc.StatusCode.NOT_FOUND


def test_push_clone_three_way_merge(server_instance, repo_2_br_no_conf, managed_tmpdir):
    from hangar import Repository

    repo_2_br_no_conf.remote.add('origin', server_instance)
    push1 = repo_2_br_no_conf.remote.push('origin', 'master')
    assert push1 == 'master'
    push2 = repo_2_br_no_conf.remote.push('origin', 'testbranch')
    assert push2 == 'testbranch'

    test_head = repo_2_br_no_conf.log(branch_name='testbranch', return_contents=True)['head']
    master_head = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['head']

    merge_cmt = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    merge_head = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['head']
    merge_order = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['order']
    merge_push = repo_2_br_no_conf.remote.push('origin', 'master')
    assert merge_push == 'master'
    assert merge_head != master_head
    assert merge_head != test_head

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

    clone_head = newRepo.log(branch_name='master', return_contents=True)['head']
    clone_order = newRepo.log(branch_name='master', return_contents=True)['order']
    assert clone_head == merge_head == merge_cmt
    assert merge_order == clone_order
    newRepo._env._close_environments()


def test_push_clone_digests_exceeding_server_nbyte_limit(server_instance, repo, managed_tmpdir):
    from hangar.remote import config
    from hangar import Repository

    config.config['server']['grpc']['fetch_max_nbytes'] = 100_000
    config.config['client']['grpc']['push_max_nbytes'] = 100_000

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(4):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datasets['dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(70):
                arr = np.random.randn(50, 20).astype(np.float32)
                d[str(sIdx)] = arr
                masterSampList.append(arr)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, masterSampList))
        co.close()

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert 'dset' in nco.datasets
        assert len(nco.datasets['dset']) == 70
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_restricted_with_right_username_password(server_instance_push_restricted, repo, managed_tmpdir):
    from hangar import Repository

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datasets['dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(70):
                arr = np.random.randn(50, 20).astype(np.float32)
                d[str(sIdx)] = arr
                masterSampList.append(arr)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, masterSampList))
        co.close()

    repo.remote.add('origin', server_instance_push_restricted)
    push1 = repo.remote.push('origin',
                             'master',
                             username='right_username',
                             password='right_password')
    assert push1 == 'master'

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir)
    newRepo.clone('Test User', 'tester@foo.com', server_instance_push_restricted, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datasets) == 1
        assert 'dset' in nco.datasets
        assert len(nco.datasets['dset']) == 70
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datasets['dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_restricted_wrong_user_and_password(server_instance_push_restricted, repo, managed_tmpdir):

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datasets['dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(70):
                arr = np.random.randn(50, 20).astype(np.float32)
                d[str(sIdx)] = arr
                masterSampList.append(arr)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, masterSampList))
        co.close()

    repo.remote.add('origin', server_instance_push_restricted)
    with pytest.raises(PermissionError):
        push1 = repo.remote.push('origin',
                                 'master',
                                 username='wrong_username',
                                 password='right_password')

    with pytest.raises(PermissionError):
        push2 = repo.remote.push('origin',
                                 'master',
                                 username='right_username',
                                 password='wrong_password')

    with pytest.raises(PermissionError):
        push3 = repo.remote.push('origin',
                                 'master',
                                 username='wrong_username',
                                 password='wrong_password')