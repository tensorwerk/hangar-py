import pytest
import numpy as np
import time
from os.path import join as pjoin
from os import mkdir
from random import randint
import platform


def test_list_all_remotes_works(repo):

    remote_spec1 = repo.remote.add('origin', 'test')
    currentRemotes = repo.remote.list_all()

    assert len(currentRemotes) == 1
    currentSpec = currentRemotes[0]
    assert len(currentSpec) == 2
    assert currentSpec.name == 'origin'
    assert currentSpec.address == 'test'

    remote_spec2 = repo.remote.add('origin2', 'test2')
    currentRemotes = repo.remote.list_all()

    assert len(currentRemotes) == 2
    currentSpec = currentRemotes[0]
    assert currentSpec == remote_spec1
    assert len(currentSpec) == 2
    assert currentSpec.name == 'origin'
    assert currentSpec.address == 'test'
    currentSpec = currentRemotes[1]
    assert currentSpec == remote_spec2
    assert currentSpec.name == 'origin2'
    assert currentSpec.address == 'test2'


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
    from hangar.records.summarize import list_history

    cmtList = []
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        sampList = []
        with co.datacells['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range(nSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                sampList.append(arr)
        cmt = co.commit(f'commit number: {cIdx}')
        cmtList.append((cmt, sampList))
        co.close()
    masterHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name='master')

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in cmtList:
        with pytest.warns(UserWarning):
            nco = newRepo.checkout(commit=cmt)
        assert len(nco.datacells) == 1
        assert '_dset' in nco.datacells
        assert len(nco.datacells['_dset']) == len(sampList)

        assert nco.datacells['_dset'].contains_remote_references is True
        remoteKeys = nco.datacells['_dset'].remote_reference_sample_keys
        assert [str(idx) for idx in range(len(sampList))] == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.datacells['_dset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.datacells['_dset'][sIdx]
        nco.close()
    cloneMasterHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name='master')
    assert cloneMasterHist == masterHist
    newRepo._env._close_environments()



@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 4], [10, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 3], [5, 5]])
def test_server_push_second_branch_with_new_commit(server_instance, repo,
                                                   array5by7, nMasterCommits,
                                                   nMasterSamples, nDevCommits,
                                                   nDevSamples):

    masterCmtList, devCmtList = [], []
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datacells['_dset'] as d:
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
        co = repo.checkout(write=True, branch=branch)
        devSampList = []
        with co.datacells['_dset'] as d:
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


@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 4], [10, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 5], [3, 5]])
def test_server_push_second_branch_with_new_commit_then_clone_partial_fetch(
        server_instance, repo, managed_tmpdir, array5by7, nMasterCommits,
        nMasterSamples, nDevCommits, nDevSamples):
    from hangar import Repository
    from hangar.records.summarize import list_history

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datacells['_dset'] as d:
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
    masterHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name='master')

    # Push dev branch test
    devCmtList = []
    branch = repo.create_branch('testbranch')
    for cIdx in range(nDevCommits):
        co = repo.checkout(write=True, branch=branch)
        devSampList = []
        with co.datacells['_dset'] as d:
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
    branchHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name=branch)

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        with pytest.warns(UserWarning):
            nco = newRepo.checkout(commit=cmt)
        assert len(nco.datacells) == 1
        assert '_dset' in nco.datacells
        assert len(nco.datacells['_dset']) == nMasterSamples

        assert nco.datacells['_dset'].contains_remote_references is True
        remoteKeys = nco.datacells['_dset'].remote_reference_sample_keys
        assert [str(idx) for idx in range(len(sampList))] == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.datacells['_dset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.datacells['_dset'][sIdx]
        nco.close()
    cloneMasterHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name='master')
    assert cloneMasterHist == masterHist

    # Fetch test
    fetch = newRepo.remote.fetch('origin', branch=branch)
    assert fetch == f'origin/{branch}'
    assert newRepo.list_branches() == ['master', 'origin/master', f'origin/{branch}']
    for cmt, sampList in devCmtList:
        # newRepo.remote.fetch_data('origin', commit=cmt)
        with pytest.warns(UserWarning):
            nco = newRepo.checkout(commit=cmt)
        assert len(nco.datacells) == 1
        assert '_dset' in nco.datacells
        assert len(nco.datacells['_dset']) == nDevSamples

        assert nco.datacells['_dset'].contains_remote_references is True
        remoteKeys = nco.datacells['_dset'].remote_reference_sample_keys
        assert [str(idx) for idx in range(len(sampList))] == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.datacells['_dset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.datacells['_dset'][sIdx]
            # assert np.allclose(nco.datacells['_dset'][str(sIdx)], samp)
        nco.close()
    cloneBranchHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name=f'origin/{branch}')
    assert cloneBranchHist == branchHist
    newRepo._env._close_environments()


@pytest.mark.filterwarnings('ignore::UserWarning')
@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[4, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[3, 24]])
@pytest.mark.parametrize('fetchBranch,fetchCommit,fetchDsetns,fetchNbytes,fetchAll_history', [
    ['master',      None,  None,        None,  False],
    ['testbranch',  None,  None,        None,  False],
    [None,          'ma',  None,        None,  False],
    [None,          'br',  None,        None,  False],
    ['master',      None,  ('_dset',),  None,  False],
    ['testbranch',  None,  ('_two',),   None,  False],
    [None,          'ma',  ('_dset',),  None,  False],
    [None,          'br',  ('_two',),   None,  False],
    ['master',      None,  None,        None,  False],
    ['testbranch',  None,  None,        None,  False],
    [None,          'ma',  None,        None,  False],
    [None,          'br',  None,        None,  False],
    ['master',      None,  ('_dset',),  None,  False],
    ['testbranch',  None,  ('_two',),   None,  False],
    [None,          'ma',  ('_dset',),  None,  False],
    [None,          'br',  ('_two',),   None,  False],
    ['master',      None,  None,        None,  True],
    ['testbranch',  None,  None,        None,  True],
    [None,          'ma',  None,        None,  True],
    [None,          'br',  None,        None,  True],
    ['master',      None,  ('_dset',),  None,  True],
    ['testbranch',  None,  ('_two',),   None,  True],
    [None,          'ma',  ('_dset',),  None,  True],
    [None,          'br',  ('_two',),   None,  True],
    ['master',      None,  None,        1000,  False],
    ['testbranch',  None,  None,        1000,  False],
    [None,          'ma',  None,        1000,  False],
    [None,          'br',  None,        1000,  False],
    ['master',      None,  ('_dset',),  1000,  False],
    ['testbranch',  None,  ('_two',),   1000,  False],
    [None,          'ma',  ('_dset',),  1000,  False],
    [None,          'br',  ('_two',),   1000,  False],
    [None,          'br',  ('_two',),   1000,  True],  # will raise error
])
def test_server_push_two_branch_then_clone_fetch_data_options(
        server_instance, repo, managed_tmpdir, array5by7, nMasterCommits,
        nMasterSamples, nDevCommits, nDevSamples, fetchBranch, fetchCommit,
        fetchDsetns, fetchNbytes, fetchAll_history):
    from hangar import Repository
    from hangar.records.summarize import list_history

    # Push master branch test
    masterCmts = {}
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='_dset', shape=(5, 7), dtype=np.float32)
    co.datacells.init_datacell(name='_two', shape=(20), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList1 = []
        masterSampList2 = []
        with co.datacells['_dset'] as d, co.datacells['_two'] as dd:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
                dd.remove(prevKey)

            for sIdx in range(nMasterSamples):
                arr1 = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr1
                masterSampList1.append(arr1)
                arr2 = np.random.randn(20).astype(np.float32)
                dd[str(sIdx)] = arr2
                masterSampList2.append(arr2)
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmts[cmt] = (masterSampList1, masterSampList2)
        co.close()

    repo.remote.add('origin', server_instance)
    push1 = repo.remote.push('origin', 'master')
    assert push1 == 'master'
    masterHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name='master')

    # Push dev branch test
    devCmts = masterCmts.copy()
    branch = repo.create_branch('testbranch')
    for cIdx in range(nDevCommits):
        co = repo.checkout(write=True, branch=branch)
        devSampList1 = []
        devSampList2 = []
        with co.datacells['_dset'] as d, co.datacells['_two'] as dd:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
                dd.remove(prevKey)

            for sIdx in range(nDevSamples):
                arr1 = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr1
                devSampList1.append(arr1)
                arr2 = np.random.randn(20).astype(np.float32)
                dd[str(sIdx)] = arr2
                devSampList2.append(arr2)
        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmts[cmt] = (devSampList1, devSampList2)
        co.close()

    push2 = repo.remote.push('origin', branch)
    assert push2 == branch
    branchHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name=branch)

    # -------------------------- end setup ------------------------------------

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    newRepo.remote.fetch('origin', branch=branch)
    newRepo.create_branch('testbranch', base_commit=branchHist['head'])
    assert newRepo.list_branches() == ['master', 'origin/master', f'origin/{branch}', branch]

    # ------------------ format arguments dependingon options -----------------

    kwargs = {
        'datacell_names': fetchDsetns,
        'max_num_bytes': fetchNbytes,
        'retrieve_all_history': fetchAll_history,
    }
    if fetchBranch is not None:
        func = branchHist if fetchBranch == 'testbranch' else masterHist
        kwargs['branch'] = fetchBranch
        kwargs['commit'] = None
    else:
        func = branchHist if fetchBranch == 'br' else masterHist
        kwargs['branch'] = None
        kwargs['commit'] = func['head']

    if fetchAll_history is True:
        commits_to_check = func['order']
    else:
        commits_to_check = [func['head']]

    # ----------------------- retrieve data with desired options --------------

    # This case should fail
    if (fetchAll_history is True) and isinstance(fetchNbytes, int):
        try:
            with pytest.raises(ValueError):
                fetch_commits = newRepo.remote.fetch_data(remote='origin', **kwargs)
        finally:
            newRepo._env._close_environments()
        return True
    # get data
    fetch_commits = newRepo.remote.fetch_data(remote='origin', **kwargs)
    assert commits_to_check == fetch_commits

    # ------------- check that you got everything you expected ----------------

    for fCmt in fetch_commits:
        co = newRepo.checkout(commit=fCmt)
        assert co.commit_hash == fCmt

        # when we are checking one dset only
        if isinstance(fetchDsetns, tuple):
            d = co.datacells[fetchDsetns[0]]
            # ensure we didn't fetch the other data simultaneously

            ds1SampList, ds2SampList = devCmts[fCmt]
            if fetchDsetns[0] == '_dset':
                compare = ds1SampList
            else:
                compare = ds2SampList

            totalSeen = 0
            for idx, samp in enumerate(compare):
                if fetchNbytes is None:
                    assert np.allclose(samp, d[str(idx)])
                else:
                    try:
                        arr = d[str(idx)]
                        assert np.allclose(samp, arr)
                        totalSeen += arr.nbytes
                    except FileNotFoundError:
                        pass
                    assert totalSeen <= fetchNbytes

        # compare both dsets at the same time
        else:
            d = co.datacells['_dset']
            dd = co.datacells['_two']
            ds1List, ds2List = devCmts[fCmt]
            totalSeen = 0
            for idx, ds1ds2 in enumerate(zip(ds1List, ds2List)):
                ds1, ds2 = ds1ds2
                if fetchNbytes is None:
                    assert np.allclose(ds1, d[str(idx)])
                    assert np.allclose(ds2, dd[str(idx)])
                else:
                    try:
                        arr1 = d[str(idx)]
                        assert np.allclose(ds1, arr1)
                        totalSeen += arr1.nbytes
                    except FileNotFoundError:
                        pass
                    try:
                        arr2 = dd[str(idx)]
                        assert np.allclose(ds2, arr2)
                        totalSeen += arr2.nbytes
                    except FileNotFoundError:
                        pass
                    assert totalSeen <= fetchNbytes
        co.close()
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

    test_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    master_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']

    merge_cmt = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    merge_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']
    merge_order = repo_2_br_no_conf.log(branch='master', return_contents=True)['order']
    merge_push = repo_2_br_no_conf.remote.push('origin', 'master')
    assert merge_push == 'master'
    assert merge_head != master_head
    assert merge_head != test_head

    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

    clone_head = newRepo.log(branch='master', return_contents=True)['head']
    clone_order = newRepo.log(branch='master', return_contents=True)['order']
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
    co.datacells.init_datacell(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(4):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datacells['dset'] as d:
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
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datacells) == 1
        assert 'dset' in nco.datacells
        assert len(nco.datacells['dset']) == 70
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datacells['dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_restricted_with_right_username_password(server_instance_push_restricted, repo, managed_tmpdir):
    from hangar import Repository

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datacells['dset'] as d:
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
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance_push_restricted, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        newRepo.remote.fetch_data('origin', commit=cmt)
        nco = newRepo.checkout(commit=cmt)
        assert len(nco.datacells) == 1
        assert 'dset' in nco.datacells
        assert len(nco.datacells['dset']) == 70
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.datacells['dset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_restricted_wrong_user_and_password(server_instance_push_restricted, repo, managed_tmpdir):

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.datacells.init_datacell(name='dset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.datacells['dset'] as d:
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
