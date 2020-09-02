import pytest

import numpy as np
import time
from os.path import join as pjoin
from os import mkdir
from random import randint
import platform



@pytest.mark.parametrize('name', [
    'invalid\n', '\ninvalid', 'inv\name', 'inva/lid', 12, ' try', 'and this ',
    'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
def test_cannot_add_invalid_remote_names(repo, name):
    with pytest.raises(ValueError):
        repo.remote.add(name, 'localhost:50051')


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


def test_server_is_started_multiple_times_via_ping_pong(server_instance,
                                                        aset_samples_initialized_repo):
    # start multiple times and test that pings go through multiple times
    aset_samples_initialized_repo.remote.add('origin', server_instance)
    roundTripTime = aset_samples_initialized_repo.remote.ping('origin')
    assert isinstance(roundTripTime, float)


@pytest.mark.parametrize('nCommits,nSamples', [[1, 10], [5, 10]])
def test_push_and_clone_master_linear_history_multiple_commits(
        server_instance, repo, managed_tmpdir, array5by7, nCommits, nSamples):
    from hangar import Repository
    from hangar.records.summarize import list_history

    cmtList = []
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        sampList = []
        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
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
        assert len(nco.columns) == 1
        assert 'writtenaset' in nco.columns
        assert len(nco.columns['writtenaset']) == len(sampList)

        assert nco.columns['writtenaset'].contains_remote_references is True
        remoteKeys = nco.columns['writtenaset'].remote_reference_keys
        assert tuple([str(idx) for idx in range(len(sampList))]) == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.columns['writtenaset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.columns['writtenaset'][sIdx]
        nco.close()
    cloneMasterHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name='master')
    assert cloneMasterHist == masterHist
    newRepo._env._close_environments()


@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 4], [5, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 3], [3, 5]])
def test_server_push_second_branch_with_new_commit(server_instance, repo,
                                                   array5by7, nMasterCommits,
                                                   nMasterSamples, nDevCommits,
                                                   nDevSamples):

    masterCmtList, devCmtList = [], []
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
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
        co = repo.checkout(write=True, branch=branch.name)
        devSampList = []
        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
            for sIdx in range(nDevSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                devSampList.append(arr)
        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmtList.append((cmt, devSampList))
        co.close()

    push2 = repo.remote.push('origin', branch.name)
    assert push2 == branch.name


@pytest.mark.parametrize('nMasterCommits,nMasterSamples', [[1, 4], [5, 10]])
@pytest.mark.parametrize('nDevCommits,nDevSamples', [[1, 5], [3, 5]])
def test_server_push_second_branch_with_new_commit_then_clone_partial_fetch(
        server_instance, repo, managed_tmpdir, array5by7, nMasterCommits,
        nMasterSamples, nDevCommits, nDevSamples):
    from hangar import Repository
    from hangar.records.summarize import list_history

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
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
        co = repo.checkout(write=True, branch=branch.name)
        devSampList = []
        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
            for sIdx in range(nDevSamples):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
                devSampList.append(arr)
        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmtList.append((cmt, devSampList))
        co.close()

    push2 = repo.remote.push('origin', branch.name)
    assert push2 == branch.name
    branchHist = list_history(repo._env.refenv, repo._env.branchenv, branch_name=branch.name)

    # Clone test (master branch)
    new_tmpdir = pjoin(managed_tmpdir, 'new')
    mkdir(new_tmpdir)
    newRepo = Repository(path=new_tmpdir, exists=False)
    newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
    assert newRepo.list_branches() == ['master', 'origin/master']
    for cmt, sampList in masterCmtList:
        with pytest.warns(UserWarning):
            nco = newRepo.checkout(commit=cmt)
        assert len(nco.columns) == 1
        assert 'writtenaset' in nco.columns
        assert len(nco.columns['writtenaset']) == nMasterSamples

        assert nco.columns['writtenaset'].contains_remote_references is True
        remoteKeys = nco.columns['writtenaset'].remote_reference_keys
        assert tuple([str(idx) for idx in range(len(sampList))]) == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.columns['writtenaset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.columns['writtenaset'][sIdx]
        nco.close()
    cloneMasterHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name='master')
    assert cloneMasterHist == masterHist

    # Fetch test
    fetch = newRepo.remote.fetch('origin', branch=branch.name)
    assert fetch == f'origin/{branch.name}'
    assert newRepo.list_branches() == ['master', 'origin/master', f'origin/{branch.name}']
    for cmt, sampList in devCmtList:

        with pytest.warns(UserWarning):
            nco = newRepo.checkout(commit=cmt)
        assert len(nco.columns) == 1
        assert 'writtenaset' in nco.columns
        assert len(nco.columns['writtenaset']) == nDevSamples

        assert nco.columns['writtenaset'].contains_remote_references is True
        remoteKeys = nco.columns['writtenaset'].remote_reference_keys
        assert tuple([str(idx) for idx in range(len(sampList))]) == remoteKeys
        for idx, _ in enumerate(sampList):
            sIdx = str(idx)
            assert sIdx in nco.columns['writtenaset']
            with pytest.raises(FileNotFoundError):
                shouldNotExist = nco.columns['writtenaset'][sIdx]
        nco.close()

    cloneBranchHist = list_history(newRepo._env.refenv, newRepo._env.branchenv, branch_name=f'origin/{branch.name}')
    assert cloneBranchHist == branchHist
    newRepo._env._close_environments()


@pytest.fixture(scope='class')
def array5by7_class():
    return np.random.random((5, 7))

@pytest.fixture(scope='class')
def two_branch_multi_commit_repo_class(server_instance_class, classrepo, array5by7_class):
    from hangar.records.summarize import list_history

    nMasterCommits = 2
    nMasterSamples = 10
    nDevCommits = 1
    nDevSamples = 16

    # Push master branch test
    masterCmts = {}
    co = classrepo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float32)
    co.add_ndarray_column(name='_two', shape=(20), dtype=np.float32)
    co.add_str_column('str_col')
    co.add_bytes_column('bytes_col')
    for cIdx in range(nMasterCommits):
        if cIdx != 0:
            co = classrepo.checkout(write=True)
        masterSampList1 = []
        masterSampList2 = []
        masterSampList3 = []
        masterSampList4 = []
        with co.columns['writtenaset'] as d,\
                co.columns['_two'] as dd,\
                co.columns['str_col'] as scol, \
                co.columns['bytes_col'] as bcol:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
                del dd[prevKey]
                del scol[prevKey]
                del bcol[prevKey]

            for sIdx in range(nMasterSamples):
                arr1 = np.random.randn(*array5by7_class.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr1
                masterSampList1.append(arr1)
                arr2 = np.random.randn(20).astype(np.float32)
                dd[str(sIdx)] = arr2
                masterSampList2.append(arr2)
                sval = f'strval master {cIdx} {sIdx}'
                scol[str(sIdx)] = sval
                masterSampList3.append(sval)
                bval = f'bytesval master {cIdx} {sIdx}'.encode()
                bcol[str(sIdx)] = bval
                masterSampList4.append(bval)

        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmts[cmt] = (masterSampList1, masterSampList2, masterSampList3, masterSampList4)
        co.close()

    classrepo.remote.add('origin', server_instance_class)
    push1 = classrepo.remote.push('origin', 'master')
    assert push1 == 'master'
    masterHist = list_history(classrepo._env.refenv, classrepo._env.branchenv, branch_name='master')

    # Push dev branch test
    devCmts = masterCmts.copy()
    branch = classrepo.create_branch('testbranch')
    for cIdx in range(nDevCommits):
        co = classrepo.checkout(write=True, branch=branch.name)
        devSampList1 = []
        devSampList2 = []
        devSampList3 = []
        devSampList4 = []
        with co.columns['writtenaset'] as d,\
                co.columns['_two'] as dd,\
                co.columns['str_col'] as scol, \
                co.columns['bytes_col'] as bcol:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
                del dd[prevKey]
                del scol[prevKey]
                del bcol[prevKey]

            for sIdx in range(nDevSamples):
                arr1 = np.random.randn(*array5by7_class.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr1
                devSampList1.append(arr1)
                arr2 = np.random.randn(20).astype(np.float32)
                dd[str(sIdx)] = arr2
                devSampList2.append(arr2)
                sval = f'strval dev {cIdx} {sIdx}'
                scol[str(sIdx)] = sval
                devSampList3.append(sval)
                bval = f'bytesval dev {cIdx} {sIdx}'.encode()
                bcol[str(sIdx)] = bval
                devSampList4.append(bval)

        cmt = co.commit(f'dev commit number: {cIdx}')
        devCmts[cmt] = (devSampList1, devSampList2, devSampList3, devSampList4)
        co.close()

    push2 = classrepo.remote.push('origin', branch.name)
    assert push2 == branch.name
    branchHist = list_history(classrepo._env.refenv, classrepo._env.branchenv, branch_name=branch.name)

    yield branch, branchHist, devCmts, masterHist, server_instance_class
    pass


class TestLargeRemoteServer:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('fetchAsetns', [
        None, ('writtenaset',), ('_two',), ('str_col',), ('bytes_col',),
    ])
    @pytest.mark.parametrize('fetchBranch', [None, 'testbranch'])
    @pytest.mark.parametrize('fetchCommit', [None, 'ma'])
    @pytest.mark.parametrize('fetchAll_history', [False, True])
    def test_server_push_two_branch_then_clone_fetch_data_options(
            self, two_branch_multi_commit_repo_class, managed_tmpdir_class, array5by7_class,
            fetchBranch, fetchCommit, fetchAsetns, fetchAll_history, tmp_path_factory):
        from hangar import Repository
        from operator import eq

        branch, branchHist, devCmts, masterHist, server_instance = two_branch_multi_commit_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)
        newRepo.remote.fetch('origin', branch=branch.name)
        newRepo.create_branch('testbranch', base_commit=branchHist['head'])
        assert newRepo.list_branches() == ['master', 'origin/master', f'origin/{branch.name}', branch.name]

        # ------------------ format arguments depending on options -----------------

        kwargs = {
            'column_names': fetchAsetns,
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

        # get data
        fetch_commits = newRepo.remote.fetch_data(remote='origin', **kwargs)
        assert commits_to_check == fetch_commits

        # ------------- check that you got everything you expected ----------------

        for fCmt in fetch_commits:
            co = newRepo.checkout(commit=fCmt)
            assert co.commit_hash == fCmt

            # when we are checking one aset only
            if isinstance(fetchAsetns, tuple):
                d = co.columns[fetchAsetns[0]]
                # ensure we didn't fetch the other data simultaneously
                ds1SampList, ds2SampList, ds3SampList, ds4SampList = devCmts[fCmt]
                if fetchAsetns[0] == 'writtenaset':
                    compare = ds1SampList
                    cmp_func = np.allclose
                elif fetchAsetns[0] == '_two':
                    compare = ds2SampList
                    cmp_func = np.allclose
                elif fetchAsetns[0] == 'str_col':
                    compare = ds3SampList
                    cmp_func = eq
                else:
                    compare = ds4SampList
                    cmp_func = eq

                for idx, samp in enumerate(compare):
                    assert cmp_func(samp, d[str(idx)])

            # compare both asets at the same time
            else:
                d = co.columns['writtenaset']
                dd = co.columns['_two']
                str_col = co.columns['str_col']
                bytes_col = co.columns['bytes_col']
                ds1List, ds2List, ds3List, ds4List = devCmts[fCmt]
                for idx, ds1ds2ds3ds4 in enumerate(zip(ds1List, ds2List, ds3List, ds4List)):
                    ds1, ds2, ds3, ds4 = ds1ds2ds3ds4
                    assert np.allclose(ds1, d[str(idx)])
                    assert np.allclose(ds2, dd[str(idx)])
                    assert ds3 == str_col[str(idx)]
                    assert ds4 == bytes_col[str(idx)]
            co.close()
        newRepo._env._close_environments()


@pytest.fixture(scope='class')
def two_multi_format_repo_class(server_instance_class, classrepo):

    co = classrepo.checkout(write=True)
    array_flat = co.add_ndarray_column(name='array_flat', shape=(5, 7), dtype=np.float32)
    array_nested = co.add_ndarray_column(name='array_nested', shape=(20,), dtype=np.float32, contains_subsamples=True)
    str_flat = co.add_str_column('str_flat')
    str_nested = co.add_str_column('str_nested', contains_subsamples=True)
    bytes_flat = co.add_bytes_column('bytes_flat')
    bytes_nested = co.add_bytes_column('bytes_nested', contains_subsamples=True)

    for i in range(5):
        arr = np.random.randn(5, 7).astype(np.float32)
        array_flat[i] = arr
    for i in range(5):
        data = {f'{idx}': np.random.randn(20).astype(np.float32) for idx in range(4)}
        array_nested[i] = data

    for i in range(5):
        str_flat[i] = f'string_{i}' * (i + 1)
    for i in range(5):
        data = {f'{idx}': f'string_{idx}' * (idx + 1) for idx in range(4)}
        str_nested[i] = data

    for i in range(5):
        bytes_flat[i] = f'bytes_{i}'.encode() * (i + 1)
    for i in range(5):
        data = {f'{idx}': f'bytes_{idx}'.encode() * (idx + 1) for idx in range(4)}
        bytes_nested[i] = data

    cmt = co.commit('first commit')
    co.close()
    classrepo.remote.add('origin', server_instance_class)
    classrepo.remote.push('origin', 'master')

    yield cmt, server_instance_class
    pass


class TestRemoteServerFetchDataSample:

    @pytest.mark.filterwarnings('ignore::UserWarning')
    @pytest.mark.parametrize('fetchOp', ['branch', 'commit'])
    @pytest.mark.parametrize('column_name,keys', [
        ('array_flat', 0),
        ('array_flat', [0, 1]),
        ('array_nested', 0),
        ('array_nested', [0, 1]),
        ('array_nested', [(0, '0')]),
        ('array_nested', [(0, '0'), (1, '1')]),
        ('array_nested', [0, (1, '1')]),
        ('array_nested', [(0,)]),
        ('array_nested', [(0, ...)]),
        ('array_nested', [(0, ...), 1, (2, '2')]),
        ('str_flat', 0),
        ('str_flat', [0, 1]),
        ('str_nested', 0),
        ('str_nested', [0, 1]),
        ('str_nested', [(0, '0')]),
        ('str_nested', [(0, '0'), (1, '1')]),
        ('str_nested', [0, (1, '1')]),
        ('str_nested', [(0,)]),
        ('str_nested', [(0, ...)]),
        ('str_nested', [(0, ...), 1, (2, '2')]),
        ('bytes_flat', 0),
        ('bytes_flat', [0, 1]),
        ('bytes_nested', 0),
        ('bytes_nested', [0, 1]),
        ('bytes_nested', [(0, '0')]),
        ('bytes_nested', [(0, '0'), (1, '1')]),
        ('bytes_nested', [0, (1, '1')]),
        ('bytes_nested', [(0,)]),
        ('bytes_nested', [(0, ...)]),
        ('bytes_nested', [(0, ...), 1, (2, '2')]),
    ])
    def test_server_fetch_data_sample(
            self, two_multi_format_repo_class, managed_tmpdir_class,
            fetchOp, column_name, keys, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        # ------------------ format arguments depending on options -----------------

        kwargs = {
            'column': column_name,
            'samples': keys
        }
        if fetchOp == 'branch':
            kwargs['branch'] = 'master'
        elif fetchOp == 'commit':
            kwargs['commit'] = cmt
        else:
            raise ValueError(f'fetchOp unknown: {fetchOp}')

        fetch_commit = newRepo.remote.fetch_data_sample(remote='origin', **kwargs)
        assert fetch_commit == cmt

        co = newRepo.checkout()
        try:
            col = co[column_name]
            if isinstance(keys, (list, tuple)):
                if column_name.endswith('flat'):
                    for key in keys:
                        assert col[key] is not None
                else:
                    for sample in keys:
                        if isinstance(sample, (list, tuple)):
                            if len(sample) == 2:
                                assert col[sample[0]][sample[1]] is not None
                            elif len(sample) == 1:
                                assert col[sample[0]][...] is not None
                        else:
                            assert col[sample][...] is not None
        finally:
            co.close()
            newRepo._env._close_environments()

    def test_server_fetch_data_sample_commit_not_existing(
            self, two_multi_format_repo_class, managed_tmpdir_class, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        with pytest.raises(ValueError, match='specified commit'):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                commit='DOESNOTEXISTCOMMIT',
                column='array_flat',
                samples=[0, 1])

        newRepo._env._close_environments()

    def test_server_fetch_data_sample_branch_not_existing(
            self, two_multi_format_repo_class, managed_tmpdir_class, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        with pytest.raises(ValueError, match='branch with name'):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='DOESNOTEXISTBRANCH',
                column='array_flat',
                samples=[0, 1])

        newRepo._env._close_environments()

    def test_server_fetch_data_sample_branch_and_commit_args_passed_fails(
            self, two_multi_format_repo_class, managed_tmpdir_class, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        with pytest.raises(ValueError, match='``branch`` and ``commit``'):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',  # actual value which might otherwise work
                commit=cmt,  # actual value which might otherwise work
                column='array_flat',
                samples=[0, 1])

        newRepo._env._close_environments()

    def test_server_fetch_data_sample_not_existing_fails(
            self, two_multi_format_repo_class, managed_tmpdir_class, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        with pytest.raises(KeyError):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',
                column='array_flat',
                samples=['DOESNOTEXIST'])

        with pytest.raises(KeyError):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',
                column='array_nested',
                samples=[(1, 'DOESNOTEXIST')])

        with pytest.raises(KeyError):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',
                column='array_nested',
                samples=[('DOESNOTEXIST', 0)])

        newRepo._env._close_environments()

    def test_server_fetch_data_sample_not_valid_type(
            self, two_multi_format_repo_class, managed_tmpdir_class, tmp_path_factory
    ):
        from hangar import Repository

        cmt, server_instance = two_multi_format_repo_class

        # Clone test (master branch)
        _new_tmpdir = tmp_path_factory.mktemp('newclone', numbered=True)
        new_tmpdir = str(_new_tmpdir)
        newRepo = Repository(path=new_tmpdir, exists=False)
        newRepo.clone('Test User', 'tester@foo.com', server_instance, remove_old=True)

        with pytest.raises(TypeError):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',
                column='array_flat',
                samples=[b'BYTES_TYPE_NOT_VALID'])

        with pytest.raises(ValueError, match='nested column specifier sequence'):
            newRepo.remote.fetch_data_sample(
                remote='origin',
                branch='master',
                column='array_nested',
                samples=[(0, 1, 'ARRAY_NOT_VALID')])

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
    co.add_str_column('test_meta')
    co['test_meta'][0] = 'lol'
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


# -----------------------------------------------------------------------------


def test_push_restricted_with_right_username_password(server_instance_push_restricted, repo, managed_tmpdir):
    from hangar import Repository

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='aset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.columns['aset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
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
        assert len(nco.columns) == 1
        assert 'aset' in nco.columns
        assert len(nco.columns['aset']) == 70
        for sIdx, samp in enumerate(sampList):
            assert np.allclose(nco.columns['aset'][str(sIdx)], samp)
        nco.close()
    newRepo._env._close_environments()


def test_push_restricted_wrong_user_and_password(server_instance_push_restricted, repo, managed_tmpdir):

    # Push master branch test
    masterCmtList = []
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='aset', shape=(50, 20), dtype=np.float32)
    for cIdx in range(1):
        if cIdx != 0:
            co = repo.checkout(write=True)
        masterSampList = []
        with co.columns['aset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
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
