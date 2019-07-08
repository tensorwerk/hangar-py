import shutil
import random

import pytest
import numpy as np

from hangar import Repository
import hangar


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    '''
    cleanup all singleton instances anyway before each test, to ensure
    no leaked state between tests.
    '''
    hangar.context.TxnRegisterSingleton._instances = {}
    monkeypatch.setitem(hangar.constants.LMDB_SETTINGS, 'map_size', 5_000_000)


@pytest.fixture()
def managed_tmpdir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture()
def repo(managed_tmpdir) -> Repository:
    repo_obj = Repository(path=managed_tmpdir)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    yield repo_obj
    repo_obj._env._close_environments()


@pytest.fixture()
def written_repo(repo):
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float64)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def variable_shape_written_repo(repo):
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(10, 10), dtype=np.float64, variable_shape=True)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def w_checkout(written_repo):
    co = written_repo.checkout(write=True)
    yield co
    co.close()


@pytest.fixture()
def array5by7(scope='session'):
    return np.random.random((5, 7))


@pytest.fixture()
def randomsizedarray():
    a = random.randint(2, 8)
    b = random.randint(2, 8)
    return np.random.random((a, b))


@pytest.fixture()
def written_two_cmt_repo(repo, array5by7):
    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(2):
        if cIdx != 0:
            co = repo.checkout(write=True)

        with co.datasets['_dset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range((cIdx + 1) * 5):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
        co.commit(f'commit number: {cIdx}')
        co.close()

    yield repo


@pytest.fixture(params=['00', '01'])
def repo_1_br_no_conf(request, repo):

    dummyData = np.arange(50)
    co1 = repo.checkout(write=True, branch_name='master')
    co1.datasets.init_dataset(
        name='dummy', prototype=dummyData, named_samples=True, backend=request.param)
    for idx in range(10):
        dummyData[:] = idx
        co1.datasets['dummy'][str(idx)] = dummyData
    co1.metadata['hello'] = 'world'
    co1.commit('first commit adding dummy data and hello meta')
    co1.close()

    repo.create_branch('testbranch')
    co2 = repo.checkout(write=True, branch_name='testbranch')
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.datasets['dummy'][str(idx)] = dummyData
    co2.metadata['foo'] = 'bar'
    co2.commit('first commit on test branch adding non-conflict data and meta')
    co2.close()
    return repo


@pytest.fixture()
def repo_2_br_no_conf(repo_1_br_no_conf):

    dummyData = np.arange(50)
    repo = repo_1_br_no_conf
    co1 = repo.checkout(write=True, branch_name='master')
    for idx in range(20, 30):
        dummyData[:] = idx
        co1.datasets['dummy'][str(idx)] = dummyData
    co1.commit('second commit on master adding non-conflict data')
    co1.close()
    return repo
