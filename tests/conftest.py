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
