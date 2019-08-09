import time
import shutil
import random
from random import randint
import platform
from os.path import join as pjoin
from os import mkdir

import pytest
import numpy as np

from hangar import Repository
import hangar


@pytest.fixture(autouse=True)
def reset_singletons(monkeypatch):
    """
    cleanup all singleton instances anyway before each test, to ensure
    no leaked state between tests.
    """
    hangar.context.TxnRegisterSingleton._instances = {}
    monkeypatch.setitem(hangar.constants.LMDB_SETTINGS, 'map_size', 2_000_000)


@pytest.fixture()
def managed_tmpdir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture()
def repo(managed_tmpdir) -> Repository:
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    yield repo_obj
    repo_obj._env._close_environments()


@pytest.fixture()
def written_repo(repo):
    co = repo.checkout(write=True)
    co.arraysets.init_arrayset(name='_aset', shape=(5, 7), dtype=np.float64)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def repo_with_20_samples(written_repo, array5by7):
    co = written_repo.checkout(write=True)
    second_aset = co.arraysets.init_arrayset('second_aset', prototype=array5by7)
    first_aset = co.arraysets['_aset']
    for i in range(20):
        array5by7[:] = i
        first_aset[str(i)] = array5by7
        second_aset[str(i)] = array5by7
    co.commit('20 samples')
    co.close()
    yield written_repo


@pytest.fixture()
def variable_shape_written_repo(repo):
    co = repo.checkout(write=True)
    co.arraysets.init_arrayset(name='_aset', shape=(10, 10), dtype=np.float64, variable_shape=True)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def w_checkout(written_repo):
    co = written_repo.checkout(write=True)
    yield co
    co.close()


@pytest.fixture()
def array5by7():
    return np.random.random((5, 7))


@pytest.fixture()
def randomsizedarray():
    a = random.randint(2, 8)
    b = random.randint(2, 8)
    return np.random.random((a, b))


@pytest.fixture()
def written_two_cmt_repo(repo, array5by7):
    co = repo.checkout(write=True)
    co.arraysets.init_arrayset(name='_aset', shape=(5, 7), dtype=np.float32)
    for cIdx in range(2):
        if cIdx != 0:
            co = repo.checkout(write=True)

        with co.arraysets['_aset'] as d:
            for prevKey in list(d.keys())[1:]:
                d.remove(prevKey)
            for sIdx in range((cIdx + 1) * 5):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
        co.commit(f'commit number: {cIdx}')
        co.close()
    yield repo


@pytest.fixture(params=['00', '10'])
def repo_1_br_no_conf(request, repo):

    dummyData = np.arange(50)
    co1 = repo.checkout(write=True, branch='master')
    co1.arraysets.init_arrayset(
        name='dummy', prototype=dummyData, named_samples=True, backend=request.param)
    for idx in range(10):
        dummyData[:] = idx
        co1.arraysets['dummy'][str(idx)] = dummyData
    co1.metadata['hello'] = 'world'
    co1.metadata['somemetadatakey'] = 'somemetadatavalue'
    co1.commit('first commit adding dummy data and hello meta')
    co1.close()

    repo.create_branch('testbranch')
    co2 = repo.checkout(write=True, branch='testbranch')
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.arraysets['dummy'][str(idx)] = dummyData
        co2.arraysets['dummy'][idx] = dummyData
    co2.metadata['foo'] = 'bar'
    co2.commit('first commit on test branch adding non-conflict data and meta')
    co2.close()
    return repo


@pytest.fixture()
def repo_2_br_no_conf(repo_1_br_no_conf):

    dummyData = np.arange(50)
    repo = repo_1_br_no_conf
    co1 = repo.checkout(write=True, branch='master')
    for idx in range(20, 30):
        dummyData[:] = idx
        co1.arraysets['dummy'][str(idx)] = dummyData
        co1.arraysets['dummy'][idx] = dummyData
    co1.commit('second commit on master adding non-conflict data')
    co1.close()
    return repo


@pytest.fixture()
def server_instance(managed_tmpdir, worker_id):
    from hangar import serve

    address = f'localhost:{randint(50000, 59999)}'
    base_tmpdir = pjoin(managed_tmpdir, f'{worker_id[-1]}')
    mkdir(base_tmpdir)
    server, hangserver, _ = serve(base_tmpdir, overwrite=True, channel_address=address)
    server.start()
    yield address

    hangserver.env._close_environments()
    server.stop(0.0)
    if platform.system() == 'Windows':
        # time for open file handles to close before tmp dir can be removed.
        time.sleep(0.5)


@pytest.fixture()
def server_instance_push_restricted(managed_tmpdir, worker_id):
    from hangar import serve

    address = f'localhost:{randint(50000, 59999)}'
    base_tmpdir = pjoin(managed_tmpdir, f'{worker_id[-1]}')
    mkdir(base_tmpdir)
    server, hangserver, _ = serve(base_tmpdir,
                                  overwrite=True,
                                  channel_address=address,
                                  restrict_push=True,
                                  username='right_username',
                                  password='right_password')
    server.start()
    yield address

    hangserver.env._close_environments()
    server.stop(0.0)
    if platform.system() == 'Windows':
        # time for open file handles to close before tmp dir can be removed.
        time.sleep(0.5)


@pytest.fixture()
def written_two_cmt_server_repo(server_instance, written_two_cmt_repo) -> tuple:
    written_two_cmt_repo.remote.add('origin', server_instance)
    success = written_two_cmt_repo.remote.push('origin', 'master')
    assert success == 'master'
    yield (server_instance, written_two_cmt_repo)
