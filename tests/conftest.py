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
from hangar.checkout import WriterCheckout

import hangar


variable_shape_backend_params = ['00', '10']
fixed_shape_backend_params = ['00', '01', '10']


@pytest.fixture(scope='class')
def classrepo(tmp_path_factory) -> Repository:
    old00_count = hangar.backends.hdf5_00.COLLECTION_COUNT
    old00_size = hangar.backends.hdf5_00.COLLECTION_SIZE
    old01_count = hangar.backends.hdf5_01.COLLECTION_COUNT
    old01_size = hangar.backends.hdf5_01.COLLECTION_SIZE
    old10_size = hangar.backends.numpy_10.COLLECTION_SIZE
    old30_lmdb_settings = hangar.backends.lmdb_30.LMDB_SETTINGS
    hangar.backends.hdf5_00.COLLECTION_COUNT = 5
    hangar.backends.hdf5_00.COLLECTION_SIZE = 20
    hangar.backends.hdf5_01.COLLECTION_COUNT = 5
    hangar.backends.hdf5_01.COLLECTION_SIZE = 20
    hangar.backends.numpy_10.COLLECTION_SIZE = 50
    hangar.backends.lmdb_30.LMDB_SETTINGS['map_size'] = 1_000_000

    old_map_size = hangar.constants.LMDB_SETTINGS['map_size']
    hangar.constants.LMDB_SETTINGS['map_size'] = 2_000_000
    hangar.txnctx.TxnRegisterSingleton._instances = {}

    pth = tmp_path_factory.mktemp('classrepo')
    repo_obj = Repository(path=str(pth), exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    yield repo_obj
    hangar.constants.LMDB_SETTINGS['map_size'] = old_map_size
    hangar.backends.hdf5_00.COLLECTION_COUNT = old00_count
    hangar.backends.hdf5_00.COLLECTION_SIZE = old00_size
    hangar.backends.hdf5_01.COLLECTION_COUNT = old01_count
    hangar.backends.hdf5_01.COLLECTION_SIZE = old01_size
    hangar.backends.numpy_10.COLLECTION_SIZE = old10_size
    hangar.backends.lmdb_30.LMDB_SETTINGS = old30_lmdb_settings
    repo_obj._env._close_environments()


@pytest.fixture()
def managed_tmpdir(monkeypatch, tmp_path):
    monkeypatch.setitem(hangar.constants.LMDB_SETTINGS, 'map_size', 2_000_000)
    monkeypatch.setitem(hangar.backends.lmdb_30.LMDB_SETTINGS, 'map_size', 1_000_000)
    monkeypatch.setattr(hangar.backends.hdf5_00, 'COLLECTION_COUNT', 5)
    monkeypatch.setattr(hangar.backends.hdf5_00, 'COLLECTION_SIZE', 20)
    monkeypatch.setattr(hangar.backends.hdf5_01, 'COLLECTION_COUNT', 5)
    monkeypatch.setattr(hangar.backends.hdf5_01, 'COLLECTION_SIZE', 20)
    monkeypatch.setattr(hangar.backends.numpy_10, 'COLLECTION_SIZE', 50)
    monkeypatch.setitem(hangar.backends.lmdb_30.LMDB_SETTINGS, 'map_size', 1_000_000)
    hangar.txnctx.TxnRegisterSingleton._instances = {}
    yield tmp_path
    shutil.rmtree(tmp_path)


@pytest.fixture()
def repo(managed_tmpdir) -> Repository:
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    yield repo_obj
    repo_obj._env._close_environments()


@pytest.fixture()
def aset_samples_initialized_repo(repo) -> Repository:
    co = repo.checkout(write=True)
    co.add_ndarray_column(name='writtenaset', shape=(5, 7), dtype=np.float64)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def aset_subsamples_initialized_repo(repo) -> Repository:
    co = repo.checkout(write=True)
    co.add_ndarray_column(
        name='writtenaset', shape=(5, 7), dtype=np.float64, contains_subsamples=True)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture(params=fixed_shape_backend_params)
def repo_20_filled_samples(request, aset_samples_initialized_repo, array5by7) -> Repository:
    co = aset_samples_initialized_repo.checkout(write=True)
    second_aset = co.add_ndarray_column('second_aset', prototype=array5by7, backend=request.param)
    first_aset = co.columns['writtenaset']
    for i in range(0, 20):
        array5by7[:] = i
        first_aset[str(i)] = array5by7
        array5by7[:] = -i
        second_aset[str(i)] = array5by7
    co.commit('20 samples')
    co.close()
    yield aset_samples_initialized_repo


@pytest.fixture(params=fixed_shape_backend_params)
def repo_300_filled_samples(request, aset_samples_initialized_repo, array5by7) -> Repository:
    co = aset_samples_initialized_repo.checkout(write=True)
    aset = co.add_ndarray_column('aset', prototype=array5by7, backend=request.param)
    with aset:
        for i in range(300):
            array5by7[:] = i
            aset[i] = array5by7
    co.commit('1000 samples')
    co.close()
    yield aset_samples_initialized_repo


@pytest.fixture()
def repo_20_filled_samples2(repo) -> Repository:
    # for diff testing
    dummyData = np.arange(50).astype(np.int64)
    co1 = repo.checkout(write=True, branch='master')
    co1.add_ndarray_column(name='dummy', prototype=dummyData)
    for idx in range(10):
        dummyData[:] = idx
        co1.columns['dummy'][idx] = dummyData
    co1.commit('first commit adding dummy data and hello meta')
    co1.close()
    return repo


@pytest.fixture(params=variable_shape_backend_params)
def aset_samples_var_shape_initialized_repo(request, repo) -> Repository:
    co = repo.checkout(write=True)
    co.add_ndarray_column(
        name='writtenaset', shape=(10, 10), dtype=np.float64, variable_shape=True, backend=request.param)
    co.commit('this is a commit message')
    co.close()
    yield repo


@pytest.fixture()
def aset_samples_initialized_w_checkout(aset_samples_initialized_repo) -> WriterCheckout:
    co = aset_samples_initialized_repo.checkout(write=True)
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


@pytest.fixture(params=fixed_shape_backend_params)
def two_commit_filled_samples_repo(request, repo, array5by7) -> Repository:
    co = repo.checkout(write=True)
    co.add_ndarray_column(
        name='writtenaset', shape=(5, 7), dtype=np.float32, backend=request.param)
    for cIdx in range(2):
        if cIdx != 0:
            co = repo.checkout(write=True)

        with co.columns['writtenaset'] as d:
            for prevKey in list(d.keys())[1:]:
                del d[prevKey]
            for sIdx in range((cIdx + 1) * 5):
                arr = np.random.randn(*array5by7.shape).astype(np.float32) * 100
                d[str(sIdx)] = arr
        co.commit(f'commit number: {cIdx}')
        co.close()
    yield repo


@pytest.fixture()
def repo_1_br_no_conf(repo) -> Repository:

    dummyData = np.arange(50)
    co1 = repo.checkout(write=True, branch='master')
    co1.add_ndarray_column(name='dummy', prototype=dummyData)
    for idx in range(10):
        dummyData[:] = idx
        co1.columns['dummy'][str(idx)] = dummyData
    co1.commit('first commit adding dummy data')
    co1.close()

    repo.create_branch('testbranch')
    co2 = repo.checkout(write=True, branch='testbranch')
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.columns['dummy'][str(idx)] = dummyData
        co2.columns['dummy'][idx] = dummyData
    co2.commit('first commit on test branch adding non-conflict data')
    co2.close()
    return repo


@pytest.fixture()
def repo_2_br_no_conf(repo_1_br_no_conf) -> Repository:

    dummyData = np.arange(50)
    repo = repo_1_br_no_conf
    co1 = repo.checkout(write=True, branch='master')
    for idx in range(20, 30):
        dummyData[:] = idx
        co1.columns['dummy'][str(idx)] = dummyData
        co1.columns['dummy'][idx] = dummyData
    co1.commit('second commit on master adding non-conflict data')
    co1.close()
    return repo


@pytest.fixture()
def server_instance(managed_tmpdir, worker_id, monkeypatch):
    from secrets import choice
    from hangar.remote.server import serve

    possibble_addresses = [x for x in range(50000, 59999)]
    chosen_address = choice(possibble_addresses)
    address = f'localhost:{chosen_address}'
    base_tmpdir = pjoin(managed_tmpdir, f'{worker_id[-1]}')
    mkdir(base_tmpdir)

    server, hangserver, _ = serve(base_tmpdir, overwrite=True, channel_address=address)
    monkeypatch.setitem(hangserver.CFG['SERVER_GRPC'], 'max_concurrent_rpcs', '20')
    monkeypatch.setitem(hangserver.CFG['SERVER_GRPC'], 'max_thread_pool_workers', '5')
    server.start()
    yield address

    hangserver.close()
    server.stop(0.1)
    server.wait_for_termination(timeout=10)


@pytest.fixture()
def written_two_cmt_server_repo(server_instance, two_commit_filled_samples_repo) -> tuple:
    time.sleep(0.1)  # wait for ready
    two_commit_filled_samples_repo.remote.add('origin', server_instance)
    success = two_commit_filled_samples_repo.remote.push('origin', 'master')
    assert success == 'master'
    yield (server_instance, two_commit_filled_samples_repo)
