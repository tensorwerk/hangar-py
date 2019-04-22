import pytest
import os
import shutil
import numpy as np

@pytest.fixture
def managed_tmpdir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path)

@pytest.fixture
def new_repository(managed_tmpdir):
    from hangar.repository import Repository
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='Test User', user_email='foo@test.bar', remove_old=True)
    yield repo


def test_create_initial_commit(new_repository):
    # from hangar.repository import Repository
    repo = new_repository
    # repo.init(user_name='Test User', user_email='foo@test.bar', remove_old=True)
    co = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int32)
    co.datasets.init_dataset(name=dset1n, prototype=dset1arr)

    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.zeros_like(dset1arr)
        arr[:] = i
        co.datasets[dset1n].add(arr, arrn)

    co.commit('first commit')
    assert len(co.datasets[dset1n]) == 5

    for i in range(5):
        checkArr = np.zeros_like(dset1arr)
        checkArr[:] = i
        assert np.all([co.datasets[dset1n][f'arr_1_{i}'] == checkArr])

    co._WriterCheckout__acquire_writer_lock()


def test_checkout_initial_commit(new_repository):
    # from hangar.repository import Repository
    # repo = Repository(path=managed_tmpdir)
    # repo.init(user_name='Test User', user_email='foo@test.bar', remove_old=True)
    repo = new_repository
    co = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int32)
    co.datasets.init_dataset(name=dset1n, prototype=dset1arr)

    ds = co.datasets[dset1n]
    # commit initial data 0:5
    for i in range(5):
        arrn = f'arr_1_{i}'
        arr = np.ones_like(dset1arr)
        arr[:] = i
        ds.add(arr, arrn)
    co.commit('first commit')

    assert len(co.datasets[dset1n]) == 5
    ds = co.datasets[dset1n]
    for i in range(5):
        arrn = f'arr_1_{i}'
        checkArr = np.zeros_like(dset1arr)
        checkArr[:] = i
        assert np.allclose(ds.get(arrn), checkArr) is True

    assert co.datasets[dset1n]['not_here'] is False


def test_write_variable_sized_dataset(new_repository):
    # from hangar.repository import Repository
    # repo = Repository(path=managed_tmpdir)
    # repo.init(user_name='Test User', user_email='foo@test.bar', remove_old=True)
    repo = new_repository
    co = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int32)
    co.datasets.init_dataset(name=dset1n, prototype=dset1arr, variable_shape=True, max_shape=(100,))

    ds = co.datasets[dset1n]
    # commit initial data 0:5
    for i in range(90):
        arrn = f'arr_1_{i}'
        arr = np.zeros(shape=(i+5), dtype=np.int32)
        arr[:] = i
        ds.add(arr, arrn)
    co.commit('first commit')

    assert len(co.datasets[dset1n]) == 90
    ds = co.datasets[dset1n]
    for i in range(90):
        arrn = f'arr_1_{i}'
        checkArr = np.zeros(shape=(i+5), dtype=np.int32)
        checkArr[:] = i
        o = ds.get(arrn)
        assert np.allclose(o, checkArr) is True


def test_remove_data_from_dataset(new_repository):
    # from hangar.repository import Repository
    # repo = Repository(path=managed_tmpdir)
    # repo.init(user_name='Test User', user_email='foo@test.bar', remove_old=True)
    repo = new_repository
    co = repo.checkout(write=True)

    dset1n = 'dset_1'
    dset1arr = np.random.randint(0, 10, size=(10), dtype=np.int32)
    co.datasets.init_dataset(name=dset1n, prototype=dset1arr, variable_shape=True, max_shape=(100,))

    ds = co.datasets[dset1n]
    # commit initial data 0:5
    for i in range(10):
        arrn = f'arr_1_{i}'
        arr = np.zeros(shape=(i+5), dtype=np.int32)
        arr[:] = i
        ds.add(arr, arrn)
    co.commit('first commit')

    assert len(co.datasets[dset1n]) == 10
    ds = co.datasets[dset1n]
    for i in range(5):
        ds.remove(f'arr_1_{i}')
    co.commit('second')

    assert len(co.datasets[dset1n]) == 5
    for i in range(5):
        assert ds.get(f'arr_1_{i}') is False
    for i in range(5, 10):
        checkArr = np.zeros(shape=(i+5), dtype=np.int32)
        checkArr[:] = i
        assert np.allclose(ds.get(f'arr_1_{i}'), checkArr)