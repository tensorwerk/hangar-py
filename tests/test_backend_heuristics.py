import pytest
import numpy as np


@pytest.mark.parametrize('prototype,expected_backend', [
    [np.random.randn(10), '01'],
    [np.random.randn(1000), '00'],
    [np.random.randn(2, 2), '00'],
    [np.random.randn(5, 2), '00'],
])
def test_heuristics_select_backend(repo, prototype, expected_backend):

    wco = repo.checkout(write=True)
    dset = wco.datasets.init_dataset('dset', prototype=prototype)
    assert dset._backend == expected_backend
    dset['0'] = prototype
    wco.commit('first commit')
    assert dset._backend == expected_backend
    assert np.allclose(prototype, dset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    ndset = nwco.datasets['dset']
    assert ndset._backend == expected_backend
    assert np.allclose(prototype, ndset['0'])
    nwco.close()


@pytest.mark.parametrize('prototype,backend', [
    [np.random.randn(10), '00'],
    [np.random.randn(10), '01'],
    [np.random.randn(1000), '00'],
    [np.random.randn(1000), '01'],
    [np.random.randn(2, 2), '00'],
    [np.random.randn(2, 2), '01'],
])
def test_manual_override_heuristics_select_backend(repo, prototype, backend):

    wco = repo.checkout(write=True)
    dset = wco.datasets.init_dataset('dset', prototype=prototype, backend=backend)
    assert dset._backend == backend
    dset['0'] = prototype
    wco.commit('first commit')
    assert dset._backend == backend
    assert np.allclose(prototype, dset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    ndset = nwco.datasets['dset']
    assert ndset._backend == backend
    assert np.allclose(prototype, ndset['0'])
    nwco.close()


def test_manual_override_heuristics_invalid_value_raises_error(repo):

    wco = repo.checkout(write=True)
    with pytest.raises(ValueError):
        dset = wco.datasets.init_dataset('dset', prototype=np.arange(10), backend='ERROR')
    wco.close()