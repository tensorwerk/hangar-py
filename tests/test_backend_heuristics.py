import pytest
import numpy as np


@pytest.mark.parametrize('prototype,expected_backend', [
    [np.random.randn(10), '10'],
    [np.random.randn(1000), '00'],
    [np.random.randn(2, 2), '00'],
    [np.random.randn(5, 2), '00'],
])
def test_heuristics_select_backend(repo, prototype, expected_backend):

    wco = repo.checkout(write=True)
    dcell = wco.datacells.init_datacell('dcell', prototype=prototype)
    assert dcell._backend == expected_backend
    dcell['0'] = prototype
    wco.commit('first commit')
    assert dcell._backend == expected_backend
    assert np.allclose(prototype, dcell['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    ndcell = nwco.datacells['dcell']
    assert ndcell._backend == expected_backend
    assert np.allclose(prototype, ndcell['0'])
    nwco.close()


@pytest.mark.parametrize('prototype,backend', [
    [np.random.randn(10), '00'],
    [np.random.randn(10), '10'],
    [np.random.randn(1000), '00'],
    [np.random.randn(1000), '10'],
    [np.random.randn(2, 2), '00'],
    [np.random.randn(2, 2), '10'],
])
def test_manual_override_heuristics_select_backend(repo, prototype, backend):

    wco = repo.checkout(write=True)
    dcell = wco.datacells.init_datacell('dcell', prototype=prototype, backend=backend)
    assert dcell._backend == backend
    dcell['0'] = prototype
    wco.commit('first commit')
    assert dcell._backend == backend
    assert np.allclose(prototype, dcell['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    ndcell = nwco.datacells['dcell']
    assert ndcell._backend == backend
    assert np.allclose(prototype, ndcell['0'])
    nwco.close()


def test_manual_override_heuristics_invalid_value_raises_error(repo):

    wco = repo.checkout(write=True)
    with pytest.raises(ValueError):
        dcell = wco.datacells.init_datacell('dcell', prototype=np.arange(10), backend='ERROR')
    wco.close()
