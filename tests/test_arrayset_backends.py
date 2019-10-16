import pytest
import numpy as np
from conftest import backend_params


@pytest.mark.parametrize('backend', backend_params)
def test_backend_property_reports_correct_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend=backend)
    assert aset.backend == backend
    aset[0] = array5by7
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend == backend
    rco.close()


@pytest.mark.parametrize('backend', backend_params)
def test_setting_backend_property_cannot_change_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend=backend)
    assert aset.backend == backend
    aset[0] = array5by7
    with pytest.raises(AttributeError):
        aset.backend = 'foo'
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend == backend
    with pytest.raises(AttributeError):
        naset.backend = 'foo'
    rco.close()


@pytest.mark.parametrize('backend', backend_params)
def test_backend_opts_property_reports_correct_defaults(repo, array5by7, backend):
    from hangar.backends import backend_opts_from_heuristics
    expected_opts = backend_opts_from_heuristics(backend, array5by7)

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend=backend)
    assert aset.backend_opts == expected_opts
    aset[0] = array5by7
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend_opts == expected_opts
    rco.close()


@pytest.mark.parametrize('backend', backend_params)
def test_setting_backend_opts_property_cannot_change_backend_opts(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend=backend)
    aset[0] = array5by7
    with pytest.raises(AttributeError):
        aset.backend_opts = {'foo': 'bar'}
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend == backend
    with pytest.raises(AttributeError):
        naset.backend = {'foo': 'bar'}
    rco.close()


@pytest.mark.parametrize('backend', backend_params)
def test_init_arrayset_with_backend_opts_works(repo, array5by7, backend):
    expected_opts = {'foo': 'bar'}

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset',
                                       prototype=array5by7,
                                       backend=backend,
                                       backend_opts=expected_opts)
    assert aset.backend_opts == expected_opts
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend_opts == expected_opts
    rco.close()


@pytest.mark.parametrize('prototype,expected_backend', [
    [np.random.randn(10), '10'],
    [np.random.randn(1000), '00'],
    [np.random.randn(2, 2), '00'],
    [np.random.randn(5, 2), '00'],
])
def test_heuristics_select_backend(repo, prototype, expected_backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=prototype)
    assert aset.backend == expected_backend
    aset['0'] = prototype
    wco.commit('first commit')
    assert aset.backend == expected_backend
    assert np.allclose(prototype, aset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.arraysets['aset']
    assert naset.backend == expected_backend
    assert np.allclose(prototype, naset['0'])
    nwco.close()


@pytest.mark.parametrize('prototype', [np.random.randn(10), np.random.randn(1000), np.random.randn(2, 2)])
@pytest.mark.parametrize('backend', backend_params)
def test_manual_override_heuristics_select_backend(repo, prototype, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=prototype, backend=backend)
    assert aset.backend == backend
    aset['0'] = prototype
    wco.commit('first commit')
    assert aset.backend == backend
    assert np.allclose(prototype, aset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.arraysets['aset']
    assert naset.backend == backend
    assert np.allclose(prototype, naset['0'])
    nwco.close()


def test_manual_override_heuristics_invalid_value_raises_error(repo):

    wco = repo.checkout(write=True)
    with pytest.raises(ValueError):
        aset = wco.arraysets.init_arrayset('aset', prototype=np.arange(10), backend='ERROR')
    wco.close()
