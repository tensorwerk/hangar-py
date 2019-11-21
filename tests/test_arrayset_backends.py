import pytest
import numpy as np
from conftest import fixed_shape_backend_params


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_backend_property_reports_correct_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backend)
    assert aset.backend == backend
    aset[0] = array5by7
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend == backend
    rco.close()


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_setting_backend_property_cannot_change_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backend)
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


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_backend_opts_property_reports_correct_defaults(repo, array5by7, backend):
    from hangar.backends import backend_opts_from_heuristics
    expected_opts = backend_opts_from_heuristics(backend,
                                                 array5by7,
                                                 named_samples=False,
                                                 variable_shape=False)
    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backend)
    assert aset.backend_opts == expected_opts
    aset[0] = array5by7
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend_opts == expected_opts
    rco.close()


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_setting_backend_opts_property_cannot_change_backend_opts(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backend)
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


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_init_arrayset_with_backend_opts_works(repo, array5by7, backend):
    expected_opts = {'foo': 'bar'}
    input_opts = {'backend': backend, **expected_opts}

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=input_opts)
    assert aset.backend_opts == expected_opts
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.arraysets['aset']
    assert naset.backend_opts == expected_opts
    rco.close()


@pytest.mark.parametrize('prototype,variable_shape,expected_backend', [
    [np.random.randn(10), True, '10'],
    [np.random.randn(1000), True, '00'],
    [np.random.randn(1000), False, '00'],
    [np.random.randn(9_999_999).astype(np.float16), False, '00'],
    [np.random.randn(10_000_000).astype(np.float16), False, '00'],
    [np.random.randn(10_000_001).astype(np.float16), False, '01'],
    [np.random.randn(10_000_001).astype(np.float16), True, '00'],
    [np.random.randn(2, 2), True, '00'],
    [np.random.randn(2, 2), False, '01'],
    [np.random.randn(5, 2), True, '00'],
    [np.random.randn(5, 2), False, '01'],
])
def test_heuristics_select_backend(repo, prototype, variable_shape, expected_backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=prototype, variable_shape=variable_shape)
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
@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_manual_override_heuristics_select_backend(repo, prototype, backend):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=prototype, backend_opts=backend)
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
        wco.arraysets.init_arrayset('aset', prototype=np.arange(10), backend_opts='ERROR')
    wco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendEnd', fixed_shape_backend_params)
def test_manual_change_backends_after_write_works(repo, array5by7, backendStart, backendEnd):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backendStart)
    assert aset.backend == backendStart
    aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.arraysets['aset']
    assert naset.backend == backendStart

    naset.change_backend(backend_opts=backendEnd)
    naset[1] = array5by7 + 1

    assert naset.backend == backendEnd
    assert np.allclose(array5by7, naset[0])
    assert np.allclose(array5by7 + 1, naset[1])
    nwco.commit('second')
    nwco.close()

    rco = repo.checkout()
    assert rco.arraysets['aset'].backend == backendEnd
    rco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendFail', ['lmao', '000'])
def test_manual_change_backend_to_invalid_fmt_code_fails(repo, array5by7, backendStart, backendFail):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backendStart)
    assert aset.backend == backendStart
    aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.arraysets['aset']
    assert naset.backend == backendStart

    with pytest.raises(ValueError):
        naset.change_backend(backend_opts=backendFail)
    assert naset.backend == backendStart
    naset[1] = array5by7 + 1
    assert np.allclose(array5by7, naset[0])
    assert np.allclose(array5by7 + 1, naset[1])
    nwco.commit('second')
    nwco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendEnd', fixed_shape_backend_params)
def test_manual_change_backend_fails_while_in_cm(repo, array5by7, backendStart, backendEnd):

    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=backendStart)
    assert aset.backend == backendStart
    aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.arraysets['aset']
    assert naset.backend == backendStart

    with nwco as c:
        with pytest.raises(RuntimeError):
            c['aset'].change_backend(backend_opts=backendEnd)
        with pytest.raises(RuntimeError):
            naset.change_backend(backend_opts=backendEnd)
        with pytest.raises(RuntimeError):
            c.arraysets['aset'].change_backend(backend_opts=backendEnd)
        with pytest.raises(RuntimeError):
            nwco.arraysets['aset'].change_backend(backend_opts=backendEnd)

    with naset as na:
        with pytest.raises(RuntimeError):
            na.change_backend(backend_opts=backendEnd)
        with pytest.raises(RuntimeError):
            naset.change_backend(backend_opts=backendEnd)
        with pytest.raises(RuntimeError):
            nwco.arraysets['aset'].change_backend(backend_opts=backendEnd)

    assert naset.backend == backendStart
    naset[1] = array5by7 + 1
    assert np.allclose(array5by7, naset[0])
    assert np.allclose(array5by7 + 1, naset[1])
    nwco.commit('second')
    nwco.close()
