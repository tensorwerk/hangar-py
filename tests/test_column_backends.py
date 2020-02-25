import pytest
import numpy as np
from conftest import fixed_shape_backend_params


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_backend_property_reports_correct_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column('aset', prototype=array5by7, backend=backend)
    assert aset.backend == backend
    aset[0] = array5by7
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.columns['aset']
    assert naset.backend == backend
    rco.close()


@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_setting_backend_property_cannot_change_backend(repo, array5by7, backend):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column('aset', prototype=array5by7, backend=backend)
    assert aset.backend == backend
    aset[0] = array5by7
    with pytest.raises(AttributeError):
        aset.backend = 'foo'
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.columns['aset']
    assert naset.backend == backend
    with pytest.raises(AttributeError):
        naset.backend = 'foo'
    rco.close()


@pytest.mark.parametrize('subsamples', [True, False])
@pytest.mark.parametrize('backend', fixed_shape_backend_params)
def test_setting_backend_opts_property_cannot_change_backend_opts(repo, array5by7, backend, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=array5by7, backend=backend, contains_subsamples=subsamples)
    if subsamples:
        aset.update({0: {0: array5by7}})
    else:
        aset[0] = array5by7
    with pytest.raises(AttributeError):
        aset.backend_options = {'foo': 'bar'}
    wco.commit('first')
    wco.close()

    rco = repo.checkout()
    naset = rco.columns['aset']
    assert naset.backend == backend
    with pytest.raises(AttributeError):
        naset.backend = {'foo': 'bar'}
    rco.close()


@pytest.mark.parametrize('shape,dtype,variable_shape,expected_backend', [
    [(10,), np.uint16, True, '10'],
    [(1000,), np.uint16, True, '00'],
    [(1000,), np.uint16, False, '00'],
    [(9_999_999,), np.uint8, False, '00'],
    [(10_000_000,), np.uint8, False, '00'],
    [(10_000_001,), np.uint8, False, '01'],
    [(10_000_001,), np.uint8, True, '00'],
    [(2, 2), np.uint16, True, '00'],
    [(2, 2), np.uint16, False, '01'],
    [(5, 2), np.uint16, True, '00'],
    [(5, 2), np.uint16, False, '01'],
])
@pytest.mark.parametrize('subsamples', [True, False])
def test_heuristics_select_backend(repo, shape, dtype, variable_shape, expected_backend, subsamples):
    wco = repo.checkout(write=True)
    prototype = np.ones(shape, dtype=dtype)
    aset = wco.add_ndarray_column(
        'aset', prototype=prototype, variable_shape=variable_shape, contains_subsamples=subsamples)
    assert aset.backend == expected_backend
    if subsamples:
        aset.update({'0': {'0': prototype}})
    else:
        aset['0'] = prototype
    wco.commit('first commit')
    assert aset.backend == expected_backend
    if subsamples:
        assert np.allclose(prototype, aset['0']['0'])
    else:
        assert np.allclose(prototype, aset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.columns['aset']
    assert naset.backend == expected_backend
    if subsamples:
        assert np.allclose(prototype, naset['0']['0'])
    else:
        assert np.allclose(prototype, naset['0'])
    nwco.close()


@pytest.mark.parametrize('prototype', [np.random.randn(10), np.random.randn(1000), np.random.randn(2, 2)])
@pytest.mark.parametrize('backend', fixed_shape_backend_params)
@pytest.mark.parametrize('subsamples', [True, False])
def test_manual_override_heuristics_select_backend(repo, prototype, backend, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=prototype, backend=backend, contains_subsamples=subsamples)
    assert aset.backend == backend
    if subsamples:
        aset.update({'0': {'0': prototype}})
    else:
        aset['0'] = prototype
    wco.commit('first commit')
    assert aset.backend == backend
    if subsamples:
        assert np.allclose(prototype, aset['0']['0'])
    else:
        assert np.allclose(prototype, aset['0'])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.columns['aset']
    assert naset.backend == backend
    if subsamples:
        assert np.allclose(prototype, naset['0']['0'])
    else:
        assert np.allclose(prototype, naset['0'])
    nwco.close()


def test_manual_override_heuristics_invalid_value_raises_error(repo):

    wco = repo.checkout(write=True)
    with pytest.raises(ValueError):
        wco.add_ndarray_column('aset', prototype=np.arange(10), backend='ERROR')
    wco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendEnd', fixed_shape_backend_params)
@pytest.mark.parametrize('subsamples', [True, False])
def test_manual_change_backends_after_write_works(repo, array5by7, backendStart, backendEnd, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=array5by7, backend=backendStart, contains_subsamples=subsamples)
    assert aset.backend == backendStart
    if subsamples:
        aset.update({0: {0: array5by7}})
    else:
        aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    if subsamples:
        assert np.allclose(array5by7, aset[0][0])
    else:
        assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.columns['aset']
    assert naset.backend == backendStart

    naset.change_backend(backend=backendEnd)
    if subsamples:
        naset.update({1: {1: array5by7+1}})
    else:
        naset[1] = array5by7 + 1

    assert naset.backend == backendEnd
    if subsamples:
        assert np.allclose(array5by7, naset[0][0])
        assert np.allclose(array5by7+1, naset[1][1])
    else:
        assert np.allclose(array5by7, naset[0])
        assert np.allclose(array5by7+1, naset[1])
    nwco.commit('second')
    nwco.close()

    rco = repo.checkout()
    assert rco.columns['aset'].backend == backendEnd
    rco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendFail', ['lmao', '000'])
@pytest.mark.parametrize('subsamples', [True, False])
def test_manual_change_backend_to_invalid_fmt_code_fails(repo, array5by7, backendStart, backendFail, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=array5by7, backend=backendStart, contains_subsamples=subsamples)
    assert aset.backend == backendStart
    if subsamples:
        aset[0] = {0: array5by7}
    else:
        aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    if subsamples:
        assert np.allclose(array5by7, aset[0][0])
    else:
        assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.columns['aset']
    assert naset.backend == backendStart

    with pytest.raises(ValueError):
        naset.change_backend(backend=backendFail)
    assert naset.backend == backendStart
    if subsamples:
        naset[1] = {1: array5by7+1}
    else:
        naset[1] = array5by7 + 1

    if subsamples:
        assert np.allclose(array5by7, naset[0][0])
        assert np.allclose(array5by7 + 1, naset[1][1])
    else:
        assert np.allclose(array5by7, naset[0])
        assert np.allclose(array5by7 + 1, naset[1])
    nwco.commit('second')
    nwco.close()


@pytest.mark.parametrize('backendStart', fixed_shape_backend_params)
@pytest.mark.parametrize('backendEnd', fixed_shape_backend_params)
@pytest.mark.parametrize('subsamples', [True, False])
def test_manual_change_backend_fails_while_in_cm(repo, array5by7, backendStart, backendEnd, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=array5by7, backend=backendStart, contains_subsamples=subsamples)
    assert aset.backend == backendStart
    if subsamples:
        aset[0] = {0: array5by7}
    else:
        aset[0] = array5by7
    wco.commit('first commit')
    assert aset.backend == backendStart
    if subsamples:
        assert np.allclose(array5by7, aset[0][0])
    else:
        assert np.allclose(array5by7, aset[0])
    wco.close()

    nwco = repo.checkout(write=True)
    naset = nwco.columns['aset']
    assert naset.backend == backendStart

    with nwco as c:
        with pytest.raises(RuntimeError):
            c['aset'].change_backend(backend=backendEnd)
        with pytest.raises(RuntimeError):
            naset.change_backend(backend=backendEnd)
        with pytest.raises(RuntimeError):
            c.columns['aset'].change_backend(backend=backendEnd)
        with pytest.raises(RuntimeError):
            nwco.columns['aset'].change_backend(backend=backendEnd)

    with naset as na:
        with pytest.raises(RuntimeError):
            na.change_backend(backend=backendEnd)
        with pytest.raises(RuntimeError):
            naset.change_backend(backend=backendEnd)
        with pytest.raises(RuntimeError):
            nwco.columns['aset'].change_backend(backend=backendEnd)

    assert naset.backend == backendStart
    if subsamples:
        naset[1] = {1: array5by7+1}
    else:
        naset[1] = array5by7 + 1

    if subsamples:
        assert np.allclose(array5by7, naset[0][0])
        assert np.allclose(array5by7 + 1, naset[1][1])
    else:
        assert np.allclose(array5by7, naset[0])
        assert np.allclose(array5by7 + 1, naset[1])
    nwco.commit('second')
    nwco.close()



@pytest.fixture(scope='class')
def dummy_writer_checkout(classrepo):
    wco = classrepo.checkout(write=True)
    yield wco
    wco.close()


class TestComplibRestrictions:

    @pytest.mark.parametrize('backend', ['01', '00'])
    @pytest.mark.parametrize('subsamples', [True, False])
    @pytest.mark.parametrize('complib', [
        'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:zlib', 'blosc:zstd'
    ])
    @pytest.mark.parametrize('dtype,shape', [
        [np.float32, (1, 1, 1)],
        [np.float32, (3,)],
        [np.float64, (1,)],
        [np.uint8, (15,)],
        [np.uint8, (3, 2, 2)],
    ])
    def test_schema_smaller_16_bytes_cannot_select_blosc_backend(
        self, dummy_writer_checkout, backend, complib, dtype, shape, subsamples
    ):
        wco = dummy_writer_checkout
        be_opts = {'complib': complib, 'complevel': 3, 'shuffle': 'byte'}

        # prototype spec
        with pytest.raises(ValueError, match='blosc clib requires'):
            proto = np.zeros(shape, dtype=dtype)
            wco.add_ndarray_column(
                'aset', prototype=proto, backend=backend,
                backend_options=be_opts, contains_subsamples=subsamples)

        # shape and dtype spec
        with pytest.raises(ValueError, match='blosc clib requires'):
            wco.add_ndarray_column(
                'aset', shape=shape, dtype=dtype, backend=backend,
                backend_options=be_opts, contains_subsamples=subsamples)


@pytest.mark.parametrize('backend', ['01', '00'])
@pytest.mark.parametrize('subsamples', [True, False])
@pytest.mark.parametrize('dtype,shape', [
    [np.float32, (1, 1, 1)],
    [np.float32, (3,)],
    [np.float64, (1,)],
    [np.uint8, (15,)],
    [np.uint8, (3, 2, 2)],
])
def test_schema_smaller_16_bytes_does_not_use_heuristic_to_select_blosc(
    repo, backend, dtype, shape, subsamples
):
    wco = repo.checkout(write=True)
    proto = np.zeros(shape, dtype=dtype)
    aset = wco.add_ndarray_column(
        'aset', prototype=proto, backend=backend, contains_subsamples=subsamples)
    bad_clibs = ['blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:zlib', 'blosc:zstd']
    assert aset.backend_options['complib'] not in bad_clibs
    if subsamples:
        aset[0] = {0: proto}
    else:
        aset[0] = proto
    assert aset.backend_options['complib'] not in bad_clibs
    wco.close()


@pytest.mark.parametrize('backend', ['01', '00'])
@pytest.mark.parametrize('subsamples', [True, False])
@pytest.mark.parametrize('complib', [
    'blosc:blosclz', 'blosc:lz4', 'blosc:lz4hc', 'blosc:zlib', 'blosc:zstd'
])
@pytest.mark.parametrize('dtype,shape', [
    [np.float32, (1, 1, 1)],
    [np.float32, (3,)],
    [np.float64, (1,)],
    [np.uint8, (15,)],
    [np.uint8, (3, 2, 2)],
])
def test_schema_smaller_16_bytes_cannot_change_to_blosc_backend(
    repo, backend, complib, shape, dtype, subsamples):

    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', shape=shape, dtype=dtype, backend=backend, contains_subsamples=subsamples)
    proto = np.zeros(shape, dtype=dtype)
    if subsamples:
        aset[0] = {0: proto}
    else:
        aset[0] = proto

    be_opts = {'complib': complib, 'complevel': 3, 'shuffle': None}
    with pytest.raises(ValueError, match='blosc clib requires'):
        aset.change_backend(backend=backend, backend_options=be_opts)
    wco.close()
