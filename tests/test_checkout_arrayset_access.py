import pytest
import numpy as np


# -------------------------- Reader Checkout ----------------------------------


@pytest.mark.parametrize("samplename", ['0', '-1', 1, 0, 1000, 'alkea'])
def test_write_single_arrayset_single_sample(written_repo, array5by7, samplename):
    wco = written_repo.checkout(write=True)
    wco['writtenaset', samplename] = array5by7
    assert np.allclose(array5by7, wco.arraysets['writtenaset'][samplename])
    wco.commit('init')
    assert np.allclose(array5by7, wco.arraysets['writtenaset'][samplename])
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(array5by7, rco.arraysets['writtenaset'][samplename])
    rco.close()


@pytest.mark.parametrize("samplenames,samplevals", [
    [('0', 1, '22', 23), (0, 1, 22, 23)],
    [('0', 1), (0, 1)],
    [('aeaaee', 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7)]
])
def test_write_single_arrayset_multiple_samples(written_repo, array5by7, samplenames, samplevals):
    wco = written_repo.checkout(write=True)

    values = []
    for val in samplevals:
        array5by7[:] = val
        values.append(array5by7)
    wco['writtenaset', samplenames] = values

    for val, name in zip(values, samplenames):
        assert np.allclose(val, wco.arraysets['writtenaset'][name])
    wco.commit('init')
    for val, name in zip(values, samplenames):
        assert np.allclose(val, wco.arraysets['writtenaset'][name])
    wco.close()

    rco = written_repo.checkout()
    for val, name in zip(values, samplenames):
        assert np.allclose(val, rco.arraysets['writtenaset'][name])
    rco.close()


def test_write_multiple_arrayset_single_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['writtenaset', 'newaset'], '0'] = [array5by7, array10]
    assert np.allclose(array5by7, wco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.commit('init')
    assert np.allclose(array5by7, wco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(array5by7, rco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, rco.arraysets['newaset']['0'])
    rco.close()


def test_write_in_context_manager_no_loop(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with wco:
        assert wco._is_conman is True
        wco[['writtenaset', 'newaset'], '0'] = [array5by7, array10]
    assert wco._is_conman is False

    assert np.allclose(array5by7, wco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.commit('init')
    assert np.allclose(array5by7, wco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(array5by7, rco.arraysets['writtenaset']['0'])
    assert np.allclose(array10, rco.arraysets['newaset']['0'])
    rco.close()


def test_write_in_context_manager_many_samples_looping(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with wco:
        assert wco._is_conman is True
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco[['writtenaset', 'newaset'], idx] = [array5by7, array10]
    assert wco._is_conman is False

    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, wco.arraysets['writtenaset'][idx])
        assert np.allclose(array10, wco.arraysets['newaset'][idx])
    wco.commit('init')
    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, wco.arraysets['writtenaset'][idx])
        assert np.allclose(array10, wco.arraysets['newaset'][idx])
    wco.close()

    rco = written_repo.checkout()
    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, rco.arraysets['writtenaset'][idx])
        assert np.allclose(array10, rco.arraysets['newaset'][idx])
    rco.close()


def test_write_fails_if_checkout_closed(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['writtenaset', 'newaset'], 0] = [array5by7, array10]
    wco.close()
    with pytest.raises((PermissionError, UnboundLocalError)):
        wco[['writtenaset', 'newaset'], 1] = [array5by7, array10]

    wco2 = written_repo.checkout(write=True)
    assert 0 in wco2.arraysets['writtenaset']
    assert 0 in wco2.arraysets['newaset']
    assert 1 not in wco2.arraysets['writtenaset']
    assert 1 not in wco2.arraysets['newaset']
    wco2.close()


def test_write_context_manager_fails_if_checkout_closed(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['writtenaset', 'newaset'], 0] = [array5by7, array10]
    wco.close()
    with pytest.raises(PermissionError):
        with wco:
            wco['writtenaset', 1] = array5by7

    wco2 = written_repo.checkout(write=True)
    assert 0 in wco2.arraysets['writtenaset']
    assert 0 in wco2.arraysets['newaset']
    assert 1 not in wco2.arraysets['writtenaset']
    wco2.close()


def test_write_fails_multiple_arrayset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with pytest.raises(SyntaxError):
        wco[['writtenaset', 'newaset'], ['0', 1]] = [[array5by7, array5by7], [array10, array10]]
    wco.close()


def test_write_fails_nonmatching_multiple_asets_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with pytest.raises(ValueError):
        wco[['writtenaset', 'newaset'], '0'] = [array5by7]
    with pytest.raises(TypeError):
        wco[['writtenaset', 'newaset'], '0'] = array5by7
    wco.close()


def test_write_fails_nonmatching_single_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    with pytest.raises(TypeError):
        wco['writtenaset', [i for i in range(10)]] = array5by7
    with pytest.raises(ValueError):
        wco['writtenaset', [i for i in range(10)]] = [array5by7 for i in range(4)]
    with pytest.raises(ValueError):
        wco['writtenaset', [i for i in range(10)]] = [array5by7 for i in range(14)]

    with pytest.raises(ValueError):
        wco['writtenaset', []] = [array5by7 for i in range(1)]
    wco.close()


def test_write_fails_multiple_asets_single_sample_not_compatible(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)

    with pytest.raises(ValueError):
        wco[['writtenaset', 'newaset'], 0] = [array10, array5by7]
    with pytest.raises(ValueError):
        wco[['writtenaset', 'newaset'], 0] = [array10, array5by7.astype(np.float16)]
    with pytest.raises(ValueError):
        fortran5by7 = np.zeros(shape=array5by7.shape, dtype=array5by7.dtype, order='F')
        wco[['writtenaset', 'newaset'], 0] = [array10, fortran5by7]
    wco.close()


def test_writer_co_read_single_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    assert np.allclose(wco['writtenaset', 0], array5by7)
    assert np.allclose(wco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(wco['writtenaset', 2], array5by7 + 2)

    wco.close()


def test_writer_co_read_single_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    res = wco['writtenaset', [0, 1, 2]]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array5by7 + 1)
    assert np.allclose(res[2], array5by7 + 2)
    wco.close()


def test_writer_co_read_multiple_aset_single_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    res = wco[['writtenaset', 'newaset'], 0]
    assert 'writtenaset' in res._fields
    assert 'newaset' in res._fields
    assert np.allclose(res[0], array5by7)
    assert np.array_equal(res[0], res.writtenaset)
    assert np.allclose(res[1], array10)
    assert np.array_equal(res[1], res.newaset)

    res = wco[['writtenaset', 'newaset'], 1]
    assert 'writtenaset' in res._fields
    assert 'newaset' in res._fields
    assert np.allclose(res[0], array5by7 + 1)
    assert np.array_equal(res[0], res.writtenaset)
    assert np.allclose(res[1], array10 + 1)
    assert np.array_equal(res[1], res.newaset)
    wco.close()


def test_writer_co_read_multtiple_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    res = wco[['writtenaset', 'newaset'], [0, 1]]
    assert isinstance(res, list)
    assert len(res) == 2

    s0 = res[0]
    assert isinstance(s0, tuple)
    assert s0._fields == ('writtenaset', 'newaset')
    assert np.allclose(s0.writtenaset, array5by7)
    assert np.allclose(s0.newaset, array10)

    s1 = res[1]
    assert isinstance(s1, tuple)
    assert s1._fields == ('writtenaset', 'newaset')
    assert np.allclose(s1.writtenaset, array5by7 + 1)
    assert np.allclose(s1.newaset, array10 + 1)
    wco.close()


def test_writer_co_read_fails_nonexistant_aset_name(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    with pytest.raises(KeyError):
        _ = wco['doesnotexist', 0]
    wco.close()


def test_writer_co_read_fails_nonexistant_sample_name(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    with pytest.raises(KeyError):
        _ = wco['doesnotexist', 124]
    wco.close()


def test_writer_co_get_returns_none_on_nonexistant_sample_name(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    out = wco.get('writtenaset', 124)
    assert out is None
    wco.close()


def test_writer_co_read_in_context_manager_no_loop(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['writtenaset', 'newaset'], '0'] = [array5by7, array10]
    with wco:
        assert wco._is_conman is True
        assert np.allclose(wco['writtenaset', '0'], array5by7)
    wco.close()


def test_writer_co_read_in_context_manager_many_samples_looping(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with wco:
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco[['writtenaset', 'newaset'], idx] = [array5by7, array10]

    with wco:
        writtenasetOut = wco['writtenaset', [i for i in range(100)]]
        newasetOut = wco['newaset', [i for i in range(100)]]
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            assert np.allclose(array5by7, wco['writtenaset', idx])
            assert np.allclose(array10, wco['newaset', idx])

            o = wco[['writtenaset', 'newaset'], idx]
            assert np.allclose(o.writtenaset, array5by7)
            assert np.allclose(o.newaset, array10)

            assert np.allclose(writtenasetOut[idx], array5by7)
            assert np.allclose(newasetOut[idx], array10)
    wco.close()


def test_writer_co_read_ellipses_select_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    o = wco[..., 0]
    assert 'writtenaset' in o._fields
    assert 'newaset' in o._fields
    assert np.allclose(o.writtenaset, array5by7)
    assert np.allclose(o.newaset, array10)
    wco.close()


def test_writer_co_read_slice_select_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    o = wco[:, 0]
    assert 'writtenaset' in o._fields
    assert 'newaset' in o._fields
    assert np.allclose(o.writtenaset, array5by7)
    assert np.allclose(o.newaset, array10)
    wco.close()


def test_writer_co_read_ellipses_select_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    out = wco[..., [0, 1]]
    assert len(out) == 2

    o1 = out[0]
    assert 'writtenaset' in o1._fields
    assert 'newaset' in o1._fields
    assert np.allclose(o1.writtenaset, array5by7)
    assert np.allclose(o1.newaset, array10)

    o2 = out[1]
    assert 'writtenaset' in o2._fields
    assert 'newaset' in o2._fields
    assert np.allclose(o2.writtenaset, array5by7 + 1)
    assert np.allclose(o2.newaset, array10 + 1)
    wco.close()


def test_writer_co_read_slice_select_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2

    out = wco[:, [0, 1]]
    assert len(out) == 2

    o1 = out[0]
    assert 'writtenaset' in o1._fields
    assert 'newaset' in o1._fields
    assert np.allclose(o1.writtenaset, array5by7)
    assert np.allclose(o1.newaset, array10)

    o2 = out[1]
    assert 'writtenaset' in o2._fields
    assert 'newaset' in o2._fields
    assert np.allclose(o2.writtenaset, array5by7 + 1)
    assert np.allclose(o2.newaset, array10 + 1)
    wco.close()


# -------------------------- Reader Checkout ----------------------------------


def test_reader_co_read_single_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(rco['writtenaset', 0], array5by7)
    assert np.allclose(rco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(rco['writtenaset', 2], array5by7 + 2)
    rco.close()


def test_reader_co_read_single_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    res = rco['writtenaset', [0, 1, 2]]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array5by7 + 1)
    assert np.allclose(res[2], array5by7 + 2)
    rco.close()


def test_reader_co_read_multiple_aset_single_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    res = rco[['writtenaset', 'newaset'], 0]
    assert 'writtenaset' in res._fields
    assert 'newaset' in res._fields
    assert np.allclose(res[0], array5by7)
    assert np.array_equal(res[0], res.writtenaset)
    assert np.allclose(res[1], array10)
    assert np.array_equal(res[1], res.newaset)

    res = rco[['writtenaset', 'newaset'], 1]
    assert 'writtenaset' in res._fields
    assert 'newaset' in res._fields
    assert np.allclose(res[0], array5by7 + 1)
    assert np.array_equal(res[0], res.writtenaset)
    assert np.allclose(res[1], array10 + 1)
    assert np.array_equal(res[1], res.newaset)
    rco.close()


def test_reader_co_read_multtiple_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    res = rco[['writtenaset', 'newaset'], [0, 1]]
    assert isinstance(res, list)
    assert len(res) == 2

    s0 = res[0]
    assert isinstance(s0, tuple)
    assert s0._fields == ('writtenaset', 'newaset')
    assert np.allclose(s0.writtenaset, array5by7)
    assert np.allclose(s0.newaset, array10)

    s1 = res[1]
    assert isinstance(s1, tuple)
    assert s1._fields == ('writtenaset', 'newaset')
    assert np.allclose(s1.writtenaset, array5by7 + 1)
    assert np.allclose(s1.newaset, array10 + 1)
    rco.close()


def test_reader_co_read_fails_nonexistant_aset_name(written_repo, array5by7):
    rco = written_repo.checkout()

    with pytest.raises(KeyError):
        _ = rco['doesnotexist', 0]
    rco.close()


def test_reader_co_read_fails_nonexistant_sample_name(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    with pytest.raises(KeyError):
        _ = rco['doesnotexist', 124]
    rco.close()


def test_reader_co_get_read_returns_none_nonexistant_sample_name(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    out = rco.get('writtenaset', 124)
    assert out is None
    rco.close()


def test_reader_co_read_in_context_manager_no_loop(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['writtenaset', 'newaset'], '0'] = [array5by7, array10]
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    with rco:
        assert rco._is_conman is True
        assert np.allclose(rco['writtenaset', '0'], array5by7)
    rco.close()


def test_reader_co_read_in_context_manager_many_samples_looping(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with wco:
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco[['writtenaset', 'newaset'], idx] = [array5by7, array10]
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    with rco:
        writtenasetOut = rco['writtenaset', [i for i in range(100)]]
        newasetOut = rco['newaset', [i for i in range(100)]]
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            assert np.allclose(array5by7, rco['writtenaset', idx])
            assert np.allclose(array10, rco['newaset', idx])

            o = rco[['writtenaset', 'newaset'], idx]
            assert np.allclose(o.writtenaset, array5by7)
            assert np.allclose(o.newaset, array10)

            assert np.allclose(writtenasetOut[idx], array5by7)
            assert np.allclose(newasetOut[idx], array10)
    rco.close()


def test_reader_co_read_ellipses_select_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    o = rco[..., 0]
    assert 'writtenaset' in o._fields
    assert 'newaset' in o._fields
    assert np.allclose(o.writtenaset, array5by7)
    assert np.allclose(o.newaset, array10)
    rco.close()


def test_reader_co_read_slice_select_aset_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    o = rco[:, 0]
    assert 'writtenaset' in o._fields
    assert 'newaset' in o._fields
    assert np.allclose(o.writtenaset, array5by7)
    assert np.allclose(o.newaset, array10)
    rco.close()


def test_reader_co_read_ellipses_select_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    out = rco[..., [0, 1]]
    assert len(out) == 2

    o1 = out[0]
    assert 'writtenaset' in o1._fields
    assert 'newaset' in o1._fields
    assert np.allclose(o1.writtenaset, array5by7)
    assert np.allclose(o1.newaset, array10)

    o2 = out[1]
    assert 'writtenaset' in o2._fields
    assert 'newaset' in o2._fields
    assert np.allclose(o2.writtenaset, array5by7 + 1)
    assert np.allclose(o2.newaset, array10 + 1)
    rco.close()


def test_reader_co_read_slice_select_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    array10[:] = 0
    wco.arraysets['newaset'][0] = array10
    wco.arraysets['newaset'][1] = array10 + 1
    wco.arraysets['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = written_repo.checkout()
    out = rco[:, [0, 1]]
    assert len(out) == 2

    o1 = out[0]
    assert 'writtenaset' in o1._fields
    assert 'newaset' in o1._fields
    assert np.allclose(o1.writtenaset, array5by7)
    assert np.allclose(o1.newaset, array10)

    o2 = out[1]
    assert 'writtenaset' in o2._fields
    assert 'newaset' in o2._fields
    assert np.allclose(o2.writtenaset, array5by7 + 1)
    assert np.allclose(o2.newaset, array10 + 1)
    rco.close()