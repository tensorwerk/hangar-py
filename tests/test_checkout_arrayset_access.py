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


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
def test_writer_co_read_two_asets_one_invalid_fieldname_is_renamed(written_repo, array5by7, invalid_name):
    wco = written_repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name, prototype=array10)
    wco['writtenaset', (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)

    out1 = wco[('writtenaset', invalid_name), 0]
    assert out1._fields == ('writtenaset', '_1')
    assert np.allclose(out1.writtenaset, array5by7)
    assert np.allclose(out1._1, array10)

    out2 = wco[(invalid_name, 'writtenaset'), 1]
    assert out2._fields == ('_0', 'writtenaset')
    assert np.allclose(out2.writtenaset, array5by7 + 1)
    assert np.allclose(out2._0, array10 + 1)

    out3 = wco[('writtenaset', invalid_name), (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert nt._fields == ('writtenaset', '_1')
        assert np.allclose(nt.writtenaset, array5by7 + idx)
        assert np.allclose(nt._1, array10 + idx)
    wco.close()


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name1', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
@pytest.mark.parametrize('invalid_name2', ['foo.bar2', '_helloworld2', 'fail-again2', '.lol2'])
def test_writer_co_read_two_asets_two_invalid_fieldname_both_renamed(repo, array5by7, invalid_name1, invalid_name2):
    wco = repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name1, prototype=array5by7)
    wco.arraysets.init_arrayset(invalid_name2, prototype=array10)
    wco[invalid_name1, (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name2, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)

    out1 = wco[(invalid_name1, invalid_name2), 0]
    assert out1._fields == ('_0', '_1')
    assert np.allclose(out1._0, array5by7)
    assert np.allclose(out1._1, array10)

    out2 = wco[(invalid_name2, invalid_name1), 1]
    assert out2._fields == ('_0', '_1')
    assert np.allclose(out2._1, array5by7 + 1)
    assert np.allclose(out2._0, array10 + 1)

    out3 = wco[(invalid_name1, invalid_name2), (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert nt._fields == ('_0', '_1')
        assert np.allclose(nt._0, array5by7 + idx)
        assert np.allclose(nt._1, array10 + idx)
    wco.close()


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name1', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
@pytest.mark.parametrize('invalid_name2', ['foo.bar2', '_helloworld2', 'fail-again2', '.lol2'])
def test_writer_co_read_all_asets_all_invalid_fieldname_both_renamed(repo, array5by7, invalid_name1, invalid_name2):
    """
    Uses Slice Syntax (:) instead of specifying names directly. We don't know
    which order the arraysets will come out in the namedtuple.
    """
    wco = repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name1, prototype=array5by7)
    wco.arraysets.init_arrayset(invalid_name2, prototype=array10)
    wco[invalid_name1, (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name2, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)

    out1 = wco[:, 0]
    assert '_0' in out1._fields
    assert '_1' in out1._fields
    assert len(out1._fields) == 2
    if out1._0.shape == (10,):
        assert np.allclose(out1._0, array10)
        assert np.allclose(out1._1, array5by7)
    else:
        assert np.allclose(out1._0, array5by7)
        assert np.allclose(out1._1, array10)

    out3 = wco[..., (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert '_0' in nt._fields
        assert '_1' in nt._fields
        assert len(nt._fields) == 2
        if nt._0.shape == (10,):
            assert np.allclose(nt._0, array10 + idx)
            assert np.allclose(nt._1, array5by7 + idx)
        else:
            assert np.allclose(nt._0, array5by7 + idx)
            assert np.allclose(nt._1, array10 + idx)
    wco.close()


@pytest.mark.parametrize('invalid_name', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
def test_writer_co_read_two_asets_one_invalid_fieldname_warns_of_field_rename(written_repo, array5by7, invalid_name):
    wco = written_repo.checkout(write=True)
    wco.arraysets['writtenaset'][0] = array5by7
    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name, prototype=array10)
    wco.arraysets[invalid_name][0] = array10

    with pytest.warns(UserWarning, match='Arrayset names contains characters'):
        wco[('writtenaset', invalid_name), 0]
    with pytest.warns(UserWarning, match='Arrayset names contains characters'):
        wco[(invalid_name, 'writtenaset'), 0]
    wco.close()


def test_writer_co_aset_finds_connection_manager_of_any_aset_in_cm(written_repo):
    wco = written_repo.checkout(write=True)
    wco.arraysets.init_arrayset('second', shape=(20,), dtype=np.uint8)
    asets = wco.arraysets

    with wco.arraysets['second'] as second_aset:
        assert wco.arraysets['second']._is_conman is True
        assert second_aset._is_conman is True
        assert wco.arraysets['writtenaset']._is_conman is False
        assert asets._any_is_conman() is True

    with wco.arraysets['writtenaset'] as written_aset:
        assert wco.arraysets['writtenaset']._is_conman is True
        assert written_aset._is_conman is True
        assert wco.arraysets['second']._is_conman is False
        assert asets._any_is_conman() is True

    assert wco.arraysets['writtenaset']._is_conman is False
    assert wco.arraysets['second']._is_conman is False
    assert asets._any_is_conman() is False
    wco.close()


def test_writer_co_aset_cm_not_allow_remove_aset(written_repo, array5by7):

    wco = written_repo.checkout(write=True)

    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2

    asets = wco.arraysets
    with asets as cm_asets:
        with pytest.raises(PermissionError):
            cm_asets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            asets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('writtenaset')

        with pytest.raises(PermissionError):
            del cm_asets['writtenaset']
        with pytest.raises(PermissionError):
            del asets['writtenaset']
        with pytest.raises(PermissionError):
            del wco.arraysets['writtenaset']

    assert len(wco['writtenaset']) == 3
    assert np.allclose(wco['writtenaset', 0], array5by7)
    assert np.allclose(wco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(wco['writtenaset', 2], array5by7 + 2)
    wco.close()


def test_writer_co_aset_instance_cm_not_allow_any_aset_removal(repo_with_20_samples):

    wco = repo_with_20_samples.checkout(write=True)
    asets = wco.arraysets
    writtenaset = wco.arraysets['writtenaset']
    second_aset = wco.arraysets['second_aset']

    with second_aset:
        with pytest.raises(PermissionError):
            asets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            asets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            del asets['writtenaset']
        with pytest.raises(PermissionError):
            del asets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['written_aset']

    with writtenaset:
        with pytest.raises(PermissionError):
            asets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            asets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            del asets['writtenaset']
        with pytest.raises(PermissionError):
            del asets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['written_aset']

    with asets:
        with pytest.raises(PermissionError):
            asets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            asets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('writtenaset')
        with pytest.raises(PermissionError):
            wco.arraysets.remove_aset('second_aset')
        with pytest.raises(PermissionError):
            del asets['writtenaset']
        with pytest.raises(PermissionError):
            del asets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['second_aset']
        with pytest.raises(PermissionError):
            del wco.arraysets['written_aset']

    wco.close()


def test_writer_co_aset_removes_all_samples_and_arrayset_still_exists(written_repo, array5by7):
    wco = written_repo.checkout(write=True)
    array5by7[:] = 0
    wco.arraysets['writtenaset'][0] = array5by7
    wco.arraysets['writtenaset'][1] = array5by7 + 1
    wco.arraysets['writtenaset'][2] = array5by7 + 2
    assert len(wco.arraysets) == 1
    assert len(wco.arraysets['writtenaset']) == 3

    with wco.arraysets['writtenaset'] as wset:
        wset.remove(0)
        wset.remove(1)
        wset.remove(2)
        # Removed all samples, now the aset's gone
        assert len(wset) == 0
        assert len(wco.arraysets) == 1
    assert len(wco.arraysets) == 1

    del wco.arraysets['writtenaset']

    assert len(wco.arraysets) == 0
    with pytest.raises(ReferenceError):
        len(wset)
    with pytest.raises(KeyError):
        len(wco.arraysets['writtenaset'])
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


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
def test_reader_co_read_two_asets_one_invalid_fieldname_is_renamed(written_repo, array5by7, invalid_name):
    wco = written_repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name, prototype=array10)
    wco['writtenaset', (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)
    wco.commit('yo')
    wco.close()

    rco = written_repo.checkout()
    out1 = rco[('writtenaset', invalid_name), 0]
    assert out1._fields == ('writtenaset', '_1')
    assert np.allclose(out1.writtenaset, array5by7)
    assert np.allclose(out1._1, array10)

    out2 = rco[(invalid_name, 'writtenaset'), 1]
    assert out2._fields == ('_0', 'writtenaset')
    assert np.allclose(out2.writtenaset, array5by7 + 1)
    assert np.allclose(out2._0, array10 + 1)

    out3 = rco[('writtenaset', invalid_name), (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert nt._fields == ('writtenaset', '_1')
        assert np.allclose(nt.writtenaset, array5by7 + idx)
        assert np.allclose(nt._1, array10 + idx)
    rco.close()


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name1', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
@pytest.mark.parametrize('invalid_name2', ['foo.bar2', '_helloworld2', 'fail-again2', '.lol2'])
def test_reader_co_read_two_asets_two_invalid_fieldname_both_renamed(repo, array5by7, invalid_name1, invalid_name2):
    wco = repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name1, prototype=array5by7)
    wco.arraysets.init_arrayset(invalid_name2, prototype=array10)
    wco[invalid_name1, (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name2, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)
    wco.commit('yo')
    wco.close()

    rco = repo.checkout()
    out1 = rco[(invalid_name1, invalid_name2), 0]
    assert out1._fields == ('_0', '_1')
    assert np.allclose(out1._0, array5by7)
    assert np.allclose(out1._1, array10)

    out2 = rco[(invalid_name2, invalid_name1), 1]
    assert out2._fields == ('_0', '_1')
    assert np.allclose(out2._1, array5by7 + 1)
    assert np.allclose(out2._0, array10 + 1)

    out3 = rco[(invalid_name1, invalid_name2), (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert nt._fields == ('_0', '_1')
        assert np.allclose(nt._0, array5by7 + idx)
        assert np.allclose(nt._1, array10 + idx)
    rco.close()


@pytest.mark.filterwarnings("ignore:Arrayset names contains characters")
@pytest.mark.parametrize('invalid_name1', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
@pytest.mark.parametrize('invalid_name2', ['foo.bar2', '_helloworld2', 'fail-again2', '.lol2'])
def test_reader_co_read_all_asets_all_invalid_fieldname_both_renamed(repo, array5by7, invalid_name1, invalid_name2):
    """
    Uses Slice Syntax (:) instead of specifying names directly. We don't know
    which order the arraysets will come out in the namedtuple.
    """
    wco = repo.checkout(write=True)
    array5by7 = np.zeros_like(array5by7)
    array10 = np.zeros((10,), dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name1, prototype=array5by7)
    wco.arraysets.init_arrayset(invalid_name2, prototype=array10)
    wco[invalid_name1, (0, 1, 2)] = (array5by7, array5by7 + 1, array5by7 + 2)
    wco[invalid_name2, (0, 1, 2)] = (array10, array10 + 1, array10 + 2)
    wco.commit('yo')
    wco.close()

    rco = repo.checkout()
    out1 = rco[:, 0]
    assert '_0' in out1._fields
    assert '_1' in out1._fields
    assert len(out1._fields) == 2
    if out1._0.shape == (10,):
        assert np.allclose(out1._0, array10)
        assert np.allclose(out1._1, array5by7)
    else:
        assert np.allclose(out1._0, array5by7)
        assert np.allclose(out1._1, array10)

    out3 = rco[..., (0, 1, 2)]
    for idx, nt in enumerate(out3):
        assert '_0' in nt._fields
        assert '_1' in nt._fields
        assert len(nt._fields) == 2
        if nt._0.shape == (10,):
            assert np.allclose(nt._0, array10 + idx)
            assert np.allclose(nt._1, array5by7 + idx)
        else:
            assert np.allclose(nt._0, array5by7 + idx)
            assert np.allclose(nt._1, array10 + idx)
    rco.close()


@pytest.mark.parametrize('invalid_name', ['foo.bar', '_helloworld', 'fail-again', '.lol'])
def test_reader_co_read_two_asets_one_invalid_fieldname_warns_of_field_rename(written_repo, array5by7, invalid_name):
    wco = written_repo.checkout(write=True)
    wco.arraysets['writtenaset'][0] = array5by7
    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset(invalid_name, prototype=array10)
    wco.arraysets[invalid_name][0] = array10
    wco.commit('commit message')
    wco.close()

    rco = written_repo.checkout()
    with pytest.warns(UserWarning, match='Arrayset names contains characters'):
        rco[('writtenaset', invalid_name), 0]
    with pytest.warns(UserWarning, match='Arrayset names contains characters'):
        rco[(invalid_name, 'writtenaset'), 0]
    rco.close()