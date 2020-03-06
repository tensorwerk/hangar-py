import pytest
import numpy as np


# -------------------------- Reader Checkout ----------------------------------


@pytest.mark.parametrize('write', [True, False])
def test_arrayset_getattr_does_not_raise_permission_error_if_alive(write, aset_samples_initialized_repo):
    co = aset_samples_initialized_repo.checkout(write=write)
    asets = co.columns

    assert hasattr(asets, 'doesnotexist') is False  # does not raise error
    assert hasattr(asets, '_mode') is True
    with pytest.raises(AttributeError):
        assert getattr(asets, 'doesnotexist')
    assert getattr(asets, '_mode') == 'a' if write else 'r'

    co.close()
    with pytest.raises(PermissionError):
        hasattr(asets, 'doesnotexist')
    with pytest.raises(PermissionError):
        hasattr(asets, '_mode')


def test_write_in_context_manager_no_loop(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    with wco:
        assert wco._is_conman is True
        wco['writtenaset']['0'] = array5by7
        wco['newaset']['0'] = array10
    assert wco._is_conman is False

    assert np.allclose(array5by7, wco.columns['writtenaset']['0'])
    assert np.allclose(array10, wco.columns['newaset']['0'])
    wco.commit('init')
    assert np.allclose(array5by7, wco.columns['writtenaset']['0'])
    assert np.allclose(array10, wco.columns['newaset']['0'])
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    assert np.allclose(array5by7, rco.columns['writtenaset']['0'])
    assert np.allclose(array10, rco.columns['newaset']['0'])
    rco.close()


def test_write_in_context_manager_many_samples_looping(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    with wco:
        assert wco._is_conman is True
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco['writtenaset'][idx] = array5by7
            wco['newaset'][idx] = array10
    assert wco._is_conman is False

    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, wco.columns['writtenaset'][idx])
        assert np.allclose(array10, wco.columns['newaset'][idx])
    wco.commit('init')
    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, wco.columns['writtenaset'][idx])
        assert np.allclose(array10, wco.columns['newaset'][idx])
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    for idx in range(100):
        array10[:] = idx
        array5by7[:] = idx
        assert np.allclose(array5by7, rco.columns['writtenaset'][idx])
        assert np.allclose(array10, rco.columns['newaset'][idx])
    rco.close()


def test_write_fails_if_checkout_closed(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)
    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    wco['writtenaset'][0] = array5by7
    wco['newaset'][0] = array10
    wco.close()
    with pytest.raises((PermissionError, UnboundLocalError)):
        wco['writtenaset'][1] = array5by7
        wco['newaset'][1] = array10

    wco2 = aset_samples_initialized_repo.checkout(write=True)
    assert 0 in wco2.columns['writtenaset']
    assert 0 in wco2.columns['newaset']
    assert 1 not in wco2.columns['writtenaset']
    assert 1 not in wco2.columns['newaset']
    wco2.close()


def test_write_context_manager_fails_if_checkout_closed(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)
    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    wco['writtenaset'][0] = array5by7
    wco['newaset'][0] = array10
    wco.close()
    with pytest.raises(PermissionError):
        with wco:
            wco['writtenaset'][1] = array5by7
    with pytest.raises(PermissionError):
        with wco:
            wco['newaset'][1] = array10

    wco2 = aset_samples_initialized_repo.checkout(write=True)
    assert 0 in wco2.columns['writtenaset']
    assert 0 in wco2.columns['newaset']
    assert 1 not in wco2.columns['writtenaset']
    assert 1 not in wco2.columns['newaset']
    wco2.close()


def test_writer_co_read_single_aset_single_sample(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    assert np.allclose(wco['writtenaset', 0], array5by7)
    assert np.allclose(wco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(wco['writtenaset', 2], array5by7 + 2)
    wco.close()


def test_writer_co_read_single_aset_multiple_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    res = wco[('writtenaset', 0), ('writtenaset', 1), ('writtenaset', 2)]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array5by7 + 1)
    assert np.allclose(res[2], array5by7 + 2)
    wco.close()


def test_writer_co_read_multiple_aset_single_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    array10[:] = 0
    wco.columns['newaset'][0] = array10
    wco.columns['newaset'][1] = array10 + 1
    wco.columns['newaset'][2] = array10 + 2

    res = wco[('writtenaset', 0), ('newaset', 0)]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array10)
    res = wco[('writtenaset', 1), ('newaset', 1)]
    assert np.allclose(res[0], array5by7 + 1)
    assert np.allclose(res[1], array10 + 1)
    wco.close()


def test_writer_co_read_multtiple_aset_multiple_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    array10[:] = 0
    wco.columns['newaset'][0] = array10
    wco.columns['newaset'][1] = array10 + 1
    wco.columns['newaset'][2] = array10 + 2

    res = wco[('writtenaset', 0), ('newaset', 0), ('writtenaset', 1), ('newaset', 1)]
    assert isinstance(res, list)
    assert len(res) == 4
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array10)
    assert np.allclose(res[2], array5by7 + 1)
    assert np.allclose(res[3], array10 + 1)
    wco.close()


def test_writer_co_read_fails_nonexistant_aset_name(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    with pytest.raises(KeyError):
        _ = wco['doesnotexist', 0]
    wco.close()


def test_writer_co_read_fails_nonexistant_sample_name(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    with pytest.raises(KeyError):
        _ = wco['doesnotexist', 124]
    wco.close()


def test_writer_co_get_returns_none_on_nonexistant_sample_name(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    out = wco.get(('writtenaset', 124))
    assert out is None
    wco.close()


def test_writer_co_read_in_context_manager_no_loop(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    wco['writtenaset']['0'] = array5by7
    wco['newaset']['0'] = array10
    with wco:
        assert wco._is_conman is True
        assert np.allclose(wco['writtenaset', '0'], array5by7)
    wco.close()


def test_writer_co_read_in_context_manager_many_samples_looping(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    with wco:
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco['writtenaset'][idx] = array5by7
            wco['newaset'][idx] = array10

    with wco:
        waset_keys = [('writtenaset', i) for i in range(100)]
        naset_keys = [('newaset', i) for i in range(100)]
        writtenasetOut = wco[waset_keys]
        newasetOut = wco[naset_keys]
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            assert np.allclose(array5by7, wco['writtenaset', idx])
            assert np.allclose(array10, wco['newaset', idx])

            o = wco[('writtenaset', idx), ('newaset', idx)]
            assert np.allclose(o[0], array5by7)
            assert np.allclose(o[1], array10)

            assert np.allclose(writtenasetOut[idx], array5by7)
            assert np.allclose(newasetOut[idx], array10)
    wco.close()


@pytest.mark.parametrize('write', [True, False])
def test_co_read_dunder_getitem_excepts_missing_sample(aset_samples_initialized_repo, write):
    co = aset_samples_initialized_repo.checkout(write=write)
    with pytest.raises(KeyError):
        res = co['writtenaset', 0]
    co.close()


@pytest.mark.parametrize('write', [True, False])
def test_co_read_get_except_missing_true_excepts_missing_sample(aset_samples_initialized_repo, write):
    co = aset_samples_initialized_repo.checkout(write=write)
    with pytest.raises(KeyError):
        res = co.get(('writtenaset', 0), except_missing=True)
    co.close()


@pytest.mark.parametrize('write', [True, False])
def test_co_read_get_except_missing_false_returns_none_on_missing_sample(aset_samples_initialized_repo, write):
    co = aset_samples_initialized_repo.checkout(write=write)
    res_1 = co.get(('writtenaset', 0))
    assert res_1 is None
    res_2 = co.get(('writtenaset', 0), except_missing=False)
    assert res_2 is None
    co.close()


def test_writer_co_aset_finds_connection_manager_of_any_aset_in_cm(aset_samples_initialized_repo):
    wco = aset_samples_initialized_repo.checkout(write=True)
    wco.add_ndarray_column('second', shape=(20,), dtype=np.uint8)
    asets = wco.columns

    with wco.columns['second'] as second_aset:
        assert wco.columns['second']._is_conman is True
        assert second_aset._is_conman is True
        assert asets._any_is_conman() is True

    with wco.columns['writtenaset'] as written_aset:
        assert wco.columns['writtenaset']._is_conman is True
        assert written_aset._is_conman is True
        assert asets._any_is_conman() is True

    assert wco.columns['writtenaset']._is_conman is False
    assert wco.columns['second']._is_conman is False
    assert asets._any_is_conman() is False
    wco.close()


def test_writer_co_aset_cm_not_allow_remove_aset(aset_samples_initialized_repo, array5by7):

    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    asets = wco.columns
    with asets as cm_asets:
        with pytest.raises(PermissionError):
            cm_asets.delete('writtenaset')
        with pytest.raises(PermissionError):
            asets.delete('writtenaset')
        with pytest.raises(PermissionError):
            wco.columns.delete('writtenaset')

        with pytest.raises(PermissionError):
            del cm_asets['writtenaset']
        with pytest.raises(PermissionError):
            del asets['writtenaset']
        with pytest.raises(PermissionError):
            del wco.columns['writtenaset']

    assert len(wco['writtenaset']) == 3
    assert np.allclose(wco['writtenaset', 0], array5by7)
    assert np.allclose(wco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(wco['writtenaset', 2], array5by7 + 2)
    wco.close()


def test_writer_co_column_instance_cm_not_allow_any_column_removal(repo_20_filled_samples):

    wco = repo_20_filled_samples.checkout(write=True)
    columns = wco.columns
    writtenaset = wco.columns['writtenaset']
    second_aset = wco.columns['second_aset']

    with second_aset:
        with pytest.raises(PermissionError):
            columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            columns.delete('second_aset')
        with pytest.raises(PermissionError):
            wco.columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            wco.columns.delete('second_aset')
        with pytest.raises(PermissionError):
            del columns['writtenaset']
        with pytest.raises(PermissionError):
            del columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['written_aset']

    with writtenaset:
        with pytest.raises(PermissionError):
            columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            columns.delete('second_aset')
        with pytest.raises(PermissionError):
            wco.columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            wco.columns.delete('second_aset')
        with pytest.raises(PermissionError):
            del columns['writtenaset']
        with pytest.raises(PermissionError):
            del columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['written_aset']

    with columns:
        with pytest.raises(PermissionError):
            columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            columns.delete('second_aset')
        with pytest.raises(PermissionError):
            wco.columns.delete('writtenaset')
        with pytest.raises(PermissionError):
            wco.columns.delete('second_aset')
        with pytest.raises(PermissionError):
            del columns['writtenaset']
        with pytest.raises(PermissionError):
            del columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['second_aset']
        with pytest.raises(PermissionError):
            del wco.columns['written_aset']

    wco.close()


def test_writer_co_aset_removes_all_samples_and_arrayset_still_exists(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)
    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2
    assert len(wco.columns) == 1
    assert len(wco.columns['writtenaset']) == 3

    with wco.columns['writtenaset'] as wset:
        del wset[0]
        del wset[1]
        del wset[2]
        # Removed all samples, now the aset's gone
        assert len(wset) == 0
        assert len(wco.columns) == 1
    assert len(wco.columns) == 1

    del wco.columns['writtenaset']

    assert len(wco.columns) == 0
    with pytest.raises(KeyError):
        len(wco.columns['writtenaset'])
    wco.close()


# -------------------------- Reader Checkout ----------------------------------


def test_reader_co_read_single_aset_single_sample(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    assert np.allclose(rco['writtenaset', 0], array5by7)
    assert np.allclose(rco['writtenaset', 1], array5by7 + 1)
    assert np.allclose(rco['writtenaset', 2], array5by7 + 2)
    rco.close()


def test_reader_co_read_single_aset_multiple_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    res = rco[('writtenaset', 0), ('writtenaset', 1), ('writtenaset', 2)]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array5by7 + 1)
    assert np.allclose(res[2], array5by7 + 2)
    rco.close()


def test_reader_co_read_multiple_aset_single_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    array10[:] = 0
    wco.columns['newaset'][0] = array10
    wco.columns['newaset'][1] = array10 + 1
    wco.columns['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    res = rco[('writtenaset', 0), ('newaset', 0)]
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array10)
    res = rco[('writtenaset', 1), ('newaset', 1)]
    assert np.allclose(res[0], array5by7 + 1)
    assert np.allclose(res[1], array10 + 1)
    rco.close()


def test_reader_co_read_multtiple_aset_multiple_samples(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.columns['writtenaset'][1] = array5by7 + 1
    wco.columns['writtenaset'][2] = array5by7 + 2

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    array10[:] = 0
    wco.columns['newaset'][0] = array10
    wco.columns['newaset'][1] = array10 + 1
    wco.columns['newaset'][2] = array10 + 2
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    res = rco[('writtenaset', 0), ('newaset', 0), ('writtenaset', 1), ('newaset', 1)]
    assert isinstance(res, list)
    assert len(res) == 4
    assert np.allclose(res[0], array5by7)
    assert np.allclose(res[1], array10)
    assert np.allclose(res[2], array5by7 + 1)
    assert np.allclose(res[3], array10 + 1)
    rco.close()


def test_reader_co_read_fails_nonexistant_aset_name(aset_samples_initialized_repo, array5by7):
    rco = aset_samples_initialized_repo.checkout()
    with pytest.raises(KeyError):
        _ = rco['doesnotexist', 0]
    rco.close()


def test_reader_co_read_fails_nonexistant_sample_name(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)
    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    with pytest.raises(KeyError):
        _ = rco['doesnotexist', 124]
    rco.close()


def test_reader_co_get_read_returns_none_nonexistant_sample_name(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)
    array5by7[:] = 0
    wco.columns['writtenaset'][0] = array5by7
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    out = rco.get(('writtenaset', 124))
    assert out is None
    rco.close()


def test_reader_co_read_in_context_manager_no_loop(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    wco['writtenaset']['0'] = array5by7
    wco['newaset']['0'] = array10
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    with rco:
        assert rco._is_conman is True
        assert np.allclose(rco['writtenaset', '0'], array5by7)
    rco.close()


def test_reader_co_read_in_context_manager_many_samples_looping(aset_samples_initialized_repo, array5by7):
    wco = aset_samples_initialized_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.add_ndarray_column('newaset', prototype=array10)
    with wco:
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            wco['writtenaset'][idx] = array5by7
            wco['newaset'][idx] = array10
    wco.commit('first')
    wco.close()

    rco = aset_samples_initialized_repo.checkout()
    with rco:
        waset_keys = [('writtenaset', i) for i in range(100)]
        naset_keys = [('newaset', i) for i in range(100)]
        writtenasetOut = rco[waset_keys]
        newasetOut = rco[naset_keys]
        for idx in range(100):
            array10[:] = idx
            array5by7[:] = idx
            assert np.allclose(array5by7, rco['writtenaset', idx])
            assert np.allclose(array10, rco['newaset', idx])

            o = rco[('writtenaset', idx), ('newaset', idx)]
            assert np.allclose(o[0], array5by7)
            assert np.allclose(o[1], array10)
            assert np.allclose(writtenasetOut[idx], array5by7)
            assert np.allclose(newasetOut[idx], array10)
    rco.close()
