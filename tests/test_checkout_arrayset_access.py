import pytest
import numpy as np


@pytest.mark.parametrize("samplename", ['0', '-1', 1, 0, 1000, 'alkea'])
def test_write_single_arrayset_single_sample(written_repo, array5by7, samplename):
    wco = written_repo.checkout(write=True)
    wco['_aset', samplename] = array5by7
    assert np.allclose(array5by7, wco.arraysets['_aset'][samplename])
    wco.commit('init')
    assert np.allclose(array5by7, wco.arraysets['_aset'][samplename])
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(array5by7, rco.arraysets['_aset'][samplename])
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
    wco['_aset', samplenames] = values

    for val, name in zip(values, samplenames):
        assert np.allclose(val, wco.arraysets['_aset'][name])
    wco.commit('init')
    for val, name in zip(values, samplenames):
        assert np.allclose(val, wco.arraysets['_aset'][name])
    wco.close()

    rco = written_repo.checkout()
    for val, name in zip(values, samplenames):
        assert np.allclose(val, rco.arraysets['_aset'][name])
    rco.close()


def test_write_multiple_arrayset_single_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    wco[['_aset', 'newaset'], '0'] = [array5by7, array10]
    assert np.allclose(array5by7, wco.arraysets['_aset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.commit('init')
    assert np.allclose(array5by7, wco.arraysets['_aset']['0'])
    assert np.allclose(array10, wco.arraysets['newaset']['0'])
    wco.close()

    rco = written_repo.checkout()
    assert np.allclose(array5by7, rco.arraysets['_aset']['0'])
    assert np.allclose(array10, rco.arraysets['newaset']['0'])
    rco.close()


def test_write_fails_multiple_arrayset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with pytest.raises(SyntaxError):
        wco[['_aset', 'newaset'], ['0', 1]] = [[array5by7, array5by7], [array10, array10]]
    wco.close()

def test_write_fails_nonmatching_multiple_asets_single_sample(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    array10 = np.arange(10, dtype=np.float32)
    wco.arraysets.init_arrayset('newaset', prototype=array10)
    with pytest.raises(ValueError):
        wco[['_aset', 'newaset'], '0'] = [array5by7]
    with pytest.raises(TypeError):
        wco[['_aset', 'newaset'], '0'] = array5by7
    wco.close()


def test_write_fails_nonmatching_sing_aset_multiple_samples(written_repo, array5by7):
    wco = written_repo.checkout(write=True)

    with pytest.raises(TypeError):
        wco['_aset', [i for i in range(10)]] = array5by7
    with pytest.raises(ValueError):
        wco['_aset', [i for i in range(10)]] = [array5by7 for i in range(4)]
    with pytest.raises(ValueError):
        wco['_aset', [i for i in range(10)]] = [array5by7 for i in range(14)]

    with pytest.raises(ValueError):
        wco['_aset', []] = [array5by7 for i in range(1)]
    wco.close()