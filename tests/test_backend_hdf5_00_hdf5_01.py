import pytest
import numpy as np


@pytest.fixture(params=['00', '01'])
def be_filehandle(request):
    if request.param == '00':
        from hangar.backends.hdf5_00 import HDF5_00_FileHandles
        return HDF5_00_FileHandles
    elif request.param == '01':
        from hangar.backends.hdf5_01 import HDF5_01_FileHandles
        return HDF5_01_FileHandles
    else:
        raise ValueError(f'request param "{request.param}" for backend code unknown.')


@pytest.mark.parametrize('clib,clibCode',
                         [('blosc:blosclz', 0), ('blosc:lz4', 1),
                          ('blosc:lz4hc', 2), ('blosc:zlib', 4),
                          ('blosc:zstd', 5)])
@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, 0), ('byte', 1), ('bit', 2)])
@pytest.mark.parametrize('beCode', ['00', '01'])
def test_blosc_filter_opts_result_in_correct_dataset_args(
        be_filehandle, clib, clibCode, clevel, cshuffle, cshuffleCode, beCode):

    out = be_filehandle._dataset_opts(complib=clib,
                                      complevel=clevel,
                                      shuffle=cshuffle)
    expected = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, clevel, cshuffleCode, clibCode),
        'shuffle': False}
    assert out == expected


@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_lzf_filter_opts_result_in_correct_dataset_args(be_filehandle, cshuffle, cshuffleCode):
    out = be_filehandle._dataset_opts(complib='lzf',
                                      complevel=None,
                                      shuffle=cshuffle)
    expected = {
        'compression': 'lzf',
        'compression_opts': None,
        'shuffle': cshuffleCode}
    assert out == expected


@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_gzip_filter_opts_result_in_correct_dataset_args(be_filehandle, clevel, cshuffle, cshuffleCode):
    out = be_filehandle._dataset_opts(complib='gzip',
                                      complevel=clevel,
                                      shuffle=cshuffle)
    expected = {
        'compression': 'gzip',
        'compression_opts': clevel,
        'shuffle': cshuffleCode}
    assert out == expected


# ------------------------- test actual compression ---------------------------


@pytest.mark.parametrize('clib,clibCode',
                         [('blosc:blosclz', 0), ('blosc:lz4', 1),
                          ('blosc:lz4hc', 2), ('blosc:zlib', 4),
                          ('blosc:zstd', 5)])
@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, 0), ('byte', 1), ('bit', 2)])
@pytest.mark.parametrize('be_code', ['00', '01'])
def test_arrayset_init_with_various_blosc_opts(repo, array5by7, clib, clibCode, clevel, cshuffle, cshuffleCode, be_code):

    opts = {
        'shuffle': cshuffle,
        'complib': clib,
        'complevel': clevel,
    }
    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column('aset', prototype=array5by7, backend=be_code, backend_options=opts)
    assert aset.backend == be_code
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    wuid = aset._be_fs[be_code].w_uid
    plist = aset._be_fs[be_code].wFp[wuid]['/0'].id.get_create_plist()
    _, _, resopts, _ = plist.get_filter(0)
    res_clevel, res_cshuffle, res_clib = resopts[4:7]
    assert res_clevel == clevel
    assert res_clib == clibCode
    assert res_cshuffle == cshuffleCode
    wco.commit('hi')
    wco.close()


@pytest.mark.parametrize('cshuffle,cshuffleCode', [(False, False), (True, True)])
@pytest.mark.parametrize('be_code', ['00', '01'])
def test_arrayset_init_with_various_lzf_opts(repo, array5by7, cshuffle, cshuffleCode, be_code):

    opts = {
        'shuffle': cshuffle,
        'complib': 'lzf',
        'complevel': None,
    }
    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column('aset', prototype=array5by7, backend=be_code, backend_options=opts)
    assert aset.backend == be_code
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    res_compression = aset._be_fs[be_code].wFp[aset._be_fs[be_code].w_uid]['/0'].compression
    res_shuffle = aset._be_fs[be_code].wFp[aset._be_fs[be_code].w_uid]['/0'].shuffle
    assert res_compression == 'lzf'
    assert res_shuffle == cshuffleCode
    wco.commit('hi')
    wco.close()


@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(False, False), (True, True)])
@pytest.mark.parametrize('be_code', ['00', '01'])
def test_arrayset_init_with_various_gzip_opts(repo, array5by7, clevel, cshuffle, cshuffleCode, be_code):

    opts = {
        'shuffle': cshuffle,
        'complib': 'gzip',
        'complevel': clevel,
    }
    wco = repo.checkout(write=True)
    aset = wco.add_ndarray_column(
        'aset', prototype=array5by7, backend=be_code, backend_options=opts)
    assert aset.backend == be_code
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    res_compression = aset._be_fs[be_code].wFp[aset._be_fs[be_code].w_uid]['/0'].compression
    res_compression_opts = aset._be_fs[be_code].wFp[aset._be_fs[be_code].w_uid]['/0'].compression_opts
    res_shuffle = aset._be_fs[be_code].wFp[aset._be_fs[be_code].w_uid]['/0'].shuffle
    assert res_compression == 'gzip'
    assert res_shuffle == cshuffleCode
    assert res_compression_opts == clevel
    wco.commit('hi')
    wco.close()


@pytest.mark.parametrize('be_code', ['00', '01'])
def test_arrayset_overflows_collection_size_collection_count(be_code, repo, monkeypatch):
    if be_code == '00':
        from hangar.backends import hdf5_00
        monkeypatch.setattr(hdf5_00, 'COLLECTION_COUNT', 5)
        monkeypatch.setattr(hdf5_00, 'COLLECTION_SIZE', 10)
    elif be_code == '01':
        from hangar.backends import hdf5_01
        monkeypatch.setattr(hdf5_01, 'COLLECTION_COUNT', 5)
        monkeypatch.setattr(hdf5_01, 'COLLECTION_SIZE', 10)
    else:
        raise ValueError(f'be_code param "{be_code}" unknown.')

    wco = repo.checkout(write=True)
    proto = np.arange(50).astype(np.uint16)
    aset = wco.add_ndarray_column('aset', prototype=proto, backend=be_code)
    with aset as cm_aset:
        for i in range(500):
            proto[:] = i
            cm_aset[i] = proto
    assert aset._be_fs[be_code].hColsRemain == 4
    assert aset._be_fs[be_code].hMaxSize == 10
    wco.commit('hello')

    with aset as cm_aset:
        for i in range(500):
            proto[:] = i
            assert np.allclose(proto, cm_aset[i])
    wco.close()

    rco = repo.checkout()
    naset = rco.columns['aset']
    with naset as ncm_aset:
        for i in range(500):
            proto[:] = i
            assert np.allclose(proto, ncm_aset[i])
    rco.close()
