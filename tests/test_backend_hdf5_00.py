import pytest
import h5py


@pytest.mark.parametrize('clib,clibCode',
                         [('blosc:blosclz', 0), ('blosc:lz4', 1),
                          ('blosc:lz4hc', 2), ('blosc:zlib', 4),
                          ('blosc:zstd', 5)])
@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, 0), ('byte', 1), ('bit', 2)])
def test_blosc_filter_opts_result_in_correct_dataset_args(
        clib, clibCode, clevel, cshuffle, cshuffleCode):
    from hangar.backends.hdf5_00 import HDF5_00_FileHandles

    out = HDF5_00_FileHandles._dataset_opts(complib=clib,
                                            complevel=clevel,
                                            shuffle=cshuffle,
                                            fletcher32=True)
    expected = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, clevel, cshuffleCode, clibCode),
        'fletcher32': True,
        'shuffle': False}

    assert out == expected


@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_lzf_filter_opts_result_in_correct_dataset_args(cshuffle, cshuffleCode):
    from hangar.backends.hdf5_00 import HDF5_00_FileHandles

    out = HDF5_00_FileHandles._dataset_opts(complib='lzf',
                                            complevel=None,
                                            shuffle=cshuffle,
                                            fletcher32=True)
    expected = {
        'compression': 'lzf',
        'compression_opts': None,
        'fletcher32': True,
        'shuffle': cshuffleCode}

    assert out == expected


@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_gzip_filter_opts_result_in_correct_dataset_args(clevel, cshuffle, cshuffleCode):
    from hangar.backends.hdf5_00 import HDF5_00_FileHandles

    out = HDF5_00_FileHandles._dataset_opts(complib='gzip',
                                            complevel=clevel,
                                            shuffle=cshuffle,
                                            fletcher32=True)
    expected = {
        'compression': 'gzip',
        'compression_opts': clevel,
        'fletcher32': True,
        'shuffle': cshuffleCode}

    assert out == expected


# ------------------------- test actual compression ---------------------------

hdf5BloscAvail = h5py.h5z.filter_avail(32001)

@pytest.mark.skipif(hdf5BloscAvail is False, reason='hdf5 blosc compression lib not installed on this system.')
@pytest.mark.parametrize('clib,clibCode',
                         [('blosc:blosclz', 0), ('blosc:lz4', 1),
                          ('blosc:lz4hc', 2), ('blosc:zlib', 4),
                          ('blosc:zstd', 5)])
@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, 0), ('byte', 1), ('bit', 2)])
def test_arrayset_init_with_various_blosc_opts(repo, array5by7, clib, clibCode, clevel, cshuffle, cshuffleCode):

    opts = {
        'backend': '00',
        'shuffle': cshuffle,
        'complib': clib,
        'complevel': clevel,
        'fletcher32': True
    }
    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=opts)
    assert aset.backend == '00'
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    plist = aset._fs['00'].wFp['/0'].id.get_create_plist()
    _, _, resopts, _ = plist.get_filter(0)
    res_clevel, res_cshuffle, res_clib = resopts[4:7]
    assert res_clevel == clevel
    assert res_clib == clibCode
    assert res_cshuffle == cshuffleCode
    wco.commit('hi')
    wco.close()


@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_arrayset_init_with_various_lzf_opts(repo, array5by7, cshuffle, cshuffleCode):

    opts = {
        'backend': '00',
        'shuffle': cshuffle,
        'complib': 'lzf',
        'complevel': None,
        'fletcher32': True
    }
    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=opts)
    assert aset.backend == '00'
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    res_compression = aset._fs['00'].wFp[aset._fs['00'].w_uid]['/0'].compression
    res_shuffle = aset._fs['00'].wFp[aset._fs['00'].w_uid]['/0'].shuffle
    assert res_compression == 'lzf'
    assert res_shuffle == cshuffleCode
    wco.commit('hi')
    wco.close()


@pytest.mark.parametrize('clevel', [1, 4, 8])
@pytest.mark.parametrize('cshuffle,cshuffleCode', [(None, False), ('byte', True)])
def test_arrayset_init_with_various_gzip_opts(repo, array5by7, clevel, cshuffle, cshuffleCode):

    opts = {
        'backend': '00',
        'shuffle': cshuffle,
        'complib': 'gzip',
        'complevel': clevel,
        'fletcher32': True
    }
    wco = repo.checkout(write=True)
    aset = wco.arraysets.init_arrayset('aset', prototype=array5by7, backend_opts=opts)
    assert aset.backend == '00'
    with aset as a:
        for i in range(10):
            a[i] = array5by7 + i

    res_compression = aset._fs['00'].wFp[aset._fs['00'].w_uid]['/0'].compression
    res_compression_opts = aset._fs['00'].wFp[aset._fs['00'].w_uid]['/0'].compression_opts
    res_shuffle = aset._fs['00'].wFp[aset._fs['00'].w_uid]['/0'].shuffle
    assert res_compression == 'gzip'
    assert res_shuffle == cshuffleCode
    assert res_compression_opts == clevel
    wco.commit('hi')
    wco.close()
