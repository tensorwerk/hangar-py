import pytest

import numpy as np


param_shapes = [
    (1,), (10,), (1000,), (1, 1), (10, 10), (100, 100),
    (623, 3, 5), (1, 6, 83), (211, 11), (2, 4, 5, 6, 1, 3)]
param_dtypes = [np.uint8, np.float32, np.float64, np.int32]
param_digest = ['digesta', 'digestaaaaaa', 'digestaaaaaaaaaaaaaaaaaaaaaaaaaa']
param_schema = ['schemaa', 'schemaaaaaaaaa', 'schemaaaaaaaaaaaaaaaa']


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


@pytest.fixture(scope='module', params=param_shapes)
def arr_shape(request):
    return request.param

@pytest.fixture(scope='module', params=param_dtypes)
def arr_dtype(request):
    return request.param

@pytest.fixture(scope='module', params=param_digest)
def ident_digest(request):
    return request.param

@pytest.fixture(scope='module', params=param_schema)
def ident_schema(request):
    return request.param


@pytest.fixture(scope='module')
def array_testcase(arr_shape, arr_dtype):
    arr = 200 * np.random.random_sample(arr_shape) - 100
    return arr.astype(arr_dtype)


@pytest.fixture(scope='module')
def ident_testcase(ident_digest, ident_schema):
    return (ident_digest, ident_schema)


def test_serialize_deserialize_array(array_testcase):
    from hangar.remote.chunks import serialize_arr
    from hangar.remote.chunks import deserialize_arr

    raw = serialize_arr(array_testcase)
    res = deserialize_arr(raw)
    assert_equal(array_testcase, res)


def test_serialize_deserialize_ident(ident_testcase):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import ArrayIdent

    digest, schema = ident_testcase
    raw = serialize_ident(digest, schema)
    res = deserialize_ident(raw)
    assert isinstance(res, ArrayIdent)
    assert res.digest == digest
    assert res.schema == schema


def test_serialize_deserialize_record(array_testcase, ident_testcase):
    from hangar.remote.chunks import serialize_record
    from hangar.remote.chunks import deserialize_record
    from hangar.remote.chunks import ArrayRecord

    digest, schema = ident_testcase
    raw = serialize_record(array_testcase, digest, schema)
    res = deserialize_record(raw)
    assert isinstance(res, ArrayRecord)
    assert_equal(res.array, array_testcase)
    assert res.digest == digest
    assert res.schema == schema


@pytest.mark.parametrize('nrecords', [1, 10, 25])
def test_serialize_deserialize_record_pack(ident_testcase, nrecords):
    from hangar.remote.chunks import serialize_record
    from hangar.remote.chunks import serialize_record_pack
    from hangar.remote.chunks import deserialize_record
    from hangar.remote.chunks import deserialize_record_pack
    from hangar.remote.chunks import ArrayRecord

    idx = 0
    ArrList, RecList = [], []
    digest, schema = ident_testcase
    for shape in param_shapes:
        for dtype in param_dtypes:
            arr = 200 * np.random.random_sample(shape) + 100
            arr = arr.astype(dtype)
            digest = f'digest{idx*len(digest)}'
            schema = f'schema{idx*len(schema)}'
            idx += 1

            ArrList.append((arr, digest, schema))
            RecList.append(serialize_record(arr, digest, schema))

    rawpack = serialize_record_pack(RecList)
    reslist = deserialize_record_pack(rawpack)

    assert reslist == RecList

    for rawres, origRec in zip(reslist, ArrList):
        resRec = deserialize_record(rawres)
        assert isinstance(resRec, ArrayRecord)
        assert_equal(resRec.array, origRec[0])
        assert resRec.digest == origRec[1]
        assert resRec.schema == origRec[2]