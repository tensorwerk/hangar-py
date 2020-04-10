import pytest

import numpy as np


param_shapes = [(1,), (1000,), (1, 1), (623, 3, 5), (2, 4, 5, 6, 1, 3)]
param_dtypes = [np.uint8, np.float32, np.float64, np.int32]
param_digest = ['0=digesta', '0=digestaaaaaa', '2=digestaaaaaaaaaaaaaaaaaaaaaaaaaa']
param_schema = ['schemaa', 'schemaaaaaaaaa', 'schemaaaaaaaaaaaaaaaa']


def assert_array_equal(arr, arr2):
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


@pytest.fixture(scope='module', params=[
    'hello', ' world', 'how are you today! ',
    '325', f'{"".join([str(i) for i in range(100)])}',
    'o\n\x01'
])
def str_testcase(request):
    return request.param


@pytest.fixture(scope='module', params=[
    b'hello', b' world', b'how are you today! ',
    b'325', b'\x01\x00\x12\x14'
])
def bytes_testcase(request):
    return request.param


@pytest.fixture(scope='module')
def ident_testcase(ident_digest, ident_schema):
    return (ident_digest, ident_schema)


def test_serialize_deserialize_array(array_testcase):
    from hangar.remote.chunks import _serialize_arr
    from hangar.remote.chunks import _deserialize_arr

    raw = _serialize_arr(array_testcase)
    res = _deserialize_arr(raw)
    assert_array_equal(array_testcase, res)


def test_serialize_deserialize_str(str_testcase):
    from hangar.remote.chunks import _serialize_str, _deserialize_str
    raw = _serialize_str(str_testcase)
    res = _deserialize_str(raw)
    assert res == str_testcase


def test_serialize_deserialize_bytes(bytes_testcase):
    from hangar.remote.chunks import _serialize_bytes, _deserialize_bytes
    raw = _serialize_bytes(bytes_testcase)
    res = _deserialize_bytes(raw)
    assert bytes_testcase == res


@pytest.mark.parametrize('expected_dtype_code,data', [
    (0, np.array([0, 1, 2, 3, 4])),
    (2, 'i am string'),
    (3, b'i am bytes')
])
def test_serialize_deserialize_data(expected_dtype_code, data):
    from hangar.remote.chunks import serialize_data, deserialize_data

    dtcode, raw = serialize_data(data)
    res = deserialize_data(dtype_code=dtcode, raw_data=raw)
    assert dtcode == expected_dtype_code
    if isinstance(res, np.ndarray):
        assert_array_equal(data, res)
    elif isinstance(res, (str, bytes)):
        assert data == res
    else:
        raise TypeError(data)


def test_serialize_deserialize_ident(ident_testcase):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import DataIdent

    digest, schema = ident_testcase
    raw = serialize_ident(digest, schema)
    res = deserialize_ident(raw)
    assert isinstance(res, DataIdent)
    assert res.digest == digest
    assert res.schema == schema


def test_serialize_deserialize_record(array_testcase, ident_testcase):
    from hangar.remote.chunks import serialize_record
    from hangar.remote.chunks import deserialize_record
    from hangar.remote.chunks import DataRecord

    digest, schema = ident_testcase
    raw = serialize_record(array_testcase, digest, schema)
    res = deserialize_record(raw)
    assert isinstance(res, DataRecord)
    assert_array_equal(res.data, array_testcase)
    assert res.digest == digest
    assert res.schema == schema


@pytest.mark.parametrize('nrecords', [1, 25])
def test_serialize_deserialize_record_pack(ident_testcase, nrecords):
    from hangar.remote.chunks import serialize_record
    from hangar.remote.chunks import serialize_record_pack
    from hangar.remote.chunks import deserialize_record
    from hangar.remote.chunks import deserialize_record_pack
    from hangar.remote.chunks import DataRecord

    idx = 0
    ArrList, RecList = [], []
    digest, schema = ident_testcase
    for shape in param_shapes:
        for dtype in param_dtypes:
            arr = 200 * np.random.random_sample(shape) + 100
            arr = arr.astype(dtype)
            digest = f'digest{str(idx) + str(digest)}'
            schema = f'schema{str(idx) + str(schema)}'
            idx += 1

            ArrList.append((arr, digest, schema))
            RecList.append(serialize_record(arr, digest, schema))

    rawpack = serialize_record_pack(RecList)
    reslist = deserialize_record_pack(rawpack)

    assert reslist == RecList

    for rawres, origRec in zip(reslist, ArrList):
        resRec = deserialize_record(rawres)
        assert isinstance(resRec, DataRecord)
        assert_array_equal(resRec.data, origRec[0])
        assert resRec.digest == origRec[1]
        assert resRec.schema == origRec[2]


def test_serialize_deserialize_ident_digest_field_only(ident_testcase):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import DataIdent

    digest, schema = ident_testcase
    raw = serialize_ident(digest, '')
    res = deserialize_ident(raw)
    assert isinstance(res, DataIdent)
    assert res.digest == digest
    assert res.schema == ''


def test_serialize_deserialize_ident_schema_field_only(ident_testcase):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import DataIdent

    digest, schema = ident_testcase
    raw = serialize_ident('', schema)
    res = deserialize_ident(raw)
    assert isinstance(res, DataIdent)
    assert res.digest == ''
    assert res.schema == schema


@pytest.mark.parametrize('nrecords', [1, 25])
def test_serialize_deserialize_ident_only_record_pack(ident_testcase, nrecords):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import serialize_record_pack
    from hangar.remote.chunks import deserialize_record_pack
    from hangar.remote.chunks import DataIdent

    idx = 0
    IdentList, RawList = [], []
    digest, schema = ident_testcase
    for idx in range(nrecords):
        digest = f'digest{str(idx) + str(digest)}'
        schema = f'schema{str(idx) + str(schema)}'

        IdentList.append((digest, schema))
        RawList.append(serialize_ident(digest, schema))

    packed_raw = serialize_record_pack(RawList)
    unpacked_raw = deserialize_record_pack(packed_raw)

    assert unpacked_raw == RawList

    for raw, origIdent in zip(unpacked_raw, IdentList):
        resIdent = deserialize_ident(raw)
        assert isinstance(resIdent, DataIdent)
        assert resIdent.digest == origIdent[0]
        assert resIdent.schema == origIdent[1]


@pytest.mark.parametrize('nrecords', [1, 25])
def test_serialize_deserialize_ident_only_digest_only_record_pack(ident_testcase, nrecords):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import serialize_record_pack
    from hangar.remote.chunks import deserialize_record_pack
    from hangar.remote.chunks import DataIdent

    idx = 0
    IdentList, RawList = [], []
    digest, schema = ident_testcase
    for idx in range(nrecords):
        digest = f'digest{str(idx)+str(digest)}'
        schema = f''

        IdentList.append((digest, schema))
        RawList.append(serialize_ident(digest, schema))

    packed_raw = serialize_record_pack(RawList)
    unpacked_raw = deserialize_record_pack(packed_raw)

    assert unpacked_raw == RawList

    for raw, origIdent in zip(unpacked_raw, IdentList):
        resIdent = deserialize_ident(raw)
        assert isinstance(resIdent, DataIdent)
        assert resIdent.digest == origIdent[0]
        assert resIdent.schema == origIdent[1]


@pytest.mark.parametrize('nrecords', [1, 25])
def test_serialize_deserialize_ident_only_schema_only_record_pack(ident_testcase, nrecords):
    from hangar.remote.chunks import serialize_ident
    from hangar.remote.chunks import deserialize_ident
    from hangar.remote.chunks import serialize_record_pack
    from hangar.remote.chunks import deserialize_record_pack
    from hangar.remote.chunks import DataIdent

    idx = 0
    IdentList, RawList = [], []
    digest, schema = ident_testcase
    for idx in range(nrecords):
        digest = f''
        schema = f'schema{str(idx)+str(schema)}'

        IdentList.append((digest, schema))
        RawList.append(serialize_ident(digest, schema))

    packed_raw = serialize_record_pack(RawList)
    unpacked_raw = deserialize_record_pack(packed_raw)

    assert unpacked_raw == RawList

    for raw, origIdent in zip(unpacked_raw, IdentList):
        resIdent = deserialize_ident(raw)
        assert isinstance(resIdent, DataIdent)
        assert resIdent.digest == origIdent[0]
        assert resIdent.schema == origIdent[1]
