import pytest
import numpy as np
from math import prod


def assert_equal(arr, arr2):
    assert np.array_equal(arr, arr2)
    assert arr.dtype == arr2.dtype


def test_bulk_importer_ndarray(repo):
    from hangar.bulk_importer import run_bulk_import
    from hangar.bulk_importer import UDF_Return

    def make_ndarray(column, key, shape, dtype, multiplier):
        size = prod(shape)
        arr = np.arange(size, dtype=dtype).reshape(shape) * multiplier
        yield UDF_Return(column=column, key=key, data=arr)

    co = repo.checkout(write=True)
    co.add_ndarray_column('arr', shape=(5, 5), dtype=np.uint32)
    co.commit('first')
    co.close()

    kwargs = []
    expected_kv = []
    for idx in range(200):
        _kw_dict = {
            'column': 'arr',
            'key': idx,
            'shape': (5, 5),
            'dtype': np.uint32,
            'multiplier': idx
        }
        kwargs.append(_kw_dict)

        for _udf_val in make_ndarray(**_kw_dict):
            expected_kv.append(_udf_val)
    assert len(expected_kv) == 200

    run_bulk_import(
        repo,
        branch_name='master',
        column_names=['arr'],
        udf=make_ndarray,
        udf_kwargs=kwargs,
        ncpus=2)

    co = repo.checkout()
    try:
        arr_col = co['arr']
        assert len(arr_col) == 200
        for _expected_udf_val in expected_kv:
            assert _expected_udf_val.key in arr_col
            assert_equal(arr_col[_expected_udf_val.key], _expected_udf_val.data)
    finally:
        co.close()


def test_bulk_importer_pystr(repo):
    from hangar.bulk_importer import run_bulk_import
    from hangar.bulk_importer import UDF_Return

    def make_pystr(column, key, str_val):
        yield UDF_Return(column=column, key=key, data=str_val)

    co = repo.checkout(write=True)
    co.add_str_column('str')
    co.commit('first')
    co.close()

    kwargs = []
    expected_kv = []
    for idx in range(200):
        _kw_dict = {
            'column': 'str',
            'key': idx,
            'str_val': f'{str(idx) * 2}',
        }
        kwargs.append(_kw_dict)

        for _udf_val in make_pystr(**_kw_dict):
            expected_kv.append(_udf_val)
    assert len(expected_kv) == 200

    run_bulk_import(
        repo,
        branch_name='master',
        column_names=['str'],
        udf=make_pystr,
        udf_kwargs=kwargs,
        ncpus=2)

    co = repo.checkout()
    try:
        str_col = co['str']
        assert len(str_col) == 200
        for _expected_udf_val in expected_kv:
            assert _expected_udf_val.key in str_col
            assert str_col[_expected_udf_val.key] == _expected_udf_val.data
    finally:
        co.close()


def test_bulk_importer_pybytes(repo):
    from hangar.bulk_importer import run_bulk_import
    from hangar.bulk_importer import UDF_Return

    def make_pybytes(column, key, str_val):
        raw = str_val.encode()
        yield UDF_Return(column=column, key=key, data=raw)

    co = repo.checkout(write=True)
    co.add_bytes_column('bytes')
    co.commit('first')
    co.close()

    kwargs = []
    expected_kv = []
    for idx in range(200):
        _kw_dict = {
            'column': 'bytes',
            'key': idx,
            'str_val': f'{str(idx) * 2}',
        }
        kwargs.append(_kw_dict)

        for _udf_val in make_pybytes(**_kw_dict):
            expected_kv.append(_udf_val)
    assert len(expected_kv) == 200

    run_bulk_import(
        repo,
        branch_name='master',
        column_names=['bytes'],
        udf=make_pybytes,
        udf_kwargs=kwargs,
        ncpus=2)

    co = repo.checkout()
    try:
        bytes_col = co['bytes']
        assert len(bytes_col) == 200
        for _expected_udf_val in expected_kv:
            assert _expected_udf_val.key in bytes_col
            assert bytes_col[_expected_udf_val.key] == _expected_udf_val.data
    finally:
        co.close()


def test_bulk_importer_two_col_pybytes_pystr(repo):
    from hangar.bulk_importer import run_bulk_import
    from hangar.bulk_importer import UDF_Return

    def _make_pystr(column, key, str_val):
        yield UDF_Return(column=column, key=key, data=str_val)

    def _make_pybytes(column, key, str_val):
        raw = str_val.encode()
        yield UDF_Return(column=column, key=key, data=raw)

    def make_pystr_pybytes(str_col, bytes_col, key, str_val):
        yield from _make_pystr(column=str_col, key=key, str_val=str_val)
        yield from _make_pybytes(column=bytes_col, key=key, str_val=str_val)

    co = repo.checkout(write=True)
    co.add_bytes_column('bytes')
    co.add_str_column('str')
    co.commit('first')
    co.close()

    kwargs = []
    expected_kv = []
    for idx in range(200):
        _kw_dict = {
            'str_col': 'str',
            'bytes_col': 'bytes',
            'key': idx,
            'str_val': f'{str(idx) * 2}',
        }
        kwargs.append(_kw_dict)

        for _udf_val in make_pystr_pybytes(**_kw_dict):
            expected_kv.append(_udf_val)
    assert len(expected_kv) == 400

    run_bulk_import(
        repo,
        branch_name='master',
        column_names=['bytes', 'str'],
        udf=make_pystr_pybytes,
        udf_kwargs=kwargs,
        ncpus=2)

    co = repo.checkout()
    try:
        pybytes_col = co['bytes']
        pystr_col = co['str']
        assert len(pybytes_col) == 200
        assert len(pystr_col) == 200
        for _expected_udf_val in expected_kv:
            assert _expected_udf_val.column in ['str', 'bytes']
            if _expected_udf_val.column == 'str':
                assert _expected_udf_val.key in pystr_col
                assert pystr_col[_expected_udf_val.key] == _expected_udf_val.data
            elif _expected_udf_val.column == 'bytes':
                assert _expected_udf_val.key in pystr_col
                assert pybytes_col[_expected_udf_val.key] == _expected_udf_val.data
            else:
                raise ValueError(_expected_udf_val.column)
    finally:
        co.close()


def test_signature_wrong(repo):
    from hangar.bulk_importer import run_bulk_import
    from hangar.bulk_importer import UDF_Return

    def wrong_sig_udf(a, b, c=None):
        yield UDF_Return(column='str', key=a, data=f'{a} {b} {c}')

    co = repo.checkout(write=True)
    co.add_str_column('str')
    co.commit('first')
    co.close()

    kwargs = []
    for idx in range(200):
        _kw_dict = {
            'a': 'bytes',
            'str_val': f'{str(idx) * 2}',
        }
        kwargs.append(_kw_dict)

    with pytest.raises(TypeError):
        run_bulk_import(
            repo,
            branch_name='master',
            column_names=['str'],
            udf=wrong_sig_udf,
            udf_kwargs=kwargs,
            ncpus=2)
