import pytest
import numpy as np


def test_add_samples_to_existing_column(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo_20_filled_samples2
    expected = '============ \n'\
               '| Branch: master \n'\
               ' \n'\
               '============ \n'\
               '| ADDED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 20 \n'\
               '|  - "dummy": 20 \n'\
               ' \n'\
               '============ \n'\
               '| DELETED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'\
               '============ \n'\
               '| MUTATED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'
    dummyData = np.arange(50).astype(np.int64)
    co2 = repo.checkout(write=True)
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.columns['dummy'][str(idx)] = dummyData
        co2.columns['dummy'][idx] = dummyData
    df = co2.diff.staged()
    co2.close()
    assert status(repo._env.hashenv, 'master', df.diff).getvalue() == expected


def test_mutate_sample_values(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo = repo_20_filled_samples2
    expected = '============ \n'\
               '| Branch: master \n'\
               ' \n'\
               '============ \n'\
               '| ADDED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'\
               '============ \n'\
               '| DELETED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'\
               '============ \n'\
               '| MUTATED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 5 \n'\
               '|  - "dummy": 5 \n'\
               ' \n'

    dummyData = np.arange(50).astype(np.int64)
    co2 = repo.checkout(write=True)
    for idx in range(5, 10):
        dummyData[:] = idx + 10
        co2.columns['dummy'][idx] = dummyData
    df = co2.diff.staged()
    co2.close()
    assert status(repo._env.hashenv, 'master', df.diff).getvalue() == expected


def test_delete_samples(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo_20_filled_samples2
    expected = '============ \n'\
               '| Branch: master \n'\
               ' \n'\
               '============ \n'\
               '| ADDED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'\
               '============ \n'\
               '| DELETED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 5 \n'\
               '|  - "dummy": 5 \n'\
               ' \n'\
               '============ \n'\
               '| MUTATED \n'\
               '|---------- \n'\
               '| Schema: 0 \n'\
               '|---------- \n'\
               '| Samples: 0 \n'\
               ' \n'

    co2 = repo.checkout(write=True)
    for idx in range(5, 10):
        del co2.columns['dummy'][idx]
    df = co2.diff.staged()
    co2.close()
    assert status(repo._env.hashenv, 'master', df.diff).getvalue() == expected


def test_add_new_column_schema_and_samples(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo_20_filled_samples2
    expected = (
        '============ \n'
        '| Branch: master \n'
        ' \n'
        '============ \n'
        '| ADDED \n'
        '|---------- \n'
        '| Schema: 1 \n'
        '|  - "new_aset": \n'
        '|       digest="1=555a833b66ab" \n'
        '|       column_layout: flat \n'
        '|       column_type: ndarray \n'
        '|       schema_hasher_tcode: 1 \n'
        '|       data_hasher_tcode: 0 \n'
        '|       schema_type: fixed_shape \n'
        '|       shape: (10, 10) \n'
        '|       dtype: float32 \n'
        '|       backend: 01 \n'
        '|       backend_options: {\'complib\': \'blosc:lz4hc\', \'complevel\': 5, \'shuffle\': \'byte\'} \n'
        '|---------- \n'
        '| Samples: 5 \n'
        '|  - "new_aset": 5 \n'
        ' \n'
        '============ \n'
        '| DELETED \n'
        '|---------- \n'
        '| Schema: 0 \n'
        '|---------- \n'
        '| Samples: 0 \n'
        ' \n'
        '============ \n'
        '| MUTATED \n'
        '|---------- \n'
        '| Schema: 0 \n'
        '|---------- \n'
        '| Samples: 0 \n'
        ' \n'
    )
    co2 = repo.checkout(write=True)
    co2.add_ndarray_column('new_aset', shape=(10, 10), dtype=np.float32)
    for idx in range(5):
        dummyData = np.random.randn(10, 10).astype(np.float32)
        co2.columns['new_aset'][idx] = dummyData
    df = co2.diff.staged()
    co2.close()
    result = status(repo._env.hashenv, 'master', df.diff).getvalue()
    assert result == expected


def test_add_new_column_schema_and_sample_and_delete_old_column(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo_20_filled_samples2
    expected = (
        '============ \n'
        '| Branch: master \n'
        ' \n'
        '============ \n'
        '| ADDED \n'
        '|---------- \n'
        '| Schema: 1 \n'
        '|  - "new_aset": \n'
        '|       digest="1=555a833b66ab" \n'
        '|       column_layout: flat \n'
        '|       column_type: ndarray \n'
        '|       schema_hasher_tcode: 1 \n'
        '|       data_hasher_tcode: 0 \n'
        '|       schema_type: fixed_shape \n'
        '|       shape: (10, 10) \n'
        '|       dtype: float32 \n'
        '|       backend: 01 \n'
        '|       backend_options: {\'complib\': \'blosc:lz4hc\', \'complevel\': 5, \'shuffle\': \'byte\'} \n'
        '|---------- \n'
        '| Samples: 5 \n'
        '|  - "new_aset": 5 \n'
        ' \n'
        '============ \n'
        '| DELETED \n'
        '|---------- \n'
        '| Schema: 1 \n'
        '|  - "dummy": \n'
        '|       digest="1=18599cd5ea25" \n'
        '|       column_layout: flat \n'
        '|       column_type: ndarray \n'
        '|       schema_hasher_tcode: 1 \n'
        '|       data_hasher_tcode: 0 \n'
        '|       schema_type: fixed_shape \n'
        '|       shape: (50,) \n'
        '|       dtype: int64 \n'
        '|       backend: 10 \n'
        '|       backend_options: {} \n'
        '|---------- \n'
        '| Samples: 10 \n'
        '|  - "dummy": 10 \n'
        ' \n'
        '============ \n'
        '| MUTATED \n'
        '|---------- \n'
        '| Schema: 0 \n'
        '|---------- \n'
        '| Samples: 0 \n'
        ' \n'
    )
    co2 = repo.checkout(write=True)
    new = co2.add_ndarray_column('new_aset', shape=(10, 10), dtype=np.float32)
    for idx in range(5):
               dummyData = np.random.randn(10, 10).astype(np.float32)
               co2.columns['new_aset'][idx] = dummyData
    del co2.columns['dummy']
    df = co2.diff.staged()
    co2.close()
    result = status(repo._env.hashenv, 'master', df.diff).getvalue()
    assert result == expected


def test_add_new_schema_and_samples_and_change_old_backend(repo_20_filled_samples2):
    from hangar.records.summarize import status
    repo = repo_20_filled_samples2
    expected = (
        '============ \n'
        '| Branch: master \n'
        ' \n'
        '============ \n'
        '| ADDED \n'
        '|---------- \n'
        '| Schema: 1 \n'
        '|  - "new_aset": \n'
        '|       digest="1=555a833b66ab" \n'
        '|       column_layout: flat \n'
        '|       column_type: ndarray \n'
        '|       schema_hasher_tcode: 1 \n'
        '|       data_hasher_tcode: 0 \n'
        '|       schema_type: fixed_shape \n'
        '|       shape: (10, 10) \n'
        '|       dtype: float32 \n'
        '|       backend: 01 \n'
        '|       backend_options: {\'complib\': \'blosc:lz4hc\', \'complevel\': 5, \'shuffle\': \'byte\'} \n'
        '|---------- \n'
        '| Samples: 5 \n'
        '|  - "new_aset": 5 \n'
        ' \n'
        '============ \n'
        '| DELETED \n'
        '|---------- \n'
        '| Schema: 0 \n'
        '|---------- \n'
        '| Samples: 0 \n'
        ' \n'
        '============ \n'
        '| MUTATED \n'
        '|---------- \n'
        '| Schema: 1 \n'
        '|  - "dummy": \n'
        '|       digest="1=5d6cc8241705" \n'
        '|       column_layout: flat \n'
        '|       column_type: ndarray \n'
        '|       schema_hasher_tcode: 1 \n'
        '|       data_hasher_tcode: 0 \n'
        '|       schema_type: fixed_shape \n'
        '|       shape: (50,) \n'
        '|       dtype: int64 \n'
        '|       backend: 00 \n'
        '|       backend_options: {\'complib\': \'blosc:lz4hc\', \'complevel\': 5, \'shuffle\': \'byte\'} \n'
        '|---------- \n'
        '| Samples: 5 \n'
        '|  - "dummy": 5 \n'
        ' \n'
    )
    co2 = repo.checkout(write=True)
    co2.columns['dummy'].change_backend('00')
    co2.add_ndarray_column('new_aset', shape=(10, 10), dtype=np.float32)
    for idx in range(5):
        dummyData = np.random.randn(10, 10).astype(np.float32)
        co2.columns['new_aset'][idx] = dummyData
        co2.columns['dummy'][idx] = np.arange(50).astype(np.int64) + idx
    df = co2.diff.staged()
    co2.close()
    result = status(repo._env.hashenv, 'master', df.diff).getvalue()
    assert result == expected
