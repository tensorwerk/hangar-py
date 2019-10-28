import pytest

import numpy as np

from hangar import Repository


backend_params = ['00', '10']


@pytest.fixture(params=backend_params)
def variable_shape_repo_co_float32(managed_tmpdir, request) -> Repository:
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo_obj.checkout(write=True)
    co.arraysets.init_arrayset(name='writtenaset',
                               shape=(10, 10, 10, 10),
                               dtype=np.float32,
                               variable_shape=True,
                               backend_opts=request.param)
    yield co
    co.close()
    repo_obj._env._close_environments()


@pytest.fixture(params=backend_params)
def variable_shape_repo_co_uint8(managed_tmpdir, request) -> Repository:
    repo_obj = Repository(path=managed_tmpdir, exists=False)
    repo_obj.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    co = repo_obj.checkout(write=True)
    co.arraysets.init_arrayset(name='writtenaset',
                               shape=(10, 10, 10, 10),
                               dtype=np.uint8,
                               variable_shape=True,
                               backend_opts=request.param)
    yield co
    co.close()
    repo_obj._env._close_environments()