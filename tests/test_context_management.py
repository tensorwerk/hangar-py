import pytest
import numpy as np

from conftest import fixed_shape_backend_params, variable_shape_backend_params

all_backend_params = list(set(fixed_shape_backend_params).union(set(variable_shape_backend_params)))


@pytest.mark.parametrize('backend1', all_backend_params)
@pytest.mark.parametrize('backend2', all_backend_params)
def test_nested_context_manager_does_not_close_all_open(repo, backend1, backend2):
    co = repo.checkout(write=True)
    fooaset = co.add_ndarray_column('foo', prototype=np.arange(10), backend=backend1)
    baraset = co.add_ndarray_column('bar', prototype=np.arange(10), backend=backend2, contains_subsamples=True)

    with co:
        assert co.columns._any_is_conman() is True
        assert fooaset._is_conman is True
        assert baraset._is_conman is True
        with fooaset as foo:
            assert co.columns._any_is_conman() is True
            assert foo._is_conman is True
            assert fooaset._is_conman is True
            assert baraset._is_conman is True
        assert co.columns._any_is_conman() is True
        assert fooaset._is_conman is True
        assert baraset._is_conman is True
    assert co.columns._any_is_conman() is False
    co.close()
