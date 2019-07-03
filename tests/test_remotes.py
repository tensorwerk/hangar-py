import pytest
import numpy as np
import time


@pytest.fixture(scope='function')
def server_instance(managed_tmpdir):
    from hangar import serve
    server = serve(managed_tmpdir, overwrite=False)
    server.start()
    yield 'localhost:50051'
    server.stop(0.2)
    time.sleep(0.2)


def test_server_is_started(server_instance, repo, array5by7):

    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    co.datasets['_dset']['0'] = array5by7.astype(np.float32)
    co.metadata['hello'] = 'world'
    co.commit('first')
    co.close()

    repo.add_remote('origin', server_instance)
    repo.push('origin', 'master')
