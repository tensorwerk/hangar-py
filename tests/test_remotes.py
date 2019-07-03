import pytest
import numpy as np
import time


@pytest.fixture(scope='function')
def server_instance(managed_tmpdir, worker_id):
    from hangar import serve

    address = 'localhost:50051' if worker_id == 'master' else f'localhost:5005{worker_id[-1]}'
    server, hangserver = serve(managed_tmpdir, overwrite=True, channel_address=address)
    server.start()
    time.sleep(0.2)
    yield address
    hangserver.env._close_environments()
    server.stop(0.2)
    time.sleep(0.2)


def test_server_is_started_via_ping_pong(server_instance, written_repo):

    written_repo.add_remote('origin', server_instance)
    reply = written_repo._ping_server('origin')
    assert reply == 'PONG'


def test_server_push_master_branch_one_commit(server_instance, repo, array5by7):

    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    co.datasets['_dset']['0'] = array5by7.astype(np.float32)
    co.metadata['hello'] = 'world'
    co.commit('first')
    co.close()

    repo.add_remote('origin', server_instance)
    repo.push('origin', 'master')


def test_server_push_second_branch_with_new_commit(server_instance, repo, array5by7):

    co = repo.checkout(write=True)
    co.datasets.init_dataset(name='_dset', shape=(5, 7), dtype=np.float32)
    co.datasets['_dset']['0'] = array5by7.astype(np.float32)
    co.metadata['hello'] = 'world'
    co.commit('first')
    co.close()

    repo.add_remote('origin', server_instance)
    repo.push('origin', 'master')

    branch = repo.create_branch('testbranch')
    co = repo.checkout(write=True, branch_name=branch)
    co.datasets['_dset']['1'] = array5by7.astype(np.float32)
    co.metadata.add('a', 'b')
    co.commit('this is a commit message')
    co.close()
    repo.push('origin', branch)