import pytest
import numpy as np


@pytest.fixture(scope='function')
def server_instance(managed_tmpdir):
    from hangar import serve
    server, channel_addr = serve(managed_tmpdir, overwrite=True)
    server.start()
    yield channel_addr
    server.stop(0)


def test_server_is_started(server_instance, written_repo):
    written_repo.add_remote('origin', server_instance)
    written_repo.push('origin', 'master')
