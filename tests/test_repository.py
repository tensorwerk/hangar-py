import pytest
import os
import shutil

@pytest.fixture
def managed_tmpdir(tmp_path):
    yield tmp_path
    shutil.rmtree(tmp_path)


def test_import():
    import hangar
    import hangar.repository
    from hangar import Repository


def test_create_empty_repository(managed_tmpdir):
    from hangar import Repository
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='Test User', user_email='foo@test.bar')
    init_branches = repo.list_branch_names()
    assert init_branches == ['master']
    assert repo._repo_path == os.path.join(managed_tmpdir, '__hangar')