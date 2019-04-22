import os
import pytest
from hangar import Repository


def test_imports():
    import hangar
    from hangar import Repository
    import hangar.repository


def test_starting_up(managed_tmpdir):
    repo = Repository(path=managed_tmpdir)
    repo.init(
        user_name='tester', user_email='foo@test.bar', remove_old=True)
    assert repo.list_branch_names() == ['master']
    assert os.path.isdir(repo._repo_path)
    assert repo._repo_path == os.path.join(managed_tmpdir, '__hangar')
    assert repo.status() == 'CLEAN'


def initial_read_checkout(managed_tmpdir):
    repo = Repository(path=managed_tmpdir)
    repo.init(
        user_name='tester', user_email='foo@test.bar', remove_old=True)
    # TODO it should do something to indicate the issue or return a read checkout
    with pytest.raises(ValueError):
        r_checkout = repo.checkout()


def test_initial_dataset(managed_tmpdir, randomsizedarray):
    repo = Repository(path=managed_tmpdir)
    repo.init(
        user_name='tester', user_email='foo@test.bar', remove_old=True)
    r_checkout = repo.checkout()
    # TODO: read only checkout of naked repo is None
    assert r_checkout is None
    w_checkout = repo.checkout(write=True)
    assert len(w_checkout.datasets) == 0
    with pytest.raises(KeyError):
        w_checkout.datasets['dset']
    dset = w_checkout.datasets.init_dataset('dset', prototype=randomsizedarray)
    assert dset._dset_name == 'dset'
    w_checkout.close()


def test_empty_commit(managed_tmpdir, caplog):
    repo = Repository(path=managed_tmpdir)
    repo.init(
        user_name='tester', user_email='foo@test.bar', remove_old=True)
    w_checkout = repo.checkout(write=True)
    no_commit_msg = 'No changes made to the repository. Cannot commit'
    w_checkout.commit()
    assert no_commit_msg in caplog.text  # TODO very vague and error prone
    w_checkout.close()
