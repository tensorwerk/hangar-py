import os
import pytest
from hangar import Repository


def test_imports():
    import hangar
    from hangar import Repository


def test_starting_up(managed_tmpdir):
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    assert repo.list_branch_names() == ['master']
    assert os.path.isdir(repo._repo_path)
    assert repo._repo_path == os.path.join(managed_tmpdir, '__hangar')
    # TODO: Fix failing test
    # assert repo.status() == 'CLEAN'


def initial_read_checkout(managed_tmpdir):
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    # TODO it should do something to indicate the issue or return a read checkout
    with pytest.raises(ValueError):
        r_checkout = repo.checkout()


def test_initial_dataset(managed_tmpdir, randomsizedarray):
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='tester', user_email='foo@test.bar', remove_old=True)

    with pytest.raises(ValueError):
        # Read only checkout of repo without commits raises as expected
        r_checkout = repo.checkout()

    w_checkout = repo.checkout(write=True)
    assert len(w_checkout.datasets) == 0
    with pytest.raises(KeyError):
        w_checkout.datasets['dset']
    dset = w_checkout.datasets.init_dataset('dset', prototype=randomsizedarray)
    assert dset._dsetn == 'dset'
    w_checkout.close()


def test_empty_commit(managed_tmpdir, caplog):
    repo = Repository(path=managed_tmpdir)
    repo.init(user_name='tester', user_email='foo@test.bar', remove_old=True)
    w_checkout = repo.checkout(write=True)
    with pytest.raises(RuntimeError):
        w_checkout.commit('this is a merge message')
    w_checkout.close()
