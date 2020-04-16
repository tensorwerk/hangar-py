import pytest


@pytest.mark.parametrize('name', [
    'dummy branch', 'origin/master', '\nmaster', '\\master', 'master\n'
    'master\r\n', 'master ', 1412, 'foo !', 'foo@', 'foo#', 'foo$', '(foo)',
    'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
def test_create_branch_fails_invalid_name(aset_samples_initialized_repo, name):
    repo = aset_samples_initialized_repo
    with pytest.raises(ValueError):
        repo.create_branch(name)


def test_list_branches_only_reports_master_upon_initialization(repo):
    branches = repo.list_branches()
    assert branches == ['master']


def test_cannot_create_new_branch_from_initialized_repo_with_no_commits(repo):
    with pytest.raises(RuntimeError):
        repo.create_branch('testbranch')


def test_can_create_new_branch_from_repo_with_one_commit(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    expected_digest = co.commit('first')
    co.close()

    branchRes = repo.create_branch('testbranch')

    assert branchRes.name == 'testbranch'
    assert branchRes.digest == expected_digest


def test_cannot_duplicate_branch_name(aset_samples_initialized_repo):
    aset_samples_initialized_repo.create_branch('testbranch')
    with pytest.raises(ValueError):
        aset_samples_initialized_repo.create_branch('testbranch')


def test_create_multiple_branches_different_name_same_commit(aset_samples_initialized_repo):
    b1 = aset_samples_initialized_repo.create_branch('testbranch1')
    b2 = aset_samples_initialized_repo.create_branch('testbranch2')
    b3 = aset_samples_initialized_repo.create_branch('testbranch3')

    assert b1.digest == b2.digest
    assert b2.digest == b3.digest
    assert b3.digest == b1.digest
    assert aset_samples_initialized_repo.list_branches() == ['master', 'testbranch1', 'testbranch2', 'testbranch3']


def test_create_branch_by_specifying_base_commit(repo):

    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first commit')
    first_digest = co.commit_hash
    co['test_meta']['foo'] = 'bar'
    second_digest = co.commit('second')
    co['test_meta']['hello'] = 'world'
    third_digest = co.commit('third')
    co['test_meta']['zen'] = 'python'
    fourth_digest = co.commit('fourth')
    co.close()

    assert repo.list_branches() == ['master']

    secBranch = repo.create_branch('dev-second', base_commit=second_digest)
    assert secBranch.name == 'dev-second'
    assert secBranch.digest == second_digest

    co = repo.checkout(branch='dev-second')
    assert len(co['test_meta']) == 1
    assert co['test_meta']['foo'] == 'bar'
    co.close()


def test_remove_branch_works_when_commits_align(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()
    repo.create_branch('testdelete')

    assert repo.list_branches() == ['master', 'testdelete']

    removedBranch = repo.remove_branch('testdelete')
    assert removedBranch.name == 'testdelete'
    assert removedBranch.digest == masterHEAD
    assert repo.list_branches() == ['master']


def test_delete_branch_raises_runtime_error_when_history_not_merged(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    repo.create_branch('testdelete')
    co = repo.checkout(write=True, branch='testdelete')
    co['test_meta']['hello'] = 'world'
    thirdDigest = co.commit('third')
    co.close()

    # checkout master so staging area is not on branch
    co = repo.checkout(write=True, branch='master')
    co.close()

    assert repo.list_branches() == ['master', 'testdelete']
    with pytest.raises(RuntimeError):
        repo.remove_branch('testdelete')


def test_delete_branch_completes_when_history_not_merged_but_force_option_set(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    repo.create_branch('testdelete')
    co = repo.checkout(write=True, branch='testdelete')
    co['test_meta']['hello'] = 'world'
    thirdDigest = co.commit('third')
    co.close()

    # checkout master so staging area is not on branch
    co = repo.checkout(write=True, branch='master')
    co.close()
    assert repo.list_branches() == ['master', 'testdelete']

    removedBranch = repo.remove_branch('testdelete', force_delete=True)
    assert removedBranch.name == 'testdelete'
    assert removedBranch.digest == thirdDigest
    assert repo.list_branches() == ['master']


def test_delete_branch_raises_value_error_if_invalid_branch_name(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    repo.create_branch('testdelete')
    co = repo.checkout(write=True, branch='testdelete')
    co['test_meta']['hello'] = 'world'
    thirdDigest = co.commit('third')
    co.close()

    assert repo.list_branches() == ['master', 'testdelete']
    with pytest.raises(ValueError):
        repo.remove_branch('doesnotexist')
    with pytest.raises(ValueError):
        repo.remove_branch('origin/master')


def test_delete_branch_raises_permission_error_if_writer_lock_held(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    repo.create_branch('testdelete')
    co = repo.checkout(write=True, branch='testdelete')
    co['test_meta']['hello'] = 'world'
    thirdDigest = co.commit('third')
    co.close()

    # checkout master so staging area is not on branch
    co = repo.checkout(write=True, branch='master')
    with pytest.raises(PermissionError):
        repo.remove_branch('testdelete')
    assert repo.list_branches() == ['master', 'testdelete']
    co.close()


def test_delete_branch_raises_permission_error_if_branch_requested_is_staging_head(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co.commit('first')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    repo.create_branch('testdelete')
    co = repo.checkout(write=True, branch='testdelete')
    co['test_meta']['hello'] = 'world'
    thirdDigest = co.commit('third')
    co.close()

    with pytest.raises(PermissionError):
        repo.remove_branch('testdelete')
    assert repo.list_branches() == ['master', 'testdelete']


def test_delete_branch_raises_permission_error_if_only_one_branch_left(repo):
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    co['test_meta']['foo'] = 'bar'
    masterHEAD = co.commit('second')
    co.close()

    assert repo.list_branches() == ['master']
    with pytest.raises(PermissionError):
        repo.remove_branch('master')
    assert repo.list_branches() == ['master']
