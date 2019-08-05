import pytest
import numpy as np


def test_merge_fails_with_invalid_branch_name(repo_1_br_no_conf):
    with pytest.raises(ValueError):
        cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'failbranchname')
    # no message passed in
    with pytest.raises(TypeError):
        cmt_hash = repo_1_br_no_conf.merge('master', 'testbranch')


def test_is_ff_merge(repo_1_br_no_conf):
    testbranch_head = repo_1_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash == testbranch_head


def test_ff_merge_no_conf_correct_contents_for_name_or_hash_checkout(repo_1_br_no_conf):
    cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')
    coByName = repo_1_br_no_conf.checkout(branch='master')
    coByHash = repo_1_br_no_conf.checkout(commit=cmt_hash)

    assert len(coByHash.datacells) == len(coByName.datacells)
    for dsetn in coByHash.datacells.keys():
        dset_byHash = coByHash.datacells[dsetn]
        dset_byName = coByName.datacells[dsetn]
        assert len(dset_byHash) == len(dset_byHash)
        for k, v in dset_byHash.items():
            assert np.allclose(v, dset_byName[k])

    assert len(coByHash.metadata) == len(coByName.metadata)
    for metaKey in coByHash.metadata.keys():
        meta_byHash = coByHash.metadata[metaKey]
        meta_byName = coByName.metadata[metaKey]
        assert meta_byHash == meta_byName
    coByHash.close()
    coByName.close()


def test_ff_merge_no_conf_updates_head_commit_of_branches(repo_1_br_no_conf):
    repo = repo_1_br_no_conf
    co = repo.checkout(write=True, branch='master')
    co.close()
    repo.create_branch('NotUpdatedBranch')
    old_branch_head = repo.log(branch='NotUpdatedBranch', return_contents=True)['head']

    cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
    master_head = repo.log(branch='master', return_contents=True)['head']
    testbranch_head = repo.log(branch='testbranch', return_contents=True)['head']
    assert master_head == testbranch_head
    assert cmt_hash == master_head

    check_old_branch = repo.log(branch='NotUpdatedBranch', return_contents=True)['head']
    assert check_old_branch == old_branch_head
    assert check_old_branch != master_head


def test_is_3_way_merge(repo_2_br_no_conf):
    testbranch_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    masterbranch_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']
    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash != testbranch_head
    assert cmt_hash != masterbranch_head


def test_3_way_merge_no_conflict_correct_contents(repo_2_br_no_conf):
    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    co = repo_2_br_no_conf.checkout(branch='master')
    # metadata
    assert len(co.metadata) == 3
    assert co.metadata['hello'] == 'world'
    assert co.metadata['foo'] == 'bar'

    # datacells
    assert len(co.datacells) == 1
    assert 'dummy' in co.datacells
    # datacell samples
    dset = co.datacells['dummy']
    assert len(dset) == 50

    # datacell sample values
    checkarr = np.zeros_like(np.arange(50))
    for k, v in dset.items():
        checkarr[:] = int(k)
        assert np.allclose(v, checkarr)

    # datacell sample keys
    dset_keys = list(dset.keys())
    for genKey in range(30):
        assert str(genKey) in dset_keys
        dset_keys.remove(str(genKey))
    assert len(dset_keys) == 20
    co.close()


def test_3_way_merge_updates_head_commit_of_branches(repo_2_br_no_conf):
    orig_testbranch_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    orig_masterbranch_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']

    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')

    new_testbranch_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    new_masterbranch_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']

    assert orig_testbranch_head == new_testbranch_head
    assert orig_masterbranch_head != new_masterbranch_head
    assert new_masterbranch_head == cmt_hash


class TestMetadataConflicts(object):

    def test_conflict_additions_same_names_different_vals(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.metadata['foo'] = 'this should be a conflict'
        co.commit('commit on master')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_removal_and_mutation(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.metadata['hello'] = 'this is the mutation of the hello key'
        co.commit('commit on master mutating hello')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.metadata.remove('hello')
        co.commit('this was the removal of the hello key on testbranch')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_mutate_with_different_vals(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.metadata['hello'] = 'this is the mutation of the hello key'
        co.commit('commit on master mutating hello')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['hello'] = 'a different mutation of the hello key'
        co.commit('this was a differnt of the hello key on testbranch')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_both_remove(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.metadata.remove('hello')
        co.commit('commit on master removing hellow')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.metadata.remove('hello')
        co.commit('this was the removal of the hello key on testbranch')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        assert 'hello' not in co.metadata
        assert co.metadata['foo'] == 'bar'

    def test_no_conflict_both_add_same(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.metadata['bothadd'] = 'this value'
        co.commit('commit on master adding kv')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['bothadd'] = 'this value'
        co.commit('this was the addition on testbranching adding kv')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        assert len(co.metadata) == 4
        assert co.metadata['bothadd'] == 'this value'
        assert co.metadata['hello'] == 'world'
        assert co.metadata['foo'] == 'bar'
        co.close()


class TestDatacellSampleConflicts(object):

    def test_conflict_additions_same_str_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.datacells['dummy']['15'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_additions_same_int_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.datacells['dummy'][15] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_additions_same_str_and_int_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.datacells['dummy'][15] = newdata
        co.datacells['dummy']['15'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_additions_same_name_and_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata[:] = 15

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.datacells['dummy']['15'] = newdata
        co.datacells['dummy'][15] = newdata
        co.commit('commit on master with same value data')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        dset = co.datacells['dummy']
        assert np.allclose(dset['15'], newdata)
        assert np.allclose(dset[15], newdata)
        co.close()

    def test_conflict_mutations_same_name_different_value(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        newdata = np.arange(50)
        co.datacells['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        newdata = newdata * 2
        co.datacells['dummy']['0'] = newdata
        co.commit('commit on testbranch with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_mutation_and_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        newdata = np.arange(50)
        co.datacells['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.datacells['dummy'].remove('0')
        co.commit('commit on testbranch with removal')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_both_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.datacells['dummy'].remove('0')
        del co.datacells['dummy'][21]
        co.commit('commit on master with removal')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.datacells['dummy'].remove('0')
        del co.datacells['dummy'][10]
        co.commit('commit on testbranch with removal')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        dset = co.datacells['dummy']
        assert '0' not in dset
        assert len(dset) == 47
