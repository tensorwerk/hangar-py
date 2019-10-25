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


def test_writer_checkout_ff_merge(repo_1_br_no_conf):
    testbranch_head = repo_1_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    co = repo_1_br_no_conf.checkout(write=True, branch='master')
    master_head = co.commit_hash
    mergeHash = co.merge('dummy message', 'testbranch')
    assert mergeHash == testbranch_head
    assert mergeHash != master_head
    assert co.branch_name == 'master'
    co.close()

    master_order = repo_1_br_no_conf.log(branch='testbranch', return_contents=True)['order']
    tesbranch_order = repo_1_br_no_conf.log(branch='master', return_contents=True)['order']
    assert master_order == tesbranch_order


def test_merge_fails_if_changes_staged(repo_1_br_no_conf):
    co = repo_1_br_no_conf.checkout(write=True, branch='master')
    co.metadata['temporary'] = 'value'
    co.close()
    with pytest.raises(RuntimeError, match='Changes are currently pending'):
        repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')


def test_writer_checkout_merge_fails_if_changes_staged(repo_1_br_no_conf):
    co = repo_1_br_no_conf.checkout(write=True, branch='master')
    co.metadata['temporary'] = 'value'
    with pytest.raises(RuntimeError, match='Changes are currently pending'):
        co.merge('merge commit', 'testbranch')
    co.close()


def test_ff_merge_no_conf_correct_contents_for_name_or_hash_checkout(repo_1_br_no_conf):
    cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')
    coByName = repo_1_br_no_conf.checkout(branch='master')
    coByHash = repo_1_br_no_conf.checkout(commit=cmt_hash)

    assert len(coByHash.arraysets) == len(coByName.arraysets)
    for asetn in coByHash.arraysets.keys():
        aset_byHash = coByHash.arraysets[asetn]
        aset_byName = coByName.arraysets[asetn]
        assert len(aset_byHash) == len(aset_byHash)
        for k, v in aset_byHash.items():
            assert np.allclose(v, aset_byName[k])

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


def test_writer_checkout_is_3_way_merge(repo_2_br_no_conf):
    testbranch_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    masterbranch_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']
    co = repo_2_br_no_conf.checkout(write=True, branch='master')
    cmt_hash = co.merge('merge commit', 'testbranch')
    co.close()
    assert cmt_hash != testbranch_head
    assert cmt_hash != masterbranch_head


def test_3_way_merge_no_conflict_correct_contents(repo_2_br_no_conf):
    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    co = repo_2_br_no_conf.checkout(branch='master')
    # metadata
    assert len(co.metadata) == 3
    assert co.metadata['hello'] == 'world'
    assert co.metadata['foo'] == 'bar'

    # arraysets
    assert len(co.arraysets) == 1
    assert 'dummy' in co.arraysets
    # arrayset samples
    aset = co.arraysets['dummy']
    assert len(aset) == 50

    # arrayset sample values
    checkarr = np.zeros_like(np.arange(50))
    for k, v in aset.items():
        checkarr[:] = int(k)
        assert np.allclose(v, checkarr)

    # arrayset sample keys
    aset_keys = list(aset.keys())
    for genKey in range(30):
        assert str(genKey) in aset_keys
        aset_keys.remove(str(genKey))
    assert len(aset_keys) == 20
    co.close()


def test_writer_checkout_3_way_merge_no_conflict_correct_contents(repo_2_br_no_conf):
    co = repo_2_br_no_conf.checkout(write=True, branch='master')
    cmt_hash = co.merge('merge commit', 'testbranch')
    # metadata
    assert len(co.metadata) == 3
    assert co.metadata['hello'] == 'world'
    assert co.metadata['foo'] == 'bar'

    # arraysets
    assert len(co.arraysets) == 1
    assert 'dummy' in co.arraysets
    # arrayset samples
    aset = co.arraysets['dummy']
    assert len(aset) == 50

    # arrayset sample values
    checkarr = np.zeros_like(np.arange(50))
    for k, v in aset.items():
        checkarr[:] = int(k)
        assert np.allclose(v, checkarr)

    # arrayset sample keys
    aset_keys = list(aset.keys())
    for genKey in range(30):
        assert str(genKey) in aset_keys
        aset_keys.remove(str(genKey))
    assert len(aset_keys) == 20
    co.close()


def test_3_way_merge_no_conflict_and_mutation_correct_contents(repo_2_br_no_conf):
    co = repo_2_br_no_conf.checkout(write=True, branch='master')
    co.arraysets['dummy']['1'] = co.arraysets['dummy']['0']
    co.commit('mutated master')
    co.close()

    co = repo_2_br_no_conf.checkout(write=True, branch='testbranch')
    co.arraysets['dummy']['2'] = co.arraysets['dummy']['0']
    co.commit('mutated testbranch')
    co.close()

    repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    co = repo_2_br_no_conf.checkout(branch='master')
    # metadata
    assert len(co.metadata) == 3
    assert co.metadata['hello'] == 'world'
    assert co.metadata['foo'] == 'bar'

    # arraysets
    assert len(co.arraysets) == 1
    assert 'dummy' in co.arraysets
    # arrayset samples
    aset = co.arraysets['dummy']
    assert len(aset) == 50

    # arrayset sample values
    checkarr = np.zeros_like(np.arange(50))
    for k, v in aset.items():
        if k == '2':
            checkarr[:] = 0
        elif k == '1':
            checkarr[:] = 0
        else:
            checkarr[:] = int(k)
        assert np.allclose(v, checkarr)

    # arrayset sample keys
    aset_keys = list(aset.keys())
    for genKey in range(30):
        assert str(genKey) in aset_keys
        aset_keys.remove(str(genKey))
    assert len(aset_keys) == 20
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


def test_writer_checkout_3_way_merge_updates_head_commit_of_branches(repo_2_br_no_conf):
    orig_testbranch_head = repo_2_br_no_conf.log(branch='testbranch', return_contents=True)['head']
    orig_masterbranch_head = repo_2_br_no_conf.log(branch='master', return_contents=True)['head']

    co = repo_2_br_no_conf.checkout(write=True, branch='master')
    cmt_hash = co.merge('merge commit', 'testbranch')
    assert cmt_hash == co.commit_hash
    co.close()

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


class TestArraysetSampleConflicts(object):

    def test_conflict_additions_same_str_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy']['15'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_additions_same_int_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy'][15] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_additions_same_str_and_int_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy'][15] = newdata
        co.arraysets['dummy']['15'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_additions_same_name_and_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata[:] = 15

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy']['15'] = newdata
        co.arraysets['dummy'][15] = newdata
        co.commit('commit on master with same value data')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        aset = co.arraysets['dummy']
        assert np.allclose(aset['15'], newdata)
        assert np.allclose(aset[15], newdata)
        co.close()

    def test_conflict_mutations_same_name_different_value(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        newdata = np.arange(50)
        co.arraysets['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        newdata = newdata * 2
        co.arraysets['dummy']['0'] = newdata
        co.commit('commit on testbranch with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_mutation_and_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        newdata = np.arange(50)
        co.arraysets['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.arraysets['dummy'].remove('0')
        co.commit('commit on testbranch with removal')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_both_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy'].remove('0')
        del co.arraysets['dummy'][21]
        co.commit('commit on master with removal')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.arraysets['dummy'].remove('0')
        del co.arraysets['dummy'][10]
        co.commit('commit on testbranch with removal')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        aset = co.arraysets['dummy']
        assert '0' not in aset
        assert len(aset) == 47
