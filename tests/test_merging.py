import pytest
import numpy as np

@pytest.fixture()
def repo_1_br_no_conf(repo):

    dummyData = np.arange(50)

    co1 = repo.checkout(write=True, branch_name='master')
    co1.datasets.init_dataset(name='dummy', prototype=dummyData, samples_are_named=True)
    for idx in range(10):
        dummyData[:] = idx
        co1.datasets['dummy'][str(idx)] = dummyData
    co1.metadata['hello'] = 'world'
    co1.commit('first commit adding dummy data and hello meta')
    co1.close()

    repo.create_branch('testbranch')
    co2 = repo.checkout(write=True, branch_name='testbranch')
    for idx in range(10, 20):
        dummyData[:] = idx
        co2.datasets['dummy'][str(idx)] = dummyData
    co2.metadata['foo'] = 'bar'
    co2.commit('first commit on test branch adding non-conflict data and meta')
    co2.close()
    return repo


@pytest.fixture()
def repo_2_br_no_conf(repo_1_br_no_conf):

    dummyData = np.arange(50)

    repo = repo_1_br_no_conf
    co1 = repo.checkout(write=True, branch_name='master')
    for idx in range(20, 30):
        dummyData[:] = idx
        co1.datasets['dummy'][str(idx)] = dummyData
    co1.commit('second commit on master adding non-conflict data')
    co1.close()
    return repo


def test_merge_fails_with_invalid_branch_name(repo_1_br_no_conf):
    with pytest.raises(ValueError):
        cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'failbranchname')
    # no message passed in
    with pytest.raises(TypeError):
        cmt_hash = repo_1_br_no_conf.merge('master', 'testbranch')


def test_is_ff_merge(repo_1_br_no_conf):
    testbranch_head = repo_1_br_no_conf.log(branch_name='testbranch', return_contents=True)['head']
    cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash == testbranch_head


def test_ff_merge_no_conf_correct_contents_for_name_or_hash_checkout(repo_1_br_no_conf):
    cmt_hash = repo_1_br_no_conf.merge('merge commit', 'master', 'testbranch')
    coByName = repo_1_br_no_conf.checkout(branch_name='master')
    coByHash = repo_1_br_no_conf.checkout(commit=cmt_hash)

    assert len(coByHash.datasets) == len(coByName.datasets)
    for dsetn in coByHash.datasets.keys():
        dset_byHash = coByHash.datasets[dsetn]
        dset_byName = coByName.datasets[dsetn]
        assert len(dset_byHash) == len(dset_byHash)
        for k, v in dset_byHash.items():
            assert np.allclose(v, dset_byName[k])

    assert len(coByHash.metadata) == len(coByName.metadata)
    for metaKey in coByHash.metadata.keys():
        meta_byHash = coByHash.metadata[metaKey]
        meta_byName = coByName.metadata[metaKey]
        assert meta_byHash == meta_byName


def test_ff_merge_no_conf_updates_head_commit_of_branches(repo_1_br_no_conf):
    repo = repo_1_br_no_conf
    co = repo.checkout(write=True, branch_name='master')
    co.close()
    repo.create_branch('NotUpdatedBranch')
    old_branch_head = repo.log(branch_name='NotUpdatedBranch', return_contents=True)['head']

    cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
    master_head = repo.log(branch_name='master', return_contents=True)['head']
    testbranch_head = repo.log(branch_name='testbranch', return_contents=True)['head']
    assert master_head == testbranch_head
    assert cmt_hash == master_head

    check_old_branch = repo.log(branch_name='NotUpdatedBranch', return_contents=True)['head']
    assert check_old_branch == old_branch_head
    assert check_old_branch != master_head


def test_is_3_way_merge(repo_2_br_no_conf):
    testbranch_head = repo_2_br_no_conf.log(branch_name='testbranch', return_contents=True)['head']
    masterbranch_head = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['head']
    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash != testbranch_head
    assert cmt_hash != masterbranch_head


def test_3_way_merge_no_conflict_correct_contents(repo_2_br_no_conf):
    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')
    co = repo_2_br_no_conf.checkout(branch_name='master')
    # metadata
    assert len(co.metadata) == 2
    assert co.metadata['hello'] == 'world'
    assert co.metadata['foo'] == 'bar'

    # datasets
    assert len(co.datasets) == 1
    assert 'dummy' in co.datasets
    # dataset samples
    dset = co.datasets['dummy']
    assert len(dset) == 30

    # dataset sample values
    checkarr = np.zeros_like(np.arange(50))
    for k, v in dset.items():
        checkarr[:] = int(k)
        assert np.allclose(v, checkarr)

    # dataset sample keys
    dset_keys = list(dset.keys())
    for genKey in range(30):
        assert str(genKey) in dset_keys
        dset_keys.remove(str(genKey))
    assert len(dset_keys) == 0


def test_3_way_merge_updates_head_commit_of_branches(repo_2_br_no_conf):
    orig_testbranch_head = repo_2_br_no_conf.log(branch_name='testbranch', return_contents=True)['head']
    orig_masterbranch_head = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['head']

    cmt_hash = repo_2_br_no_conf.merge('merge commit', 'master', 'testbranch')

    new_testbranch_head = repo_2_br_no_conf.log(branch_name='testbranch', return_contents=True)['head']
    new_masterbranch_head = repo_2_br_no_conf.log(branch_name='master', return_contents=True)['head']

    assert orig_testbranch_head == new_testbranch_head
    assert orig_masterbranch_head != new_masterbranch_head
    assert new_masterbranch_head == cmt_hash


class TestMetadataConflicts(object):

    def test_conflict_additions_same_names_different_vals(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.metadata['foo'] = 'this should be a conflict'
        co.commit('commit on master')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_removal_and_mutation(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.metadata['hello'] = 'this is the mutation of the hello key'
        co.commit('commit on master mutating hello')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.metadata.remove('hello')
        co.commit('this was the removal of the hello key on testbranch')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_mutate_with_different_vals(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.metadata['hello'] = 'this is the mutation of the hello key'
        co.commit('commit on master mutating hello')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.metadata['hello'] = 'a different mutation of the hello key'
        co.commit('this was a differnt of the hello key on testbranch')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_both_remove(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.metadata.remove('hello')
        co.commit('commit on master removing hellow')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.metadata.remove('hello')
        co.commit('this was the removal of the hello key on testbranch')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        assert 'hello' not in co.metadata
        assert co.metadata['foo'] == 'bar'

    def test_no_conflict_both_add_same(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.metadata['bothadd'] = 'this value'
        co.commit('commit on master adding kv')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.metadata['bothadd'] = 'this value'
        co.commit('this was the addition on testbranching adding kv')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        assert len(co.metadata) == 3
        assert co.metadata['bothadd'] == 'this value'
        assert co.metadata['hello'] == 'world'
        assert co.metadata['foo'] == 'bar'


class TestDatasetSampleConflicts(object):

    def test_conflict_additions_same_name_different_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata = newdata * 2

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.datasets['dummy']['15'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_additions_same_name_and_value(self, repo_2_br_no_conf):
        newdata = np.arange(50)
        newdata[:] = 15

        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.datasets['dummy']['15'] = newdata
        co.commit('commit on master with same value data')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        dset = co.datasets['dummy']
        assert np.allclose(dset['15'], newdata)

    def test_conflict_mutations_same_name_different_value(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        newdata = np.arange(50)
        co.datasets['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        newdata = newdata * 2
        co.datasets['dummy']['0'] = newdata
        co.commit('commit on testbranch with conflicting data')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_conflict_mutation_and_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        newdata = np.arange(50)
        co.datasets['dummy']['0'] = newdata
        co.commit('commit on master with conflicting data')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.datasets['dummy'].remove('0')
        co.commit('commit on testbranch with removal')
        co.close()

        with pytest.raises(ValueError):
            repo.merge('merge commit', 'master', 'testbranch')

    def test_no_conflict_both_removal(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        co = repo.checkout(write=True, branch_name='master')
        co.datasets['dummy'].remove('0')
        co.commit('commit on master with removal')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.datasets['dummy'].remove('0')
        co.commit('commit on testbranch with removal')
        co.close()

        cmt_hash = repo.merge('merge commit', 'master', 'testbranch')
        co = repo.checkout(commit=cmt_hash)
        dset = co.datasets['dummy']
        assert '0' not in dset
        assert len(dset) == 29
