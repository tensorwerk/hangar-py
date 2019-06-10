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


@pytest.fixture()
def repo_2_br_with_conf(repo_2_br_no_conf):
    dummyData = np.arange(50)
    repo = repo_2_br_no_conf
    dummyData[:] = 1234
    co = repo.checkout(write=True, branch_name='master')
    co.datasets['dummy']['15'] = dummyData
    co.commit('final commit on master, making a conflict')
    co.close()
    return repo


class TestReaderDiff:

    def test_diff_by_commit_and_branch(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch_name='testbranch')
        masterco = repo.checkout(branch_name='master')
        commit_diffs = masterco.diff.commit(testco.commit_hash)
        branch_diffs = masterco.diff.branch('testbranch')
        assert commit_diffs == branch_diffs

    def test_diff_with_wrong_commit_hash(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch_name='testbranch')
        masterco = repo.checkout(branch_name='master')
        wrong_commit_hash = testco.commit_hash + 'WrongHash'
        with pytest.raises(ValueError):
            masterco.diff.commit(wrong_commit_hash)

    def test_diff_with_wrong_branch_name(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        masterco = repo.checkout(branch_name='master')
        with pytest.raises(ValueError):
            masterco.diff.branch('wrong_branch_name')

    def test_diff_data_samples(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # mutating and removing data from testbranch
        testco = repo.checkout(write=True, branch_name='testbranch')
        testco.datasets['dummy']['1'] = dummyData
        del testco.datasets['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co = repo.checkout(branch_name='master')
        diffdata = co.diff.branch('testbranch')
        conflict_dict = diffdata[1]
        assert conflict_dict['conflict_found'] is False
        diffs = diffdata[0]
        from pprint import pprint
        pprint(diffs)
        breakpoint()

        # testing datasets and metadata that has no change
        assert diffs['datasets']['master']['additions'] == {}
        assert diffs['datasets']['master']['mutations'] == {}
        assert diffs['datasets']['master']['removals'] == {}
        assert 'dummy' in diffs['datasets']['master']['unchanged'].keys()
        assert 'foo' in diffs['metadata']['master']['additions'].keys()
        assert 'hello' in diffs['metadata']['master']['unchanged'].keys()
        assert diffs['metadata']['master']['mutations'] == {}
        assert diffs['metadata']['master']['removals'] == {}

        # testing datarecords for addition, unchanged mutated, removed
        for datarecord in diffs['samples']['master']['dummy']['additions']:
            assert 9 < int(datarecord.data_name) < 20
        for datarecord in diffs['samples']['master']['dummy']['unchanged']:
            assert 0 <= int(datarecord.data_name) < 10
        for removed in diffs['samples']['master']['dummy']['removals']:
            removed.data_name == 2
        for mutated in diffs['samples']['master']['dummy']['mutations']:
            mutated.data_name == 1
