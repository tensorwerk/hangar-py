import pytest
import numpy as np


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

    def test_comparing_diffs_of_dev_and_master(self, repo_1_br_no_conf):
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
        diffs1 = diffdata[0]

        co = repo.checkout(branch_name='testbranch')
        diffdata = co.diff.branch('master')
        diffs2 = diffdata[0]
        assert diffs1['samples']['dev']['dummy']['additions'] == diffs2['samples']['master']['dummy']['additions']
        assert diffs1['samples']['dev']['dummy']['mutations'] == diffs2['samples']['master']['dummy']['mutations']
        assert diffs1['samples']['dev']['dummy']['removals'] == diffs2['samples']['master']['dummy']['removals']
        assert diffs1['samples']['dev']['dummy']['unchanged'] == diffs2['samples']['master']['dummy']['unchanged']

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

        # testing datasets and metadata that has no change
        assert diffs['datasets']['dev']['additions'] == {}
        assert diffs['datasets']['dev']['mutations'] == {}
        assert diffs['datasets']['dev']['removals'] == {}
        assert 'dummy' in diffs['datasets']['master']['unchanged'].keys()
        assert 'foo' in diffs['metadata']['dev']['additions'].keys()
        assert len(diffs['metadata']['master']['additions'].keys()) == 0
        assert 'hello' in diffs['metadata']['master']['unchanged'].keys()
        assert 'hello' in diffs['metadata']['dev']['unchanged'].keys()
        assert diffs['metadata']['dev']['mutations'] == {}
        assert diffs['metadata']['dev']['removals'] == {}

        # testing datarecords for addition, unchanged mutated, removed
        for datarecord in diffs['samples']['dev']['dummy']['additions']:
            assert 9 < int(datarecord.data_name) < 20
        for datarecord in diffs['samples']['dev']['dummy']['unchanged']:
            assert 0 <= int(datarecord.data_name) < 10
        for removed in diffs['samples']['dev']['dummy']['removals']:
            removed.data_name == 2
        for mutated in diffs['samples']['dev']['dummy']['mutations']:
            mutated.data_name == 1

    def test_basic_sample_conflict(self, repo_2_br_with_conf):
        repo = repo_2_br_with_conf
        co = repo.checkout()
        diffdata = co.diff.branch('testbranch')
        conflict_data = diffdata[1]
        assert conflict_data['conflict_found'] is True
        assert conflict_data['sample']['dummy'].t1[0].data_name == '15'
        assert conflict_data['sample']['dummy'].t1[0].dset_name == 'dummy'

    def test_sample_addition_conflict(self, repo_1_br_no_conf):
        # t1
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # adding data in master
        co = repo.checkout(write=True)
        dummyData[:] = 123
        co.datasets['dummy']['55'] = dummyData
        co.commit('Adding data in master')
        co.close()

        # adding data in testbranch
        co = repo.checkout(write=True, branch_name='testbranch')
        dummyData[:] = 234
        co.datasets['dummy']['55'] = dummyData
        co.commit('adding data in testbranch')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert conflicts['conflict_found'] is True
        assert len(conflicts['sample']['dummy'].t1) == 1
        assert conflicts['sample']['dummy'].t1[0].data_name == '55'

    def test_sample_removal_conflict(self, repo_1_br_no_conf):
        # t21 and t22
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True)
        del co.datasets['dummy']['6']
        co.datasets['dummy']['7'] = dummyData
        co.commit('removal & mutation in master')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        co.datasets['dummy']['6'] = dummyData
        del co.datasets['dummy']['7']
        co.commit('removal & mutation in dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['sample']['dummy'].t21) == 1
        assert len(conflicts['sample']['dummy'].t22) == 1
        assert conflicts['sample']['dummy'].t21[0].data_name == '6'
        assert conflicts['sample']['dummy'].t22[0].data_name == '7'

    def test_sample_mutation_conflict(self, repo_1_br_no_conf):
        # t3
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True)
        co.datasets['dummy']['7'] = dummyData
        co.commit('mutation in master')
        co.close()

        co = repo.checkout(write=True, branch_name='testbranch')
        dummyData[:] = 234
        co.datasets['dummy']['7'] = dummyData
        co.commit('mutation in dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['sample']['dummy'].t3) == 1
        assert conflicts['sample']['dummy'].t3[0].data_name == '7'
