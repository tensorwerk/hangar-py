import pytest
import numpy as np


def create_meta_nt(name):
    from hangar.records.parsing import MetadataRecordKey
    res = MetadataRecordKey(name)
    return res

class TestReaderDiff(object):

    def test_diff_by_commit_and_branch(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch='testbranch')
        masterco = repo.checkout(branch='master')
        commit_diffs = masterco.diff.commit(testco.commit_hash)
        branch_diffs = masterco.diff.branch('testbranch')
        assert commit_diffs == branch_diffs
        testco.close()
        masterco.close()

    def test_diff_with_wrong_commit_hash(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch='testbranch')
        masterco = repo.checkout(branch='master')
        wrong_commit_hash = testco.commit_hash + 'WrongHash'
        with pytest.raises(ValueError):
            masterco.diff.commit(wrong_commit_hash)
        testco.close()
        masterco.close()

    def test_diff_with_wrong_branch_name(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        masterco = repo.checkout(branch='master')
        with pytest.raises(ValueError):
            masterco.diff.branch('wrong_branch_name')
        masterco.close()

    def test_comparing_diffs_of_dev_and_master(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # mutating and removing data from testbranch
        testco = repo.checkout(write=True, branch='testbranch')
        testco.datacells['dummy']['1'] = dummyData
        del testco.datacells['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co = repo.checkout(branch='master')
        diffdata = co.diff.branch('testbranch')
        diffs1 = diffdata[0]

        co = repo.checkout(branch='testbranch')
        diffdata = co.diff.branch('master')
        diffs2 = diffdata[0]
        assert diffs1['samples']['dev']['dummy']['additions'] == diffs2['samples']['master']['dummy']['additions']
        assert diffs1['samples']['dev']['dummy']['mutations'] == diffs2['samples']['master']['dummy']['mutations']
        assert diffs1['samples']['dev']['dummy']['removals'] == diffs2['samples']['master']['dummy']['removals']
        assert diffs1['samples']['dev']['dummy']['unchanged'] == diffs2['samples']['master']['dummy']['unchanged']
        co.close()

    def test_diff_data_samples(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # mutating and removing data from testbranch
        testco = repo.checkout(write=True, branch='testbranch')
        testco.datacells['dummy']['1'] = dummyData
        del testco.datacells['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co = repo.checkout(branch='master')
        diffdata = co.diff.branch('testbranch')
        conflict_dict = diffdata[1]
        assert conflict_dict['conflict_found'] is False

        diffs = diffdata[0]

        # testing datacells and metadata that has no change
        assert diffs['datacells']['dev']['additions'] == {}
        assert diffs['datacells']['dev']['mutations'] == {}
        assert diffs['datacells']['dev']['removals'] == {}
        assert 'dummy' in diffs['datacells']['master']['unchanged'].keys()
        assert create_meta_nt('foo' ) in diffs['metadata']['dev']['additions'].keys()
        assert len(diffs['metadata']['master']['additions'].keys()) == 0
        assert create_meta_nt('hello') in diffs['metadata']['master']['unchanged'].keys()
        assert create_meta_nt('hello') in diffs['metadata']['dev']['unchanged'].keys()
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
        co.close()

    def test_sample_addition_conflict(self, repo_1_br_no_conf):
        # t1
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # adding data in master
        co = repo.checkout(write=True)
        dummyData[:] = 123
        co.datacells['dummy']['55'] = dummyData
        co.commit('Adding data in master')
        co.close()

        # adding data in testbranch
        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.datacells['dummy']['55'] = dummyData
        co.commit('adding data in testbranch')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert conflicts['conflict_found'] is True
        assert len(conflicts['sample']['dummy'].t1) == 1
        assert conflicts['sample']['dummy'].t1[0].data_name == '55'
        co.close()

    def test_sample_removal_conflict(self, repo_1_br_no_conf):
        # t21 and t22
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True)
        del co.datacells['dummy']['6']
        co.datacells['dummy']['7'] = dummyData
        co.commit('removal & mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.datacells['dummy']['6'] = dummyData
        del co.datacells['dummy']['7']
        co.commit('removal & mutation in dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['sample']['dummy'].t21) == 1
        assert len(conflicts['sample']['dummy'].t22) == 1
        assert conflicts['sample']['dummy'].t21[0].data_name == '6'
        assert conflicts['sample']['dummy'].t22[0].data_name == '7'
        co.close()

    def test_sample_mutation_conflict(self, repo_1_br_no_conf):
        # t3
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True)
        co.datacells['dummy']['7'] = dummyData
        co.commit('mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.datacells['dummy']['7'] = dummyData
        co.commit('mutation in dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['sample']['dummy'].t3) == 1
        assert conflicts['sample']['dummy'].t3[0].data_name == '7'
        co.close()

    def test_dset_addition_conflict(self, written_repo):
        # t1
        repo = written_repo

        repo.create_branch('testbranch')
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='testing_dset', shape=(5, 7), dtype=np.float64)
        co.commit('dset init in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.datacells.init_datacell(name='testing_dset', shape=(7, 7), dtype=np.float64)
        co.commit('dset init in dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['dset'].t1) == 1
        assert conflicts['dset'].t1[0] == 'testing_dset'
        co.close()

    def test_dset_removal_conflict(self, written_repo):
        # t21 and t22
        repo = written_repo
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='testing_dset1', shape=(5, 7), dtype=np.float64)
        co.datacells.init_datacell(name='testing_dset2', shape=(5, 7), dtype=np.float64)
        co.commit('added dsets')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True)
        del co.datacells['testing_dset1']
        del co.datacells['testing_dset2']
        co.datacells.init_datacell(name='testing_dset2', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.datacells['testing_dset1']
        del co.datacells['testing_dset2']
        co.datacells.init_datacell(name='testing_dset1', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['dset'].t21) == 1
        assert len(conflicts['dset'].t22) == 1
        assert conflicts['dset'].t21[0] == 'testing_dset1'
        assert conflicts['dset'].t22[0] == 'testing_dset2'
        co.close()

    def test_dset_mutation_conflict(self, written_repo):
        # t3
        repo = written_repo
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='testing_dset', shape=(5, 7), dtype=np.float64)
        co.commit('added dset')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True)
        del co.datacells['testing_dset']
        co.datacells.init_datacell(name='testing_dset', shape=(7, 7), dtype=np.float64)
        co.commit('mutation from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.datacells['testing_dset']
        co.datacells.init_datacell(name='testing_dset', shape=(5, 7), dtype=np.float32)
        co.commit('mutation from dev')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts['dset'].t3) == 1
        assert conflicts['dset'].t3[0] == 'testing_dset'
        co.close()

    def test_meta_addition_conflict(self, repo_1_br_no_conf):
        # t1
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['metatest'] = 'value1'
        co.commit('metadata addition')
        co.close()

        co = repo.checkout(write=True)
        co.metadata['metatest'] = 'value2'
        co.commit('metadata addition')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert conflicts['meta'].t1[0] == create_meta_nt('metatest')
        assert len(conflicts['meta'].t1) == 1
        co.close()

    def test_meta_removal_conflict(self, repo_1_br_no_conf):
        # t21 and t22
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['hello'] = 'again'  # this is world in master
        del co.metadata['somemetadatakey']
        co.commit('removed & mutated')
        co.close()

        co = repo.checkout(write=True)
        del co.metadata['hello']
        co.metadata['somemetadatakey'] = 'somemetadatavalue - not anymore'
        co.commit('removed & mutation')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert conflicts['meta'].t21[0] == create_meta_nt('hello')
        assert len(conflicts['meta'].t21) == 1
        assert conflicts['meta'].t22[0] == create_meta_nt('somemetadatakey')
        assert len(conflicts['meta'].t22) == 1
        co.close()

    def test_meta_mutation_conflict(self, repo_1_br_no_conf):
        # t3
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['hello'] = 'again'  # this is world in master
        co.commit('mutated')
        co.close()

        co = repo.checkout(write=True)
        co.metadata['hello'] = 'again and again'
        co.commit('mutation')
        co.close()

        co = repo.checkout()
        conflicts = co.diff.branch('testbranch')[1]
        assert conflicts['meta'].t3[0] == create_meta_nt('hello')
        assert len(conflicts['meta'].t3) == 1
        co.close()

    def test_commits_inside_cm(self, written_repo, array5by7):
        repo = written_repo
        repo.create_branch('testbranch')
        co = repo.checkout(write=True, branch='testbranch')
        dset = co.datacells['_dset']
        dset2 = co.datacells.init_datacell('dset2', prototype=array5by7)
        dset2[1] = array5by7
        with dset, co.metadata:
            dset[100] = array5by7
            co.metadata['crazykey'] = 'crazyvalue'
            co.commit('inside cm')
            dset[101] = array5by7
            co.commit('another commit inside cm')
        co.close()
        co = repo.checkout(branch='testbranch')
        assert np.allclose(co.datacells['_dset'][101], array5by7)
        diff = co.diff.branch('master')[0]
        assert create_meta_nt('crazykey') in diff['metadata']['master']['additions'].keys()
        assert 'dset2' in diff['datacells']['master']['additions'].keys()
        for record in diff['samples']['master']['_dset']['additions']:
            assert record.data_name in [100, 101]
        co.close()


class TestWriterDiff(object):

    def test_status_and_staged_meta(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        co.metadata['hello_from_test'] = 'hai to test'
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()[0]
        assert create_meta_nt('hello_from_test') in diff['metadata']['master']['additions']
        co.commit('init metadata')
        assert co.diff.status() == 'CLEAN'
        co.close()

    def test_status_and_staged_samples(self, written_repo):
        dummyData = np.zeros((5, 7))
        repo = written_repo
        co = repo.checkout()
        with pytest.raises(AttributeError):
            co.diff.status()  # Read checkout doesn't have status()

        co = repo.checkout(write=True)
        co.datacells['_dset']['45'] = dummyData
        assert co.diff.status() == 'DIRTY'
        diffs = co.diff.staged()[0]
        for key in diffs['samples']['master']['_dset']['additions'].keys():
            assert key.data_name == '45'
        co.commit('adding')
        assert co.diff.status() == 'CLEAN'
        co.close()

    def test_status_and_staged_dset(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        co.datacells.init_datacell(name='sampledset', shape=(3, 5), dtype=np.float32)
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()[0]
        assert 'sampledset' in diff['datacells']['master']['additions'].keys()
        assert '_dset' in diff['datacells']['master']['unchanged'].keys()
        co.commit('init dset')
        assert co.diff.status() == 'CLEAN'
        co.close()
