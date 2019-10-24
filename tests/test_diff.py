import pytest
import numpy as np


def create_meta_nt(name):
    from hangar.records.parsing import MetadataRecordKey
    res = MetadataRecordKey(name)
    return res


class TestReaderWriterDiff(object):

    @pytest.mark.parametrize('writer', [False, True])
    def test_diff_by_commit_and_branch(self, repo_2_br_no_conf, writer):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch='testbranch')
        masterco = repo.checkout(write=writer, branch='master')
        commit_diffs = masterco.diff.commit(testco.commit_hash)
        branch_diffs = masterco.diff.branch('testbranch')
        assert commit_diffs == branch_diffs
        testco.close()
        masterco.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_diff_with_wrong_commit_hash(self, repo_2_br_no_conf, writer):
        repo = repo_2_br_no_conf
        testco = repo.checkout(branch='testbranch')
        masterco = repo.checkout(write=writer, branch='master')
        wrong_commit_hash = testco.commit_hash + 'WrongHash'
        with pytest.raises(ValueError):
            masterco.diff.commit(wrong_commit_hash)
        testco.close()
        masterco.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_diff_with_wrong_branch_name(self, repo_1_br_no_conf, writer):
        repo = repo_1_br_no_conf
        masterco = repo.checkout(write=writer, branch='master')
        with pytest.raises(ValueError):
            masterco.diff.branch('wrong_branch_name')
        masterco.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_comparing_diffs_of_dev_and_master(self, repo_1_br_no_conf, writer):
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # mutating and removing data from testbranch
        testco = repo.checkout(write=True, branch='testbranch')
        testco.arraysets['dummy']['1'] = dummyData
        del testco.arraysets['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co1 = repo.checkout(write=writer, branch='master')
        diffdata1 = co1.diff.branch('testbranch')
        diffs1 = diffdata1.diff
        co1.close()

        co2 = repo.checkout(write=writer, branch='testbranch')
        diffdata2 = co2.diff.branch('master')
        diffs2 = diffdata2.diff
        co2.close()

        assert diffs1.added.samples == diffs2.added.samples
        assert diffs1.deleted.samples == diffs2.deleted.samples
        assert diffs1.mutated.samples == diffs2.mutated.samples

    @pytest.mark.parametrize('writer', [False, True])
    def test_diff_data_samples(self, repo_1_br_no_conf, writer):
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # mutating and removing data from testbranch
        testco = repo.checkout(write=True, branch='testbranch')
        testco.arraysets['dummy']['1'] = dummyData
        del testco.arraysets['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co = repo.checkout(write=writer, branch='master')
        diffdata = co.diff.branch('testbranch')
        conflicts = diffdata.conflict
        assert conflicts.conflict is False

        diffs = diffdata.diff

        # testing arraysets and metadata that has no change
        assert len(diffs.added.samples) == 20
        assert len(diffs.mutated.samples) == 1
        assert len(diffs.deleted.samples) == 1

        assert len(diffs.added.metadata) == 1
        assert len(diffs.deleted.metadata) == 0
        assert len(diffs.mutated.metadata) == 0

        assert len(diffs.added.schema) == 0
        assert len(diffs.deleted.schema) == 0
        assert len(diffs.mutated.schema) == 0

        for datarecord in diffs.added.samples:
            assert 9 < int(datarecord.data_name) < 20
        for mutated in diffs.mutated.samples:
            assert mutated.data_name == '1'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_addition_conflict(self, repo_1_br_no_conf, writer):
        # t1
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # adding data in master
        co = repo.checkout(write=True, branch='master')
        dummyData[:] = 123
        co.arraysets['dummy']['55'] = dummyData
        co.commit('Adding data in master')
        co.close()

        # adding data in testbranch
        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.arraysets['dummy']['55'] = dummyData
        co.commit('adding data in testbranch')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert conflicts.conflict is True
        assert len(conflicts.t1.samples) == 1
        for k in conflicts.t1.samples:
            assert k.data_name == '55'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_removal_conflict(self, repo_1_br_no_conf, writer):
        # t21 and t22
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='master')
        del co.arraysets['dummy']['6']
        co.arraysets['dummy']['7'] = dummyData
        co.commit('removal & mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.arraysets['dummy']['6'] = dummyData
        del co.arraysets['dummy']['7']
        co.commit('removal & mutation in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t21.samples) == 1
        assert len(conflicts.t22.samples) == 1
        for k in conflicts.t21.samples.keys():
            assert k.data_name == '6'
        for k in conflicts.t22.samples.keys():
            assert k.data_name == '7'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_mutation_conflict(self, repo_1_br_no_conf, writer):
        # t3
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.arraysets['dummy']['7'] = dummyData
        co.commit('mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.arraysets['dummy']['7'] = dummyData
        co.commit('mutation in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t3.samples) == 1
        for k in conflicts.t3.samples:
            assert k.data_name == '7'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_addition_conflict(self, written_repo, writer):
        # t1
        repo = written_repo

        repo.create_branch('testbranch')
        co = repo.checkout(write=True, branch='master')
        co.arraysets.init_arrayset(name='testing_aset', shape=(5, 7), dtype=np.float64)
        co.commit('aset init in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.arraysets.init_arrayset(name='testing_aset', shape=(7, 7), dtype=np.float64)
        co.commit('aset init in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t1.schema) == 1
        for k in conflicts.t1.schema:
            assert k == 'testing_aset'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_removal_conflict(self, written_repo, writer):
        # t21 and t22
        repo = written_repo
        co = repo.checkout(write=True, branch='master')
        co.arraysets.init_arrayset(name='testing_aset1', shape=(5, 7), dtype=np.float64)
        co.arraysets.init_arrayset(name='testing_aset2', shape=(5, 7), dtype=np.float64)
        co.commit('added asets')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True, branch='master')
        del co.arraysets['testing_aset1']
        del co.arraysets['testing_aset2']
        co.arraysets.init_arrayset(name='testing_aset2', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.arraysets['testing_aset1']
        del co.arraysets['testing_aset2']
        co.arraysets.init_arrayset(name='testing_aset1', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts.t21.schema) == 1
        assert len(conflicts.t22.schema) == 1
        assert list(conflicts.t21.schema.keys()) == ['testing_aset1']
        assert list(conflicts.t22.schema.keys()) == ['testing_aset2']
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_mutation_conflict(self, written_repo, writer):
        # t3
        repo = written_repo
        co = repo.checkout(write=True, branch='master')
        co.arraysets.init_arrayset(name='testing_aset', shape=(5, 7), dtype=np.float64)
        co.commit('added aset')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True, branch='master')
        del co.arraysets['testing_aset']
        co.arraysets.init_arrayset(name='testing_aset', shape=(7, 7), dtype=np.float64)
        co.commit('mutation from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.arraysets['testing_aset']
        co.arraysets.init_arrayset(name='testing_aset', shape=(5, 7), dtype=np.float32)
        co.commit('mutation from dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts.t3.schema) == 1
        assert list(conflicts.t3.schema.keys()) == ['testing_aset']
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_meta_addition_conflict(self, repo_1_br_no_conf, writer):
        # t1
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['metatest'] = 'value1'
        co.commit('metadata addition')
        co.close()

        co = repo.checkout(write=True, branch='master')
        co.metadata['metatest'] = 'value2'
        co.commit('metadata addition')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        for k in conflicts.t1.metadata:
            assert k == create_meta_nt('metatest')
        assert len(conflicts.t1.metadata) == 1
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_meta_removal_conflict(self, repo_1_br_no_conf, writer):
        # t21 and t22
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['hello'] = 'again'  # this is world in master
        del co.metadata['somemetadatakey']
        co.commit('removed & mutated')
        co.close()

        co = repo.checkout(write=True, branch='master')
        del co.metadata['hello']
        co.metadata['somemetadatakey'] = 'somemetadatavalue - not anymore'
        co.commit('removed & mutation')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        co.close()

        assert len(conflicts.t21.metadata) == 1
        for k in conflicts.t21.metadata:
            assert k == create_meta_nt('hello')
        assert len(conflicts.t22.metadata) == 1
        for k in conflicts.t22.metadata:
            assert k == create_meta_nt('somemetadatakey')

    @pytest.mark.parametrize('writer', [False, True])
    def test_meta_mutation_conflict(self, repo_1_br_no_conf, writer):
        # t3
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='testbranch')
        co.metadata['hello'] = 'again'  # this is world in master
        co.commit('mutated')
        co.close()

        co = repo.checkout(write=True, branch='master')
        co.metadata['hello'] = 'again and again'
        co.commit('mutation')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts.t3.metadata) == 1
        for k in conflicts.t3.metadata:
            assert k == create_meta_nt('hello')
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_commits_inside_cm(self, written_repo, array5by7, writer):
        repo = written_repo
        repo.create_branch('testbranch')
        co = repo.checkout(write=True, branch='testbranch')
        aset = co.arraysets['writtenaset']
        aset2 = co.arraysets.init_arrayset('aset2', prototype=array5by7)
        aset2[1] = array5by7
        with aset, co.metadata:
            aset[100] = array5by7
            co.metadata['crazykey'] = 'crazyvalue'
            co.commit('inside cm')
            aset[101] = array5by7
            co.commit('another commit inside cm')
        co.close()
        co = repo.checkout(write=writer, branch='testbranch')
        assert np.allclose(co.arraysets['writtenaset'][101], array5by7)
        diff = co.diff.branch('master').diff
        assert create_meta_nt('crazykey') in diff.added.metadata.keys()
        assert 'aset2' in diff.added.schema.keys()
        calledWithAset = False
        for record in diff.added.samples:
            if record.aset_name == 'writtenaset':
                calledWithAset = True
                assert record.data_name in [100, 101]
        assert calledWithAset is True
        co.close()


class TestWriterDiff(object):

    def test_status_and_staged_meta(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        co.metadata['hello_from_test'] = 'hai to test'
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged().diff
        assert create_meta_nt('hello_from_test') in diff.added.metadata
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
        co.arraysets['writtenaset']['45'] = dummyData
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()
        calledWithAset = False
        for record in diff.diff.added.samples:
            if record.aset_name == 'writtenaset':
                calledWithAset = True
                assert record.data_name in '45'
        assert calledWithAset is True
        co.commit('adding')
        assert co.diff.status() == 'CLEAN'
        co.close()

    def test_status_and_staged_aset(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset(name='sampleaset', shape=(3, 5), dtype=np.float32)
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()
        assert 'sampleaset' in diff.diff.added.schema
        co.commit('init aset')
        assert co.diff.status() == 'CLEAN'
        co.close()