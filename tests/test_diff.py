import pytest
import numpy as np


def create_meta_nt(name):
    from hangar.records import MetadataRecordKey
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
        testco.columns['dummy']['1'] = dummyData
        del testco.columns['dummy']['2']
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
        testco.columns['dummy']['1'] = dummyData
        del testco.columns['dummy']['2']
        testco.commit("mutation and removal")
        testco.close()

        co = repo.checkout(write=writer, branch='master')
        diffdata = co.diff.branch('testbranch')
        conflicts = diffdata.conflict
        assert conflicts.conflict is False

        diffs = diffdata.diff

        # testing columns and metadata that has no change
        assert len(diffs.added.samples) == 20
        assert len(diffs.mutated.samples) == 1
        assert len(diffs.deleted.samples) == 1

        assert len(diffs.added.schema) == 0
        assert len(diffs.deleted.schema) == 0
        assert len(diffs.mutated.schema) == 0

        for datarecord in diffs.added.samples:
            assert 9 < int(datarecord.sample) < 20
        for mutated in diffs.mutated.samples:
            assert mutated.sample == '1'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_addition_conflict(self, repo_1_br_no_conf, writer):
        # t1
        repo = repo_1_br_no_conf
        dummyData = np.arange(50)

        # adding data in master
        co = repo.checkout(write=True, branch='master')
        dummyData[:] = 123
        co.columns['dummy']['55'] = dummyData
        co.commit('Adding data in master')
        co.close()

        # adding data in testbranch
        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.columns['dummy']['55'] = dummyData
        co.commit('adding data in testbranch')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert conflicts.conflict is True
        assert len(conflicts.t1.samples) == 1
        for k in conflicts.t1.samples:
            assert k.sample == '55'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_removal_conflict(self, repo_1_br_no_conf, writer):
        # t21 and t22
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='master')
        del co.columns['dummy']['6']
        co.columns['dummy']['7'] = dummyData
        co.commit('removal & mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.columns['dummy']['6'] = dummyData
        del co.columns['dummy']['7']
        co.commit('removal & mutation in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t21.samples) == 1
        assert len(conflicts.t22.samples) == 1
        for k in conflicts.t21.samples:
            assert k.sample == '6'
        for k in conflicts.t22.samples:
            assert k.sample == '7'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_sample_mutation_conflict(self, repo_1_br_no_conf, writer):
        # t3
        dummyData = np.arange(50)
        dummyData[:] = 123
        repo = repo_1_br_no_conf
        co = repo.checkout(write=True, branch='master')
        co.columns['dummy']['7'] = dummyData
        co.commit('mutation in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        dummyData[:] = 234
        co.columns['dummy']['7'] = dummyData
        co.commit('mutation in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t3.samples) == 1
        for k in conflicts.t3.samples:
            assert k.sample == '7'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_addition_conflict(self, aset_samples_initialized_repo, writer):
        # t1
        repo = aset_samples_initialized_repo

        repo.create_branch('testbranch')
        co = repo.checkout(write=True, branch='master')
        co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float64)
        co.commit('aset init in master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        co.add_ndarray_column(name='testing_aset', shape=(7, 7), dtype=np.float64)
        co.commit('aset init in dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch').conflict
        assert len(conflicts.t1.schema) == 1
        for k in conflicts.t1.schema:
            assert k.column == 'testing_aset'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_removal_conflict(self, aset_samples_initialized_repo, writer):
        # t21 and t22
        repo = aset_samples_initialized_repo
        co = repo.checkout(write=True, branch='master')
        co.add_ndarray_column(name='testing_aset1', shape=(5, 7), dtype=np.float64)
        co.add_ndarray_column(name='testing_aset2', shape=(5, 7), dtype=np.float64)
        co.commit('added asets')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True, branch='master')
        del co.columns['testing_aset1']
        del co.columns['testing_aset2']
        co.add_ndarray_column(name='testing_aset2', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.columns['testing_aset1']
        del co.columns['testing_aset2']
        co.add_ndarray_column(name='testing_aset1', shape=(5, 7), dtype=np.float32)
        co.commit('mutation and removal from dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts.t21.schema) == 1
        assert len(conflicts.t22.schema) == 1
        assert list(conflicts.t21.schema.keys())[0].column == 'testing_aset1'
        assert list(conflicts.t22.schema.keys())[0].column == 'testing_aset2'
        co.close()

    @pytest.mark.parametrize('writer', [False, True])
    def test_aset_mutation_conflict(self, aset_samples_initialized_repo, writer):
        # t3
        repo = aset_samples_initialized_repo
        co = repo.checkout(write=True, branch='master')
        co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float64)
        co.commit('added aset')
        co.close()
        repo.create_branch('testbranch')

        co = repo.checkout(write=True, branch='master')
        del co.columns['testing_aset']
        co.add_ndarray_column(name='testing_aset', shape=(7, 7), dtype=np.float64)
        co.commit('mutation from master')
        co.close()

        co = repo.checkout(write=True, branch='testbranch')
        del co.columns['testing_aset']
        co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float32)
        co.commit('mutation from dev')
        co.close()

        co = repo.checkout(write=writer, branch='master')
        conflicts = co.diff.branch('testbranch')[1]
        assert len(conflicts.t3.schema) == 1
        assert list(conflicts.t3.schema.keys())[0].column == 'testing_aset'
        co.close()


    @pytest.mark.parametrize('writer', [False, True])
    def test_commits_inside_cm(self, aset_samples_initialized_repo, array5by7, writer):
        repo = aset_samples_initialized_repo
        repo.create_branch('testbranch')
        co = repo.checkout(write=True, branch='testbranch')
        aset = co.columns['writtenaset']
        aset2 = co.add_ndarray_column('aset2', prototype=array5by7)
        aset2[1] = array5by7
        with aset:
            aset[100] = array5by7
            co.commit('inside cm')
            aset[101] = array5by7
            co.commit('another commit inside cm')
        co.close()
        co = repo.checkout(write=writer, branch='testbranch')
        assert np.allclose(co.columns['writtenaset'][101], array5by7)
        diff = co.diff.branch('master').diff
        assert 'aset2' in [x.column for x in diff.added.schema.keys()]
        calledWithAset = False
        for record in diff.added.samples:
            if record.column == 'writtenaset':
                calledWithAset = True
                assert record.sample in [100, 101]
        assert calledWithAset is True
        co.close()


class TestWriterDiff(object):

    def test_status_and_staged_column(self, aset_samples_initialized_repo):
        repo = aset_samples_initialized_repo
        co = repo.checkout(write=True)
        co.add_str_column('DOESNOTEXIST')
        co['DOESNOTEXIST'][1] = 'foo'
        assert co.diff.status() == 'DIRTY'
        co.commit('init metadata')
        assert co.diff.status() == 'CLEAN'
        co.close()

    def test_status_and_staged_samples(self, aset_samples_initialized_repo):
        dummyData = np.zeros((5, 7))
        repo = aset_samples_initialized_repo
        co = repo.checkout()
        with pytest.raises(AttributeError):
            co.diff.status()  # Read checkout doesn't have status()

        co = repo.checkout(write=True)
        co.columns['writtenaset']['45'] = dummyData
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()
        calledWithAset = False
        for record in diff.diff.added.samples:
            if record.column == 'writtenaset':
                calledWithAset = True
                assert record.sample in '45'
        assert calledWithAset is True
        co.commit('adding')
        assert co.diff.status() == 'CLEAN'
        co.close()

    def test_status_and_staged_aset(self, aset_samples_initialized_repo):
        repo = aset_samples_initialized_repo
        co = repo.checkout(write=True)
        co.add_ndarray_column(name='sampleaset', shape=(3, 5), dtype=np.float32)
        assert co.diff.status() == 'DIRTY'
        diff = co.diff.staged()
        assert 'sampleaset' in [x.column for x in diff.diff.added.schema]
        co.commit('init aset')
        assert co.diff.status() == 'CLEAN'
        co.close()


def test_repo_diff_method_branch_names(aset_samples_initialized_repo):
    # t3
    repo = aset_samples_initialized_repo
    co = repo.checkout(write=True, branch='master')
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float64)
    co.commit('added aset')
    co.close()
    repo.create_branch('testbranch')

    co = repo.checkout(write=True, branch='master')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(7, 7), dtype=np.float64)
    masterHEAD = co.commit('mutation from master')
    co.close()

    co = repo.checkout(write=True, branch='testbranch')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float32)
    devHEAD = co.commit('mutation from dev')
    co.close()

    co = repo.checkout(write=False, branch='master')
    co_diff = co.diff.branch('testbranch')
    co.close()

    repo_diff = repo.diff('master', 'testbranch')
    assert co_diff == repo_diff


def test_repo_diff_method_commit_digests(aset_samples_initialized_repo):
    # t3
    repo = aset_samples_initialized_repo
    co = repo.checkout(write=True, branch='master')
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float64)
    co.commit('added aset')
    co.close()
    repo.create_branch('testbranch')

    co = repo.checkout(write=True, branch='master')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(7, 7), dtype=np.float64)
    masterHEAD = co.commit('mutation from master')
    co.close()

    co = repo.checkout(write=True, branch='testbranch')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float32)
    devHEAD = co.commit('mutation from dev')
    co.close()

    co = repo.checkout(write=False, branch='master')
    co_diff = co.diff.commit(devHEAD)
    co.close()

    repo_diff = repo.diff(masterHEAD, devHEAD)
    assert co_diff == repo_diff




def test_repo_diff_method_one_branch_one_commit_digest(aset_samples_initialized_repo):
    # t3
    repo = aset_samples_initialized_repo
    co = repo.checkout(write=True, branch='master')
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float64)
    co.commit('added aset')
    co.close()
    repo.create_branch('testbranch')

    co = repo.checkout(write=True, branch='master')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(7, 7), dtype=np.float64)
    masterHEAD = co.commit('mutation from master')
    co.close()

    co = repo.checkout(write=True, branch='testbranch')
    del co.columns['testing_aset']
    co.add_ndarray_column(name='testing_aset', shape=(5, 7), dtype=np.float32)
    devHEAD = co.commit('mutation from dev')
    co.close()

    co = repo.checkout(write=False, branch='master')
    co_diff = co.diff.commit(devHEAD)
    co.close()

    repo_diff1 = repo.diff('master', devHEAD)
    assert co_diff == repo_diff1

    repo_diff2 = repo.diff(masterHEAD, 'testbranch')
    assert co_diff == repo_diff2
