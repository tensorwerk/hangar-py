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
        masterco = repo.checkout('master')
        commit_diffs = masterco.diff.commit(testco.commit_hash)
        branch_diffs = masterco.diff.branch('testbranch')
        assert commit_diffs == branch_diffs

    def test_diff_with_wrong_commit_hash(self, repo_2_br_no_conf):
        repo = repo_2_br_no_conf
        testco = repo.checkout('testbranch')
        masterco = repo.checkout('master')
        wrong_commit_hash = testco.commit_hash + 'WrongHash'
        with pytest.raises(ValueError):
            masterco.diff.commit(wrong_commit_hash)

        # diff = masterco.diff.commit(testco.commit_hash)
        breakpoint()
        repo.merge('dummy', 'master', 'testbranch')
        masterco = repo.checkout('master')

    def test_diff_with_wrong_branch_name(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        masterco = repo.checkout('master')
        with pytest.raises(ValueError):
            masterco.diff.branch('wrong_branch_name')

    def test_diff_data(self, repo_1_br_no_conf):
        repo = repo_1_br_no_conf
        co = repo.checkout('master')
        diffdata = co.diff.branch('testbranch')
        breakpoint()

