import pytest
import numpy as np

@pytest.fixture()
def single_branch_repo(repo):

    dummyData = np.arange(50)

    co = repo.checkout(write=True, branch_name='master')
    co.datasets.init_dataset(name='dummy', prototype=dummyData, samples_are_named=True)
    for idx in range(10):
        dummyData[:] = idx
        co.datasets['dummy'][str(idx)] = dummyData
    co.metadata['hello'] = 'world'
    co.commit('first commit adding dummy data and hello meta')
    co.close()

    repo.create_branch('testbranch')
    co = repo.checkout(write=True, branch_name='testbranch')
    for idx in range(10, 20):
        dummyData[:] = idx
        co.datasets['dummy'][str(idx)] = dummyData
    co.metadata['foo'] = 'bar'
    co.commit('first commit on test branch adding non-conflict data and meta')
    co.close()
    return repo


@pytest.fixture()
def multi_branch_repo(single_branch_repo):

    dummyData = np.arange(50)

    repo = single_branch_repo
    co = repo.checkout(write=True, branch_name='master')
    for idx in range(20, 30):
        dummyData[:] = idx
        co.datasets['dummy'][str(idx)] = dummyData
    co.commit('second commit on master adding non-conflict data')
    co.close()
    return repo


def test_merge_fails_with_invalid_branch_name(single_branch_repo):
    with pytest.raises(ValueError):
        cmt_hash = single_branch_repo.merge('merge commit', 'master', 'failbranchname')
    # no message passed in
    with pytest.raises(TypeError):
        cmt_hash = single_branch_repo.merge('master', 'testbranch')


def test_is_ff_merge(single_branch_repo):
    testbranch_head = single_branch_repo.log(branch_name='testbranch', return_contents=True)['head']
    cmt_hash = single_branch_repo.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash == testbranch_head


def test_ff_merge_returns_correct_contents_for_name_or_hash_checkout(single_branch_repo):
    cmt_hash = single_branch_repo.merge('merge commit', 'master', 'testbranch')
    coByName = single_branch_repo.checkout(branch_name='master')
    coByHash = single_branch_repo.checkout(commit=cmt_hash)

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


def test_ff_merge_updates_head_commit_of_branches_correctly(single_branch_repo):
    repo = single_branch_repo
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


def test_is_3_way_merge(multi_branch_repo):
    testbranch_head = multi_branch_repo.log(branch_name='testbranch', return_contents=True)['head']
    masterbranch_head = multi_branch_repo.log(branch_name='master', return_contents=True)['head']
    cmt_hash = multi_branch_repo.merge('merge commit', 'master', 'testbranch')
    assert cmt_hash != testbranch_head
    assert cmt_hash != masterbranch_head
