import pytest

import numpy as np


@pytest.fixture()
def diverse_repo(repo):
    co = repo.checkout(write=True)
    co.arraysets.init_arrayset('test', prototype=np.arange(10))
    co.arraysets['test'][0] = np.arange(10)
    co.arraysets['test'][1] = np.arange(10) + 1
    co.arraysets['test'][2] = np.arange(10) + 2
    co.arraysets['test'][3] = np.arange(10) + 3
    co.arraysets['test'][4] = np.arange(10) + 4
    co.metadata['hi'] = 'foo'
    co.metadata['aea'] = 'eeae'
    co.commit('hello world')

    sample_trimg = np.arange(50).reshape(5, 10).astype(np.uint8)
    sample_trlabel = np.array([0], dtype=np.int64)
    sample_vimg = np.zeros(50).reshape(5, 10).astype(np.uint8)
    sample_vlabel = np.array([1], dtype=np.int64)

    co.close()
    repo.create_branch('dev')
    co = repo.checkout(write=True, branch='dev')
    dset_trlabels = co.arraysets.init_arrayset(name='train_labels', prototype=sample_trlabel)
    dset_trimgs = co.arraysets.init_arrayset('train_images', prototype=sample_trimg, backend_opts='01')
    dset_trlabels[0] = sample_trlabel
    dset_trlabels[1] = sample_trlabel + 1
    dset_trlabels[2] = sample_trlabel + 2
    dset_trimgs[0] = sample_trimg
    dset_trimgs[1] = sample_trimg + 1
    dset_trimgs[2] = sample_trimg + 2
    co.commit('second on dev')
    co.close()

    co = repo.checkout(write=True, branch='master')
    dset_vimgs = co.arraysets.init_arrayset('valid_images', prototype=sample_vimg)
    dset_vlabels = co.arraysets.init_arrayset('valid_labels', prototype=sample_vlabel)
    dset_vlabels[0] = sample_vlabel
    dset_vlabels[1] = sample_vlabel + 1
    dset_vlabels[2] = sample_vlabel + 2
    dset_vimgs[0] = sample_vimg
    dset_vimgs[1] = sample_vimg + 1
    dset_vimgs[2] = sample_vimg + 2
    co.metadata['second'] = 'on master now'
    co.commit('second on master')
    co.close()

    base = repo.merge('merge commit', 'master', 'dev')
    repo.create_branch('newbranch', base_commit=base)
    co = repo.checkout(write=True, branch='master')
    co.metadata['newmeta'] = 'wow'
    co.commit('on master after merge')
    co.close()

    co = repo.checkout(write=True, branch='newbranch')
    ds_trimgs = co.arraysets['train_images']
    ds_trlabels = co.arraysets['train_labels']
    ds_trlabels[3] = sample_trlabel + 3
    ds_trlabels[4] = sample_trlabel + 4
    ds_trlabels[5] = sample_trlabel + 5
    ds_trimgs[3] = sample_trimg + 3
    ds_trimgs[4] = sample_trimg + 4
    ds_trimgs[5] = sample_trimg + 5
    co.commit('on newdev after merge')
    co.close()

    base = repo.merge('new merge commit', 'master', 'newbranch')
    return repo


def test_verify_correct(diverse_repo):
    assert diverse_repo.verify_repo_integrity() is True


class TestVerifyCommitRefDigests(object):

    def test_remove_array_digest_is_caught(self, diverse_repo):
        from hangar.records import hashs
        from hangar.diagnostics.integrity import _verify_commit_ref_digests_exist

        hq = hashs.HashQuery(diverse_repo._env.hashenv)
        keys_to_remove = list(hq.gen_all_hash_keys_db())

        for key_removed in keys_to_remove:
            with diverse_repo._env.hashenv.begin(write=True) as txn:
                val_removed = txn.get(key_removed)
                txn.delete(key_removed)

            with pytest.raises(RuntimeError):
                _verify_commit_ref_digests_exist(hashenv=diverse_repo._env.hashenv,
                                                 labelenv=diverse_repo._env.labelenv,
                                                 refenv=diverse_repo._env.refenv)

            with diverse_repo._env.hashenv.begin(write=True) as txn:
                txn.put(key_removed, val_removed)

    def test_remove_metadata_digest_is_caught(self, diverse_repo):
        from hangar.records import hashs
        from hangar.diagnostics.integrity import _verify_commit_ref_digests_exist

        hq = hashs.HashQuery(diverse_repo._env.labelenv)
        keys_to_remove = list(hq.gen_all_hash_keys_db())

        for key_removed in keys_to_remove:
            with diverse_repo._env.labelenv.begin(write=True) as txn:
                val_removed = txn.get(key_removed)
                txn.delete(key_removed)

            with pytest.raises(RuntimeError):
                _verify_commit_ref_digests_exist(hashenv=diverse_repo._env.hashenv,
                                                 labelenv=diverse_repo._env.labelenv,
                                                 refenv=diverse_repo._env.refenv)

            with diverse_repo._env.labelenv.begin(write=True) as txn:
                txn.put(key_removed, val_removed)

    def test_remove_schema_digest_is_caught(self, diverse_repo):
        from hangar.records import hashs
        from hangar.diagnostics.integrity import _verify_commit_ref_digests_exist

        hq = hashs.HashQuery(diverse_repo._env.hashenv)
        keys_to_remove = list(hq.gen_all_schema_keys_db())
        for key_removed in keys_to_remove:
            with diverse_repo._env.hashenv.begin(write=True) as txn:
                val_removed = txn.get(key_removed)
                txn.delete(key_removed)

            with pytest.raises(RuntimeError):
                _verify_commit_ref_digests_exist(hashenv=diverse_repo._env.hashenv,
                                                 labelenv=diverse_repo._env.labelenv,
                                                 refenv=diverse_repo._env.refenv)

        with diverse_repo._env.hashenv.begin(write=True) as txn:
            txn.put(key_removed, val_removed)


class TestVerifyCommitTree(object):

    def test_parent_ref_digest_of_cmt_does_not_exist(self, diverse_repo):
        from hangar.diagnostics.integrity import _verify_commit_tree_integrity
        from hangar.records.parsing import commit_parent_db_key_from_raw_key

        repo = diverse_repo
        history = repo.log(return_contents=True)
        all_commits = history['order']
        for cmt in all_commits:
            parentKey = commit_parent_db_key_from_raw_key(cmt)
            with repo._env.refenv.begin(write=True) as txn:
                parentVal = txn.get(parentKey)
                txn.delete(parentKey)

            with pytest.raises(RuntimeError, match='Data corruption detected for parent ref'):
                _verify_commit_tree_integrity(repo._env.refenv)

            with repo._env.refenv.begin(write=True) as txn:
                txn.put(parentKey, parentVal)

    def test_parent_ref_references_nonexisting_commits(self, diverse_repo):
        from hangar.diagnostics.integrity import _verify_commit_tree_integrity
        from hangar.records.parsing import commit_parent_db_key_from_raw_key
        from hangar.records.parsing import commit_parent_raw_val_from_db_val
        from hangar.records.parsing import commit_parent_db_val_from_raw_val

        repo = diverse_repo
        history = repo.log(return_contents=True)
        all_commits = history['order']
        for cmt in all_commits:
            parentKey = commit_parent_db_key_from_raw_key(cmt)
            with repo._env.refenv.begin(write=True) as txn:
                parentVal = txn.get(parentKey)
                parent_raw = commit_parent_raw_val_from_db_val(parentVal)
                parent = parent_raw.ancestor_spec

                if parent.dev_ancestor:
                    modifiedVal = commit_parent_db_val_from_raw_val(
                        master_ancestor=parent.master_ancestor,
                        dev_ancestor='corrupt',
                        is_merge_commit=parent.is_merge_commit)
                elif parent.master_ancestor:
                    modifiedVal = commit_parent_db_val_from_raw_val(
                        master_ancestor='corrupt',
                        dev_ancestor=parent.dev_ancestor,
                        is_merge_commit=parent.is_merge_commit)
                else:
                    continue

                txn.put(parentKey, modifiedVal.raw, overwrite=True)

            with pytest.raises(RuntimeError, match='Data corruption detected in commit tree'):
                _verify_commit_tree_integrity(repo._env.refenv)

            with repo._env.refenv.begin(write=True) as txn:
                txn.put(parentKey, parentVal)

    def test_parent_ref_has_two_initial_commits(self, diverse_repo):
        from hangar.diagnostics.integrity import _verify_commit_tree_integrity
        from hangar.records.parsing import commit_parent_db_key_from_raw_key

        repo = diverse_repo
        repo = diverse_repo
        history = repo.log(return_contents=True)
        all_commits = history['order']
        initial_commit = all_commits[-1]
        for cmt in all_commits:
            if cmt == initial_commit:
                continue

            parentKey = commit_parent_db_key_from_raw_key(cmt)
            with repo._env.refenv.begin(write=True) as txn:
                parentVal = txn.get(parentKey)
                txn.put(parentKey, b'', overwrite=True)

            with pytest.raises(RuntimeError, match='Commit tree integrity compromised. Multiple "initial"'):
                _verify_commit_tree_integrity(repo._env.refenv)

            with repo._env.refenv.begin(write=True) as txn:
                txn.put(parentKey, parentVal, overwrite=True)


class TestBranchIntegrity(object):

    def test_atleast_one_branch_exists(self, diverse_repo):
        from hangar.records.heads import get_branch_names
        from hangar.records.parsing import repo_branch_head_db_key_from_raw_key
        from hangar.diagnostics.integrity import _verify_branch_integrity

        branch_names = get_branch_names(diverse_repo._env.branchenv)
        with diverse_repo._env.branchenv.begin(write=True) as txn:
            for bname in branch_names:
                branchKey = repo_branch_head_db_key_from_raw_key(bname)
                txn.delete(branchKey)

        with pytest.raises(
                RuntimeError,
                match='Branch map compromised. Repo must contain atleast one branch'
        ):
            _verify_branch_integrity(diverse_repo._env.branchenv, diverse_repo._env.refenv)

    def test_branch_name_head_commit_digests_exist(self, diverse_repo):
        from hangar.records.heads import get_branch_names, get_branch_head_commit
        from hangar.records.parsing import commit_ref_db_key_from_raw_key
        from hangar.records.parsing import commit_parent_db_key_from_raw_key
        from hangar.records.parsing import commit_spec_db_key_from_raw_key
        from hangar.diagnostics.integrity import _verify_branch_integrity

        branch_names = get_branch_names(diverse_repo._env.branchenv)
        for bname in branch_names:
            bhead = get_branch_head_commit(diverse_repo._env.branchenv, branch_name=bname)
            with diverse_repo._env.refenv.begin(write=True) as txn:
                cmtRefKey = commit_ref_db_key_from_raw_key(bhead)
                cmtSpecKey = commit_spec_db_key_from_raw_key(bhead)
                cmtParentKey = commit_parent_db_key_from_raw_key(bhead)

                cmtRefVal = txn.get(cmtRefKey)
                cmtSpecVal = txn.get(cmtSpecKey)
                cmtParentVal = txn.get(cmtParentKey)

                txn.delete(cmtRefKey)
                txn.delete(cmtSpecKey)
                txn.delete(cmtParentKey)

            with pytest.raises(RuntimeError, match='Branch commit map compromised. Branch name'):
                _verify_branch_integrity(diverse_repo._env.branchenv, diverse_repo._env.refenv)

            with diverse_repo._env.refenv.begin(write=True) as txn:
                txn.put(cmtRefKey, cmtRefVal)
                txn.put(cmtSpecKey, cmtSpecVal)
                txn.put(cmtParentKey, cmtParentVal)

    def test_staging_head_branch_name_exists(self, diverse_repo):
        from hangar.records.heads import get_staging_branch_head
        from hangar.records.parsing import repo_branch_head_db_key_from_raw_key
        from hangar.diagnostics.integrity import _verify_branch_integrity

        bname = get_staging_branch_head(diverse_repo._env.branchenv)
        with diverse_repo._env.branchenv.begin(write=True) as txn:
            branchKey = repo_branch_head_db_key_from_raw_key(bname)
            txn.delete(branchKey)

        with pytest.raises(
                RuntimeError,
                match='Brach commit map compromised. Staging head refers to branch name'
        ):
            _verify_branch_integrity(diverse_repo._env.branchenv, diverse_repo._env.refenv)
