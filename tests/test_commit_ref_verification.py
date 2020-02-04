import pytest


def test_verify_corruption_in_commit_ref_alerts(two_commit_filled_samples_repo):
    from hangar.records.parsing import commit_ref_db_key_from_raw_key
    from hangar.records.parsing import commit_ref_raw_val_from_db_val
    from hangar.records.parsing import commit_ref_db_val_from_raw_val

    repo = two_commit_filled_samples_repo
    history = repo.log(return_contents=True)
    head_commit = history['head']

    refKey = commit_ref_db_key_from_raw_key(head_commit)
    with repo._env.refenv.begin(write=True) as txn:
        refVal = txn.get(refKey)
        ref_unpacked = commit_ref_raw_val_from_db_val(refVal)

        modified_ref = list(ref_unpacked.db_kvs)
        modified_ref[0] = list(modified_ref[0])
        modified_ref[0][1] = b'corrupt!'
        modified_ref[0] = tuple(modified_ref[0])
        modified_ref = tuple(modified_ref)
        modifiedVal = commit_ref_db_val_from_raw_val(modified_ref)

        txn.put(refKey, modifiedVal.raw, overwrite=True)

    with pytest.raises(IOError):
        _ = repo.checkout(write=True)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False, commit=head_commit)


def test_verify_corruption_in_commit_parent_val_alerts(two_commit_filled_samples_repo):
    from hangar.records.parsing import commit_parent_db_key_from_raw_key
    from hangar.records.parsing import commit_parent_raw_val_from_db_val
    from hangar.records.parsing import commit_parent_db_val_from_raw_val

    repo = two_commit_filled_samples_repo
    history = repo.log(return_contents=True)
    head_commit = history['head']

    parentKey = commit_parent_db_key_from_raw_key(head_commit)
    with repo._env.refenv.begin(write=True) as txn:
        parentVal = txn.get(parentKey)

        parent_raw = commit_parent_raw_val_from_db_val(parentVal)
        parent = parent_raw.ancestor_spec
        modifiedVal = commit_parent_db_val_from_raw_val(
            master_ancestor='corrupt',
            dev_ancestor=parent.dev_ancestor,
            is_merge_commit=parent.is_merge_commit)

        txn.put(parentKey, modifiedVal.raw, overwrite=True)

    with pytest.raises(IOError):
        _ = repo.checkout(write=True)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False, commit=head_commit)


def test_verify_corruption_in_spec_val_alerts(two_commit_filled_samples_repo):
    from hangar.records.parsing import commit_spec_db_key_from_raw_key
    from hangar.records.parsing import commit_spec_db_val_from_raw_val
    from hangar.records.parsing import commit_spec_raw_val_from_db_val

    repo = two_commit_filled_samples_repo
    history = repo.log(return_contents=True)
    head_commit = history['head']

    specKey = commit_spec_db_key_from_raw_key(head_commit)
    with repo._env.refenv.begin(write=True) as txn:
        specVal = txn.get(specKey)

        spec_raw = commit_spec_raw_val_from_db_val(specVal)
        modified_spec = spec_raw.user_spec
        modified_spec = modified_spec._replace(commit_time=10.42)
        modifiedVal = commit_spec_db_val_from_raw_val(*modified_spec)

        txn.put(specKey, modifiedVal.raw, overwrite=True)

    with pytest.raises(IOError):
        _ = repo.checkout(write=True)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False)
    with pytest.raises(IOError):
        _ = repo.checkout(write=False, commit=head_commit)
