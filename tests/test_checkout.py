import atexit
import numpy as np
import pytest
from conftest import fixed_shape_backend_params


class TestCheckout(object):

    def test_write_checkout_specifying_commit_not_allowed_if_commit_exists(self, aset_samples_initialized_repo):
        cmt_digest = aset_samples_initialized_repo.log(return_contents=True)['head']
        with pytest.raises(ValueError):
            aset_samples_initialized_repo.checkout(write=True, commit=cmt_digest)

    def test_write_checkout_specifying_commit_not_allowed_if_commit_does_not_exists(self, aset_samples_initialized_repo):
        cmt_digest = 'notrealcommit'
        with pytest.raises(ValueError):
            aset_samples_initialized_repo.checkout(write=True, commit=cmt_digest)

    def test_two_write_checkouts(self, repo):
        w1_checkout = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            repo.checkout(write=True)
        w1_checkout.close()

    def test_two_read_checkouts(self, repo, array5by7):
        w_checkout = repo.checkout(write=True)
        arrayset_name = 'aset'
        r_ds = w_checkout.add_ndarray_column(name=arrayset_name, prototype=array5by7)
        r_ds['1'] = array5by7
        w_checkout.commit('init')
        r1_checkout = repo.checkout()
        r2_checkout = repo.checkout()
        assert np.allclose(r1_checkout.columns['aset']['1'], array5by7)
        assert np.allclose(
            r1_checkout.columns['aset']['1'], r2_checkout.columns['aset']['1'])
        r1_checkout.close()
        r2_checkout.close()
        w_checkout.close()

    def test_write_with_read_checkout(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout()
        with pytest.raises(AttributeError):
            co.add_ndarray_column(name='aset', shape=(5, 7), dtype=np.float64)
        with pytest.raises(AttributeError):
            co.add_str_column('test_meta')
        co.close()

    def test_writer_aset_obj_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        asets = co.columns
        aset = co.columns['writtenaset']
        co.close()

        with pytest.raises(PermissionError):
            asets.iswriteable
        with pytest.raises(PermissionError):
            shouldFail = asets['writtenaset']
        with pytest.raises(PermissionError):
            aset.iswriteable

    def test_writer_aset_obj_arrayset_iter_values_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        oldObjs = []
        for oldObj in co.columns.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(PermissionError):
                oldObj.column

    def test_writer_aset_obj_arrayset_iter_items_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        oldObjs = {}
        for oldName, oldObj in co.columns.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(PermissionError):
                obj.column

    def test_writer_aset_obj_not_accessible_after_commit_and_close(self, aset_samples_initialized_repo, array5by7):
        repo = aset_samples_initialized_repo
        co = repo.checkout(write=True)
        asets = co.columns
        aset = co.columns['writtenaset']
        aset['1'] = array5by7
        co.commit('hey there')
        co.close()

        with pytest.raises(PermissionError):
            asets.iswriteable
        with pytest.raises(PermissionError):
            shouldFail = asets['writtenaset']
        with pytest.raises(PermissionError):
            aset.iswriteable
        with pytest.raises(PermissionError):
            shouldFail = aset['1']

    def test_reader_aset_obj_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=False)
        asets = co.columns
        aset = co.columns['writtenaset']
        co.close()

        with pytest.raises(PermissionError):
            asets.iswriteable
        with pytest.raises(PermissionError):
            shouldFail = asets['writtenaset']
        with pytest.raises(PermissionError):
            aset.iswriteable

    def test_reader_aset_obj_column_iter_values_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=False)
        oldObjs = []
        for oldObj in co.columns.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(PermissionError):
                oldObj.column

    def test_reader_aset_obj_arrayset_iter_items_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=False)
        oldObjs = {}
        for oldName, oldObj in co.columns.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(PermissionError):
                obj.column

    def test_reader_arrayset_context_manager_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=False)
        aset = co.columns['writtenaset']
        klist = []
        with aset as ds:
            for k in ds.keys():
                klist.append(k)
                a = ds
        co.close()

        with pytest.raises(PermissionError):
            a.column
        with pytest.raises(PermissionError):
            ds.column
        with pytest.raises(PermissionError):
            aset[klist[0]]

    def test_writer_arrayset_context_manager_not_accessible_after_close(self, two_commit_filled_samples_repo):
        repo = two_commit_filled_samples_repo
        co = repo.checkout(write=True)
        aset = co.columns['writtenaset']
        with aset as ds:
            # for k in ds.keys():
            #     klist.append(k)
            a = ds
            a['1232'] = np.random.randn(5, 7).astype(np.float32)
        co.close()

        with pytest.raises(PermissionError):
            a.column
        with pytest.raises(PermissionError):
            ds.column
        with pytest.raises(PermissionError):
            aset['1232']

    def test_close_read_does_not_invalidate_write_checkout(self, aset_samples_initialized_repo, array5by7):
        repo = aset_samples_initialized_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)
        r_co.close()

        with pytest.raises(PermissionError):
            shouldFail = r_co.columns

        aset = w_co.columns['writtenaset']
        aset['1'] = array5by7
        assert np.allclose(w_co.columns['writtenaset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(PermissionError):
            aset.column

    def test_close_write_does_not_invalidate_read_checkout(self, aset_samples_initialized_repo, array5by7):
        repo = aset_samples_initialized_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)

        aset = w_co.columns['writtenaset']
        aset['1'] = array5by7
        assert np.allclose(w_co.columns['writtenaset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        assert 'writtenaset' in r_co.columns
        with pytest.raises(PermissionError):
            aset.column
        r_co.close()
        with pytest.raises(PermissionError):
            r_co.columns

    def test_operate_on_arrayset_after_closing_old_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            aset['1'] = array5by7
            co.commit('this is a commit message')
        co.close()
        with pytest.raises(PermissionError):
            aset['1']

    def test_operate_on_closed_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        co.add_ndarray_column('aset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        with pytest.raises(PermissionError):
            co.columns['aset']['1'] = array5by7

    @pytest.mark.parametrize("aset_backend", fixed_shape_backend_params)
    def test_operate_on_arrayset_samples_after_commiting_but_not_closing_checkout(self, aset_backend, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=array5by7, backend=aset_backend)
        aset['1'] = array5by7
        co.commit('hi')

        aset['2'] = array5by7  # this raises Exception since the reference to aset i gon
        co.commit('hello 2')
        assert np.allclose(aset['2'], array5by7)
        co.close()

        with pytest.raises(PermissionError):
            aset.name

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    def test_operate_on_arraysets_after_commiting_but_not_closing_checkout(self, aset1_backend, aset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        asets = co.columns
        aset = co.add_ndarray_column('aset', prototype=array5by7, backend=aset1_backend)
        aset['1'] = array5by7
        co.commit('hi')

        aset2 = co.add_ndarray_column('arange', prototype=np.arange(50), backend=aset2_backend)
        aset2['0'] = np.arange(50)
        co.commit('hello 2')
        assert np.allclose(aset2['0'], np.arange(50))
        co.close()

        with pytest.raises(PermissionError):
            co.columns
        with pytest.raises(PermissionError):
            asets.iswriteable
        with pytest.raises(PermissionError):
            aset2.name

    def test_with_wrong_argument_value(self, repo):
        # It is intuitive to a user to pass branchname as positional
        # argument but hangar expect permission as first argument
        with pytest.raises(ValueError):
            repo.checkout('branchname')
        with pytest.raises(ValueError):
            repo.checkout(write='True')
        with pytest.raises(ValueError):
            repo.checkout(branch=True)
        co = repo.checkout(True)  # This should not raise any excpetion
        # unregister close operation as conftest will close env before this is called.
        atexit.unregister(co.close)


    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    def test_reset_staging_area_no_changes_made_does_not_work(self, aset1_backend, aset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=array5by7, backend=aset1_backend)
        aset2 = co.add_ndarray_column('arange', prototype=np.arange(50), backend=aset2_backend)
        aset['1'] = array5by7
        aset2['0'] = np.arange(50)
        co.commit('hi')

        # verifications before reset
        assert np.allclose(aset2['0'], np.arange(50))
        assert len(co.columns) == 2
        assert co.columns['arange'].iswriteable

        with pytest.raises(RuntimeError, match='No changes made'):
            co.reset_staging_area()

        # verifications after reset
        assert np.allclose(aset2['0'], np.arange(50))
        assert len(co.columns) == 2
        assert co.columns['arange'].iswriteable
        co.close()

    @pytest.mark.parametrize("aset1_backend", fixed_shape_backend_params)
    @pytest.mark.parametrize("aset2_backend", fixed_shape_backend_params)
    def test_reset_staging_area_clears_arraysets(self, aset1_backend, aset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.add_ndarray_column('aset', prototype=array5by7, backend=aset1_backend)
        aset['1'] = array5by7
        co.commit('hi')

        aset2 = co.add_ndarray_column('arange', prototype=np.arange(50), backend=aset2_backend)
        aset2['0'] = np.arange(50)
        # verifications before reset
        assert np.allclose(aset2['0'], np.arange(50))
        assert len(co.columns) == 2
        assert co.columns['arange'].iswriteable

        co.reset_staging_area()
        # behavior expected after reset
        assert len(co.columns) == 1
        with pytest.raises(PermissionError):
            aset2['0']
        with pytest.raises(KeyError):
            co.columns['arange']
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_dunder_contains_method(self, repo_20_filled_samples, write):
        co = repo_20_filled_samples.checkout(write=write)
        assert 'writtenaset' in co
        assert 'second_aset' in co
        assert 'doesnotexist' not in co
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_dunder_len_method(self, repo_20_filled_samples, write):
        co = repo_20_filled_samples.checkout(write=write)
        assert len(co) == 2
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_dunder_iter_method(self, repo_20_filled_samples, write):
        from typing import Iterable
        co = repo_20_filled_samples.checkout(write=write)
        it = iter(co)
        assert isinstance(it, Iterable)
        icount = 0
        for k in it:
            assert k in ['writtenaset', 'second_aset']
            icount += 1
        assert icount == 2
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_keys_method(self, repo_20_filled_samples, write):
        co = repo_20_filled_samples.checkout(write=write)
        keys = list(co.keys())
        assert len(keys) == 2
        for k in ['writtenaset', 'second_aset']:
            assert k in keys
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_values_method(self, repo_20_filled_samples, write):
        from hangar.columns.layout_nested import NestedSampleWriter, NestedSampleReader
        from hangar.columns.layout_flat import FlatSampleWriter, FlatSampleReader
        possible_classes = (
            NestedSampleWriter, NestedSampleReader, FlatSampleReader, FlatSampleWriter)

        co = repo_20_filled_samples.checkout(write=write)
        icount = 0
        for col in co.values():
            assert isinstance(col, possible_classes)
            icount += 1
        assert icount == 2
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_items_method(self, repo_20_filled_samples, write):
        from hangar.columns.layout_nested import NestedSampleWriter, NestedSampleReader
        from hangar.columns.layout_flat import FlatSampleWriter, FlatSampleReader
        possible_classes = (
            NestedSampleWriter, NestedSampleReader, FlatSampleReader, FlatSampleWriter)

        co = repo_20_filled_samples.checkout(write=write)
        icount = 0
        for k, col in co.items():
            assert k in ['writtenaset', 'second_aset']
            assert isinstance(col, possible_classes)
            icount += 1
        assert icount == 2
        co.close()

    @pytest.mark.parametrize('write', [True, False])
    def test_checkout_log_method(self, repo_20_filled_samples, write):
        repo_log = repo_20_filled_samples.log(return_contents=True)
        co = repo_20_filled_samples.checkout(write=write)
        co_log = co.log(return_contents=True)
        co.close()
        assert repo_log == co_log


class TestBranchingMergingInCheckout(object):

    def test_merge(self, aset_samples_initialized_repo, array5by7):
        branch = aset_samples_initialized_repo.create_branch('testbranch')
        assert isinstance(branch.name, str)
        assert isinstance(branch.digest, str)
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch.name)
        assert co._branch_name == branch.name
        co.add_str_column('test_meta')
        co.columns['writtenaset']['1'] = array5by7
        co['test_meta'].update({'a': 'b'})
        co.commit('this is a commit message')
        co.close()
        aset_samples_initialized_repo.merge('test merge', 'master', branch.name)
        co = aset_samples_initialized_repo.checkout()
        assert (co.columns['writtenaset']['1'] == array5by7).all()
        assert co['test_meta'].get('a') == 'b'
        co.close()

    def test_merge_without_closing_previous_checkout(self, aset_samples_initialized_repo, array5by7):
        branch = aset_samples_initialized_repo.create_branch('testbranch')
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch.name)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(PermissionError):
            aset_samples_initialized_repo.merge('test merge', 'master', branch.name)
        # unregister close operation as conftest will close env before this is called.
        atexit.unregister(co.close)

    def test_merge_multiple_checkouts_same_aset(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.add_str_column('test_meta')
        co.commit('test meta commit')
        co.close()
        branch1 = aset_samples_initialized_repo.create_branch('testbranch1')
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        co.columns['writtenaset']['1'] = array5by7
        co['test_meta'].update({'a1': 'b1'})
        co.commit('this is a commit message')
        co.close()

        branch2 = aset_samples_initialized_repo.create_branch('testbranch2')
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)
        co.columns['writtenaset']['2'] = array5by7
        co['test_meta'].update({'a2': 'b2'})
        co.commit('this is a commit message')
        co.close()

        aset_samples_initialized_repo.merge('test merge 1', 'master', branch1.name)
        aset_samples_initialized_repo.merge('test merge 2', 'master', branch2.name)

        co = aset_samples_initialized_repo.checkout(branch='master')
        assert len(co.columns) == 2
        assert len(co.columns['writtenaset']) == 2
        assert list(co['test_meta'].keys()) == ['a1', 'a2']
        co.close()

    def test_merge_multiple_checkouts_multiple_aset(self, aset_samples_initialized_repo, array5by7):
        branch1 = aset_samples_initialized_repo.create_branch('testbranch1')
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        co.columns['writtenaset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        branch2 = aset_samples_initialized_repo.create_branch('testbranch2')
        co = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)
        second_aset = co.add_ndarray_column(name='second_aset', prototype=array5by7)
        second_aset['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        aset_samples_initialized_repo.merge('test merge 1', 'master', branch1.name)
        aset_samples_initialized_repo.merge('test merge 2', 'master', branch2.name)

        co = aset_samples_initialized_repo.checkout(branch='master')
        assert len(co.columns) == 2
        assert len(co.columns['writtenaset']) == 1
        assert len(co.columns['second_aset']) == 1
        co.close()

    def test_merge_diverged_conflict(self, aset_samples_initialized_repo, array5by7):
        co = aset_samples_initialized_repo.checkout(write=True)
        co.add_str_column('test_meta')
        co.commit('test meta commit')
        co.close()
        branch1 = aset_samples_initialized_repo.create_branch('testbranch1')
        branch2 = aset_samples_initialized_repo.create_branch('testbranch2')

        co = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        co.columns['writtenaset']['1'] = array5by7
        co['test_meta'].update({'a': 'b'})
        co.commit('this is a commit message')
        co.close()

        co = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)
        newarray = np.zeros_like(array5by7)
        co.columns['writtenaset']['1'] = newarray
        co['test_meta'].update({'a': 'c'})
        co.commit('this is a commit message')
        co.close()

        aset_samples_initialized_repo.merge('commit message', 'master', branch1.name)

        with pytest.raises(ValueError):
            aset_samples_initialized_repo.merge('commit message', 'master', branch2.name)

    def test_new_branch_from_where(self, aset_samples_initialized_repo, array5by7):
        branch1 = aset_samples_initialized_repo.create_branch('testbranch1')
        branch2 = aset_samples_initialized_repo.create_branch('testbranch2')
        co1 = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        h1 = aset_samples_initialized_repo.log(branch=co1.branch_name, return_contents=True)
        co1.close()

        co2 = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)
        co2.add_ndarray_column('aset2', prototype=array5by7)
        co2.columns['aset2']['2'] = array5by7
        co2.commit('this is a merge message')
        co2.close()
        h2 = aset_samples_initialized_repo.log(branch=branch2.name, return_contents=True)

        branch3 = aset_samples_initialized_repo.create_branch('testbranch3')
        co3 = aset_samples_initialized_repo.checkout(write=True, branch=branch3.name)
        h3 = aset_samples_initialized_repo.log(branch=co3.branch_name, return_contents=True)
        co3.close()

        assert h2['head'] == h3['head']
        assert h2['ancestors'][h2['head']] == h3['ancestors'][h3['head']]
        assert h1['head'] in h2['ancestors'][h2['head']]

    def test_cannot_checkout_branch_with_staged_changes(self, aset_samples_initialized_repo, array5by7):
        branch1 = aset_samples_initialized_repo.create_branch('testbranch1')
        branch2 = aset_samples_initialized_repo.create_branch('testbranch2')
        co1 = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        initial_cmt = co1.commit_hash
        co1.add_ndarray_column('aset2', prototype=array5by7)
        co1.columns['aset2']['2'] = array5by7
        co1.close()

        with pytest.raises(ValueError):
            con = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)

        co1 = aset_samples_initialized_repo.checkout(write=True, branch=branch1.name)
        co1.commit('hi')
        assert co1.commit_hash != initial_cmt
        assert co1.branch_name == branch1.name
        co1.close()

        co2 = aset_samples_initialized_repo.checkout(write=True, branch=branch2.name)
        assert co2.branch_name == branch2.name
        assert co2.commit_hash == initial_cmt
        co2.close()


def test_full_from_short_commit_digest(two_commit_filled_samples_repo):
    from hangar.records.commiting import expand_short_commit_digest

    repo = two_commit_filled_samples_repo
    history = repo.log(branch='master', return_contents=True)
    commits = history['order']
    for full_cmt in commits:
        short_cmt = full_cmt[:18]
        found_cmt = expand_short_commit_digest(repo._env.refenv, short_cmt)
        assert found_cmt == full_cmt

    with pytest.raises(KeyError, match='No matching commit hash found starting with'):
        expand_short_commit_digest(repo._env.refenv, 'zzzzzzzzzzzzzzzzzzzzzzzzzzzz')


def test_writer_context_manager_objects_are_gc_removed_after_co_close(two_commit_filled_samples_repo):

    repo = two_commit_filled_samples_repo
    co = repo.checkout(write=True)
    co.add_str_column('test_meta')
    with co['test_meta'] as m:
        m['aa'] = 'bb'
        cmt1 = co.commit('here is the first commit')
        with co.columns['writtenaset'] as d:
            d['2422'] = d['0'] + 213
            cmt2 = co.commit('here is the second commit')

    assert co.close() is None
    with pytest.raises(PermissionError):
        _ = m.__dict__
    with pytest.raises(PermissionError):
        _ = d.column
    with pytest.raises(PermissionError):
        _ = co.columns
    assert co.__dict__ == {}

    co = repo.checkout(commit=cmt1)
    assert 'aa' in co['test_meta']
    assert co['test_meta']['aa'] == 'bb'
    co.close()

    co = repo.checkout(commit=cmt2)
    assert 'aa' in co['test_meta']
    assert co['test_meta']['aa'] == 'bb'
    assert '2422' in co.columns['writtenaset']
    assert np.allclose(co.columns['writtenaset']['2422'],
                       co.columns['writtenaset']['0'] + 213)
    co.close()


def test_reader_context_manager_objects_are_gc_removed_after_co_close(two_commit_filled_samples_repo):

    repo = two_commit_filled_samples_repo
    co = repo.checkout(write=False)
    with co.columns['writtenaset'] as d:
        ds = d['2']

    assert d.iswriteable is False
    assert np.allclose(ds, d.get('2'))
    assert np.allclose(ds, co.columns['writtenaset'].get('2'))

    assert co.close() is None

    with pytest.raises(PermissionError):
        d.column
    with pytest.raises(AttributeError):
        co._columns
    with pytest.raises(PermissionError):
        str(co.columns.get('writtenaset'))
    with pytest.raises(PermissionError):
        co.columns
    with pytest.raises(PermissionError):
        repr(co)
    assert co.__dict__ == {}


def test_checkout_branch_not_existing_does_not_hold_writer_lock(two_commit_filled_samples_repo):
    repo = two_commit_filled_samples_repo
    assert 'doesnotexist' not in repo.list_branches()
    assert repo.writer_lock_held is False
    with pytest.raises(ValueError):
        co = repo.checkout(write=True, branch='doesnotexist')
    assert repo.writer_lock_held is False
    with pytest.raises(NameError):
        co.branch_name  # should not even exist
