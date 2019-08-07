import atexit
import numpy as np
import pytest


class TestCheckout(object):

    def test_two_write_checkouts(self, repo):
        w1_checkout = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            repo.checkout(write=True)
        w1_checkout.close()

    def test_two_read_checkouts(self, repo, array5by7):
        w_checkout = repo.checkout(write=True)
        arrayset_name = 'aset'
        r_ds = w_checkout.arraysets.init_arrayset(name=arrayset_name, prototype=array5by7)
        r_ds['1'] = array5by7
        w_checkout.metadata.add('init', 'array5by7 added')
        w_checkout.commit('init')
        r1_checkout = repo.checkout()
        r2_checkout = repo.checkout()
        assert np.allclose(r1_checkout.arraysets['aset']['1'], array5by7)
        assert np.allclose(
            r1_checkout.arraysets['aset']['1'], r2_checkout.arraysets['aset']['1'])
        assert r1_checkout.metadata.get('init') == 'array5by7 added'
        assert r2_checkout.metadata.get('init') == 'array5by7 added'
        r1_checkout.close()
        r2_checkout.close()
        w_checkout.close()

    def test_write_with_read_checkout(self, written_repo, array5by7):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            co.arraysets.init_arrayset(name='aset', shape=(5, 7), dtype=np.float64)
        with pytest.raises(AttributeError):
            co.metadata.add('a', 'b')
        co.close()

    def test_writer_aset_obj_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        asets = co.arraysets
        aset = co.arraysets['_aset']
        co.close()

        with pytest.raises(ReferenceError):
            asets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = asets['_aset']
        with pytest.raises(ReferenceError):
            aset.__dict__

    def test_writer_aset_obj_arrayset_iter_values_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        oldObjs = []
        for oldObj in co.arraysets.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(ReferenceError):
                oldObj.__dict__

    def test_writer_aset_obj_arrayset_iter_items_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        oldObjs = {}
        for oldName, oldObj in co.arraysets.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(ReferenceError):
                obj.__dict__

    def test_writer_aset_obj_not_accessible_after_commit_and_close(self, written_repo, array5by7):
        repo = written_repo
        co = repo.checkout(write=True)
        asets = co.arraysets
        aset = co.arraysets['_aset']
        aset['1'] = array5by7
        co.commit('hey there')
        co.close()

        with pytest.raises(ReferenceError):
            asets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = asets['_aset']
        with pytest.raises(ReferenceError):
            aset.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = aset['1']

    def test_reader_aset_obj_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=False)
        asets = co.arraysets
        aset = co.arraysets['_aset']
        co.close()

        with pytest.raises(ReferenceError):
            asets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = asets['_aset']
        with pytest.raises(ReferenceError):
            aset.__dict__

    def test_reader_aset_obj_arrayset_iter_values_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=False)
        oldObjs = []
        for oldObj in co.arraysets.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(ReferenceError):
                oldObj.__dict__

    def test_reader_aset_obj_arrayset_iter_items_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=False)
        oldObjs = {}
        for oldName, oldObj in co.arraysets.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(ReferenceError):
                obj.__dict__

    def test_reader_arrayset_context_manager_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=False)
        aset = co.arraysets['_aset']
        klist = []
        with aset as ds:
            for k in ds.keys():
                klist.append(k)
                a = ds
        co.close()

        with pytest.raises(ReferenceError):
            a.__dict__
        with pytest.raises(ReferenceError):
            ds.__dict__
        with pytest.raises(ReferenceError):
            aset[klist[0]]

    def test_writer_arrayset_context_manager_not_accessible_after_close(self, written_two_cmt_repo):
        repo = written_two_cmt_repo
        co = repo.checkout(write=True)
        aset = co.arraysets['_aset']
        with aset as ds:
            # for k in ds.keys():
            #     klist.append(k)
            a = ds
            a['1232'] = np.random.randn(5, 7).astype(np.float32)
        co.close()

        with pytest.raises(ReferenceError):
            a.__dict__
        with pytest.raises(ReferenceError):
            ds.__dict__
        with pytest.raises(ReferenceError):
            aset['1232']

    def test_writer_metadata_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        md = co.metadata
        co.close()

        with pytest.raises(ReferenceError):
            md.__dict__

    def test_writer_metadata_obj_not_accessible_after_commit_and_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        md = co.metadata
        md['hello'] = 'world'
        co.commit('test commit')
        co.close()

        with pytest.raises(ReferenceError):
            md.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = md['hello']

    def test_reader_metadata_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        md = co.metadata
        co.close()
        with pytest.raises(ReferenceError):
            md.__dict__

    def test_close_read_does_not_invalidate_write_checkout(self, written_repo, array5by7):
        repo = written_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)
        r_co.close()

        with pytest.raises(PermissionError):
            shouldFail = r_co.arraysets

        aset = w_co.arraysets['_aset']
        aset['1'] = array5by7
        assert np.allclose(w_co.arraysets['_aset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            aset.__dict__

    def test_close_write_does_not_invalidate_read_checkout(self, written_repo, array5by7):
        repo = written_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)

        aset = w_co.arraysets['_aset']
        aset['1'] = array5by7
        assert np.allclose(w_co.arraysets['_aset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            aset.__dict__

        assert '_aset' in r_co.arraysets
        assert len(r_co.metadata) == 0

        r_co.close()
        with pytest.raises(PermissionError):
            r_co.arraysets

    def test_operate_on_arrayset_after_closing_old_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout(write=True)
        with pytest.raises(ReferenceError):
            aset.add(array5by7, '1')
            co.commit('this is a commit message')
        co.close()
        with pytest.raises(ReferenceError):
            aset['1']

    def test_operate_on_closed_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        co.arraysets.init_arrayset('aset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        with pytest.raises(PermissionError):
            co.arraysets['aset']['1'] = array5by7
        with pytest.raises(PermissionError):
            co.metadata.add('a', 'b')

    @pytest.mark.parametrize("aset_backend", ['00', '10'])
    def test_operate_on_arrayset_samples_after_commiting_but_not_closing_checkout(self, aset_backend, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=array5by7, backend=aset_backend)
        aset.add(array5by7, '1')
        co.commit('hi')

        aset.add(array5by7, '2')  # this raises ReferenceError since the reference to aset is gone
        co.commit('hello 2')
        assert np.allclose(aset['2'], array5by7)
        co.close()

        with pytest.raises(ReferenceError):
            aset.name

    def test_operate_on_metadata_after_commiting_but_not_closing_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        md = co.metadata
        md['hello'] = 'world'
        co.commit('hi')

        md['foo'] = 'bar'
        co.commit('hello 2')
        assert md.get('hello') == 'world'
        assert md['foo'] == 'bar'
        co.close()

        with pytest.raises(ReferenceError):
            md.get('foo')
        with pytest.raises(ReferenceError):
            md['hello']

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    def test_operate_on_arraysets_after_commiting_but_not_closing_checkout(self, aset1_backend, aset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        asets = co.arraysets
        aset = co.arraysets.init_arrayset('aset', prototype=array5by7, backend=aset1_backend)
        aset.add(array5by7, '1')
        co.commit('hi')

        aset2 = co.arraysets.init_arrayset('arange', prototype=np.arange(50), backend=aset2_backend)
        aset2['0'] = np.arange(50)
        co.commit('hello 2')
        assert np.allclose(aset2['0'], np.arange(50))
        co.close()

        with pytest.raises(PermissionError):
            co.arraysets
        with pytest.raises(ReferenceError):
            asets.iswriteable
        with pytest.raises(ReferenceError):
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

    @pytest.mark.parametrize("aset1_backend", ['00', '10'])
    @pytest.mark.parametrize("aset2_backend", ['00', '10'])
    def test_reset_staging_area_clears_arraysets(self, aset1_backend, aset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        aset = co.arraysets.init_arrayset('aset', prototype=array5by7, backend=aset1_backend)
        aset.add(array5by7, '1')
        co.commit('hi')

        aset2 = co.arraysets.init_arrayset('arange', prototype=np.arange(50), backend=aset2_backend)
        aset2['0'] = np.arange(50)
        # verifications before reset
        assert np.allclose(aset2['0'], np.arange(50))
        assert len(co.arraysets) == 2
        assert co.arraysets['arange'].iswriteable

        co.reset_staging_area()
        # behavior expected after reset
        assert len(co.arraysets) == 1
        with pytest.raises(ReferenceError):
            aset2['0']
        with pytest.raises(KeyError):
            co.arraysets['arange']
        co.close()

    def test_reset_staging_area_clears_metadata(self, repo):
        co = repo.checkout(write=True)
        md = co.metadata
        md['hello'] = 'world'
        co.commit('hi')

        md['foo'] = 'bar'
        co.metadata['bar'] = 'baz'
        # verifications before reset
        assert len(co.metadata) == 3
        assert len(md) == 3
        assert co.metadata['hello'] == 'world'
        assert co.metadata['foo'] == 'bar'
        assert co.metadata['bar'] == 'baz'
        assert md['foo'] == 'bar'

        co.reset_staging_area()
        # behavior expected after reset
        assert len(co.metadata) == 1
        assert co.metadata['hello'] == 'world'
        with pytest.raises(ReferenceError):
            assert len(md) == 1
        with pytest.raises(ReferenceError):
            assert md['hello']
        with pytest.raises(KeyError):
            co.metadata['foo']
        with pytest.raises(KeyError):
            co.metadata['bar']
        co.close()

class TestBranching(object):

    def test_invalid_name(self, repo):
        with pytest.raises(ValueError):
            repo.create_branch('dummy branch')

    def test_cannot_create_new_branch_from_new_repo(self, repo):
        with pytest.raises(RuntimeError):
            repo.create_branch('testbranch')

    def test_merge(self, written_repo, array5by7):
        branch = written_repo.create_branch('testbranch')
        assert type(branch) is str
        co = written_repo.checkout(write=True, branch=branch)
        assert co._branch_name == branch
        co.arraysets['_aset']['1'] = array5by7
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        co.close()
        written_repo.merge('test merge', 'master', branch)
        co = written_repo.checkout()
        assert (co.arraysets['_aset']['1'] == array5by7).all()
        assert co.metadata.get('a') == 'b'
        co.close()

    def test_merge_without_closing_previous_checkout(self, written_repo, array5by7):
        branch = written_repo.create_branch('testbranch')
        co = written_repo.checkout(write=True, branch=branch)
        co.arraysets['_aset']['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(PermissionError):
            written_repo.merge('test merge', 'master', branch)
        # unregister close operation as conftest will close env before this is called.
        atexit.unregister(co.close)

    def test_name_conflict(self, written_repo, array5by7):
        written_repo.create_branch('testbranch')
        with pytest.raises(ValueError):
            written_repo.create_branch('testbranch')

    def test_delete_branch(self, written_repo):
        with pytest.raises(NotImplementedError):
            # TODO should raise "branch doesn't exist error"
            written_repo.remove_branch('testbranch')
        written_repo.create_branch('testbranch')
        with pytest.raises(NotImplementedError):
            written_repo.remove_branch('testbranch')

    def test_merge_multiple_checkouts_same_aset(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        co = written_repo.checkout(write=True, branch=branch1)
        co.arraysets['_aset']['1'] = array5by7
        co.metadata.add('a1', 'b1')
        co.commit('this is a commit message')
        co.close()

        branch2 = written_repo.create_branch('testbranch2')
        co = written_repo.checkout(write=True, branch=branch2)
        co.arraysets['_aset']['2'] = array5by7
        co.metadata.add('a2', 'b2')
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('test merge 1', 'master', branch1)
        written_repo.merge('test merge 2', 'master', branch2)

        co = written_repo.checkout(branch='master')
        assert len(co.arraysets) == 1
        assert len(co.arraysets['_aset']) == 2
        assert list(co.metadata.keys()) == ['a1', 'a2']
        co.close()

    def test_merge_multiple_checkouts_multiple_aset(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        co = written_repo.checkout(write=True, branch=branch1)
        co.arraysets['_aset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        branch2 = written_repo.create_branch('testbranch2')
        co = written_repo.checkout(write=True, branch=branch2)
        second_aset = co.arraysets.init_arrayset(name='second_aset', prototype=array5by7)
        second_aset['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('test merge 1', 'master', branch1)
        written_repo.merge('test merge 2', 'master', branch2)

        co = written_repo.checkout(branch='master')
        assert len(co.arraysets) == 2
        assert len(co.arraysets['_aset']) == 1
        assert len(co.arraysets['second_aset']) == 1
        co.close()

    def test_merge_diverged_conflict(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')

        co = written_repo.checkout(write=True, branch=branch1)
        co.arraysets['_aset']['1'] = array5by7
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True, branch=branch2)
        newarray = np.zeros_like(array5by7)
        co.arraysets['_aset']['1'] = newarray
        co.metadata.add('a', 'c')
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('commit message', 'master', branch1)

        with pytest.raises(ValueError):
            written_repo.merge('commit message', 'master', branch2)

    def test_new_branch_from_where(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')
        co1 = written_repo.checkout(write=True, branch=branch1)
        h1 = written_repo.log(branch=co1.branch_name, return_contents=True)
        co1.close()

        co2 = written_repo.checkout(write=True, branch=branch2)
        co2.arraysets.init_arrayset('aset2', prototype=array5by7)
        co2.arraysets['aset2']['2'] = array5by7
        co2.commit('this is a merge message')
        co2.close()
        h2 = written_repo.log(branch=branch2, return_contents=True)

        branch3 = written_repo.create_branch('testbranch3')
        co3 = written_repo.checkout(write=True, branch=branch3)
        h3 = written_repo.log(branch=co3.branch_name, return_contents=True)
        co3.close()

        assert h2['head'] == h3['head']
        assert h2['ancestors'][h2['head']] == h3['ancestors'][h3['head']]
        assert h1['head'] in h2['ancestors'][h2['head']]

    def test_cannot_checkout_branch_with_staged_changes(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')
        co1 = written_repo.checkout(write=True, branch=branch1)
        initial_cmt = co1.commit_hash
        co1.arraysets.init_arrayset('aset2', prototype=array5by7)
        co1.arraysets['aset2']['2'] = array5by7
        co1.close()

        with pytest.raises(ValueError):
            con = written_repo.checkout(write=True, branch=branch2)

        co1 = written_repo.checkout(write=True, branch=branch1)
        co1.commit('hi')
        assert co1.commit_hash != initial_cmt
        assert co1.branch_name == branch1
        co1.close()

        co2 = written_repo.checkout(write=True, branch=branch2)
        assert co2.branch_name == branch2
        assert co2.commit_hash == initial_cmt
        co2.close()


def test_full_from_short_commit_digest(written_two_cmt_repo):
    from hangar.records.commiting import expand_short_commit_digest

    repo = written_two_cmt_repo
    history = repo.log(branch='master', return_contents=True)
    commits = history['order']
    for full_cmt in commits:
        short_cmt = full_cmt[:18]
        found_cmt = expand_short_commit_digest(repo._env.refenv, short_cmt)
        assert found_cmt == full_cmt

    with pytest.raises(KeyError):
        expand_short_commit_digest(repo._env.refenv, 'zzzzzzzzzzzzzzzzzzzzzzzzzzzz')


def test_writer_context_manager_objects_are_gc_removed_after_co_close(written_two_cmt_repo):

    repo = written_two_cmt_repo
    co = repo.checkout(write=True)
    with co.metadata as m:
        m['aa'] = 'bb'
        cmt1 = co.commit('here is the first commit')
        with co.arraysets['_aset'] as d:
            d['2422'] = d['0'] + 213
            cmt2 = co.commit('here is the second commit')

    assert co.close() is None
    with pytest.raises(ReferenceError):
        m.__dict__
    with pytest.raises(ReferenceError):
        d.__dict__
    with pytest.raises(PermissionError):
        co.arraysets
    assert co.__dict__ == {}

    co = repo.checkout(commit=cmt1)
    assert 'aa' in co.metadata
    assert co.metadata['aa'] == 'bb'
    co.close()

    co = repo.checkout(commit=cmt2)
    assert 'aa' in co.metadata
    assert co.metadata['aa'] == 'bb'
    assert '2422' in co.arraysets['_aset']
    assert np.allclose(co.arraysets['_aset']['2422'],
                       co.arraysets['_aset']['0'] + 213)
    co.close()


def test_reader_context_manager_objects_are_gc_removed_after_co_close(written_two_cmt_repo):

    repo = written_two_cmt_repo
    co = repo.checkout(write=False)
    with co.metadata as m:
        k = list(m.keys())
        with co.arraysets['_aset'] as d:
            ds = d['2']

    assert m.iswriteable is False
    assert d.iswriteable is False
    assert k == list(m.keys())
    assert k == list(co.metadata.keys())
    assert np.allclose(ds, d.get('2'))
    assert np.allclose(ds, co.arraysets['_aset'].get('2'))

    assert co.close() is None

    with pytest.raises(ReferenceError):
        m.__dict__
    with pytest.raises(ReferenceError):
        d.__dict__
    with pytest.raises(AttributeError):
        co._arraysets
    with pytest.raises(AttributeError):
        co._metadata
    with pytest.raises(PermissionError):
        str(co.arraysets.get('_aset'))
    with pytest.raises(PermissionError):
        repr(co.metadata)
    with pytest.raises(PermissionError):
        co.arraysets
    with pytest.raises(PermissionError):
        repr(co)
    assert co.__dict__ == {}
