import numpy as np
import pytest
import shutil
import platform


class TestCheckout(object):

    def test_two_write_checkouts(self, repo):
        w1_checkout = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            repo.checkout(write=True)
        w1_checkout.close()

    def test_two_read_checkouts(self, repo, array5by7):
        w_checkout = repo.checkout(write=True)
        dataset_name = 'dset'
        r_ds = w_checkout.datasets.init_dataset(name=dataset_name, prototype=array5by7)
        r_ds['1'] = array5by7
        w_checkout.metadata.add('init', 'array5by7 added')
        w_checkout.commit('init')
        r1_checkout = repo.checkout()
        r2_checkout = repo.checkout()
        assert np.allclose(r1_checkout.datasets['dset']['1'], array5by7)
        assert np.allclose(
            r1_checkout.datasets['dset']['1'], r2_checkout.datasets['dset']['1'])
        assert r1_checkout.metadata.get('init') == 'array5by7 added'
        assert r2_checkout.metadata.get('init') == 'array5by7 added'
        r1_checkout.close()
        r2_checkout.close()
        w_checkout.close()

    def test_write_with_read_checkout(self, written_repo, array5by7):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            co.datasets.init_dataset(name='dset', shape=(5, 7), dtype=np.float64)
        with pytest.raises(AttributeError):
            co.metadata.add('a', 'b')
        co.close()

    def test_writer_dset_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        dsets = co.datasets
        dset = co.datasets['_dset']
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_writer_dset_obj_dataset_iter_values_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        oldObjs = []
        for oldObj in co.datasets.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(ReferenceError):
                oldObj.__dict__

    def test_writer_dset_obj_dataset_iter_items_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=True)
        oldObjs = {}
        for oldName, oldObj in co.datasets.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(ReferenceError):
                obj.__dict__

    def test_writer_dset_obj_not_accessible_after_commit_and_close(self, written_repo, array5by7):
        repo = written_repo
        co = repo.checkout(write=True)
        dsets = co.datasets
        dset = co.datasets['_dset']
        dset['1'] = array5by7
        co.commit('hey there')
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dset['1']

    def test_reader_dset_obj_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        dsets = co.datasets
        dset = co.datasets['_dset']
        co.close()

        with pytest.raises(ReferenceError):
            dsets.__dict__
        with pytest.raises(ReferenceError):
            shouldFail = dsets['_dset']
        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_reader_dset_obj_dataset_iter_values_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        oldObjs = []
        for oldObj in co.datasets.values():
            oldObjs.append(oldObj)
        co.close()

        for oldObj in oldObjs:
            with pytest.raises(ReferenceError):
                oldObj.__dict__

    def test_reader_dset_obj_dataset_iter_items_not_accessible_after_close(self, written_repo):
        repo = written_repo
        co = repo.checkout(write=False)
        oldObjs = {}
        for oldName, oldObj in co.datasets.items():
            oldObjs[oldName] = oldObj
        co.close()

        for name, obj in oldObjs.items():
            assert isinstance(name, str)
            with pytest.raises(ReferenceError):
                obj.__dict__

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
            shouldFail = r_co.datasets

        dset = w_co.datasets['_dset']
        dset['1'] = array5by7
        assert np.allclose(w_co.datasets['_dset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            dset.__dict__

    def test_close_write_does_not_invalidate_read_checkout(self, written_repo, array5by7):
        repo = written_repo
        r_co = repo.checkout(write=False)
        w_co = repo.checkout(write=True)

        dset = w_co.datasets['_dset']
        dset['1'] = array5by7
        assert np.allclose(w_co.datasets['_dset']['1'], array5by7)
        w_co.commit('hello commit')
        w_co.close()

        with pytest.raises(ReferenceError):
            dset.__dict__

        assert '_dset' in r_co.datasets
        assert len(r_co.metadata) == 0

        r_co.close()
        with pytest.raises(PermissionError):
            r_co.datasets

    def test_operate_on_dataset_after_closing_old_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout(write=True)
        with pytest.raises(ReferenceError):
            dset.add(array5by7, '1')
            co.commit('this is a commit message')
        co.close()
        with pytest.raises(ReferenceError):
            dset['1']

    def test_operate_on_closed_checkout(self, repo, array5by7):
        co = repo.checkout(write=True)
        co.datasets.init_dataset('dset', prototype=array5by7)
        co.commit('this is a commit message')
        co.close()
        with pytest.raises(PermissionError):
            co.datasets['dset']['1'] = array5by7
        with pytest.raises(PermissionError):
            co.metadata.add('a', 'b')

    @pytest.mark.parametrize("dset_backend", ['00', '01'])
    def test_operate_on_dataset_samples_after_commiting_but_not_closing_checkout(self, dset_backend, repo, array5by7):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=array5by7, backend=dset_backend)
        dset.add(array5by7, '1')
        co.commit('hi')

        dset.add(array5by7, '2')  # this raises ReferenceError since the reference to dset is gone
        co.commit('hello 2')
        assert np.allclose(dset['2'], array5by7)
        co.close()

        with pytest.raises(ReferenceError):
            dset.name

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

    @pytest.mark.parametrize("dset1_backend", ['00', '01'])
    @pytest.mark.parametrize("dset2_backend", ['00', '01'])
    def test_operate_on_datasets_after_commiting_but_not_closing_checkout(self, dset1_backend, dset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        dsets = co.datasets
        dset = co.datasets.init_dataset('dset', prototype=array5by7, backend=dset1_backend)
        dset.add(array5by7, '1')
        co.commit('hi')

        dset2 = co.datasets.init_dataset('arange', prototype=np.arange(50), backend=dset2_backend)
        dset2['0'] = np.arange(50)
        co.commit('hello 2')
        assert np.allclose(dset2['0'], np.arange(50))
        co.close()

        with pytest.raises(PermissionError):
            co.datasets
        with pytest.raises(ReferenceError):
            dsets.iswriteable
        with pytest.raises(ReferenceError):
            dset2.name

    def test_with_wrong_argument_value(self, repo):
        # It is intuitive to a user to pass branchname as positional
        # argument but hangar expect permission as first argument
        with pytest.raises(ValueError):
            repo.checkout('branchname')
        with pytest.raises(ValueError):
            repo.checkout(write='True')
        with pytest.raises(ValueError):
            repo.checkout(branch_name=True)
        repo.checkout(True)  # This should not raise any excpetion


    @pytest.mark.parametrize("dset1_backend", ['00', '01'])
    @pytest.mark.parametrize("dset2_backend", ['00', '01'])
    def test_reset_staging_area_clears_datasets(self, dset1_backend, dset2_backend, repo, array5by7):
        co = repo.checkout(write=True)
        dset = co.datasets.init_dataset('dset', prototype=array5by7, backend=dset1_backend)
        dset.add(array5by7, '1')
        co.commit('hi')

        dset2 = co.datasets.init_dataset('arange', prototype=np.arange(50), backend=dset2_backend)
        dset2['0'] = np.arange(50)
        # verifications before reset
        assert np.allclose(dset2['0'], np.arange(50))
        assert len(co.datasets) == 2
        assert co.datasets['arange'].iswriteable

        co.reset_staging_area()
        # behavior expected after reset
        assert len(co.datasets) == 1
        with pytest.raises(ReferenceError):
            dset2['0']
        with pytest.raises(KeyError):
            co.datasets['arange']
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
        co = written_repo.checkout(write=True, branch_name=branch)
        assert co._branch_name == branch
        co.datasets['_dset']['1'] = array5by7
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        co.close()
        written_repo.merge('test merge', 'master', branch)
        co = written_repo.checkout()
        assert (co.datasets['_dset']['1'] == array5by7).all()
        assert co.metadata.get('a') == 'b'

    def test_merge_without_closing_previous_checkout(self, written_repo, array5by7):
        branch = written_repo.create_branch('testbranch')
        co = written_repo.checkout(write=True, branch_name=branch)
        co.datasets['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        with pytest.raises(PermissionError):
            written_repo.merge('test merge', 'master', branch)

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

    def test_merge_multiple_checkouts_same_dset(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        co = written_repo.checkout(write=True, branch_name=branch1)
        co.datasets['_dset']['1'] = array5by7
        co.metadata.add('a1', 'b1')
        co.commit('this is a commit message')
        co.close()

        branch2 = written_repo.create_branch('testbranch2')
        co = written_repo.checkout(write=True, branch_name=branch2)
        co.datasets['_dset']['2'] = array5by7
        co.metadata.add('a2', 'b2')
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('test merge 1', 'master', branch1)
        written_repo.merge('test merge 2', 'master', branch2)

        co = written_repo.checkout(branch_name='master')
        assert len(co.datasets) == 1
        assert len(co.datasets['_dset']) == 2
        assert list(co.metadata.keys()) == ['a1', 'a2']

    def test_merge_multiple_checkouts_multiple_dset(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        co = written_repo.checkout(write=True, branch_name=branch1)
        co.datasets['_dset']['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        branch2 = written_repo.create_branch('testbranch2')
        co = written_repo.checkout(write=True, branch_name=branch2)
        second_dset = co.datasets.init_dataset(name='second_dset', prototype=array5by7)
        second_dset['1'] = array5by7
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('test merge 1', 'master', branch1)
        written_repo.merge('test merge 2', 'master', branch2)

        co = written_repo.checkout(branch_name='master')
        assert len(co.datasets) == 2
        assert len(co.datasets['_dset']) == 1
        assert len(co.datasets['second_dset']) == 1

    def test_merge_diverged_conflict(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')

        co = written_repo.checkout(write=True, branch_name=branch1)
        co.datasets['_dset']['1'] = array5by7
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        co.close()

        co = written_repo.checkout(write=True, branch_name=branch2)
        newarray = np.zeros_like(array5by7)
        co.datasets['_dset']['1'] = newarray
        co.metadata.add('a', 'c')
        co.commit('this is a commit message')
        co.close()

        written_repo.merge('commit message', 'master', branch1)

        with pytest.raises(ValueError):
            written_repo.merge('commit message', 'master', branch2)

    def test_new_branch_from_where(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')
        co1 = written_repo.checkout(write=True, branch_name=branch1)
        h1 = written_repo.log(branch_name=co1.branch_name, return_contents=True)
        co1.close()

        co2 = written_repo.checkout(write=True, branch_name=branch2)
        co2.datasets.init_dataset('dset2', prototype=array5by7)
        co2.datasets['dset2']['2'] = array5by7
        co2.commit('this is a merge message')
        co2.close()
        h2 = written_repo.log(branch_name=branch2, return_contents=True)

        branch3 = written_repo.create_branch('testbranch3')
        co3 = written_repo.checkout(write=True, branch_name=branch3)
        h3 = written_repo.log(branch_name=co3.branch_name, return_contents=True)
        co3.close()

        assert h2['head'] == h3['head']
        assert h2['ancestors'][h2['head']] == h3['ancestors'][h3['head']]
        assert h1['head'] in h2['ancestors'][h2['head']]

    def test_cannot_checkout_branch_with_staged_changes(self, written_repo, array5by7):
        branch1 = written_repo.create_branch('testbranch1')
        branch2 = written_repo.create_branch('testbranch2')
        co1 = written_repo.checkout(write=True, branch_name=branch1)
        initial_cmt = co1.commit_hash
        co1.datasets.init_dataset('dset2', prototype=array5by7)
        co1.datasets['dset2']['2'] = array5by7
        co1.close()

        with pytest.raises(ValueError):
            con = written_repo.checkout(write=True, branch_name=branch2)

        co1 = written_repo.checkout(write=True, branch_name=branch1)
        co1.commit('hi')
        assert co1.commit_hash != initial_cmt
        assert co1.branch_name == branch1
        co1.close()

        co2 = written_repo.checkout(write=True, branch_name=branch2)
        assert co2.branch_name == branch2
        assert co2.commit_hash == initial_cmt
        co2.close()


def test_full_from_short_commit_digest(written_two_cmt_repo):
    from hangar.records.commiting import expand_short_commit_digest

    repo = written_two_cmt_repo
    history = repo.log(branch_name='master', return_contents=True)
    commits = history['order']
    for full_cmt in commits:
        short_cmt = full_cmt[:18]
        found_cmt = expand_short_commit_digest(repo._env.refenv, short_cmt)
        assert found_cmt == full_cmt

    with pytest.raises(KeyError):
        expand_short_commit_digest(repo._env.refenv, 'zzzzzzzzzzzzzzzzzzzzzzzzzzzz')
