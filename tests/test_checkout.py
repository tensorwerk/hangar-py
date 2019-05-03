import numpy as np
import pytest
import shutil


class TestCheckout(object):

    def test_two_write_checkouts(self, repo):
        w1_checkout = repo.checkout(write=True)
        with pytest.raises(PermissionError):
            w2_checkout = repo.checkout(write=True)
        w1_checkout.close()

    def test_two_read_checkouts(self, repo, array5by7):
        w_checkout = repo.checkout(write=True)
        dataset_name = 'dset'
        r_ds = w_checkout.datasets.init_dataset(name=dataset_name, prototype=array5by7)
        r_ds['1'] = array5by7
        w_checkout.commit('init')
        r1_checkout = repo.checkout()
        r2_checkout = repo.checkout()
        assert np.allclose(r1_checkout.datasets['dset']['1'], array5by7)
        assert np.allclose(r1_checkout.datasets['dset']['1'], r2_checkout.datasets['dset']['1'])
        r1_checkout.close()
        r2_checkout.close()
        w_checkout.close()

    def test_write_with_r_checkout(self, written_repo, array5by7):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            co.datasets.init_dataset(name='dset', shape=(5, 7), dtype=np.float64)
            co.datasets['_dset']['1'] = array5by7
            co.datasets['_dset'].add('1', array5by7)
        co.close()

    def test_write_after_repo_deletion(self, written_repo, array5by7):
        co = written_repo.checkout(write=True)
        shutil.rmtree(written_repo._repo_path)
        with pytest.raises(OSError):
            co.datasets.init_dataset('dset', prototype=array5by7)
        dset = co.datasets['_dset']
        dset['1'] = array5by7
        # todo: shouldn't it be atleast same as above
        with pytest.raises(FileNotFoundError):
            co.commit()
        co.close()


@pytest.mark.skip(reason='not implemented')
class TestBranching(object):

    def test_merge(self):
        pass

    def test_name_conflict(self):
        pass

    def test_delete_branch(self):
        pass

    def test_merge_multiple_checkouts(self):
        pass

    def test_merge_diverged(self):
        pass

    def test_merge_diverged_conflict(self):
        pass


class TestTimeTravel(object):

    def test_merge_data_removal(self, written_repo, array5by7, get_history):
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['1'] = array5by7
        co.datasets['_dset']['2'] = array5by7
        co.metadata.add('a1', 'b1')
        co.metadata.add('a2', 'b2')
        co.commit('Commit with two data point')
        co.close()
        branch = written_repo.create_branch('testbranch')
        co = written_repo.checkout(write=True, branch_name=branch)
        co.datasets['_dset'].remove('2')
        co.metadata.remove('a2')
        co.commit('removing one data point')
        co.close()
        written_repo.merge('removal test merge', 'master', branch)
        co = written_repo.checkout()
        assert len(co.datasets['_dset']) == 1
        assert '1' in co.datasets['_dset'].keys()
        assert list(co.metadata.keys()) == ['a1']

        # TODO: better way to get history? right now, only possible
        # with write checkout?
        co = written_repo.checkout(write=True)
        history = get_history(co._refenv, co._branchenv)
        co.close()
        zeroth_commit = history['order'][2]
        first_commit = history['order'][1]
        second_commit = history['order'][0]
        zeroth_co = written_repo.checkout(commit=zeroth_commit)
        first_co = written_repo.checkout(commit=first_commit)
        second_co = written_repo.checkout(commit=second_commit)
        assert len(zeroth_co.datasets['_dset']) == 0
        assert len(first_co.datasets['_dset']) == 2
        assert len(second_co.datasets['_dset']) == 1
        assert np.allclose(
            first_co.datasets['_dset']['1'],
            second_co.datasets['_dset']['1'])
        with pytest.raises(KeyError):
            second_co.datasets['_dset']['2']
        with pytest.raises(KeyError):
            zeroth_co.datasets['_dset']['1']

        assert second_co.metadata['a1'] == 'b1'
        assert first_co.metadata['a1'] == 'b1'
        assert first_co.metadata['a2'] == 'b2'
        with pytest.raises(KeyError):
            second_co.metadata['a2']
        with pytest.raises(KeyError):
            second_co.metadata.get('a2')

    def test_remove_and_add_dset(self, written_repo, array5by7, get_history):
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['1'] = array5by7
        co.datasets['_dset']['2'] = array5by7
        co.metadata.add('a1', 'b1')
        co.commit()
        co.close()
        co = written_repo.checkout(write=True)
        co.datasets.remove_dset('_dset')
        co.commit()
        co.close()
        co = written_repo.checkout(write=True)
        dset = co.datasets.init_dataset('_dset', prototype=array5by7)
        dset['1'] = np.zeros_like(array5by7)
        co.commit()
        history = get_history(co._refenv, co._branchenv)
        co.close()
        second_commit = history['order'][2]
        third_commit = history['order'][1]
        fourth_commit = history['order'][0]
        second_co = written_repo.checkout(commit=second_commit)
        third_co = written_repo.checkout(commit=third_commit)
        fourth_co = written_repo.checkout(commit=fourth_commit)
        assert list(second_co.datasets['_dset'].keys()) == ['1', '2']
        assert np.allclose(second_co.datasets['_dset']['1'], array5by7)
        with pytest.raises(KeyError):
            third_co.datasets['_dset']
        assert np.allclose(fourth_co.datasets['_dset']['1'], np.zeros_like(array5by7))

    def test_3way_merges_with_mutation(self, written_repo, array5by7, get_history):
        branch1 = written_repo.create_branch('testbranch1')
        co = written_repo.checkout(write=True)
        co.datasets['_dset']['1'] = array5by7
        first_commit = co.commit()
        co.close()
        branch2 = written_repo.create_branch('testbranch2')
        co = written_repo.checkout(write=True, branch_name=branch1)
        co.datasets['_dset']['2'] = np.ones_like(array5by7) * 4
        second_commit = co.commit()
        co.close()
        branch3 = written_repo.create_branch('testbranch3')
        co = written_repo.checkout(write=True, branch_name=branch3)
        tmparray = co.datasets['_dset']['2']
        tmparray[3] *= 2
        co.datasets['_dset']['2'] = tmparray
        third_commit = co.commit()
        co.close()
        co = written_repo.checkout(write=True, branch_name=branch2)
        co.datasets['_dset'].remove('1')
        fourth_commit = co.commit()
        co.close()
        merge_commit1 = written_repo.merge('zeroth and first', 'master', branch1)
        merge_commit2 = written_repo.merge('data deletion branch', 'master', branch2)
        # TODO: This is sort of a merge conflict. The error should say something else
        with pytest.raises(KeyError):
            merge_commit3 = written_repo.merge('data mutation', 'master', branch3)
        co = written_repo.checkout(write=True)
        history = get_history(co._refenv, co._branchenv)
        co.close()
        # TODO: This is what needs to be tested once the above TODO is fixed
        # assert history['order'][:-1] == [
        #     merge_commit3, merge_commit2, merge_commit1,
        #     fourth_commit, third_commit, second_commit, first_commit]
