import pytest


class TestMetadata(object):

    def test_invalid_name(self, w_checkout):
        # TODO: It's a value error in dataset and branch
        with pytest.raises(SyntaxError):
            w_checkout.metadata.add('a a', 'b')

    def test_name_conflict(self, w_checkout):
        w_checkout.metadata.add('a', 'b')
        w_checkout.commit('tehis is a merge message')
        # TODO: shouldn't it raise a warning atleast?
        w_checkout.metadata.add('a', 'c')
        w_checkout.commit('second time')
        assert w_checkout.metadata.get('a') == 'c'

    def test_other_datatype(self, w_checkout):
        # TODO: should raise a proper error
        with pytest.raises(AttributeError):
            w_checkout.metadata.add(1, '1')
        with pytest.raises(AttributeError):
            w_checkout.metadata.add('1', 1)

    def test_read_only_mode(self, written_repo):
        co = written_repo.checkout()
        # TODO: should be permissionerror
        with pytest.raises(AttributeError):
            co.metadata.add('a', 'b')

    def test_add_remove_without_enough_args(self, repo):
        co = repo.checkout(write=True)
        with pytest.raises(TypeError):
            co.metadata.add('1')
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        with pytest.raises(TypeError):
            co.metadata.remove()
        co.metadata.remove('a')
        co.commit('this is a commit message')
        assert list(co.metadata.keys()) == []

    def test_get_and_remove_wrong_key(self, repo):
        co = repo.checkout(write=True)
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        with pytest.raises(KeyError):
            co.metadata.get('randome')
        with pytest.raises(KeyError):
            co.metadata.remove('randome')

    def test_loop_through(self, repo):
        co = repo.checkout(write=True)
        limit = 10
        for i in range(limit):
            co.metadata.add(f'k_{i}', f'v_{i}')
        co.commit('this is a commit message')
        co.close()
        co = repo.checkout()
        for i, (k, v) in enumerate(co.metadata.items()):
            assert k == f'k_{i}'
            assert v == f'v_{i}'
        for i, k in enumerate(co.metadata.keys()):
            assert co.metadata[k] == f'v_{i}'
        for i, k in enumerate(co.metadata):
            assert co.metadata[k] == f'v_{i}'
        for i, v in enumerate(co.metadata.values()):
            assert co.metadata[f'k_{i}'] == v
        for i in range(limit):
            assert f'k_{i}' in co.metadata
