import pytest


class TestMetadata(object):

    @pytest.mark.parametrize('name', [
        'invalid\n', '\ninvalid', 'inv name', 'inva@lid',' trythis', 'andthis ',
        'VeryLongNameIsInvalidOver64CharactersNotAllowedVeryLongNameIsInva'])
    def test_writer_cannot_add_key_contains_whitespace(self, w_checkout, name):
        with pytest.raises(ValueError):
            w_checkout.metadata.add(name, 'b')

    def test_writer_add_can_overwrite_key_with_new_value(self, w_checkout):
        w_checkout.metadata.add('a', 'b')
        w_checkout.commit('this is a merge message')
        w_checkout.metadata.add('a', 'c')
        w_checkout.commit('second time')
        assert w_checkout.metadata.get('a') == 'c'

    def test_writer_add_requires_string_type_values(self, w_checkout):
        w_checkout.metadata.add('1', 'test')
        with pytest.raises(ValueError):
            w_checkout.metadata.add('1', 1)
        assert w_checkout.metadata.get('1') == 'test'
        assert list(w_checkout.metadata.keys()) == ['1']

    def test_writer_add_mixed_string_int_type_keys(self, w_checkout):
        w_checkout.metadata.add('1', 'test')
        w_checkout.metadata.add(1, 'test number')
        w_checkout.metadata['2'] = 'test2'
        w_checkout.metadata[2] = 'test2 number'

        assert w_checkout.metadata.get('1') == 'test'
        assert w_checkout.metadata.get(1) == 'test number'
        assert w_checkout.metadata['2'] == 'test2'
        assert w_checkout.metadata[2] == 'test2 number'
        assert list(w_checkout.metadata.keys()) == ['1', 1, '2', 2]

    def test_writer_remove_mixed_string_int_type_type_keys(self, w_checkout):
        w_checkout.metadata.add('1', 'test')
        w_checkout.metadata.add(2, 'test2')
        w_checkout.metadata[3] = 'test3'

        with pytest.raises(KeyError):
            w_checkout.metadata.remove(1)
        with pytest.raises(KeyError):
            del w_checkout.metadata['2']

        succ = w_checkout.metadata.remove('1')
        assert succ == '1'
        del w_checkout.metadata[2]

        assert w_checkout.metadata.get(3) == 'test3'
        assert list(w_checkout.metadata.keys()) == [3]

    def test_writer_dict_style_add_get_works(self, w_checkout):
        w_checkout.metadata['1'] = 'test'
        assert w_checkout.metadata['1'] == 'test'
        assert list(w_checkout.metadata.keys()) == ['1']

    def test_writer_dict_style_add_delete_works(self, w_checkout):
        w_checkout.metadata['1'] = 'test'
        w_checkout.metadata['2'] = 'test two'
        del w_checkout.metadata['2']
        with pytest.raises(KeyError):
            w_checkout.metadata['2']
        assert list(w_checkout.metadata.keys()) == ['1']

        w_checkout.commit('test commit')
        w_checkout.metadata['2'] = 'test two.two'
        del w_checkout.metadata['1']
        with pytest.raises(KeyError):
            w_checkout.metadata['1']
        assert w_checkout.metadata['2'] == 'test two.two'
        assert list(w_checkout.metadata.keys()) == ['2']

        w_checkout.commit('commit two')
        assert w_checkout.metadata['2'] == 'test two.two'
        assert list(w_checkout.metadata.keys()) == ['2']

    def test_writer_remove_requires_arguments(self, repo):
        co = repo.checkout(write=True)
        with pytest.raises(TypeError):
            co.metadata.add('1')
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        assert co.metadata['a'] == 'b'
        assert list(co.metadata.keys()) == ['a']

        with pytest.raises(TypeError):
            co.metadata.remove()
        assert co.metadata['a'] == 'b'
        assert list(co.metadata.keys()) == ['a']

        co.metadata.remove('a')
        co.commit('this is a commit message')
        assert list(co.metadata.keys()) == []
        co.close()

    def test_writer_get_does_not_succeed_if_key_does_not_exist(self, repo):
        co = repo.checkout(write=True)
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        with pytest.raises(KeyError):
            co.metadata.get('randome')
        co.close()

    def test_writer_remove_does_not_succeed_if_key_does_not_exist(self, repo):
        co = repo.checkout(write=True)
        co.metadata.add('a', 'b')
        co.commit('this is a commit message')
        with pytest.raises(KeyError):
            co.metadata.remove('randome')
        co.close()

    def test_writer_len_magic_works(self, repo):
        co = repo.checkout(write=True)
        co.metadata.add('a', 'b')
        co.metadata['1'] = '2'
        assert len(co.metadata) == 2
        del co.metadata['1']
        assert len(co.metadata) == 1
        del co.metadata['a']
        assert len(co.metadata) == 0
        co.close()

    def test_writer_contains_magic_works(self, repo):
        co = repo.checkout(write=True)
        co.metadata.add('a', 'b')
        co.metadata['1'] = '2'
        assert 'a' in co.metadata
        assert '1' in co.metadata
        assert 'foo' not in co.metadata
        co.close()

    def test_writer_iswriteable_property_is_true(self, repo):
        co = repo.checkout(write=True)
        assert co.metadata.iswriteable is True
        co.close()

    def test_reader_get_accepts_str_int_type_arguments(self, repo):
        w_checkout = repo.checkout(write=True)
        w_checkout.metadata.add('1', 'test')
        w_checkout.metadata.add(2, 'test2')

        w_checkout.commit('test commit')
        w_checkout.close()

        r_checkout = repo.checkout()
        with pytest.raises(KeyError):
            r_checkout.metadata.get(1)
        with pytest.raises(KeyError):
            r_checkout.metadata['2']

        assert r_checkout.metadata.get('1') == 'test'
        assert r_checkout.metadata[2] == 'test2'
        assert list(r_checkout.metadata.keys()) == [2, '1']
        r_checkout.close()

    def test_reader_dict_style_get_works(self, repo):
        w_checkout = repo.checkout(write=True)
        w_checkout.metadata.add('1', 'test')
        w_checkout.commit('test commit')
        w_checkout.close()

        r_checkout = repo.checkout()
        assert r_checkout.metadata['1'] == 'test'
        assert list(r_checkout.metadata.keys()) == ['1']
        r_checkout.close()

    def test_reader_add_not_permitted(self, written_repo):
        co = written_repo.checkout()
        with pytest.raises(AttributeError):
            co.metadata.add('a', 'b')
        co.close()

    def test_reader_dict_style_add_not_permitted(self, written_repo):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            co.metadata['a'] = 'b'
        co.close()

    def test_reader_remove_not_permitted(self, written_repo):
        co = written_repo.checkout()
        with pytest.raises(TypeError):
            del co.metadata['a']
        co.close()

    def test_reader_len_magic_works(self, repo):
        wco = repo.checkout(write=True)
        wco.metadata.add('a', 'b')
        wco.metadata['1'] = '2'
        wco.commit('test commit')
        wco.close()

        rco = repo.checkout()
        assert len(rco.metadata) == 2
        rco.close()

    def test_reader_contains_magic_works(self, repo):
        wco = repo.checkout(write=True)
        wco.metadata.add('a', 'b')
        wco.metadata['1'] = '2'
        wco.commit('test commit')
        wco.close()

        rco = repo.checkout()
        assert 'a' in rco.metadata
        assert '1' in rco.metadata
        assert 'foo' not in rco.metadata
        rco.close()

    def test_reader_iswriteable_property_is_false(self, written_repo):
        co = written_repo.checkout(write=False)
        assert co.metadata.iswriteable is False
        co.close()

    def test_metadata_correct_after_mutating_same_key_in_multiple_commits(self, repo):

        cmt_hashs = []
        co = repo.checkout(write=True)
        for idx in range(10):
            co.metadata['key'] = f'value {idx}'
            cmt_hashs.append(co.commit(f'this is commit {idx}'))
        co.close()

        for idx, cmt in enumerate(cmt_hashs):
            rco = repo.checkout(write=False, commit=cmt)
            assert rco.metadata['key'] == f'value {idx}'
            assert list(rco.metadata.keys()) == ['key']
            rco.close()

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
        co.close()


def test_get_multi_threading_pool(repo):
    from multiprocessing import dummy

    masterCmtList = []
    co = repo.checkout(write=True)
    masterSampKeyList = []
    masterSampValList = []
    for cIdx in range(2):
        if cIdx != 0:
            co = repo.checkout(write=True)
        with co.metadata as m:
            kstart = 500 * cIdx
            for sIdx in range(500):
                sName = str(sIdx + kstart)
                m[sName] = f'{cIdx}_{sIdx}'
                masterSampKeyList.append(sName)
                masterSampValList.append(f'{cIdx}_{sIdx}')
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, list(masterSampKeyList), list(masterSampValList)))
        co.close()

    for cmt, sampKeyList, sampValList in masterCmtList:
        nco = repo.checkout(write=False, commit=cmt)
        with dummy.Pool(2) as p:
            with nco.metadata as m:
                out = p.map(m.get, sampKeyList)
        for expected, received in zip(sampValList, out):
            assert expected == received
        nco.close()


def test_get_multi_process_pool_fails(repo):
    from multiprocessing import get_context

    masterCmtList = []
    co = repo.checkout(write=True)
    masterSampKeyList = []
    masterSampValList = []
    for cIdx in range(2):
        if cIdx != 0:
            co = repo.checkout(write=True)
        with co.metadata as m:
            kstart = 500 * cIdx
            for sIdx in range(500):
                sName = str(sIdx + kstart)
                m[sName] = f'{cIdx}_{sIdx}'
                masterSampKeyList.append(sName)
                masterSampValList.append(f'{cIdx}_{sIdx}')
        cmt = co.commit(f'master commit number: {cIdx}')
        masterCmtList.append((cmt, list(masterSampKeyList), list(masterSampValList)))
        co.close()

    for cmt, sampKeyList, sampValList in masterCmtList:
        nco = repo.checkout(write=False, commit=cmt)
        with pytest.raises(TypeError):
            with get_context('spawn').Pool(2) as p:
                out = p.map(nco.metadata.get, sampKeyList)
        with nco.metadata as m:
            for idx, k in enumerate(sampKeyList):
                out = m.get(k)
                assert out == sampValList[idx]
        nco.close()