import pytest

from hangar.optimized_utils import SizedDict


def test_sizeddict_maxsize_property():
    d = SizedDict(maxsize=5)
    assert d.maxsize == 5
    d2 = SizedDict(maxsize=10)
    assert d2.maxsize == 10


def test_sizeddict_setitem_no_overflow_retains_keys_and_len():
    d = SizedDict(maxsize=5)
    for i in range(5):
        d[i] = i

    assert len(d) == 5
    for i in range(5):
        assert i in d
        assert d[i] == i


def test_sizeddict_setitem_overflow_truncates_keys_and_len():
    d = SizedDict(maxsize=5)
    for i in range(10):
        d[i] = i

    assert len(d) == 5
    for i in range(0, 5):
        assert i not in d
        with pytest.raises(KeyError):
            _ = d[i]
    for i in range(5, 10):
        assert i in d
        assert d[i] == i


def test_sizeddict_update_no_overflow_retains_keys_and_len():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    assert len(d) == 5
    for i in range(5):
        assert i in d
        assert d[i] == i


def test_sizeddict_updateoverflow_truncates_keys_and_len():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(10)}
    d.update(inp)

    assert len(d) == 5
    for i in range(0, 5):
        assert i not in d
        with pytest.raises(KeyError):
            _ = d[i]
    for i in range(5, 10):
        assert i in d
        assert d[i] == i


def test_sizeddict_get_returns_default_on_missing_key():
    d = SizedDict()
    res = d.get('doesnotexist')
    assert res is None
    res = d.get('doesnotexist', default='foo')
    assert res == 'foo'


def test_sizeddict_delitem():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    del d[3]
    assert len(d) == 4
    assert 3 not in d

    d['new'] = 'foo'
    assert len(d) == 5
    assert 'new' in d


def test_sizeddict_pop():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    res = d.pop(0)
    assert res == 0
    assert len(d) == 4
    res = d.pop('doesnotexist', default='foo')
    assert res == 'foo'
    assert len(d) == 4


def test_sizeddict_popitem():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    res = d.popitem()
    assert res == (4, 4)
    assert len(d) == 4
    res = d.popitem()
    assert res == (3, 3)
    assert len(d) == 3

    d['foo'] = 'bar'
    assert len(d) == 4
    res = d.popitem()
    assert res == ('foo', 'bar')
    assert len(d) == 3


def test_sizeddict_keys():
    d = SizedDict(maxsize=5)
    inp = {str(i): i for i in range(5)}
    d.update(inp)

    assert list(d.keys()) == list(inp.keys())
    for res_k, expected_k in zip(d.keys(), inp.keys()):
        assert res_k == expected_k


def test_sizeddict_values():
    d = SizedDict(maxsize=5)
    inp = {str(i): i for i in range(5)}
    d.update(inp)

    assert list(d.values()) == list(inp.values())
    for res_v, expected_v in zip(d.values(), inp.values()):
        assert res_v == expected_v


def test_sizeddict_keys():
    d = SizedDict(maxsize=5)
    inp = {str(i): i for i in range(5)}
    d.update(inp)

    assert list(d.items()) == list(inp.items())
    for res_kv, expected_kv in zip(d.items(), inp.items()):
        assert res_kv == expected_kv


def test_sizeddict_setdefault():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    res = d.setdefault('doesnotexist')
    assert res is None
    assert len(d) == 5
    assert 'doesnotexist' in d
    assert d['doesnotexist'] is None

    res = d.setdefault('doesnotexist2', default=True)
    assert res is True
    assert len(d) == 5
    assert 'doesnotexist2' in d
    assert d['doesnotexist2'] is True

    res = d.setdefault(2, default=True)
    assert res == 2
    assert len(d) == 5
    assert 2 in d
    assert d[2] == 2


def test_sizeddict_clear():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    assert len(d) == 5
    d.clear()
    assert len(d) == 0
    assert len(d._stack) == 0
    assert len(d._data) == 0
    assert d._stack_size == 0
    assert d._maxsize == 5


def test_sizeddict_repr():
    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    expected = repr(inp)
    res = repr(d)
    assert res == expected


def test_sizeddict_is_pickleable():
    import pickle

    d = SizedDict(maxsize=5)
    inp = {i: i for i in range(5)}
    d.update(inp)

    pick = pickle.dumps(d, protocol=pickle.HIGHEST_PROTOCOL)
    res = pickle.loads(pick)

    assert res._maxsize == d._maxsize
    assert res._stack == d._stack
    assert res._stack_size == d._stack_size
    assert res._data == d._data
