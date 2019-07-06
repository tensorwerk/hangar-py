import pytest


@pytest.mark.parametrize('arg,expected', [
    [1, '1 B'],
    [1234, '1.23 kB'],
    [12345678, '12.35 MB'],
    [1234567890, '1.23 GB'],
    [1234567890000, '1.23 TB'],
    [1234567890000000, '1.23 PB']
])
def test_format_bytes(arg, expected):
    from hangar.utils import format_bytes

    res = format_bytes(arg)
    assert res == expected


@pytest.mark.parametrize('arg,expected', [
    ['100', 100],
    ['100 MB', 100000000],
    ['100M', 100000000],
    ['5kB', 5000],
    ['5.4 kB', 5400],
    ['1kiB', 1024],
    ['1e6', 1000000],
    ['1e6 kB', 1000000000],
    ['MB', 1000000]
])
def test_parse_bytes(arg, expected):
    from hangar.utils import parse_bytes

    res = parse_bytes(arg)
    assert res == expected


@pytest.mark.parametrize('arg,expected', [
    ['3s', 3],
    ['3.5 seconds', 3.5],
    ['300ms', 0.3]
])
def test_parse_timedelta(arg, expected):
    from hangar.utils import parse_timedelta

    res = parse_timedelta(arg)
    assert res == expected
