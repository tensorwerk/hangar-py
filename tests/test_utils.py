import pytest


@pytest.mark.parametrize('arg,key,expected', [
    ['AAABBBCCC', None, ['A', 'B', 'C']],
    ['AAABbBCcC', str.lower, ['A', 'B', 'C']],
    ['ABACBACDA', None, ['A', 'B', 'C', 'D']],
    ['ABacBaCAd', str.upper, ['A', 'B', 'c', 'd']],
])
def test_unique_everseen(arg, key, expected):
    from hangar.utils import unique_everseen

    res = list(unique_everseen(arg, key=key))
    assert res == expected


@pytest.mark.parametrize('pth', [pytest.File, None, 123])
def test_valid_directory_path_errors_on_invalid_path_arg(pth):
    from hangar.utils import is_valid_directory_path
    with pytest.raises(TypeError, match='Path arg `p`'):
        is_valid_directory_path(pth)


def test_valid_directory_path_recognizes_not_a_directory(managed_tmpdir):
    from hangar.utils import is_valid_directory_path
    from pathlib import Path

    test_pth = Path(managed_tmpdir, 'test.txt').resolve()
    with test_pth.open('w+') as f:
        f.write('hello')
    with pytest.raises(NotADirectoryError):
        is_valid_directory_path(test_pth)


@pytest.mark.parametrize('arg,expected', [
    [1, '1.00 B'],
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
    [0, 2],
    [1, 2],
    [2, 2],
    [3, 3],
    [4, 5],
    [7, 7],
    [174, 179],
    [10065, 10067],
    [104721, 104723],
])
def test_find_next_prime(arg, expected):
    from hangar.utils import find_next_prime

    res = find_next_prime(arg)
    assert res == expected
