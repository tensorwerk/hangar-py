import logging
import os
import random
import re
import string
import weakref
from functools import partial
from typing import Union, Any

import blosc
import wrapt

logger = logging.getLogger(__name__)


def set_blosc_nthreads() -> int:
    '''set the blosc library to two less than the core count on the system.

    If less than 2 cores are ncores-2, we set the value to two.

    Returns
    -------
    int
        ncores blosc will use on the system
    '''
    nCores = blosc.detect_number_of_cores()
    if nCores <= 2:
        nUsed = 1
    elif nCores <= 4:
        nUsed = nCores - 1
    else:
        nUsed = nCores - 2
    blosc.set_nthreads(nUsed)
    return nUsed


def random_string(n: int = 6) -> str:
    '''Generate a case random string of ascii letters and digits of some length.

    Parameters
    ----------
    n: int, optional
        The number of characters which the output string will have. Default = 6
    '''
    letters = ''.join([string.ascii_letters, string.digits])
    return ''.join(random.choice(letters) for i in range(n))


def cm_weakref_obj_proxy(obj: Any) -> wrapt.ObjectProxy:
    '''Creates a weakproxy reference honoring optional use context managers.

    This is required because (for some unknown reason) `weakproxy`
    references will not actually pass through the `__enter__` attribute of
    the reffered object's instance. As such, these are manual set to the
    appropriate values on the `weakproxy` object. The final `weakproxy` is
    in turn referenced by a object proxy so that all calls to the
    methods/attributes are passed through uniformly.

    Parameters
    ----------
    obj: Any
        object instance implementing the __enter__ and __exit__ methods which
        should be passed through as a weakref proxy object

    Returns
    -------
    ObjectProxy
        object analogous to a plain weakproxy object.
    '''
    wr = weakref.proxy(obj)
    setattr(wr, '__enter__', partial(obj.__class__.__enter__, wr))
    setattr(wr, '__exit__', partial(obj.__class__.__exit__, wr))
    obj_proxy = wrapt.ObjectProxy(wr)
    return obj_proxy


def symlink_rel(src: os.PathLike, dst: os.PathLike):
    '''Create symbolic links which actually work like they should

    Parameters
    ----------
    src : os.PathLike
        create a symbolic link pointic to src
    dst : os.PathLike
        create a link named dst
    '''
    rel_path_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_path_src, dst)


_SuitableCharRE = re.compile(r'[\w\.\-\_]+$', flags=re.ASCII)


def is_suitable_user_key(key: Union[str, int]) -> bool:
    '''Checks if string contains only alpha-numeric ascii chars or ['.', '-' '_'] (no whitespace)

    Necessary because python 3.6 does not have a str.isascii() method.

    Parameters
    ----------
    key : Union[str, int]
        string to check if it contains only ascii characters

    Returns
    -------
    bool
        True if only ascii characters in the string, else False.
    '''
    try:
        if isinstance(key, int) and (key >= 0):
            str_data = str(key)
        elif isinstance(key, str):
            str_data = str(key)
        else:
            raise TypeError
        return bool(_SuitableCharRE.match(str_data))
    except TypeError:
        return False


def is_ascii(str_data: str) -> bool:
    '''Checks if string contains only ascii chars.

    Necessary because python 3.6 does not have a str.isascii() method.

    Parameters
    ----------
    str_data : str
        string to check if it contains only ascii characters

    Returns
    -------
    bool
        True if only ascii characters in the string, else False.
    '''
    try:
        str_data.encode('ascii')
    except UnicodeEncodeError:
        return False
    return True


def find_next_prime(N: int) -> int:
    '''Find next prime >= N

    Parameters
    ----------
    N : int
        Starting point to find the next prime >= N.

    Returns
    -------
    int
        the next prime found after the number N
    '''
    def is_prime(n):
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i != 0:
                i += 2
            else:
                return False
        return True

    if N < 3:
        return 2
    if N % 2 == 0:
        N += 1
    for n in range(N, 2 * N, 2):
        if is_prime(n):
            return n


def file_size(p: os.PathLike) -> int:
    '''Query the file size of a specific file

    Parameters
    ----------
    p : os.PathLike
        path to a file that exists on disk.

    Raises
    ------
    FileNotFoundError
        if the file does not exist

    Returns
    -------
    int
        nbytes the file consumes on disk.
    '''
    if not os.path.isfile(p):
        err = f'Cannot query size of: {p}. File does not exist'
        raise FileNotFoundError(err)
    nbytes = os.stat(p).st_size
    return nbytes


def folder_size(p: os.PathLike, *, recurse: bool = False) -> int:
    '''size of all files in a folder.

    Default is to not include subdirectories. Set "recurse=True"
    to enable recursive calculation.

    Parameters
    ----------
    p : os.PathLike
        path to the repository on disk.
    recurse : bool, kwarg-only
        to calculate the full size of the repo (Default value = False)

    Returns
    -------
    int
        number of bytes used up in the repo_path
    '''
    total = 0
    for entry in os.scandir(p):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif (recurse is True) and (entry.is_dir() is True):
            total += folder_size(entry.path, recurse=True)
    return total


def is_valid_directory_path(p: os.PathLike) -> os.PathLike:
    '''Check if path is directory which user has write permission to.

    Parameters
    ----------
    p : os.PathLike
        path to some location on disk

    Returns
    -------
    os.PathLike
        If successful, the path with any user constructions expanded
        (ie. `~/somedir` -> `/home/foo/somedir`)

    Raises
    ------
    TypeError
        If the provided path argument is not a pathlike object
    OSError
        If the path does not exist, or is not a directory on disk
    PermissionError
        If the user does not have write access to the specified path
    '''
    try:
        usr_path = os.path.expanduser(p)
        isDir = os.path.isdir(usr_path)
        isWriteable = os.access(usr_path, os.W_OK)
    except TypeError:
        msg = f'Path arg `p`: {p} of type: {type(p)} is not valid path specifier'
        raise TypeError(msg)

    if not isDir:
        msg = f'Path arg `p`: {p} is not a directory.'
        raise OSError(msg)
    elif not isWriteable:
        msg = f'User does not have permission to write to directory path: {p}'
        raise PermissionError(msg)

    return usr_path


# ----------------- human & machine nbytes ------------------------------------


def format_bytes(n: int) -> str:
    """ Format bytes as text
    >>> format_bytes(1)
    '1.00 B'
    >>> format_bytes(1234)
    '1.23 kB'
    >>> format_bytes(12345678)
    '12.35 MB'
    >>> format_bytes(1234567890)
    '1.23 GB'
    >>> format_bytes(1234567890000)
    '1.23 TB'
    >>> format_bytes(1234567890000000)
    '1.23 PB'
    """
    for x in ['B', 'kB', 'MB', 'GB', 'TB', 'PB']:
        if n < 1000.0:
            return "%3.2f %s" % (n, x)
        n /= 1000.0


_byte_sizes = {
    'kb': 1000,
    'mb': 1000000,
    'gb': 1000000000,
    'tb': 1000000000000,
    'pb': 1000000000000000,
    'kib': 1024,
    'mib': 1048576,
    'gib': 1073741824,
    'tib': 1099511627776,
    'pib': 1125899906842624,
    'b': 1,
    '': 1,
    'k': 1000,
    'm': 1000000,
    'g': 1000000000,
    't': 1000000000000,
    'p': 1000000000000000,
    'ki': 1024,
    'mi': 1048576,
    'gi': 1073741824,
    'ti': 1099511627776,
    'pi': 1125899906842624
}


def parse_bytes(s: str) -> int:
    """ Parse byte string to numbers
    >>> parse_bytes('100')
    100
    >>> parse_bytes('100 MB')
    100000000
    >>> parse_bytes('100M')
    100000000
    >>> parse_bytes('5kB')
    5000
    >>> parse_bytes('5.4 kB')
    5400
    >>> parse_bytes('1kiB')
    1024
    >>> parse_bytes('1e6')
    1000000
    >>> parse_bytes('1e6 kB')
    1000000000
    >>> parse_bytes('MB')
    1000000
    """
    s = s.replace(' ', '').lower()
    s = f'1{s}' if not s[0].isdigit() else s
    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break

    n = float(s[:i + 1])
    mult = _byte_sizes[s[i + 1:]]
    return int(n * mult)
