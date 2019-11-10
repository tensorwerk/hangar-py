import os
import random
import re
import time
import string
import weakref
from io import StringIO
from functools import partial
from itertools import tee, filterfalse
from typing import Union, Any
import importlib
import types

import blosc
import wrapt

from . import __version__
from .constants import DIR_HANGAR


class LazyImporter(types.ModuleType):
    """Lazily import a module. `_load` adds the attributes of
    importing module to `LazyLoader` instance. Hence `__getattr__` of `LazyLoader` will
    be invoked only once. We might need to extend the `LazyLoader` class to have
    functions like `__dir__` later.
    """

    def __init__(self, name: str):
        super(LazyImporter, self).__init__(name)

    def _import_module(self):
        module = importlib.import_module(self.__name__)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._import_module()  # this happens only once
        return getattr(module, item)


def set_blosc_nthreads() -> int:
    """set the blosc library to two less than the core count on the system.

    If less than 2 cores are ncores-2, we set the value to two.

    Returns
    -------
    int
        ncores blosc will use on the system
    """
    nCores = blosc.detect_number_of_cores()
    if nCores == 1:
        nUsed = 1
    elif nCores == 2:
        nUsed = 2
    elif nCores <= 4:
        nUsed = nCores - 1
    else:
        nUsed = nCores - 2
    blosc.set_nthreads(nUsed)
    return nUsed


def random_string(n: int = 6) -> str:
    """Generate a case random string of ascii letters and digits of some length.

    Parameters
    ----------
    n: int, optional
        The number of characters which the output string will have. Default = 6
    """
    letters = ''.join([string.ascii_letters, string.digits])
    return ''.join(random.choice(letters) for i in range(n))


def cm_weakref_obj_proxy(obj: Any) -> wrapt.ObjectProxy:
    """Creates a weakproxy reference honoring optional use context managers.

    This is required because (for some unknown reason) `weakproxy`
    references will not actually pass through the `__enter__` attribute of
    the referred object's instance. As such, these are manually set to the
    appropriate values on the `weakproxy` object. The final `weakproxy` is
    in turn referenced by an object proxy so that all calls to the
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
    """
    wr = weakref.proxy(obj)
    setattr(wr, '__enter__', partial(obj.__class__.__enter__, wr))
    setattr(wr, '__exit__', partial(obj.__class__.__exit__, wr))
    obj_proxy = wrapt.ObjectProxy(wr)
    return obj_proxy


def symlink_rel(src: os.PathLike, dst: os.PathLike, *, is_dir=False):
    """Create symbolic links which actually work like they should

    Parameters
    ----------
    src : os.PathLike
        create a symbolic link pointic to src
    dst : os.PathLike
        create a link named dst
    is_dir : bool, kwarg-only, optional
        if pointing to a directory, set to true. Default = False
    """
    rel_path_src = os.path.relpath(src, os.path.dirname(dst))
    os.symlink(rel_path_src, dst, target_is_directory=is_dir)


_SuitableCharRE = re.compile(r'[\w\.\-\_]+\Z', flags=re.ASCII)


def is_suitable_user_key(key: Union[str, int]) -> bool:
    """Checks if only alpha-numeric ascii chars or ['.', '-' '_'] (no whitespace)

    Necessary because python 3.6 does not have a str.isascii() method. In
    addition, checks that all input keys are less than 64 characters long.

    Parameters
    ----------
    key : Union[str, int]
        string to check if it contains only ascii characters

    Returns
    -------
    bool
        True if only ascii characters in the string, else False.
    """
    try:
        if isinstance(key, int) and (key >= 0):
            str_data = str(key)
        elif isinstance(key, str):
            str_data = str(key)
        else:
            raise TypeError
        if len(str_data) > 64:
            return False
        return bool(_SuitableCharRE.match(str_data))
    except TypeError:
        return False


def is_ascii(str_data: str) -> bool:
    """Checks if string contains only ascii chars.

    Necessary because python 3.6 does not have a str.isascii() method.

    Parameters
    ----------
    str_data : str
        string to check if it contains only ascii characters

    Returns
    -------
    bool
        True if only ascii characters in the string, else False.
    """
    try:
        str_data.encode('ascii')
    except (UnicodeEncodeError, AttributeError):
        return False
    return True


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def unique_everseen(iterable, key=None):
    """List unique elements, preserving order. Remember all elements ever seen.

    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    >>> list(unique_everseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'D']
    """
    seen = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def find_next_prime(N: int) -> int:
    """Find next prime >= N

    Parameters
    ----------
    N : int
        Starting point to find the next prime >= N.

    Returns
    -------
    int
        the next prime found after the number N
    """
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


def file_size(p: os.PathLike) -> int:  # pragma: no cover
    """Query the file size of a specific file

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
    """
    if not os.path.isfile(p):
        err = f'Cannot query size of: {p}. File does not exist'
        raise FileNotFoundError(err)
    nbytes = os.stat(p).st_size
    return nbytes


def folder_size(p: os.PathLike, *, recurse: bool = False) -> int:
    """size of all files in a folder.

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
    """
    total = 0
    for entry in os.scandir(p):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif (recurse is True) and (entry.is_dir(follow_symlinks=False) is True):
            total += folder_size(entry.path, recurse=True)
    return total


def is_valid_directory_path(p: os.PathLike) -> os.PathLike:
    """Check if path is directory which user has write permission to.

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
    NotADirectoryError
        If the path does not exist, or is not a directory on disk
    PermissionError
        If the user does not have write access to the specified path
    """
    try:
        usr_path = os.path.expanduser(p)
    except TypeError:
        msg = f'Path arg `p`: {p} of type: {type(p)} is not valid path specifier'
        raise TypeError(msg)

    if not os.path.isdir(usr_path):
        msg = f'Path arg `p`: {p} is not a directory.'
        raise NotADirectoryError(msg)
    if not os.access(usr_path, os.W_OK):  # pragma: no cover
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


def readme_contents(user_name: str, user_email: str) -> StringIO:
    """Create the Hangar README.txt contents used to fill out file on repo initialization

    Parameters
    ----------
    user_name : str
        name of the user initializing the repository on the machine.
    user_email : str
        email of the user initializing the repository on the machine.

    Returns
    -------
    StringIO
        Buffered string text ready to be sent to a file writer.
    """
    buf = StringIO()
    buf.write(f'This directory has been used to initialize a Hangar Repository\n')
    buf.write(f'\n')
    buf.write(f'This repository was initialized by:\n')
    buf.write(f'    User Name:        {user_name}\n')
    buf.write(f'    User Email:       {user_email}\n')
    buf.write(f'    Creation Time:    {time.asctime(time.gmtime())} UTC\n')
    buf.write(f'    Software Version: {__version__}\n')
    buf.write(f'\n')
    buf.write(f'NOTE: The repository may have been updated to work with newer Hangar versions\n')
    buf.write(f'since initialization.\n')
    buf.write(f'\n')
    buf.write(f'Do not modify the contents of this `{DIR_HANGAR}` folder under any circumstances.\n')
    buf.write(f'The contents are not meant to be understood by humans. Doing so will result\n')
    buf.write(f'in data loss / corruption.\n')
    buf.write(f'\n')
    buf.write(f'The project homepage can be found at: https://github.com/tensorwerk/hangar-py/ \n')
    buf.write(f'Documention is available at: https://hangar-py.readthedocs.io/en/latest/ \n')
    buf.write(f'\n')
    buf.write(f'NOTE: If this Repository has been initialized in a directory under traditional\n')
    buf.write(f'version control systems, please add `{DIR_HANGAR}` as an ignored directory path.\n')
    buf.write(f'Failure to do so will result in undesireable performance of version control\n')
    buf.write(f'systems meant for text/code such as Git, Mercurial, Subversion, etc.\n')

    return buf
