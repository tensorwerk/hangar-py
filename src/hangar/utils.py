import importlib
import os
import re
import secrets
import string
import sys
import time
import types
from collections import deque
from io import StringIO
from itertools import tee, filterfalse, count
from pathlib import Path
from typing import Union

import blosc


class LazyLoader(types.ModuleType):
    """Lazily import a module, mainly to avoid pulling in large dependencies.
    `tensorflow`, and `pytorch` are examples of modules that are large and not always
    needed, and this allows them to only be loaded when they are used.
    """

    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super(LazyLoader, self).__init__(name)

    def _load(self):
        """Load the module and insert it into the parent's globals.
        """
        # Import the target module and insert it into the parent's namespace
        module = importlib.import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module

        # Update this object's dict so that if someone keeps a reference to the
        #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
        #   that fail).
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


def is_64bits():
    """bool indicating if running on atleast a 64 bit machine
    """
    return sys.maxsize > 2 ** 32


def set_blosc_nthreads() -> int:  # pragma: no cover
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


def random_string(
    n: int = 8,
    *, _ALPHABET=''.join([string.ascii_lowercase, string.digits])
) -> str:
    """Generate a random string of lowercase ascii letters and digits.

    Parameters
    ----------
    n: int, optional
        The number of characters which the output string will have. Default=8
    """
    token = [secrets.choice(_ALPHABET) for i in range(n)]
    return ''.join(token)


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


def valfilter(predicate, d, factory=dict):
    """ Filter items in dictionary by values that are true.

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilter(iseven, d)
    {1: 2, 3: 4}

    See Also:
        valfilterfalse
    """
    rv = factory()
    for k, v in d.items():
        if predicate(v):
            rv[k] = v
    return rv


def valfilterfalse(predicate, d, factory=dict):
    """ Filter items in dictionary by values which are false.

    >>> iseven = lambda x: x % 2 == 0
    >>> d = {1: 2, 2: 3, 3: 4, 4: 5}
    >>> valfilterfalse(iseven, d)
    {2: 3, 4: 5}

    See Also:
        valfilter
    """
    rv = factory()
    for k, v in d.items():
        if not predicate(v):
            rv[k] = v
    return rv


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


def ilen(iterable):
    """Return the number of items in *iterable*.

        >>> ilen(x for x in range(1000000) if x % 3 == 0)
        333334
        >>> it = iter([0, 1, 2, False])
        >>> ilen(it)
        4

    This consumes the iterable, so handle with care.
    """
    counter = count()
    deque(zip(iterable, counter), maxlen=0)
    return next(counter)


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


def file_size(p: Path) -> int:  # pragma: no cover
    """Query the file size of a specific file

    Parameters
    ----------
    p : Path
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
    if not p.is_file():
        err = f'Cannot query size of: {str(p)}. File does not exist'
        raise FileNotFoundError(err)
    return p.stat().st_size


def folder_size(p: Path, *, recurse: bool = False) -> int:
    """size of all files in a folder.

    Default is to not include subdirectories. Set "recurse=True"
    to enable recursive calculation.

    Parameters
    ----------
    p : Path
        path to the repository on disk.
    recurse : bool, kwarg-only
        to calculate the full size of the repo (Default value = False)

    Returns
    -------
    int
        number of bytes used up in the repo_path
    """
    total = 0
    for entry in p.iterdir():
        if entry.is_file() and not entry.is_symlink():
            total += entry.stat().st_size
        elif recurse and entry.is_dir() and not entry.is_symlink():
            total += folder_size(entry.resolve(), recurse=True)
    return total


def is_valid_directory_path(p: Path) -> Path:
    """Check if path is directory which user has write permission to.

    Parameters
    ----------
    p : Path
        path to some location on disk

    Returns
    -------
    Path
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
    if not isinstance(p, Path):
        msg = f'Path arg `p`: {p} of type: {type(p)} is not valid path specifier'
        raise TypeError(msg)

    usr_path = p.expanduser().resolve(strict=True)

    if not usr_path.is_dir():
        msg = f'Path arg `p`: {p} is not a directory.'
        raise NotADirectoryError(msg)
    if not os.access(str(usr_path), os.W_OK):  # pragma: no cover
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
    from . import __version__
    from .constants import DIR_HANGAR

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
