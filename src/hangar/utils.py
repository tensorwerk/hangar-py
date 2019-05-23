import os
import weakref
from functools import partial
from contextlib import contextmanager
from datetime import timedelta
from numbers import Number

import wrapt


def cm_weakref_obj_proxy(obj):
    '''Creates a weakproxy reference honoring optional use context managers.

    This is required because (for some unknown reason) `weakproxy`
    references will not actually pass through the `__enter__` attribute of
    the reffered object's instance. As such, these are manual set to the
    appropriate values on the `weakproxy` object. The final `weakproxy` is
    in turn referenced by a object proxy so that all calls to the
    methods/attributes are passed through uniformly.

    Parameters
    ----------
    obj : class
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


def is_ascii_alnum(str_data: str):
    '''Checks if string contains only alpha-numeric ascii chars (no whitespace)

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
        asciiStrIsAlnum = False if any(c.isspace() for c in str_data) else True
        return asciiStrIsAlnum
    except UnicodeEncodeError:
        return False


def is_ascii(str_data: str):
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


def find_next_prime(N):
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
            if n % i:
                i += 2
            else:
                return False
        return True
    if N < 3:
        return 2
    if N % 2 == 0:
        N += 1
    for n in range(N, 2*N, 2):
        if is_prime(n):
            return n


def file_size(file_path):
    '''Query the file size of a specific file

    Parameters
    ----------
    file_path : str
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
    if not os.path.isfile(file_path):
        err = f'Cannot query size of: {file_path}. File does not exist'
        raise FileNotFoundError(err)
    nbytes = os.stat(file_path).st_size
    return nbytes


def folder_size(repo_path, *, recurse_directories=False):
    '''size of all files in a folder.

    Default is to not include subdirectories. Set "recurse_directories=True"
    to enable recursive calculation.

    Parameters
    ----------
    repo_path : str
        path to the repository on disk.
    recurse_directories : bool
        to calculate the full size of the repo (Default value = False)

    Returns
    -------
    int
        number of bytes used up in the repo_path
    '''
    total = 0
    for entry in os.scandir(repo_path):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
        elif (recurse_directories is True) and (entry.is_dir() is True):
            total += folder_size(entry.path, recurse_directories=True)
    return total


def is_valid_directory_path(path: str) -> str:
    '''Check if path is directory which user has write permission to.

    Parameters
    ----------
    path : str
        path to some location on disk

    Returns
    -------
    str
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
        usr_path = os.path.expanduser(path)
        isDir = os.path.isdir(usr_path)
        isWriteable = os.access(usr_path, os.W_OK)
    except TypeError:
        msg = f'HANGAR TYPE ERROR:: `path` arg: {path} of type: {type(path)} '\
              f'is not valid path specifier'
        raise TypeError(msg)

    if not isDir:
        msg = f'HANGAR VALUE ERROR:: `path` arg: {path} is not a directory.'
        raise OSError(msg)
    elif not isWriteable:
        msg = f'HANGAR PERMISSION ERROR:: user does not have permission to write '\
              f'to directory `path` arg: {path}'
        raise PermissionError(msg)

    return usr_path


'''
Methods following this notice have been taken & modified from the Dask Distributed project
url: https://github.com/dask/distributed

From file: distributed/utils.py
commit_hash: f50b239b8e6420fb87646f7183edaafb4b8e20be
Access Date: 09 APR 2019

Dask Distributed License
-------------------------------------------------------------------------------
Copyright (c) 2015-2017, Anaconda, Inc. and contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of Anaconda nor the names of any contributors
may be used to endorse or promote products derived from this software
without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
'''

# ----------------- human & machine nbytes ------------------------------------


@contextmanager
def ignoring(*exceptions):
    try:
        yield
    except exceptions as e:
        pass


def asciitable(columns, rows):
    """Formats an ascii table for given columns and rows.
    Parameters
    ----------
    columns : list
        The column names
    rows : list of tuples
        The rows in the table. Each tuple must be the same length as
        ``columns``.
    """
    rows = [tuple(str(i) for i in r) for r in rows]
    columns = tuple(str(i) for i in columns)
    widths = tuple(max(max(map(len, x)), len(c))
                   for x, c in zip(zip(*rows), columns))
    row_template = ('|' + (' %%-%ds |' * len(columns))) % widths
    header = row_template % tuple(columns)
    bar = '+%s+' % '+'.join('-' * (w + 2) for w in widths)
    data = '\n'.join(row_template % r for r in rows)
    return '\n'.join([bar, header, bar, data, bar])


def format_bytes(n):
    """ Format bytes as text
    >>> format_bytes(1)
    '1 B'
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
    if n > 1e15:
        return '%0.2f PB' % (n / 1e15)
    if n > 1e12:
        return '%0.2f TB' % (n / 1e12)
    if n > 1e9:
        return '%0.2f GB' % (n / 1e9)
    if n > 1e6:
        return '%0.2f MB' % (n / 1e6)
    if n > 1e3:
        return '%0.2f kB' % (n / 1000)
    return '%d B' % n


byte_sizes = {
    'kB': 10**3,
    'MB': 10**6,
    'GB': 10**9,
    'TB': 10**12,
    'PB': 10**15,
    'KiB': 2**10,
    'MiB': 2**20,
    'GiB': 2**30,
    'TiB': 2**40,
    'PiB': 2**50,
    'B': 1,
    '': 1,
}
byte_sizes = {k.lower(): v for k, v in byte_sizes.items()}
byte_sizes.update({k[0]: v for k, v in byte_sizes.items() if k and 'i' not in k})
byte_sizes.update({k[:-1]: v for k, v in byte_sizes.items() if k and 'i' in k})


def parse_bytes(s):
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
    s = s.replace(' ', '')
    if not s[0].isdigit():
        s = '1' + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:]

    n = float(prefix)

    multiplier = byte_sizes[suffix.lower()]

    result = n * multiplier
    return int(result)


def memory_repr(num):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0


# ----------- Time Deltas -----------------------------------------------------

timedelta_sizes = {
    's': 1,
    'ms': 1e-3,
    'us': 1e-6,
    'ns': 1e-9,
    'm': 60,
    'h': 3600,
    'd': 3600 * 24,
}

tds2 = {
    'second': 1,
    'minute': 60,
    'hour': 60 * 60,
    'day': 60 * 60 * 24,
    'millisecond': 1e-3,
    'microsecond': 1e-6,
    'nanosecond': 1e-9,
}
tds2.update({k + 's': v for k, v in tds2.items()})
timedelta_sizes.update(tds2)
timedelta_sizes.update({k.upper(): v for k, v in timedelta_sizes.items()})


def parse_timedelta(s, default='seconds'):
    """ Parse timedelta string to number of seconds
    Examples
    --------
    >>> parse_timedelta('3s')
    3
    >>> parse_timedelta('3.5 seconds')
    3.5
    >>> parse_timedelta('300ms')
    0.3
    >>> parse_timedelta(timedelta(seconds=3))  # also supports timedeltas
    3
    """
    if isinstance(s, timedelta):
        return s.total_seconds()
    if isinstance(s, Number):
        s = str(s)
    s = s.replace(' ', '')
    if not s[0].isdigit():
        s = '1' + s

    for i in range(len(s) - 1, -1, -1):
        if not s[i].isalpha():
            break
    index = i + 1

    prefix = s[:index]
    suffix = s[index:] or default

    n = float(prefix)

    multiplier = timedelta_sizes[suffix.lower()]

    result = n * multiplier
    if int(result) == result:
        result = int(result)
    return result
