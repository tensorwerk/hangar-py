import os
from contextlib import contextmanager
from datetime import timedelta
from numbers import Number


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
