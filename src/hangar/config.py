'''
Methods in this file have been taken & modified from the Dask project
https://github.com/dask/dask

From file: dask/config.py
commit_hash: caa2da0c4412ab8ebf467437e58a7bd447332b70
Access Date: 09 APR 2019

Dask License
-------------------------------------------------------------------------------
Copyright (c) 2014-2018, Anaconda, Inc. and contributors
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

Neither the name of Anaconda nor the names of any contributors may be used to
endorse or promote products derived from this software without specific prior
written permission.

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

import os
import threading
from functools import lru_cache
from os import makedirs

import yaml

no_default = '__no_default__'
global_config = config = {}
config_lock = threading.Lock()
defaults = []


def update(old, new, priority='new'):
    ''' Update a nested dictionary with values from another

    This is like dict.update except that it smoothly merges nested values

    This operates in-place and modifies old

    Parameters
    ----------
    priority: string {'old', 'new'}
        If new (default) then the new dictionary has preference.
        Otherwise the old dictionary does.

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b)  # doctest: +SKIP
    {'x': 2, 'y': {'a': 2, 'b': 3}}

    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'x': 2, 'y': {'b': 3}}
    >>> update(a, b, priority='old')  # doctest: +SKIP
    {'x': 1, 'y': {'a': 2, 'b': 3}}
    '''
    for k, v in new.items():
        if k not in old and isinstance(v, dict):
            old[k] = {}

        if isinstance(v, dict):
            if old[k] is None:
                old[k] = {}
            update(old[k], v, priority=priority)
        else:
            if priority == 'new' or k not in old:
                old[k] = v

    return old


def merge(*dicts):
    ''' Update a sequence of nested dictionaries

    This prefers the values in the latter dictionaries to those in the former

    Examples
    --------
    >>> a = {'x': 1, 'y': {'a': 2}}
    >>> b = {'y': {'b': 3}}
    >>> merge(a, b)  # doctest: +SKIP
    {'x': 1, 'y': {'a': 2, 'b': 3}}
    '''
    result = {}
    for d in dicts:
        update(result, d)
    return result


def collect_yaml(paths):
    ''' Collect configuration from yaml files

    This searches through a list of paths, expands to find all yaml or json
    files, and then parses each file.
    '''
    # Find all paths
    file_paths = []
    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                try:
                    file_paths.extend(sorted([
                        os.path.join(path, p)
                        for p in os.listdir(path)
                        if os.path.splitext(p)[1].lower() in ('.yaml', '.yml')
                    ]))
                except OSError:
                    # Ignore permission errors
                    pass
            else:
                file_paths.append(path)

    configs = []

    # Parse yaml files
    for path in file_paths:
        try:
            with open(path) as f:
                data = yaml.safe_load(f.read()) or {}
                configs.append(data)
        except (OSError, IOError):
            # Ignore permission errors
            pass

    return configs


def ensure_file(source, destination=None, comment=False):
    '''
    Copy file to default location if it does not already exist

    This tries to move a default configuration file to a default location if
    if does not already exist.  It also comments out that file by default.

    This is to be used on repository initialization or downstream module that
    may have default configuration files that they wish to include in the
    default configuration path.

    Parameters
    ----------
    source : string, filename
        Source configuration file, typically within a source directory.
    destination : string, directory
        Destination directory.
    comment : bool, False by default
        Whether or not to comment out the config file when copying.
    '''
    if destination is None:
        raise FileNotFoundError('not currenly accepting no-destination option')

    # destination is a file and already exists, never overwrite
    if os.path.isfile(destination):
        return

    # If destination is not an existing file, interpret as a directory,
    # use the source basename as the filename
    directory = destination
    destination = os.path.join(directory, os.path.basename(source))

    try:
        if not os.path.exists(destination):
            makedirs(directory, exist_ok=True)

            # Atomically create destination.  Parallel testing discovered
            # a race condition where a process can be busy creating the
            # destination while another process reads an empty config file.
            tmp = '%s.tmp.%d' % (destination, os.getpid())
            with open(source) as f:
                lines = list(f)

            if comment:
                lines = ['# ' + line
                         if line.strip() and not line.startswith('#')
                         else line
                         for line in lines]

            with open(tmp, 'w') as f:
                f.write(''.join(lines))

            try:
                os.rename(tmp, destination)
            except OSError:
                os.remove(tmp)
    except OSError:
        pass


class conf_set(object):
    ''' Temporarily set configuration values within a context manager

    Examples
    --------
    >>> import hangar
    >>> with hangar.config.set({'foo': 123}):
    ...     pass

    See Also
    --------
    hangar.config.get
    '''
    def __init__(self, arg=None, config=config, lock=config_lock, **kwargs):
        get.cache_clear()
        if arg and not kwargs:
            kwargs = arg

        with lock:
            self.config = config
            self.old = {}

            for key, value in kwargs.items():
                self._assign(key.split('.'), value, config, old=self.old)

    def __enter__(self):
        get.cache_clear()
        return self.config

    def __exit__(self, type, value, traceback):
        get.cache_clear()
        for keys, value in self.old.items():
            if value == '--delete--':
                d = self.config
                try:
                    while len(keys) > 1:
                        d = d[keys[0]]
                        keys = keys[1:]
                    del d[keys[0]]
                except KeyError:
                    pass
            else:
                self._assign(keys, value, self.config)

    @classmethod
    def _assign(cls, keys, value, d, old=None, path=[]):
        ''' Assign value into a nested configuration dictionary

        Optionally record the old values in old

        Parameters
        ----------
        keys: Sequence[str]
            The nested path of keys to assign the value, similar to toolz.put_in
        value: object
        d: dict
            The part of the nested dictionary into which we want to assign the
            value
        old: dict, optional
            If provided this will hold the old values
        path: List[str]
            Used internally to hold the path of old values
        '''
        get.cache_clear()
        # key = normalize_key(keys[0])
        key = keys[0]
        if len(keys) == 1:
            if old is not None:
                path_key = tuple(path + [key])
                if key in d:
                    old[path_key] = d[key]
                else:
                    old[path_key] = '--delete--'
            d[key] = value
        else:
            if key not in d:
                d[key] = {}
                if old is not None:
                    old[tuple(path + [key])] = '--delete--'
                old = None
            cls._assign(keys[1:], value, d[key], path=path + [key], old=old)


def collect(paths=[], env=None):
    '''
    Collect configuration from paths and environment variables

    Parameters
    ----------
    paths : List[str]
        A list of paths to search for yaml config files

    env : dict
        The system environment variables

    Returns
    -------
    config: dict
    '''
    if env is None:
        env = os.environ
    configs = []

    if yaml:
        configs.extend(collect_yaml(paths=paths))

    return merge(*configs)


def refresh(config=config, defaults=defaults, **kwargs):
    '''
    Update configuration by re-reading yaml files and env variables

    This mutates the global hangar.config.config, or the config parameter if
    passed in.

    This goes through the following stages:

    1.  Clearing out all old configuration
    2.  Updating from the stored defaults from downstream libraries
        (see update_defaults)
    3.  Updating from yaml files and environment variables

    Note that some functionality only checks configuration once at startup and
    may not change behavior, even if configuration changes.  It is recommended
    to restart your python process if convenient to ensure that new
    configuration changes take place.
    '''
    get.cache_clear()
    config.clear()

    for d in defaults:
        update(config, d, priority='new')

    update(config, collect(**kwargs))


@lru_cache(maxsize=128)
def get(key, default=no_default, config=config):
    '''
    Get elements from global config

    Use '.' for nested access

    Examples
    --------
    >>> from hangar import config
    >>> config.get('foo')  # doctest: +SKIP
    {'x': 1, 'y': 2}

    >>> config.get('foo.x')  # doctest: +SKIP
    1

    >>> config.get('foo.x.y', default=123)  # doctest: +SKIP
    123
    '''
    keys = key.split('.')
    result = config
    for k in keys:
        try:
            result = result[k]
        except (TypeError, IndexError, KeyError):
            if default is not no_default:
                return default
            else:
                raise
    return result


def rename(aliases, config=config):
    ''' Rename old keys to new keys

    This helps migrate older configuration versions over time
    '''
    get.cache_clear()
    old = list()
    new = dict()
    for o, n in aliases.items():
        value = get(o, None, config=config)
        if value is not None:
            old.append(o)
            new[n] = value

    for k in old:
        del config[k]  # TODO: support nested keys

    conf_set(new, config=config)


def update_defaults(new, config=config, defaults=defaults):
    '''Add a new set of defaults to the configuration

    It does two things:

    1.  Add the defaults to a global collection to be used by refresh later
    2.  Updates the global config with the new configuration
        prioritizing older values over newer ones
    '''
    get.cache_clear()
    defaults.append(new)
    update(config, new, priority='old')


fn = os.path.join(os.path.dirname(__file__), 'config_hangar.yml')
with open(fn) as f:
    conf = yaml.safe_load(f)

update_defaults(conf)
