"""
Portions of this code have been adapted from the "scikit-image" project.


URL:      https://github.com/scikit-image/scikit-image
Commit:   bde5a9bc3106d68ab9a4ca3dfed4f866fdd6a129
Accessed: 06 AUG 2019

Scikit-Image License
-------------------------------------------------------------------------------
Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

-------------------------------------------------------------------------------

Handle image reading, writing and plotting plugins.

To improve performance, plugins are only loaded as needed. As a result, there
can be multiple states for a given plugin:

    available: Defined in an *ini file located in `skimage.io._plugins`.
        See also `skimage.io.available_plugins`.
    partial definition: Specified in an *ini file, but not defined in the
        corresponding plugin module. This will raise an error when loaded.
    available but not on this system: Defined in `skimage.io._plugins`, but
        a dependent library (e.g. Qt, PIL) is not available on your system.
        This will raise an error when loaded.
    loaded: The real availability is determined when it's explicitly loaded,
        either because it's one of the default plugins, or because it's
        loaded explicitly by the user.
"""
from configparser import ConfigParser
import os.path
from glob import glob


# The plugin store will save a list of *loaded* io functions for each io type
# (e.g. 'imread', 'imsave', etc.). Plugins are loaded as requested.
plugin_store = None
# Dictionary mapping plugin names to a list of functions they provide.
plugin_provides = {}
# The module names for the plugins in `hangar.plugins._plugins`.
plugin_module_name = {}
# Meta-data about plugins provided by *.ini files.
plugin_meta_data = {}
# For each plugin type, default to the first available plugin as defined by
# the following preferences.
preferred_plugins = {
    # Default plugins for all types (overridden by specific types below).
    'all': ['pil', 'matplotlib'],
    'imshow': ['matplotlib'],
}


def _clear_plugins():
    """Clear the plugin state to the default, i.e., where no plugins are loaded

    """
    global plugin_store
    plugin_store = {'imread': [],
                    'imsave': [],
                    'imshow': [],
                    '_app_show': []}


_clear_plugins()


def _load_preferred_plugins():
    # Load preferred plugin for each io function.
    io_types = ['imsave', 'imshow', 'imread']
    for p_type in io_types:
        _set_plugin(p_type, preferred_plugins['all'])

    plugin_types = (p for p in preferred_plugins.keys() if p != 'all')
    for p_type in plugin_types:
        _set_plugin(p_type, preferred_plugins[p_type])


def _set_plugin(plugin_type, plugin_list):
    for plugin in plugin_list:
        if plugin not in available_plugins:
            continue
        try:
            use_plugin(plugin, kind=plugin_type)
            break
        except (ImportError, RuntimeError, OSError):
            pass


def reset_plugins():
    _clear_plugins()
    _load_preferred_plugins()


def _parse_config_file(filename):
    """Return plugin name and meta-data dict from plugin config file."""
    parser = ConfigParser()
    parser.read(filename)
    name = parser.sections()[0]

    meta_data = {}
    for opt in parser.options(name):
        meta_data[opt] = parser.get(name, opt)

    return name, meta_data


def _scan_plugins():
    """Scan the plugins directory for .ini files and parse them
    to gather plugin meta-data.

    """
    pd = os.path.dirname(__file__)
    config_files = glob(os.path.join(pd, '_plugins', '*.ini'))

    for filename in config_files:
        name, meta_data = _parse_config_file(filename)
        plugin_meta_data[name] = meta_data

        provides = [s.strip() for s in meta_data['provides'].split(',')]
        valid_provides = [p for p in provides if p in plugin_store]

        for p in provides:
            if p not in plugin_store:
                print(f"Plugin: {name} wants to provide non-existent: {p}. Ignoring.")

        plugin_provides[name] = valid_provides
        plugin_module_name[name] = os.path.basename(filename)[:-4]


_scan_plugins()


def find_available_plugins(loaded=False):
    """List available plugins.

    Parameters
    ----------
    loaded : bool
        If True, show only those plugins currently loaded.  By default,
        all plugins are shown.

    Returns
    -------
    p : dict
        Dictionary with plugin names as keys and exposed functions as
        values.

    """
    active_plugins = set()
    for plugin_func in plugin_store.values():
        for plugin, func in plugin_func:
            active_plugins.add(plugin)

    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin]
                         if not f.startswith('_')]

    return d


available_plugins = find_available_plugins()


def call_plugin(kind, *args, **kwargs):
    """Find the appropriate plugin of 'kind' and execute it.

    Parameters
    ----------
    kind : {'imshow', 'imsave', 'imread'}
        Function to look up.
    plugin : str, optional
        Plugin to load.  Defaults to None, in which case the first
        matching plugin is used.
    *args, **kwargs : arguments and keyword arguments
        Passed to the plugin function.

    """
    if kind not in plugin_store:
        raise ValueError('Invalid function (%s) requested.' % kind)

    plugin_funcs = plugin_store[kind]
    if len(plugin_funcs) == 0:
        msg = ("No suitable plugin registered for %s.\n\n"
               "You may load I/O plugins with the `hangar.cli.io.use_plugin` "
               "command.  A list of all available plugins are shown in the "
               "`hangar.cli.io.plugins` docstring.")
        raise RuntimeError(msg % kind)

    plugin = kwargs.pop('plugin', None)
    if plugin is None:
        _, func = plugin_funcs[0]
    else:
        _load(plugin)
        try:
            func = [f for (p, f) in plugin_funcs if p == plugin][0]
        except IndexError:
            raise RuntimeError(f'Could not find the plugin "{plugin}" for {kind}.')

    return func(*args, **kwargs)


def use_plugin(name, kind=None):
    """Set the default plugin for a specified operation.  The plugin
    will be loaded if it hasn't been already.

    Parameters
    ----------
    name : str
        Name of plugin.
    kind : {'imsave', 'imread', 'imshow'}, optional
        Set the plugin for this function.  By default,
        the plugin is set for all functions.

    See Also
    --------
    available_plugins : List of available plugins

    Examples
    --------

    To use Matplotlib as the default image reader, you would write:

        >>> from hangar.cli import io
        >>> io.use_plugin('matplotlib', 'imread')

    To see a list of available plugins run ``plugins.available_plugins``. Note
    that this lists plugins that are defined, but the full list may not be
    usable if your system does not have the required libraries installed.

    """
    if kind is None:
        kind = plugin_store.keys()
    else:
        if kind not in plugin_provides[name]:
            raise RuntimeError(f"Plugin {name} does not support `{kind}`.")

        if kind == 'imshow':
            kind = [kind, '_app_show']
        else:
            kind = [kind]

    _load(name)

    for k in kind:
        if k not in plugin_store:
            raise RuntimeError(f"'{k}' is not a known plugin function.")

        funcs = plugin_store[k]

        # Shuffle the plugins so that the requested plugin stands first
        # in line
        funcs = [(n, f) for (n, f) in funcs if n == name] + \
                [(n, f) for (n, f) in funcs if n != name]

        plugin_store[k] = funcs


def _load(plugin):
    """Load the given plugin.

    Parameters
    ----------
    plugin : str
        Name of plugin to load.

    See Also
    --------
    plugins : List of available plugins

    """
    if plugin in find_available_plugins(loaded=True):
        return
    if plugin not in plugin_module_name:
        raise ValueError(f"Plugin {plugin} not found.")
    else:
        modname = plugin_module_name[plugin]
        plugin_module = __import__('hangar.cli.io._plugins.' + modname, fromlist=[modname])

    provides = plugin_provides[plugin]
    for p in provides:
        if not hasattr(plugin_module, p):
            print(f"Plugin {plugin} does not provide {p} as advertised.  Ignoring.")
            continue

        store = plugin_store[p]
        func = getattr(plugin_module, p)
        if not (plugin, func) in store:
            store.append((plugin, func))


def plugin_info(plugin):
    """Return plugin meta-data.

    Parameters
    ----------
    plugin : str
        Name of plugin.

    Returns
    -------
    m : dict
        Meta data as specified in plugin ``.ini``.

    """
    try:
        return plugin_meta_data[plugin]
    except KeyError:
        raise ValueError('No information on plugin "%s"' % plugin)


def plugin_order():
    """Return the currently preferred plugin order.

    Returns
    -------
    p : dict
        Dictionary of preferred plugin order, with function name as key and
        plugins (in order of preference) as value.

    """
    p = {}
    for func in plugin_store:
        p[func] = [plugin_name for (plugin_name, f) in plugin_store[func]]
    return p