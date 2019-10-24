"""
High level methods let user interact with hangar without diving into the internal
methods of hangar. We have enabled four basic entry points as high level methods
1. load
2. save
3. show
4. board_show

These entry points by itself is not capable of doing anything. But they are entry
points to the same methods in the `hangar.external` plugins available in pypi. These
high level entry points are used by the CLI for doing import, export and view
operations as well as the `hangarboard <https://github.com/tensorwerk/hangarboard>`_
for visualization (using `board_show`)
"""
from typing import Tuple

import numpy as np
from .plugin_manager import PluginManager

pm = PluginManager()


def load(fpath: str, plugin: str = None,
         extension: str = None, **plugin_kwargs) -> Tuple[np.ndarray, str]:
    """
    Wrapper to load data from file into memory as numpy arrays using
    plugin's `load` method

    Parameters
    ----------
    fpath : string
        Data file path, e.g. ``path/to/test.jpg``
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be `None` if
        `extension` is also `None`
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be `None` if
        `plugin` is also `None`

    Other Parameters
    ----------------
    plugin_kwargs : keywords
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be collected
        as `plugin_kwargs`

    Returns
    -------
    img_array : ndarray
        ndarray returned from the given plugin

    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('load', plugin=plugin, extension=extension)
    return func(fpath, **plugin_kwargs)


def save(fpath: str, arr: np.ndarray, plugin: str = None,
         extension: str = None, **plugin_kwargs):
    """
    Wrapper to save data in numpy ndarray format to file using plugin's
    `save` method

    Parameters
    ----------
    fpath : str
        Target file path
    arr : ndarray
        Numpy array to be saved to file
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be `None` if
        `extension` is also `None`
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be `None` if
        `plugin` is also `None`

    Other Parameters
    ----------------
    plugin_kwargs : keywords
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be collected
        as `plugin_kwargs`
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('save', plugin=plugin, extension=extension)
    func(fpath, arr, **plugin_kwargs)


def show(arr: np.ndarray, plugin: str = None,
         extension: str = None, **plugin_kwargs):  # pragma: no cover
    """
    Wrapper to display the numpy array using the `show` method of the plugin

    Parameters
    ----------
    arr : ndarray
        Data as ndarray
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be `None` if
        `extension` is also `None`
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be `None` if
        `plugin` is also `None`

    Other Parameters
    ----------------
    plugin_kwargs : keywords
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be collected
        as `plugin_kwargs`
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('show', plugin=plugin, extension=extension)
    return func(arr, **plugin_kwargs)


def board_show(arr: np.ndarray, plugin: str = None,
               extension: str = None, **plugin_kwargs):
    """
    Wrapper to convert the numpy array using the `board_show` method
    of the plugin to make it displayable in the web UI

    Parameters
    ----------
    arr : ndarray
        Data as ndarray
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be `None` if
        `extension` is also `None`
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be `None` if
        `plugin` is also `None`

    Other Parameters
    ----------------
    plugin_kwargs : keywords
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be collected
        as `plugin_kwargs`
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('board_show', plugin=plugin, extension=extension)
    return func(arr, **plugin_kwargs)
