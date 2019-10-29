"""
High level methods let user interact with hangar without diving into the internal
methods of hangar. We have enabled four basic entry points as high level methods

1. :func:`.load`
2. :func:`.save`
3. :func:`.show`
4. :func:`.board_show`

These entry points by itself is not capable of doing anything. But they are entry
points to the same methods in the `hangar.external` plugins available in pypi. These
high level entry points are used by the CLI for doing import, export and view
operations as well as the `hangarboard <https://github.com/tensorwerk/hangarboard>`_
for visualization (using ``board_show``)
"""
from typing import Tuple

import numpy as np

from .plugin_manager import PluginManager


pm = PluginManager()


def load(fpath: str,
         plugin: str = None,
         extension: str = None,
         **plugin_kwargs) -> Tuple[np.ndarray, str]:
    """
    Wrapper to load data from file into memory as numpy arrays using
    plugin's `load` method

    Parameters
    ----------
    fpath : str
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
    plugin_kwargs : dict
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be collected
        as ``plugin_kwargs``

    Returns
    -------
    img_array : :class:`numpy.ndarray`
        data returned from the given plugin.

    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('load', plugin=plugin, extension=extension)
    return func(fpath, **plugin_kwargs)


def save(arr: np.ndarray, outdir: str, sample_det: str, extension: str,
         plugin: str = None, **plugin_kwargs):
    """Wrapper plugin ``save`` methods which dump :class:`numpy.ndarray` to disk.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Numpy array to be saved to file
    outdir : str
        Target directory
    sample_det : str
        Sample name and type of the sample name formatted as
        ``sample_name_type:sample_name``
    extension : str
        Format of the file. This is used to infer which plugin to use in case
        plugin name is not provided. This cannot be ``None`` if ``plugin`` is
        also ``None``
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the given
        file format tried until a suitable. This cannot be ``None`` if
        ``extension`` is also ``None``

    Other Parameters
    ----------------
    plugin_kwargs : dict
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be
        collected as ``plugin_kwargs``

    Notes
    -----
    CLI or this method does not create the file name where to save. Instead
    they pass the required details downstream to the plugins to do that once
    they verify the given ``outdir`` is a valid directory. It is because we
    expect to get data entries where one data entry is one file (like images)
    and also data entries where multiple entries goes to single file (like
    CSV). With these ambiguous cases in hand, it's more sensible to let the
    plugin handle the file handling accordingly.
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('save', plugin=plugin, extension=extension)
    func(arr, outdir, sample_det, extension, **plugin_kwargs)


def show(arr: np.ndarray, plugin: str = None,
         extension: str = None, **plugin_kwargs):  # pragma: no cover
    """Wrapper to display :class:`numpy.ndarray` via plugin ``show`` method.

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Data to process into some human understandable representation.
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be ``None`` if
        ``extension`` is also ``None``
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be ``None`` if
        ``plugin`` is also ``None``

    Other Parameters
    ----------------
    plugin_kwargs : dict
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be
        collected as ``plugin_kwargs``
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('show', plugin=plugin, extension=extension)
    return func(arr, **plugin_kwargs)


def board_show(arr: np.ndarray, plugin: str = None,
               extension: str = None, **plugin_kwargs):
    """
    Wrapper to convert the numpy array using the ``board_show`` method
    of the plugin to make it displayable in the web UI

    Parameters
    ----------
    arr : :class:`numpy.ndarray`
        Data to process into some human understandable representation.
    plugin : str, optional
        Name of plugin to use.  By default, the preferred plugin for the
        given file format tried until a suitable. This cannot be ``None`` if
        ``extension`` is also ``None``
    extension : str, optional
        Format of the file. This is used to infer which plugin to use
        in case plugin name is not provided. This cannot be ``None`` if
        ``plugin`` is also ``None``

    Other Parameters
    ----------------
    plugin_kwargs : dict
        Plugin specific keyword arguments. If the function is being called from
        command line argument, all the unknown keyword arguments will be
        collected as ``plugin_kwargs``
    """
    if not pm.plugins_loaded:
        pm.reset_plugins()
    func = pm.get_plugin('board_show', plugin=plugin, extension=extension)
    return func(arr, **plugin_kwargs)
