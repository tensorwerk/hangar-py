import os
import importlib
from tqdm import tqdm

from ._base import ImportExportBase


def guess_plugin(file_format):
    # read it from disk and make it configurable
    available_plugins = {'jpg': ['hangar.plugins.image', 'ImagePlugin']}
    try:
        plugin_details = available_plugins[file_format]
    except KeyError:
        print(f'No plugins has been registered for reading files of type {file_format}')
        raise
    return plugin_details


def get_plugin(details):
    plugin_module = details[0]
    plugin_class = details[1]
    return getattr(importlib.import_module(plugin_module), plugin_class)


def guess_and_get_plugin(directory):
    print('Finding files..')
    for root, dirs, files in os.walk(directory):
        allfiles = [os.path.join(root, file) for file in files]
    # TODO: assuming all files have same extension. May be verify
    file_format = os.path.splitext(allfiles[0])[1][1:]
    plugin_details = guess_plugin(file_format)
    plugin = get_plugin(plugin_details)
    return plugin(allfiles)

