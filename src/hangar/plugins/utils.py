import os
import importlib

# TODO: read it from disk and make it configurable
format2plugin = {'jpg': 'ImagePlugin'}
plugin2details = {'ImagePlugin': ['hangar.plugins.image', 'ImagePlugin']}


def list_files(path):
    allfiles = []
    for root, dirs, files in os.walk(path):
        allfiles.extend([os.path.join(root, file) for file in files])


def guess_plugin(file):
    file_format = os.path.splitext(file)[1][1:]
    try:
        plugin_name = format2plugin[file_format]  # TODO keyerror
    except KeyError:
        print(f'No plugins has been registered for reading files of type {file_format}')
        raise
    return plugin_name


def get_executable_plugin(name):
    plugin_module, plugin_class = plugin2details[name]  # TODO keyerror
    plugin_exec = getattr(importlib.import_module(plugin_module), plugin_class)
    return plugin_exec()
