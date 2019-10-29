import pkg_resources
from typing import Callable


class PluginManager(object):
    """
    Container class that holds the information about available plugins and
    provides required method to fetch and clean up the plugin systems
    """

    valid_provides = ['load', 'save', 'show', 'board_show']

    def __init__(self):
        self._plugin_store = {}  # ex: {'pil': loaded_pil_module}
        self._default_plugins = {}  # ex: {'jpg': {'save': 'pil'}}
        self.plugins_loaded = False

    def reset_plugins(self):
        """
        Reset plugin clears the existing storages and then scans
        `hangar.external.plugins` for plugins. Once `plugin store` is populated,
        it creates finds the default plugins to make auto-inference based on
        file format possible
        """
        self._clear_plugins()
        self._scan_plugins()
        self._read_defaults()

    def _clear_plugins(self):
        """
        Clear the plugin state to the default, i.e., where no plugins are loaded
        """
        self._plugin_store.clear()
        self._default_plugins.clear()
        self.plugins_loaded = False

    def _scan_plugins(self):
        """
        Scan for entry points, find the plugins and store them in provided storage
        containers
        """
        for entry_point in pkg_resources.iter_entry_points('hangar.external.plugins'):
            PluginClass = entry_point.load()
            self._plugin_store[entry_point.name] = PluginClass()
        self.plugins_loaded = True

    def _read_defaults(self):
        """
        Populate default plugin dict that maps file formats to plugins and methods. This
        is used to infer which plugin to use at runtime based on file format
        """
        for fname, plugin in self._plugin_store.items():
            generator = ((ext, method) for ext in plugin.accepts for method in plugin.provides)
            for pair in generator:
                if pair not in self._default_plugins:
                    self._default_plugins[pair] = fname

    def get_plugin(self, method: str, plugin: str = None, extension: str = None) -> Callable:
        """Load installed plugin.

        User either needs to specify which plugin to load or should provide
        file format to infer which plugin to use

        Parameters
        ----------
        method : str
            Which method to import from the plugin. Methods implemented by the
            extension author should be declared as arguments passed into the
            BasePlugin superclass constructor
        plugin : str, optional
            Which plugin to load the method from. Cannot leave as ``None`` if
            ``extension`` is also ``None``
        extension : str, optional
            format of the data on the disk. This information is used to infer
            which plugin to use in case ``plugin`` is not provided explicitly.
            Cannot leave as ``None`` if ``plugin`` is also ``None``

        Returns
        -------
        plugin_method : function
            requested method from the plugin
        """

        if not plugin:
            if not extension:
                raise ValueError("Both `plugin` and `extension` cannot be empty together")

            plugin = self._default_plugins.get((extension, method))
            if plugin is None:
                raise ValueError(f"No plugins found for the file extension {extension} that could "
                                 f"do {method}")
        else:
            if plugin not in self._plugin_store:
                raise ValueError(f"Plugin {plugin} not found")
        loaded_plugin = self._plugin_store[plugin]
        try:
            return getattr(loaded_plugin, method)
        except AttributeError:
            raise RuntimeError(f"Method {method} found in `plugin.provides` but could "
                               f"not invoke from {plugin}. You might have forgot to define "
                               f"the function")
