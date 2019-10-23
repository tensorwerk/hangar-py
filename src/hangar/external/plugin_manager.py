import pkg_resources
from typing import Callable


class PluginManager(object):
    valid_provides = ['load', 'save', 'show', 'board_show']

    def __init__(self):
        self.reset_plugins()
        self._read_defaults()

    def _clear_plugins(self):
        """
        Clear the plugin state to the default, i.e., where no plugins are loaded
        """
        self._plugin_store = {}  # ex: {'pil': loaded_pil_module}
        self._default_plugins = {}  # ex: {'jpg': {'save': 'pil'}}

    def _scan_plugins(self):
        """
        Scan for entry points, find the plugins and store them in provided storage
        containers
        """
        for entry_point in pkg_resources.iter_entry_points('hangar.external.plugins'):
            PluginClass = entry_point.load()
            self._plugin_store[entry_point.name] = PluginClass()

    def _read_defaults(self):
        """
        Populate preferred plugin dict that maps file formats to plugins and methods. This
        is used to infer which plugin to use at runtime based on file format
        """
        for fname, plugin in self._plugin_store.items():
            generator = ((ext, method) for ext in plugin.accepts for method in plugin.provides)
            for pair in generator:
                if pair not in self._default_plugins:
                    self._default_plugins[pair] = fname

    def reset_plugins(self):
        self._clear_plugins()
        self._scan_plugins()

    def get_plugin(self, method: str, plugin: str = None, extension: str = None) -> Callable:
        """
        Load installed plugin. User either needs to specify which plugin to load or should
        provide file format to infer which plugin to use

        Parameters
        ----------
        method : str
            Which method to load. Allowed methods are stored in `valid_provides`
        plugin : str, optional
            Which plugin to load the method from. Cannot leave as `None` if `extension` is also None
        extension : str, optional
            format of the data on the disk. This information is used to infer which plugin to use
            in case `plugin` not provided explicitly. Cannot leave as `None` if `plugin` is also None

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
