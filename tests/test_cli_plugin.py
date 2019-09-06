import pytest

from hangar.cli import io
from hangar.cli.io import manage_plugins


priority_plugin = 'pil'


@pytest.fixture()
def plugin_reset():
    io.use_plugin('pil')
    yield None
    io.reset_plugins()


@pytest.fixture()
def protect_preferred_plugins():
    """Contexts where `preferred_plugins` can be modified w/o side-effects."""
    preferred_plugins = manage_plugins.preferred_plugins.copy()
    yield None
    manage_plugins.preferred_plugins = preferred_plugins


def test_failed_use(plugin_reset):
    with pytest.raises(ValueError):
        manage_plugins.use_plugin('asd')


def test_use_priority(plugin_reset):
    manage_plugins.use_plugin(priority_plugin)
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert plug == priority_plugin

    manage_plugins.use_plugin('matplotlib')
    plug, func = manage_plugins.plugin_store['imread'][0]
    assert plug == 'matplotlib'


def test_load_preferred_plugins_all(plugin_reset, protect_preferred_plugins):
    from hangar.cli.io._plugins import pil_plugin, matplotlib_plugin

    manage_plugins.preferred_plugins = {'all': ['pil'],
                                        'imshow': ['matplotlib']}
    manage_plugins.reset_plugins()

    for plugin_type in ('imread', 'imsave'):
        plug, func = manage_plugins.plugin_store[plugin_type][0]
        assert func == getattr(pil_plugin, plugin_type)
    plug, func = manage_plugins.plugin_store['imshow'][0]
    assert func == getattr(matplotlib_plugin, 'imshow')


def test_load_preferred_plugins_imread(plugin_reset, protect_preferred_plugins):
    from hangar.cli.io._plugins import pil_plugin, matplotlib_plugin

    manage_plugins.preferred_plugins['imread'] = ['pil']
    manage_plugins.reset_plugins()

    plug, func = manage_plugins.plugin_store['imread'][0]
    assert func == pil_plugin.imread
    plug, func = manage_plugins.plugin_store['imshow'][0]
    assert func == matplotlib_plugin.imshow, func.__module__


def test_load_preferred_plugins_order(plugin_reset, protect_preferred_plugins):
    order = manage_plugins.plugin_order()

    assert order['imread'] == ['pil', 'matplotlib']
    assert order['imsave'] == ['pil']
    assert order['imshow'] == ['matplotlib']
    assert order['_app_show'] == ['matplotlib']
