"""
Hangar's external plugin system is designed to make it flexible for users to
write custom plugins for custom data formats. External plugins should be python
installables and should make itself discoverable using package meta data. A
`detailed documentation <https://packaging.python.org/guides/creating-and-discovering-plugins/#using-package-metadata>`_
can be found in the official python doc. But for a headstart and to avoid going
through this somewhat complex process, we have made a `cookiecutter
<https://github.com/tensorwerk/hangar-external-plugin-cookiecutter>`_ package.
All the hangar plugins follow the naming standard similar to Flask plugins i.e
`hangar_pluginName`
"""

import os


class BasePlugin(object):
    """Base plugin class from where all the external plugins should be inherited.

    Child classes can have four methods to expose - ``load``, ``save``,
    ``show`` and ``board_show``. These are considered as valid methods and
    should be passed as the first argument while initializing the parent from
    child. Child should also inform the parent about the acceptable file
    formats by passing that as second argument. :class:`.BasePlugin` accepts
    ``provides`` and ``accepts`` on init and exposes them which is then used by
    plugin manager while loading the modules. BasePlugin also provides
    ``sample_name`` function to figure out the sample name from the file path.
    This function is used by ``load`` method to return the sample name which is
    then used by hangar as a key to save the data
    """
    def __init__(self, provides, accepts):
        self._provides = provides
        self._accepts = accepts

    @property
    def provides(self):
        return self._provides

    @property
    def accepts(self):
        return self._accepts

    def load(self, fpath, *args, **kwargs):
        """Load some data file on disk to recover it in :class:`numpy.ndarray` form.

        Loads the data provided from the disk for the file path given and
        returns the data as :class:`numpy.ndarray` and name of the data sample.
        Names returned from this function will be used by the import cli system
        as the key for the returned data. This function can return either a
        single :class:`numpy.ndarray`, sample name, combination, or a generator
        that produces one of the the above combinations. This helps when the
        input file is not a single data entry like an image but has multiple
        data points like CSV files.

        An example implementation that returns a single data point:

        .. code-block:: python

            def load(self, fpath, *args, **kwargs):
                data = create_np_array('myimg.jpg')
                name = create_sample_name('myimg.jpg')  # could use `self.sample_name`
                return data, name

        An example implementation that returns a generator could look like this:

        .. code-block:: python

            def load(self, fpath, *args, **kwargs):
                for i, line in enumerate('myfile.csv'):
                    data = create_np_array(line)
                    name = create_sample_name(fpath, i)
                    yield data, name
        """
        raise NotImplementedError

    def save(self, fname, arr, *args, **kwargs):
        """Save data in a :class:`numpy.ndarray` to a specific file format on disk.

        If the plugin is developed for files like CSV, JSON, etc - where
        multiple data entry would go to the same file - this should check
        whether the file exist already and weather it should modify / append
        the new data entry to the structure, instead of overwriting it or
        throwing an exception.
        """
        raise NotImplementedError

    def show(self, arr, *args, **kwargs):
        """Show/Display the data to the user.

        This function should process the input :class:`numpy.ndarray` and show
        that to the user using a data dependant display mechanism. A good
        example for such a system is ``matplotlib.pyplot``'s ``plt.show``,
        which displays the image data inline in the running terminal / kernel
        ui.
        """
        raise NotImplementedError

    def board_show(self, arr, *args, **kwargs):
        """Show/display data in hangarboard format.

        Hangarboard is capable of displaying three most common data formats:
        image, text and audio. This function should process the input
        :class:`numpy.ndarray` data and convert it to any of the supported
        formats.
        """
        raise NotImplementedError

    @staticmethod
    def sample_name(fpath: os.PathLike) -> str:
        """Sample the name from file path.

        This function comes handy since the :meth:`.load` method needs to
        ``yield`` or ``return`` both data and sample name. If there no specific
        requirements regarding sample name creation, you can use this function
        which removes the extension from the file name and returns just the
        name. For example, if filepath is ``/path/to/myfile.ext``, then it
        returns ``myfile``

        Parameters
        ----------
        fpath : os.PathLike
            Path to the file which is being loaded by `load`
        """
        return os.path.splitext(os.path.basename(fpath))[0]
