import os


class BasePlugin(object):
    def __init__(self, provides, accepts):
        # TODO: Doc strings
        self._provides = provides
        self._accepts = accepts

    @property
    def provides(self):
        return self._provides

    @property
    def accepts(self):
        return self._accepts

    @staticmethod
    def sample_name(fpath):
        return os.path.splitext(os.path.basename(fpath))[0]
