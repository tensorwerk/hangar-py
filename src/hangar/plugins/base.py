class ImportExportBase:

    def __init__(self):
        # Variables required by __next__. Developers should not interact with it
        self._current_index = 0
        self._files = []

        # Variables required for plugin registration and auto selection. This should/can
        # be overridden by child classes
        self.file_formats = []
        self.name = self.__class__.__name__  # TODO: may be a property

    def __iter__(self):
        self._current_index = 0
        return self

    def __len__(self):
        return len(self._files)

    def __next__(self):
        # TODO: make sure this implementation doesnt have any bug/corner cases
        try:
            file = self._files[self._current_index]
            data = self.load(file)
        except IndexError:
            raise StopIteration
        self._current_index += 1
        return file, data

    def _register(self, plugin_details):
        assert self.name == plugin_details[1]
        # TODO check if plugin name already exist
        # TODO write to a file
        pass

    def load(self, file):
        raise NotImplemented("Child class should override `load` function")

    def save(self, file, data):
        raise NotImplemented("Child class should override `save` function")
