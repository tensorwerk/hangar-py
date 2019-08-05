class ImportExportBase:

    def __init__(self):
        self.__current_index = 0
        self.files = []
        # TODO write which plugin to use for exporting in the metadata

    def __iter__(self):
        self.__current_index = 0
        return self

    def load(self, file):
        raise NotImplemented("Child class should override `load` function")

    def __next__(self):
        # TODO: make sure this implementation doesnt have any bug/corner cases
        try:
            file = self.files[self.__current_index]
            data = self.load(file)
        except IndexError:
            raise StopIteration
        self.__current_index += 1
        return file, data
