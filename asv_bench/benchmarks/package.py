

class TimeImport(object):

    processes = 2
    repeat = (5, 10, 10.0)

    def timeraw_import(self):
        return """
        from hangar import Repository
        """