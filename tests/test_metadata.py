import pytest


@pytest.mark.skip(reason='not implemented')
class TestMetadata(object):

    def test_invalid_name(self, repo):
        pass

    def test_name_conflict(self):
        pass

    def test_other_datatype(self):
        pass

    def test_read_only_mode(self):
        pass

    def test_add_without_enough_args(self):
        pass

    def test_remove_without_enough_args(self):
        pass

    def test_get_wrong_key(self):
        pass

    def test_loop_through(self):
        pass

    def test_context_manager(self):
        pass
