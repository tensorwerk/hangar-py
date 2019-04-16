__version__ = '0.0.0'

from . import config_logging
from .remote.hangar_server import serve
from .repository import Repository

config_logging.setup_logging()


__all__ = ['Repository', 'serve']
