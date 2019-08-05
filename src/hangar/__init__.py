__version__ = '0.1.1'

from .remote.server import serve
from .repository import Repository

__all__ = ['Repository', 'serve']

try:
    from .dataloaders import make_tf_dataset
    __all__.append('make_tf_dataset')
except Exception:
    pass

try:
    from .dataloaders import make_torch_dataset
    __all__.append('make_torch_dataset')
except Exception:
    pass