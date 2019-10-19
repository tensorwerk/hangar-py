__version__ = '0.4.0b0'
__all__ = ['Repository', 'serve', 'make_tf_dataset', 'make_torch_dataset']

from functools import partial
from .remote.server import serve
from .repository import Repository


def raise_ImportError(message, *args, **kwargs):  # pragma: no cover
    raise ImportError(message)


try:                 # pragma: no cover
    from .dataloaders.tfloader import make_tf_dataset
except ImportError:  # pragma: no cover
    make_tf_dataset = partial(raise_ImportError, "Could not import tensorflow. Install dependencies")

try:                 # pragma: no cover
    from .dataloaders.torchloader import make_torch_dataset
except ImportError:  # pragma: no cover
    make_torch_dataset = partial(raise_ImportError, "Could not import torch. Install dependencies")
