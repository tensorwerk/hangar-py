__version__ = '0.5.1'
__all__ = ('make_torch_dataset', 'make_tf_dataset', 'Repository')

from functools import partial
from .repository import Repository


def raise_ImportError(message, *args, **kwargs):
    raise ImportError(message)


try:
    from .dataloaders.tfloader import make_tf_dataset
except ImportError:
    make_tf_dataset = partial(raise_ImportError, "Could not import tensorflow. Install dependencies")

try:
    from .dataloaders.torchloader import make_torch_dataset
except ImportError:
    make_torch_dataset = partial(raise_ImportError, "Could not import torch. Install dependencies")

