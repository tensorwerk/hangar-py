__version__ = '0.5.0.dev0'
__all__ = ['Repository', 'make_tf_dataset', 'make_torch_dataset']

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
