__version__ = '0.1.1'
__all__ = ['Repository', 'serve', 'make_tf_dataset', 'make_torch_dataset']

from functools import partial
from .remote.server import serve
from .repository import Repository



def raise_RuntimeError(message, *args, **kwargs):
    raise RuntimeError(message)


try:
    from .dataloaders import make_tf_dataset
except Exception:
    make_tf_dataset = partial(raise_RuntimeError, "Could not import tensorflow. Install dependencies")

try:
    from .dataloaders import make_torch_dataset
except Exception:
    make_torch_dataset = partial(raise_RuntimeError, "Could not import torch. Install dependencies")
