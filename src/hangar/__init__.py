__version__ = '0.5.0'
__all__ = ('make_torch_dataset', 'make_tensorflow_dataset', 'make_numpy_dataset', 'Repository')

from .repository import Repository
from .dataset import make_numpy_dataset


def __getattr__(name):
    """Lazy Loader that defers the loading of heavy packages such as tensorflow & pytorch
    """
    if name == 'make_torch_dataset':
        from .dataset import make_torch_dataset
        return make_torch_dataset
    if name == 'make_tensorflow_dataset':
        from .dataset import make_tensorflow_dataset
        return make_tensorflow_dataset
    raise AttributeError(f"module {__name__} has no attribute {name}")

