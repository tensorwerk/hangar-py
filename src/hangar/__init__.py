__version__ = '0.5.0'
__all__ = ('Repository', 'make_numpy_dataset', 'make_torch_dataset', 'make_tensorflow_dataset')

from .repository import Repository
import warnings


def __dir__():
    return list(globals().keys()) + ['make_numpy_dataset',
                                     'make_torch_dataset',
                                     'make_tensorflow_dataset']


def __getattr__(name):
    warnings.warn("Dataloaders are experimental in the current release.", UserWarning)
    if name == 'make_numpy_dataset':
        from .dataset import make_numpy_dataset
        return make_numpy_dataset
    elif name == 'make_torch_dataset':
        from .dataset import make_torch_dataset
        return make_torch_dataset
    elif name == 'make_tensorflow_dataset':
        from .dataset import make_tensorflow_dataset
        return make_tensorflow_dataset
    raise AttributeError(f"module {__name__} has no attribute {name}")

