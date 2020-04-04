__version__ = '0.5.0'
__all__ = ('make_torch_dataset', 'make_tf_dataset', 'Repository')

from .repository import Repository


def __getattr__(name):  # lazy loader
    if name == 'make_tf_dataset':
        from .dataloaders import make_tf_dataset
        return make_tf_dataset

    if name == 'make_torch_dataset':
        from .dataloaders import make_torch_dataset
        return make_torch_dataset

    raise AttributeError(f"module {__name__} has no attribute {name}")
