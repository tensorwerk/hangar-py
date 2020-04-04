__all__ = ('make_numpy_dataset', 'make_tensorflow_dataset', 'make_torch_dataset')

from .numpy_dset import make_numpy_dataset


def __getattr__(name):
    global make_tensorflow_dataset, make_torch_dataset

    if name == 'make_tensorflow_dataset':
        from .tensorflow_dset import make_tensorflow_dataset as mk_tf_dset
        make_tensorflow_dataset = mk_tf_dset
        return mk_tf_dset

    if name == 'make_torch_dataset':
        from .torch_dset import make_torch_dataset as mk_torch_dset
        make_torch_dataset = mk_torch_dset
        return mk_torch_dset

    raise AttributeError(f"module {__name__} has no attribute {name}")
