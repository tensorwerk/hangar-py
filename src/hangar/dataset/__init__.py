__all__ = ('make_numpy_dataset', 'make_torch_dataset', 'make_tensorflow_dataset')


# lazy loader
def __getattr__(name):
    global make_numpy_dataset, make_tensorflow_dataset, make_torch_dataset

    if name == 'make_numpy_dataset':
        from .numpy import make_numpy_dataset as np_dset
        make_numpy_dataset = np_dset
        return np_dset

    if name == 'make_tensorflow_dataset':
        from .tensorflow import make_tf_dataset as tf_dset
        make_tensorflow_dataset = tf_dset
        return tf_dset

    if name == 'make_torch_dataset':
        from .torch import make_torch_dataset as torch_dset
        make_torch_dataset = torch_dset
        return torch_dset

    raise AttributeError(f"module {__name__} has no attribute {name}")
