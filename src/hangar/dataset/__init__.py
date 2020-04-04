__all__ = ('make_numpy_dataset', 'make_tensorflow_dataset', 'make_torch_dataset')


# lazy loader
def __getattr__(name):
    global make_numpy_dataset, make_tensorflow_dataset, make_torch_dataset

    if name == 'make_numpy_dataset':
        from .numpy import make_numpy_dataset as mk_np_dataset
        make_numpy_dataset = mk_np_dataset
        return mk_np_dataset

    if name == 'make_tensorflow_dataset':
        from .tensorflow import make_tensorflow_dataset as mk_tf_dset
        make_tensorflow_dataset = mk_tf_dset
        return mk_tf_dset

    if name == 'make_torch_dataset':
        from .torch import make_torch_dataset as mk_torch_dset
        make_torch_dataset = mk_torch_dset
        return mk_torch_dset

    raise AttributeError(f"module {__name__} has no attribute {name}")
