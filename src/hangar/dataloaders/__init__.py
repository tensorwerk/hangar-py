__all__ = ('make_torch_dataset', 'make_tf_dataset')


# lazy loader
def __getattr__(name):
    global make_tf_dataset, make_torch_dataset

    if name == 'make_tf_dataset':
        from .tfloader import make_tf_dataset as tf_dset
        make_tf_dataset = tf_dset
        return tf_dset

    if name == 'make_torch_dataset':
        from .torchloader import make_torch_dataset as torch_dset
        make_torch_dataset = torch_dset
        return torch_dset

    raise AttributeError(f"module {__name__} has no attribute {name}")
