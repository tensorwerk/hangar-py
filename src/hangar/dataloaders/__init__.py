from collections import namedtuple
from functools import partial
from .common import GroupedDsets
from .tfutils import yield_data


def make_torch_dataset(hangar_datasets, keys=None, index_range=None, field_names=None):
    """
    Returns a `torch.utils.data.Dataset` object which can be loaded into a
    `torch.utils.data.DataLoader`.

    Parameters
    ----------
    hangar_datasets : `hangar.dataset.DatasetDataReader` or a Collection
        A dataset object, a tuple of dataset object or a list of dataset objects`
    keys : list or tuple
        An array of sample names. If given only those samples will fetched from the dataset
    index_range : slice
        A python slice object which will be used to find the subset of dataset.
        Argument `keys` takes priority over `index_range` i.e. if both are given, keys
        will be used and `index_range` will be ignored
    field_names : list or tuple of str
        An array of field names used as the `field_names` for the returned namedtuple. If not
        given, dataset names will be used as the field_names.

    Examples
    --------
    >>> from hangar import Repository
    >>> from torch.utils.data import DataLoader
    >>> from hangar.dataloaders import make_torch_dataset
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> dset = co.datasets['dummy_dset']
    >>> torch_dset = make_torch_dataset(dset, index_range=slice(1, 100))
    >>> loader = DataLoader(torch_dset, batch_size=16)
    >>> for batch in loader:
    ...     train_model(batch)

    Returns
    -------
    `torch.utils.data.Dataset` object

    """
    from .torchutils import TorchDataset
    gdsets = GroupedDsets(hangar_datasets, keys, index_range)
    if field_names:
        if not isinstance(field_names, (list, tuple)):
            raise TypeError(f"`field_names` must be a list/tuple, not {type(field_names)}")
        if len(field_names) != len(hangar_datasets):
            raise RuntimeError("length mismatch: `field_names` and `hangar_datasets` must "
                               "have same number of elements")
        ret_wrapper = namedtuple('BatchTuple', field_names=field_names)
    else:
        ret_wrapper = namedtuple('BatchTuple', field_names=gdsets.dataset_names, rename=True)
    return TorchDataset(gdsets.dataset_array, gdsets.sample_names, ret_wrapper)


def make_tf_dataset(hangar_datasets, keys=None, index_range=None, shuffle=True):
    """
    Uses the hangar datasets to make a tensorflow dataset. It uses `from_generator`
    function from `tensorflow.data.Dataset` with a generator function that wraps all
    the hangar datasets. In such instances tensorflow Dataset does shuffle by loading
    the subset of data which can fit into the memory and shuffle that subset. Since it
    is not really a global shuffle `make_tf_dataset` accepts a `shuffle` argument which
    will be used by the generator to shuffle each time it is being called.

    .. warning::

        `tf.data.Dataset.from_generator` currently uses `tf.compat.v1.py_func()` internally.
        Hence the serialization function (`yield_data`) will not be serialized in a
        GraphDef. Therefore, you won't be able to serialize your model and restore it
        in a different environment if you use `make_tf_dataset`.
        The operation must run in the same address space as the Python program that calls
        tf.compat.v1.py_func(). If you are using distributed TensorFlow, you must run a
        tf.distribute.Server in the same process as the program that calls
        tf.compat.v1.py_func() and you must pin the created operation to a device in that
        server (e.g. using with tf.device():)

    Parameters
    ----------
    hangar_datasets : `hangar.dataset.DatasetDataReader` or a Collection
        A dataset object, a tuple of dataset object or a list of dataset objects`
    keys : list or tuple
        An array of sample names. If given only those samples will fetched from the dataset
    index_range : slice
        A python slice object which will be used to find the subset of dataset.
        Argument `keys` takes priority over `index_range` i.e. if both are given, keys
        will be used and `index_range` will be ignored
    shuffle : bool
        generator uses this to decide a global shuffle accross all the samples is required
        or not. But user doesn't have any restriction on doing`dataset.shuffle()` on the
        returned dataset

    Examples
    --------
    >>> from hangar import Repository
    >>> from hangar.dataloaders import make_tf_dataset
    >>> import tensorflow as tf
    >>> tf.compat.v1.enable_eager_execution()
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> data = co.datasets['mnist_data']
    >>> target = co.datasets['mnist_target']
    >>> tf_dset = make_tf_dataset([data, target])
    >>> tf_dset = tf_dset.batch(512)
    >>> for bdata, btarget in tf_dset:
    ...     print(bdata.shape, btarget.shape)


    Returns
    -------
    `tf.data.Dataset` object
    """
    try:
        import tensorflow as tf
    except (ImportError, ModuleNotFoundError):
        raise RuntimeError("Could not import Tensorflow. Install dependencies")
    gdsets = GroupedDsets(hangar_datasets, keys, index_range)
    generator = partial(yield_data, gdsets.dataset_array, gdsets.sample_names, shuffle)
    # TODO: pass proper shapes for fixed shape input
    return tf.data.Dataset.from_generator(generator, gdsets.get_types(converter=tf.as_dtype),
                                          output_shapes=gdsets.get_shapes(converter=tf.TensorShape))
