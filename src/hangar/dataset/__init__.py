__all__ = ('make_numpy_dataset', 'make_torch_dataset', 'make_tensorflow_dataset')

from typing import Sequence, Callable, TYPE_CHECKING, Union, List, Tuple

if TYPE_CHECKING:
    from ..columns import ModifierTypes as Columns
    from .torch_dset import TorchDataset
    from .numpy_dset import NumpyDataset
    from .tensorflow_dset import tf_Dataset
    KeyType = Union[str, int, List, Tuple]


def make_numpy_dataset(
        columns: Sequence['Columns'],
        keys: 'KeyType' = None,
        batch_size: int = None,
        drop_last: bool = False,
        shuffle: bool = True,
        collate_fn: Callable = None) -> 'NumpyDataset':
    """Group column into a single numpy dataset, provides iterative looping over data.

    This API also provides the options to batch the data which is a major difference
    between other dataset APIs. In traditional Machine learning applications, it's quite
    natural to load the whole dataset as a single batch because it's possible to fit into
    the system memory. Passing the size of the dataset as the batch size would make it
    possible here to do just that. This API also acts as an entry point for other
    non-supported frameworks to load data from hangar as batches into the training loop.

    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A column object, a tuple of column object or a list of column
        objects.
    keys : Union[str, int, List, Tuple]
        An sequence collection of sample names. If given only those samples will
        fetched from the column
    batch_size : int
        Size of the batch. This will batch the dataset on the zeroth dimension. For
        example, if the data is of the shape (H x W x C) the batched data will be shaped
        as (B x H x W x C) where B is the batch size
    drop_last : bool
        Should the last uncompleted batch be dropped
    shuffle : bool
        Should the data be shuffled on each epoch
    collate_fn : Callable
        A function to collate samples together in a batch. In case this option is absent,
        the heuristics to collate the batch is
            1. If the column is an ndarray flat column, then `np.stack` will be used
            2. If the column is with any other properties, `list.append` will be used
        Note that the batch of data that comes to callate_fn will have each elements consist
        of datapoints from all the columns. For example, if the columns from where the data
        being fetched are col1 and col2 then the batch would look like

        ```python
        [
            (data0_col1, data0_col2),
            (data1_col1, data1_col2),
            (data2_col1, data2_col2),
            ...
        ]
        ```

    Examples
    --------
    >>> from hangar import Repository
    >>> from hangar.dataset import make_numpy_dataset
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> imgcol = co.columns['images']
    >>> classcol = co.columns['classes']
    >>> dataset = make_numpy_dataset((imgcol, classcol), batch_size=64)
    >>> for batch in dataset:
    ...     out = train_model(batch[0])
    ...     loss = loss_fn(out, batch[1])

    Returns
    -------
    :class: `~.numpy_dset.NumpyDataset`
    """
    from .numpy_dset import _make_numpy_dataset
    return _make_numpy_dataset(
        columns=columns,
        keys=keys,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        collate_fn=collate_fn)


def make_torch_dataset(
        columns: Sequence['Columns'],
        keys: 'KeyType' = None,
        as_dict: bool = False) -> 'TorchDataset':
    """Returns a :class:`torch.utils.data.Dataset` object which can be loaded into
    a :class:`torch.utils.data.DataLoader`.

    .. note::

        PyTorch's :class:`torch.utils.data.DataLoader` can effectively do custom
        operations such as shuffling, batching, multiprocessed read etc and hence we
        limit the surface area of the dataset API here just to open the channel for
        reading. Use DataLoaders for such operations

    .. warning::

       On Windows systems, setting the parameter ``num_workers`` in the
       resulting :class:`torch.utils.data.DataLoader` method will result in a
       RuntimeError or deadlock. This is due to limitations of multiprocess
       start methods on Windows itself. Using the default argument value
       (``num_workers=0``) will let the DataLoader work in single process mode
       as expected.

    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A column object, a tuple of column object or a list of column
        objects.
    keys : Union[str, int, List, Tuple]
        An sequence collection of sample names. If given only those samples will
        fetched from the column
    as_dict : bool
        Return the data as an OrderedDict with column names as keys. If False, it returns
        a tuple of arrays

    Examples
    --------
    >>> from hangar import Repository
    >>> from torch.utils.data import DataLoader
    >>> from hangar.dataset import make_torch_dataset
    >>> from collections import namedtuple
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> imgcol = co.columns['images']
    >>> classcol = co.columns['classes']
    >>> dataset = make_torch_dataset((imgcol, classcol), as_dict=True)
    >>> loader = DataLoader(dataset, batch_size=16)
    >>> for batch in loader:
    ...     out = train_model(batch['images'])
    ...     loss = loss_fn(out, batch['classes'])

    Returns
    -------
    :class:`torch.utils.data.Dataset`
    """
    from .torch_dset import _make_torch_dataset
    return _make_torch_dataset(columns=columns, keys=keys, as_dict=as_dict)


def make_tensorflow_dataset(
        columns: Sequence['Columns'],
        keys: 'KeyType' = None,
        shuffle: bool = False) -> 'tf_Dataset':
    """Make a tensorflow dataset from a hangar column.

    This method uses `from_generator` function from `tensorflow.data.Dataset` with a
    generator function that wraps all the hangar columns. This function also accepts an
    optional ``shuffle`` argument that does a global shuffle across all the samples.
    This is convenient since Tensorflow Dataset does shuffling by loading the subset
    of data which can fit into the memory and shuffle that subset.

    .. warning::

        This function relies on `tf.data.Dataset.from_generator` and which calls into the
        python interpreter for running the generator funciton. This generator function
        will not be serialized in a GraphDef and hence has limited portability. The
        operation must run in the same address space as the Python program that calls
        'make_tensorflow_dataset'. Also, since it calls back into the python interpreter,
        we'll have the GIL problem and is not parellel-izable even with a `Dataset.map`
        call. In fact, any attempts to parellelize the read will result in worse
        performance

    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A column object, a tuple of column object or a list of column objects`
    keys : Union[str, int, List, Tuple]
        An sequence of sample names. If given only those samples will fetched from
        the column
    shuffle : bool
        The generator uses this to decide a global shuffle across all the samples is
        required or not. But user doesn't have any restriction on doing`column.shuffle()`
        on the returned column

    Examples
    --------
    >>> from hangar import Repository
    >>> from hangar.dataset import make_tensorflow_dataset
    >>> import tensorflow as tf
    >>> tf.compat.v1.enable_eager_execution()
    >>> repo = Repository('')
    >>> co = repo.checkout()
    >>> data = co.columns['mnist_data']
    >>> target = co.columns['mnist_target']
    >>> tf_dset = make_tensorflow_dataset([data, target])
    >>> tf_dset = tf_dset.batch(512)
    >>> for bdata, btarget in tf_dset:
    ...     print(bdata.shape, btarget.shape)

    Returns
    -------
    :class:`tf_Dataset`
    """
    from .tensorflow_dset import _make_tensorflow_dataset
    return _make_tensorflow_dataset(columns=columns, keys=keys, shuffle=shuffle)
