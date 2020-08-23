from typing import Sequence, Callable, List, Tuple, Union
import typing
from functools import partial
import random

try:
    import tensorflow as tf
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        'Could not import "tensorflow" library. Ensure library is '
        'installed correctly to use tensorflow dataloader functions') from None

from .common import HangarDataset

if typing.TYPE_CHECKING:
    tf_TensorType = tf.python.framework.dtypes.DType
    tf_TensorShape = tf.TensorShape
    tf_Dataset = tf.data.Dataset
    KeyType = Union[str, int, List, Tuple]
    from ..columns.column import ModifierTypes as Columns
    import numpy as np


def yield_data(dataset: HangarDataset, indices: list,
               shuffle: bool) -> Tuple['np.ndarray']:
    if shuffle:
        random.shuffle(indices)
    for i in indices:
        out = dataset.index_get(i)
        yield out if isinstance(out, tuple) else (out,)


def _make_tensorflow_dataset(columns: Sequence['Columns'],
                             keys: 'KeyType' = None,
                             shuffle: bool = False) -> 'tf_Dataset':
    """Make a tensorflow dataset from a hangar column.

    This method uses `from_generator` function from `tensorflow.data.Dataset` with a
    generator function that wraps all the hangar columns. This function also accepts an
    optional ``shuffle`` argument that does a global shuffle across all the samples.
    This is convenient since Tensorflow Dataset does shuffling by loading the subset
    of data which can fit into the memory and shuffle that subset.

    .. Note::

        Column with layouts ``str`` or ``ndarray nested`` are not compatible with the
        dataset APIs in the current release. So making dataset is only possible for
        columns with layout ``ndarray flat``

    .. warning::

        This function relies on `tf.data.Dataset.from_generator` and which calls into the
        python interpreter for running the generator funciton. This generator function
        will not be serialized in a GraphDef and hence has limited portability. The
        operation must run in the same address space as the Python program that calls
        'make_tensorflow_dataset'. Also, since it calls back into the python interpreter,
        we'll have the GIL problem and is not parellel-izable even with a `Dataset.map`
        call. In fact, any attempts to parellelize the read will result in worse
        performance

    .. note::

        This is an experimental method in the current Hangar version. Please be aware
        that Significant changes may be introduced in future releases without advance
        notice or deprication warnings.

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

    DEVELOPER NOTE
    --------------
    - Any update to this function signature or docstring must be reflected in the
      equivalent loader function in hangar/dataset/__init__.py. This function is
      "coppied" to a top level __init__.py to allow unified API and lazyloader access
    """

    dataset = HangarDataset(columns, keys)
    indices = list(range(len(dataset)))
    generator: Callable = partial(yield_data, dataset, indices, shuffle)
    shapes: List[tf_TensorShape] = []
    types: List[tf_TensorType] = []

    for col in dataset.columns.values():
        if col.schema_type == 'variable_shape':
            shape = (None,) * len(col.shape)
        else:
            shape = col.shape
        shapes.append(tf.TensorShape(shape))
        types.append(tf.as_dtype(col.dtype))

    return tf.data.Dataset.from_generator(generator=generator,
                                          output_types=tuple(types),
                                          output_shapes=tuple(shapes))
