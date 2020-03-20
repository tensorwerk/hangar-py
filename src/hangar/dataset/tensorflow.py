from typing import Sequence, Callable, List, Tuple
import typing
from functools import partial
import random

import tensorflow as tf
from .common import HangarDataset
from ..utils import experimental

if typing.TYPE_CHECKING:
    import numpy as np
    tf_TensorType = tf.python.framework.dtypes.DType
    tf_TensorShape = tf.TensorShape
    from hangar.columns.column import ModifierTypes as Columns


def yield_data(dataset: HangarDataset, shuffle: bool) -> Tuple['np.ndarray']:
    if shuffle:
        random.shuffle(dataset.keys)
    for name in dataset.keys:
        yield dataset[name]


@experimental
def make_tensorflow_dataset(columns: Sequence['Columns'], keys: Sequence[str] = None,
                            shuffle: bool = False) -> tf.data.Dataset:
    """
    Uses the hangar columns to make a tensorflow dataset. It uses `from_generator`
    function from `tensorflow.data.Dataset` with a generator function that wraps all the
    hangar columns. This function also accepts an optional ``shuffle`` argument that does
    a global shuffle across all the samples. This is convenient since Tensorflow Dataset
    does shuffling by loading the subset of data which can fit into the memory and
    shuffle that subset

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


    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A column object, a tuple of column object or a list of column objects`
    keys : Sequence[str]
        An sequence of sample names. If given only those samples will fetched from
        the column
    shuffle : bool
        The generator uses this to decide a global shuffle across all the samples is
        required or not. But user doesn't have any restriction on doing`column.shuffle()`
        on the returned column


    Examples
    --------
    >>> from hangar import Repository
    >>> from hangar import make_tf_dataset
    >>> import tensorflow as tf
    >>> tf.compat.v1.enable_eager_execution()
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> data = co.columns['mnist_data']
    >>> target = co.columns['mnist_target']
    >>> tf_dset = make_tf_dataset([data, target])
    >>> tf_dset = tf_dset.batch(512)
    >>> for bdata, btarget in tf_dset:
    ...     print(bdata.shape, btarget.shape)


    Returns
    -------
    :class:`tf.data.Dataset`
    """

    dataset = HangarDataset(columns, keys)
    generator: Callable = partial(yield_data, dataset, shuffle)
    types: List[tf_TensorType] = []
    shapes: List[tf_TensorShape] = []

    for col in dataset.columns:

        # get shape
        if col.schema_type == 'variable_shape':
            shape = (None,) * len(col.shape)
        else:
            shape = col.shape
        shapes.append(tf.TensorShape(shape))

        # get type
        types.append(tf.as_dtype(col.dtype))
    return tf.data.Dataset.from_generator(generator=generator, output_types=tuple(types),
                                          output_shapes=tuple(shapes))
