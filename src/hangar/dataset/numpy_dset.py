from typing import Sequence, Callable, TYPE_CHECKING, Union, List, Tuple
import random

import numpy as np

from .common import HangarDataset

if TYPE_CHECKING:
    from ..columns import ModifierTypes
    Columns = ModifierTypes
    KeyType = Union[str, int, List, Tuple]


def default_collate_fn(col_functions):
    def wrapper(data_arr):
        # data_arr -> array of data samples from each column
        collated = []
        for fn in col_functions:
            grouped = fn(data_arr)
            collated.append(grouped)
        return tuple(collated)
    return wrapper


class NumpyDataset:
    """NumpyDataset class provides interfaces for users to iterate over the batches of
    data from different columns. The only user facing APIs it exposes are ``__len__`` and
    ``__iter__``. Batch and shuffle operations are handled by `:func:`make_numpy_dataset`
    based on the arguments it gets and hence user should not interact with this class for
    such operations. Note that, user would never instantiate this class directly. Instead
    `:func:`make_numpy_dataset` act as the entry point and return an object of this class
    to the user

    Parameters
    ----------
    dataset : :class:`~hangar.dataset.common.Dataset` object
        Hangar's Dataset object that groups columns for downstream processing
    batch_size : int
        Size of the individual batch. If specified batches of this size will be returned
        on each iteration
    drop_last : bool
        Should drop the last incomplete batch
    shuffle : bool
        Should shuffle the batch on each epoch
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
    """
    def __init__(self, dataset: HangarDataset, batch_size: int, drop_last: bool,
                 shuffle: bool, collate_fn: Callable = None):
        self._dataset = dataset
        self._num_batches = None
        self._batch_size = None
        if batch_size:
            if not collate_fn:
                collate_colfn = []
                for col in dataset.columns.values():
                    print(col.column_type)
                    if col.column_type == 'ndarray' and col.column_layout == 'flat':
                        collate_colfn.append(np.stack)
                    else:
                        collate_colfn.append(lambda x: x)

                self.collate_fn = default_collate_fn(collate_colfn)
            else:
                self.collate_fn = collate_fn
            self._batch(batch_size, drop_last)
        else:
            if collate_fn:
                raise RuntimeError("Found `collate_fn` in the argument which is a no-op "
                                   "since batching is not enabled")
        self._shuffle = shuffle
        self._indices = list(range(len(self._dataset)))

    @property
    def dataset(self):
        return self._dataset

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int):
        if not isinstance(value, int):
            raise TypeError(f'Expected integer type, recieved {type(value)}')
        elif value < 1:
            raise ValueError(f'batch_size value must be >= 1, recieved {value}')
        self._batch_size = value

    @property
    def shuffle(self):
        return self._shuffle

    @shuffle.setter
    def shuffle(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError(f'Expected bool type, recieved {type(value)}')
        self._shuffle = value

    def __len__(self):
        return len(self._dataset)

    def _batch(self, batch_size, drop_last=True) -> None:
        """Private function to this class to calculate the batch parameters. These
        calculated parameters will be considered by the ``__iter__`` method while
        fetching the batches for downstream process. This function will be called at
        the time of object instantiation and should not be triggered independently

        Parameters
        ----------
        batch_size : int
            Size of the individual batch. If specified batches of this size will be returned
            on each iteration
        drop_last : bool
            Should drop the last incomplete batch
        """
        num_batches, has_last = divmod(len(self._dataset), batch_size)
        if num_batches == 0:
            raise RuntimeError("Batch size exceeded the number of samples")
        if has_last and not drop_last:
            num_batches += 1
        self._num_batches = num_batches
        self._batch_size = batch_size

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._indices)
        if self._num_batches is None:
            for i in self._indices:
                yield self._dataset.index_get(i)
        else:
            start = 0
            end = self._batch_size
            for i in range(self._num_batches):
                batch = self._indices[start:end]
                out = [self._dataset.index_get(i) for i in batch]

                start = end
                end = end + self._batch_size
                yield self.collate_fn(out)


def _make_numpy_dataset(columns: Sequence['Columns'],
                        keys: 'KeyType' = None,
                        batch_size: int = None,
                        drop_last: bool = False,
                        shuffle: bool = True,
                        collate_fn: Callable = None) -> NumpyDataset:
    """Group column into a single numpy dataset, provides iterative looping over data.

    This API also provides the options to batch the data which is a major difference
    between other dataset APIs. In traditional Machine learning applications, it's quite
    natural to load the whole dataset as a single batch because it's possible to fit into
    the system memory. Passing the size of the dataset as the batch size would make it
    possible here to do just that. This API also acts as an entry point for other
    non-supported frameworks to load data from hangar as batches into the training loop.

    .. note::

        Column with layouts ``str`` or ``ndarray nested`` are not compatible with the
        dataset APIs in the current release. So making dataset is only possible for
        columns with layout ``ndarray flat``

    .. note::

        This is an experimental method in the current Hangar version. Please be aware
        that Significant changes may be introduced in future releases without advance
        notice or deprication warnings.

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

    Returns
    -------
    :class: `.NumpyDataset`

    DEVELOPER NOTE
    --------------
    - Any update to this function signature or docstring must be reflected in the
      equivalent loader function in hangar/dataset/__init__.py. This function is
      "copied" to a top level __init__.py to allow unified API and lazyloader access
    """
    dataset = HangarDataset(columns, keys)
    dataset = NumpyDataset(dataset, batch_size, drop_last, shuffle, collate_fn)
    return dataset



