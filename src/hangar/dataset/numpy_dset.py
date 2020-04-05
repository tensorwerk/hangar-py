import typing
from typing import Sequence
import random
import warnings

import numpy as np

from .common import HangarDataset


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
    """
    def __init__(self, dataset: HangarDataset, batch_size: int,
                 drop_last: bool, shuffle: bool):
        self._dataset = dataset
        self._num_batches = None
        self._batch_size = None
        if batch_size:
            self._batch(batch_size, drop_last)
        self._shuffle = shuffle

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
        return len(self._dataset.keys)

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
        num_batches, has_last = divmod(len(self._dataset.keys), batch_size)
        if num_batches == 0:
            raise RuntimeError("Batch size exceeded the number of samples")
        if not has_last or drop_last:
            total = batch_size * num_batches
            # TODO: Do not access private attribute... probably want to
            #       allow batching parameters to more directly influence __iter__.
            self._dataset._keys = self._dataset._keys[:total]
        else:
            num_batches += 1
        self._num_batches = num_batches
        self._batch_size = batch_size

    def __iter__(self):
        if self._shuffle:
            random.shuffle(self._dataset.keys)
        if self._num_batches is None:
            for k in self._dataset.keys:
                yield self._dataset[k]
        else:
            start = 0
            end = self._batch_size
            for i in range(self._num_batches):
                batch = self._dataset.keys[start:end]
                start = end
                end = end + self._batch_size
                out = []
                for name in batch:
                    out.append(self._dataset[name])
                yield tuple([np.stack(d) for d in zip(*out)])


def make_numpy_dataset(columns: Sequence['Columns'], keys: Sequence[str] = None,
                       batch_size: int = None, drop_last: bool = False,
                       shuffle: bool = True) -> NumpyDataset:
    """Groups the column objects into a single numpy dataset and provide iterator to loop
    over the data. This API also provides the options to batch the data which is a major
    difference between other dataset APIs. In traditional Machine learning applications,
    it's quite natural to load the whole dataset as a single batch because it's possible
    to fit into the system memory. Passing the size of the dataset as the batch size
    would make it possible here to do just that. This API also acts as an entry point
    for other non-supported frameworks to load data from hangar as batches into the
    training loop

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
    keys : Sequence[str]
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

    Returns
    -------
        :class: `.NumpyDataset`
    """
    warn_msg = ('This is an experimental method in the current Hangar version. '
                'Please be aware that Significant changes may be introduced in '
                'future releases without advance notice / deprication warnings.')
    warnings.warn(warn_msg, UserWarning)
    dataset = HangarDataset(columns, keys)
    dataset = NumpyDataset(dataset, batch_size, drop_last, shuffle)
    return dataset



