from typing import Sequence, Union, List, Iterable

import numpy as np

from ..arrayset import ArraysetDataReader


# -------------------------- typehints ---------------------------------------


ArraysetSampleNames = Sequence[Union[str, int]]


# --------------------- sampling functions ------------------------------------


def _p_normalize(p: Sequence[float]) -> List[float]:
    ptot = np.sum(p)
    if not np.allclose(ptot, 1):
        p = [i / ptot for i in p]
    return p


def _multinomial(num_samples: int, p: Sequence[float], replacement: bool) -> List[int]:
    """Draw samples from a multinomial distribution.

    The multinomial distribution is a multivariate generalization of the
    binomial distribution.  Take an experiment with one of ``p`` possible
    outcomes.  An example of such an experiment is throwing a dice, where the
    outcome can be 1 through 6.  Each sample drawn from the distribution
    represents `n` such experiments.  Its values, ``X_i = [X_0, X_1, ...,
    X_p]``, represent the number of times the outcome was ``i``.

    Parameters
    ----------
    num_samples : int
        number of samples to draw from the probabilities

    p : Sequence[float]
        Input list containing probabilities of drawing a specific catagory. The
        elements in ``p`` do not need to sum to one (in which case we normalize and
        use the values as weights), but must be non-negative, finite and have a
        non-zero sum.

    replacement : bool
        Wheather to draw without replacement or not

    Returns
    -------
    List[int]
        Contains ``num_samples`` indices sampled from the multinomial probability
        distribution located in the corresponding row of ``p`` probabilities.

    Raises
    ------
    ValueError
        If probability arg is not a Sequence (list or tuple) of len > 0
    """
    if not isinstance(p, Sequence) or (len(p) == 0):
        raise ValueError(f'probability arg must be sequence of len > 0, {p} is invalid')
    if not all((i >= 0 for i in p)) or not any(
            (i > 0 for i in p)) or (np.Infinity in p) or (np.NaN in p):
        raise ValueError(f'probs {p} invalid. all must be >= 0, finite, and have non-zero sum')

    valid_p = _p_normalize(p)
    if not replacement:
        idxs = np.random.multinomial(num_samples, valid_p)
    else:
        idxs = np.random.choice(np.arange(len(p)), replace=True, size=num_samples, p=valid_p)
    return idxs.tolist()


# -------------------------- sampler methods ----------------------------------


class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an :meth:`__iter__` method, providing a
    way to iterate over indices of dataset elements, and a :meth:`__len__` method
    that returns the length of the returned iterators.

    .. note::

        The :meth:`__len__` method isn't strictly required by
        :class:`~torch.utils.data.DataLoader`, but is expected in any calculation
        involving the length of a :class:`~torch.utils.data.DataLoader`.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    # NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
    #
    # Many times we have an abstract class representing a collection/iterable of
    # data, e.g., `torch.utils.data.Sampler`, with its subclasses optionally
    # implementing a `__len__` method. In such cases, we must make sure to not
    # provide a default implementation, because both straightforward default
    # implementations have their issues:
    #
    #   + `return NotImplemented`:
    #     Calling `len(subclass_instance)` raises:
    #       TypeError: 'NotImplementedType' object cannot be interpreted as an integer
    #
    #   + `raise NotImplementedError()`:
    #     This prevents triggering some fallback behavior. E.g., the built-in
    #     `list(X)` tries to call `len(X)` first, and executes a different code
    #     path if the method is not found or `NotImplemented` is returned, while
    #     raising an `NotImplementedError` will propagate and and make the call
    #     fail where it could have use `__iter__` to complete the call.
    #
    # Thus, the only two sensible things to do are
    #
    #   + **not** provide a default `__len__`.
    #
    #   + raise a `TypeError` instead, which is what Python uses when users call
    #     a method that is not defined on an object.
    #     (@ssnl verifies that this works on at least Python 3.7.)


class SequentialSampler(Sampler):
    """Samples elements sequentially, always in the same order.

    Order of keys is numeric first, then lexicographically sorted strings.

    Parameters
    ----------
    data_source : ArraysetDataReader
        arrayset to derive sample names from.

    TODO
    ----
    -  Discussion: ordering holds so long as the ArraysetDataReader is not
       write enabled (the _sspecs dict is not able to mutate.) Even at that,
       this is only guarrenteed due to the implicit ordered dictionary methods
       with python 3.6. However, this could potentially change in the future if
       we ever decided to store the sample keys in a different format or change
       up how the backend actually stores data. Is sorted order something we
       want to guarrenttee?
    """

    def __init__(self, data_source: ArraysetDataReader):
        self.data_source = data_source

    def __iter__(self) -> Iterable[ArraysetSampleNames]:
        return iter(self.data_source.keys())

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler):
    """Sample names randomly from an arrayset.

    If without replacemement, then sample from a shuffled set of names. If with
    replacement, then user can specify :attr:`num_samples` to draw.

    Parameters
    ----------
    data_source : ArraysetDataReader
        arrayset to sample names from
    replacement : bool, optional
        Samples are draw with replacement if ``True``, by default ``False``
    num_samples : int, optional
        number of samples to draw, default=`len(dataset)`. This argument is
        supposed to be specified only when `replacement` is ``True``, by
        default ``None``
    """

    def __init__(self,
                 data_source: ArraysetDataReader,
                 replacement: bool = False,
                 num_samples: int = None):

        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(f"replacement must be boolean. Not {self.replacement}")
        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")
        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be int >= 1, not {self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterable[ArraysetSampleNames]:
        if self.replacement:
            n = len(self.data_source)
            indices = np.random.randint(0, high=n, size=(self.num_samples,), dtype=np.int64)
        else:
            indices = np.random.permutation(self.num_samples)
        keys = list(self.data_source.keys())
        return (keys[idx] for idx in indices.tolist())

    def __len__(self) -> int:
        return self.num_samples


class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of sample names, without replacement.

    Parameters
    ----------
    sample_names : ArraysetSampleNames
        a sequence of sample names
    """
    def __init__(self, sample_names: ArraysetSampleNames):
        self.sample_names = sample_names

    def __iter__(self) -> Iterable[ArraysetSampleNames]:
        indices = np.random.permutation(len(self.sample_names))
        return (self.sample_names[idx] for idx in indices)

    def __len__(self) -> int:
        return len(self.sample_names)


class WeightedRandomSampler(Sampler):
    """Samples elements from``[0,..,len(weights)-1]`` with given probabilities (weights).

    Examples
    --------

        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]

    Parameters
    ----------
    weights : Sequence[float]
        a sequence of weights, not necessarily summing up to one
    num_samples : int
        number of samples to draw
    replacement : bool
        if ``True``, samples are drawn with replacement. If not, they are
        drawn without replacement, meaning that when a sample name is drawn
        once in a row, it cannot be drawn again for that same row
    group_names : List[np.ndarray], optional
        If provided, iteration across sampler will return corresponding arrayset
        group value rather then a generic positional index identifying the selected
        probability/weight. If set, ``len(group_names)`` must exactaly equal ``len(weights)``.
        If not specified, or set to ``None``, returns position index corresponding to the weight
        selected. defaults to ``None``.
    """

    def __init__(self,
                 weights: Sequence[float],
                 num_samples: int,
                 replacement: bool,
                 group_names: List[np.ndarray] = None):

        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        if group_names is not None:
            if not isinstance(group_names, Sequence) or not all(
                (isinstance(item, np.ndarray) for item in group_names)) or (
                    len(group_names) != len(weights)):
                raise ValueError(f'if provided, groupnames must be list of `numpy.ndarray` with '
                                 f'len(groups) == len(weights), not {group_names} ')
        self.weights = tuple(weights)
        self.num_samples = num_samples
        self.replacement = replacement
        self.group_names = group_names

    def __iter__(self) -> Iterable[int]:
        indices = _multinomial(self.num_samples, self.weights, self.replacement)
        if self.group_names:
            indices = (self.group_names[idx] for idx in indices)
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    """Wraps another sampler to yield a mini-batch of sample names

    Parameters
    ----------
    sampler : Sampler
        sampler instance inhereting from :class:'.Sampler`
    batch_size : int
        size of the mini-batch
    drop_last : bool
        If ``True, the sampler will drop the last batch if its size would be
        less tahn ``batch_size``.

    Examples
    --------

        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                f"sampler must be hangar.dataloader.Sampler instance. Not {sampler}")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterable[List[ArraysetSampleNames]]:
        batch = []
        for sample_name in self.sampler:
            batch.append(sample_name)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size