try:
    from torch.utils.data import Dataset
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        'Could not import "pytorch" library. Ensure library is '
        'installed correctly to use pytorch dataloader functions')

from collections import namedtuple
from typing import Sequence
import warnings
from .common import GroupedAsets


def make_torch_dataset(arraysets,
                       keys: Sequence[str] = None,
                       index_range: slice = None,
                       field_names: Sequence[str] = None):
    """
    Returns a `torch.utils.data.Dataset` object which can be loaded into a
    `torch.utils.data.DataLoader`.

    Parameters
    ----------
    arraysets : :class:`~hangar.arrayset.ArraysetDataReader` or Sequence
        A arrayset object, a tuple of arrayset object or a list of arrayset
        objects.
    keys : Sequence[str]
        An iterable collection of sample names. If given only those samples will
        fetched from the arrayset
    index_range : slice
        A python slice object which will be used to find the subset of arrayset.
        Argument `keys` takes priority over `range` i.e. if both are given, keys
        will be used and `range` will be ignored
    field_names : list or tuple of str
        An array of field names used as the `field_names` for the returned
        namedtuple. If not given, arrayset names will be used as the field_names.

    Examples
    --------
    >>> from hangar import Repository
    >>> from torch.utils.data import DataLoader
    >>> from hangar import make_torch_dataset
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> aset = co.arraysets['dummy_aset']
    >>> torch_dset = make_torch_dataset(aset, index_range=slice(1, 100))
    >>> loader = DataLoader(torch_dset, batch_size=16)
    >>> for batch in loader:
    ...     train_model(batch)

    Returns
    -------
    :class:`torch.utils.data.Dataset`
    """
    warnings.warn("Dataloaders are experimental in the current release", UserWarning)
    if keys:
        if not isinstance(keys, (list, tuple, set)):
            raise TypeError(f'type(keys): {type(keys)} != (list, tuple, set)')

    gasets = GroupedAsets(arraysets, keys, index_range)

    if field_names:
        if not isinstance(field_names, (list, tuple, set)):
            raise TypeError(
                f'type(field_names): {type(field_names)} != (list, tuple, set)')
        if len(field_names) != len(arraysets):
            m = f'len(field_names): {len(field_names)} != len(arraysets): {len(arraysets)}'
            raise ValueError(m)
        wrapper = namedtuple('BatchTuple', field_names=field_names)
    else:
        wrapper = namedtuple('BatchTuple', field_names=gasets.arrayset_names, rename=True)

    return TorchDataset(gasets.arrayset_array, gasets.sample_names, wrapper)


class TorchDataset(Dataset):
    """A wrapper around torch Dataset

    TorchDataset inherits `torch.utils.data.Dataset` and accepts few convenient
    arguments to wrap hangar arraysets to be used in `torch.utils.data.DataLoaders`.

    .. warning::

        TorchDataset returns a namedtuple with `field_names` inferred from the
        arrayset names if not explicitly passed on init. Python named tuples has
        few restrictions on naming the fields such it should not with an underscore
        etc. If your datatset names are not good candidates for being `field_names`
        and if you haven't provided a good set of `field_names` we'll rename those
        fields with namedtuples `rename=True`.

    .. note::

        From PyTorch 1.1 onwards, if Dataset returns namedtuple, DataLoader also
        returns namedtuple

    Parameters
    ----------
    arraysets : :class:`~hangar.arrayset.ArraysetDataReader` or Sequence
        A list/tuple of hangar_arrayset objects with same length and contains same
        keys. This class doesn't do any explicit check for length or the key names
        and assumes those all the arraysets are valid as per the requirement
    sample_names : tuple of allowed sample names/keys
        User can select a subset of all the available samples and pass the names
        for only those
    wrapper : namedtuple
        A namedtuple wrapper use to wrap the output from __getitem__
    """

    def __init__(self, hangar_arraysets, sample_names, wrapper):
        self.hangar_arraysets = hangar_arraysets
        self.sample_names = sample_names
        self.wrapper = wrapper

    def __len__(self):
        """
        Length of the available and allowed samples

        """
        return len(self.sample_names)

    def __getitem__(self, index):
        """Use data names array to find the sample name at an index and loop
        through the array of hangar arraysets to return the sample.

        Returns
        -------
        namedtuple
            One sample with the given name from all the provided arraysets
        """
        key = self.sample_names[index]
        out = []
        for aset in self.hangar_arraysets:
            out.append(aset.get(key))
        return self.wrapper(*out)
