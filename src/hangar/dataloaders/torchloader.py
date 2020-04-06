import warnings
from collections import namedtuple
from typing import Sequence

from .common import GroupedColumns
from ..utils import LazyLoader


try:
    torchdata = LazyLoader('torchdata', globals(), 'torch.utils.data')
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        'Could not import "pytorch" library. Ensure library is '
        'installed correctly to use pytorch dataloader functions')


def make_torch_dataset(columns,
                       keys: Sequence[str] = None,
                       index_range: slice = None,
                       field_names: Sequence[str] = None):
    """
    Returns a :class:`torch.utils.data.Dataset` object which can be loaded into
    a :class:`torch.utils.data.DataLoader`.

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
    keys : Sequence[str]
        An iterable collection of sample names. If given only those samples will
        fetched from the column
    index_range : slice
        A python slice object which will be used to find the subset of column.
        Argument `keys` takes priority over `range` i.e. if both are given, keys
        will be used and `range` will be ignored
    field_names : Sequence[str], optional
        An array of field names used as the `field_names` for the returned
        dict keys. If not given, column names will be used as the field_names.

    Examples
    --------
    >>> from hangar import Repository
    >>> from torch.utils.data import DataLoader
    >>> from hangar import make_torch_dataset
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> aset = co.columns['dummy_aset']
    >>> torch_dset = make_torch_dataset(aset, index_range=slice(1, 100))
    >>> loader = DataLoader(torch_dset, batch_size=16)
    >>> for batch in loader:
    ...     train_model(batch)

    Returns
    -------
    :class:`torch.utils.data.Dataset`
    """
    warnings.warn("Dataloaders are experimental in the current release.", UserWarning)
    if keys:
        if not isinstance(keys, (list, tuple, set)):
            raise TypeError(f'type(keys): {type(keys)} != (list, tuple, set)')

    gcols = GroupedColumns(columns, keys, index_range)
    if field_names:
        if not isinstance(field_names, (list, tuple, set)):
            raise TypeError(f'type(field_names): {type(field_names)} not collection')
        if len(field_names) != len(columns):
            err = f'# field_names {len(field_names)} != # columns: {len(columns)}'
            raise ValueError(err)
        BTName = '_'.join(['BatchTuple', *field_names])
        BTFieldNames = field_names
    else:
        BTName = '_'.join(['BatchTuple', *gcols.column_names])
        BTFieldNames = gcols.column_names

    wrapper = namedtuple(BTName, field_names=BTFieldNames, rename=True)
    globals()[BTName] = wrapper
    return TorchDataset(hangar_columns=gcols.columns_col,
                        sample_names=gcols.sample_names,
                        wrapper=wrapper)


class TorchDataset(torchdata.Dataset):
    """A wrapper around torch Dataset

    TorchDataset inherits :class:`torch.utils.data.Dataset` and accepts few
    convenient arguments to wrap hangar columns to be used in
    :class:`torch.utils.data.DataLoaders`.

    .. note::

        From PyTorch 1.1 onwards, if Dataset returns dict, DataLoader also
        returns dict

    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A list/tuple of hangar_arrayset objects with same length and contains
        same keys. This class doesn't do any explicit check for length or the
        key names and assumes those all the columns are valid as per the
        requirement
    sample_names : tuple of allowed sample names/keys
        User can select a subset of all the available samples and pass the
        names for only those
    wrapper : namedtuple
        namedtuple placed in global memory used to wrap the output from
        __getitem__
    """

    def __init__(self, hangar_columns, sample_names, wrapper):
        self.hangar_columns = hangar_columns
        self.sample_names = sample_names
        self.wrapper: namedtuple = wrapper

    def __len__(self) -> int:
        """
        Length of the available and allowed samples

        Returns
        -------
        int
            number of samples retrievable by the dataloader.
        """
        return len(self.sample_names)

    def __getitem__(self, index: int):
        """Use data names array to find the sample name at an index and loop
        through the array of hangar columns to return the sample.

        Parameters
        ----------
        index : int
            some sample index location.

        Returns
        -------
        namedtuple[:class:`torch.Tensor`]
            One sample with the given name from all the provided columns
        """
        key = self.sample_names[index]
        out = []
        for aset in self.hangar_columns:
            out.append(aset.get(key))
        return self.wrapper._make(out)
