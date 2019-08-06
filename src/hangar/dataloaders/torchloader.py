from collections import namedtuple
import warnings

try:
    from torch.utils.data import Dataset
except (ImportError, ModuleNotFoundError):
    raise RuntimeError("Could not import pytorch, Install dependencies")

from .common import GroupedDsets


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
    >>> from hangar import make_torch_dataset
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
    warnings.warn("Dataloaders are experimental in the current release.", UserWarning)
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



class TorchDataset(Dataset):
    """
    TorchDataset inherits `torch.utils.data.Dataset` and accepts few convenient arguments
    to wrap hangar datasets to be used in `torch.utils.data.DataLoaders`.

    .. warning::

        TorchDataset returns a namedtuple with `field_names` inferred from the dataset names
        if not explicitly passed on init. Python named tuples has few restrictions on naming
        the fields such it should not with an underscore etc. If your datatset names are not
        good candidates for being `field_names` and if you haven't provided a good set of
        `field_names` we'll rename those fields with namedtuples `rename=True`.

    .. note::

        From PyTorch 1.1 onwards, if Dataset returns namedtuple, DataLoader also returns
        namedtuple

    Parameters
    ----------
    hangar_datasets : A Collection of hangar datasets
        A list/tuple of hangar_dataset objects with same length and contains same keys. This
        class doesn't do any explicit check for length or the key names and assumes those all
        the datasets are valid as per the requirement
    sample_names : tuple of allowed sample names/keys
        User can select a subset of all the available samples and pass the names for only those
    ret_wrapper : namedtuple
        A namedtuple wrapper use to wrap the output from __getitem__
    """

    def __init__(self, hangar_datasets, sample_names, ret_wrapper):
        self.hangar_datasets = hangar_datasets
        self.sample_names = sample_names
        self.ret_wrapper = ret_wrapper

    def __len__(self):
        """
        Length of the available and allowed samples

        """
        return len(self.sample_names)

    def __getitem__(self, index):
        """
        Use data names array to find the sample name at an index and loop through the array
        of hangar datasets to return the sample.

        Returns
        -------
        namedtuple
            One sample with the given name from all the provided datasets

        """
        key = self.sample_names[index]
        out = []
        for dset in self.hangar_datasets:
            out.append(dset.get(key))
        return self.ret_wrapper(*out)
