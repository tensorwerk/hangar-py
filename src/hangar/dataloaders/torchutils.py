try:
    from torch.utils.data import Dataset
except (ImportError, ModuleNotFoundError):
    raise RuntimeError("Could not import torch. Install dependencies")


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
