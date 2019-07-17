from ..dataset import DatasetDataReader
try:
    import torch
    from torch.utils.data import Dataset, DataLoader
except (ImportError, ModuleNotFoundError):
    raise RuntimeError("Could not import torch. Try installing extra dependencies")


class TorchDataSet(Dataset):
    """
    TorchDataSet inherits `torch.utils.data.Dataset` and override
    `__len__` & `__getitem__`. Not needed to use directly by the user.

    Parameters
    ----------
    hangar_datasets : `hangar.dataset.DatasetDataReader` or a Collection
        A dataset object or a list of dataset objects which will be
        consumed by :class:`TorchDataSet`
    """

    def __init__(self, hangar_datasets):
        if isinstance(hangar_datasets, DatasetDataReader):
            hangar_datasets = [hangar_datasets]

        try:
            self._dataset_len = len(hangar_datasets[0])
        except IndexError:
            raise
        except Exception:
            raise TypeError("`hangar_datasets` has to be a valid hangar dataset or an"
                            "iterable that can hold other objects, like a list.")

        for dset in hangar_datasets:
            if not isinstance(dset, DatasetDataReader):
                raise TypeError("`hangar_dataset` contains invalid hangar dataset(s)")
            if len(dset) != self._dataset_len:
                raise RuntimeError("Got datasets with different length")
        self._number_of_datasets = len(hangar_datasets)
        self.hangar_datasets = hangar_datasets
        self._data_names = tuple(self.hangar_datasets[0].keys())

    def __len__(self):
        return self._dataset_len

    def __getitem__(self, index):
        key = self._data_names[index]
        if self._number_of_datasets == 1:
            return torch.from_numpy(self.hangar_datasets[0][key])
        out = []
        for dset in self.hangar_datasets:
            out.append(torch.from_numpy(dset[key]))
        return out


class TorchLoader(DataLoader):
    """
    TorchLoader inherits and offloads all the operations to PyTorch DataLoader.
    TorchLoader is a convenient class to accept hangar dataset and convert it to PyTorch
    dataset implicitly. The arguments / signature of the class itself is absolutely
    similar to PyTorch DataLoader except that here it only accepts a hangar dataset.

    Parameters
    ----------
    hangar_datasets : `hangar.dataset.DatasetDataReader` or a Collection
        A dataset object or a collection of dataset objects which will be
        consumed by :class:`TorchDataSet`
    batch_size : int
        how many samples per batch to load
    shuffle : bool
        Decides whether the sampling should be shuffled or not
    sampler : torch.utils.data.Sampler
        defines the strategy to draw samples from the dataset. If specified, attribute
        `shuffle` must be False
    batch_sampler : torch.utils.data.Sampler
        like `sampler`, but returns a batch of indices at a time. Mutually exclusive
        with `batch_size`, `shuffle`, `sampler`, and `drop_last`.
    num_workers : int
        how many subprocesses to use for data loading. Value of this is default to zero
        which means the data will be loaded in the main process itself
    collate_fn : callable
        merges a list of samples to form a mini-batch of Tensor(s)
    pin_memory : bool
        the data loader will copy Tensors into CUDA pinned memory before returning them. For
        advanced usecases, checkout PyTorch documentation
    drop_last : bool
        to drop the last incomplete batch
    timeout : int or float
        the timeout value for collecting a batch from workers. Should always be non-negative.
        Default to `0`
    worker_init_fn : callable
        If not `None`, this will be called on each worker subprocess with the worker id as
        input, after seeding and before data loading. Default to `None`
    """

    def __init__(self, hangar_datasets, *args, **kwargs):
        self.torch_dataset = TorchDataSet(hangar_datasets)
        super().__init__(self.torch_dataset, *args, **kwargs)
