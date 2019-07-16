from ..dataset import DatasetDataReader
import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataSet(Dataset):
    """
    TorchDataSet inherits `torch.utils.data.Dataset` and override
    `__len__` & `__getitem__`. Not needed to use directly by the user.
    """

    def __init__(self, hangar_dataset):
        if not isinstance(hangar_dataset, DatasetDataReader):
            raise TypeError("`hangar_dataset` provided is not a valid hangar dataset")
        self.hangar_dataset = hangar_dataset
        self._data_names = list(self.hangar_dataset.keys())

    def __len__(self):
        return len(self._data_names)

    def __getitem__(self, index):
        key = self._data_names[index]
        return torch.from_numpy(self.hangar_dataset[key])


class TorchLoader(DataLoader):
    """
    TorchLoader inherits and offloads all the operations to PyTorch loaders but this class
    kept here as a convenient class to accept hangar dataset and convert it to PyTorch
    dataset implicitly. The arguments of the class itself is absolutely similar to PyTorch
    DataLoader except that here it only accepts a hangar dataset.

    Parameters
    ----------
    hangar_dataset : An instance of :class:`hangar.dataset.DatasetDataReader`
        A dataset object which will be consumed by :class:`TorchDataSet`
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

    def __init__(self, hangar_dataset, *args, **kwargs):
        self.torch_dataset = TorchDataSet(hangar_dataset)
        super().__init__(self.torch_dataset, *args, **kwargs)
