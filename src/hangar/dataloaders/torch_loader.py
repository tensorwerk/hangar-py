from ..dataset import DatasetDataReader
import torch
from torch.utils.data import Dataset, DataLoader


class TorchDataSet(Dataset):
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
    TorchLoader inherits and offloads all the operations to PyTorch loaders but kept here as
    a convenient class to accept hangar dataset and convert it to PyTorch dataset
    implicitly. The arguments of the class itself is absolutely similar to PyTorch
    DataLoader except that here it only accepts a hangar dataset. It supports both
    """

    def __init__(self, hangar_dataset, *args, **kwargs):
        self.torch_dataset = TorchDataSet(hangar_dataset)
        super().__init__(self.torch_dataset, *args, **kwargs)
