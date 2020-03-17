from torch.utils.data.dataset import Dataset
from .common import Dataset as HangarDataset


class TorchDataset(Dataset):

    def __init__(self, hangar_dataset, wrapper):
        self.dataset = hangar_dataset
        self.wrapper = wrapper

    def __len__(self) -> int:
        return len(self.dataset.keys)

    def __getitem__(self, index: int):
        key = self.dataset.keys[index]
        data = self.dataset[key]
        return self.wrapper(*data) if self.wrapper else data


def make_torch_dataset(columns, keys=None, wrapper=None):
    hangar_dataset = HangarDataset(columns, keys)
    return TorchDataset(hangar_dataset, wrapper)
