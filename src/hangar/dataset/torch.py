import torch
from torch.utils.data.dataset import Dataset


class TorchDataset(Dataset):

    def __init__(self, accessors, keys):
        self.accessors = accessors
        self.keys = keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, index: int):
        key = self.keys[index]
        return tuple([torch.from_numpy(acc.get_by_name(key)) for acc in self.accessors])
