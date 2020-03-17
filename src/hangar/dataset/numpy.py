import random
import numpy as np
from .common import Dataset


class NumpyDataset:
    def __init__(self, dataset, batch_size, drop_last, shuffle):
        self.dataset = dataset
        self.num_batches = None
        self.batch_size = None
        if batch_size:
            self._batch(batch_size, drop_last)
        self.shuffle = shuffle

    def __len__(self):
        return len(self.keys)

    def _batch(self, batch_size, drop_last=True):
        num_batches, has_last = divmod(len(self.dataset.keys), batch_size)
        if num_batches == 0:
            raise RuntimeError("Batch size exceeded the number of samples")
        if not has_last or drop_last:
            total = batch_size * num_batches
            self.dataset.keys = self.dataset.keys[:total]
        else:
            num_batches += 1
        self.num_batches = num_batches
        self.batch_size = batch_size

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.dataset.keys)
        if self.num_batches is None:
            for k in self.dataset.keys:
                yield self.dataset[k]
        else:
            start = 0
            end = self.batch_size
            for i in range(self.num_batches):
                batch = self.dataset.keys[start:end]
                start = end
                end = end + self.batch_size
                out = []
                for name in batch:
                    out.append(self.dataset[name])
                yield tuple([np.stack(d) for d in zip(*out)])


def make_numpy_dataset(columns, keys=None, batch_size=None, drop_last=False, shuffle=True):
    dataset = Dataset(columns, keys)
    dataset = NumpyDataset(dataset, batch_size, drop_last, shuffle)
    return dataset



