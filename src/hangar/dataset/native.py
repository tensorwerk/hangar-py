from functools import wraps
import numpy as np


# TODO: document that Dataset can only load ndarray
# TODO: Work on returning nested arrays

def data_process_decorator(f, processor):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return processor(f(*args, **kwargs))
    return wrapper


class DataAccessor:
    def __init__(self, column):
        if isinstance(column, tuple):
            process_data = column[1]
            column = column[0]
            self.get_by_name = data_process_decorator(self.get_by_name, process_data)
        self._be_fs = column._be_fs
        self._samples = column._samples
        self.schema_type = column.schema_type

    def get_by_name(self, name):
        try:
            spec = self._samples[name]
            return self._be_fs[spec.backend].read_data(spec)
        except KeyError:
            # TODO: manage data non locality also
            raise KeyError(f"Requested item not found {name}")


class Dataset:
    def __init__(self, columns=None, keys=None):
        if columns is None:
            return
        accessors = []
        for c in columns:
            accessors.append(DataAccessor(c))
        if keys is None:
            first = columns[0]
            if isinstance(first, (tuple, list)):
                first = first[0]
            keys = list(first.keys())
        batch_size = 1
        num_batches = len(keys)
        self._setup(accessors, keys, batch_size, num_batches)

    def _setup(self, accessors, keys, batch_size, num_batches):
        self.accessors = accessors
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.num_batches = len(keys)
        self.keys = keys

    @classmethod
    def _make_another(cls, **setup_args):
        new = cls()
        new._setup(**setup_args)
        return new

    def batch(self, batch_size, drop_last=True):
        batch_size = self.batch_size * batch_size
        num_batches, has_last = divmod(len(self.keys), batch_size)
        if num_batches == 0:
            raise RuntimeError("Batch size exceeded the number of samples")
        if not has_last or drop_last:
            total = batch_size * num_batches
            keys = self.keys[:total]
        else:
            num_batches += 1
            keys = self.keys
        return Dataset._make_another(accessors=self.accessors, keys=keys,
                                     batch_size=batch_size, num_batches=num_batches)

    def torch(self):
        # TODO: check whether batch, shuffle or any such operations are registered
        from .torch import TorchDataset
        return TorchDataset(self.accessors, self.keys)

    def tensorflow(self):
        from .tensorflow import make_tf_dataset
        return make_tf_dataset(self.accessors, self.keys)

    def __getitem__(self, item):
        # TODO: return namedtuples if found useful
        return tuple([acc.get_by_name(item) for acc in self.accessors])

    def __iter__(self):
        start = 0
        end = self.batch_size
        batched_keys = []
        for i in range(self.num_batches):
            batched_keys.append(self.keys[start:end])
            start = end
            end = end + self.batch_size

        for batch in batched_keys:
            out = []
            for accessor in self.accessors:
                # TODO: np stack or filling empty array
                out.append(
                    np.stack(
                        [accessor.get_by_name(name) for name in batch]))
            yield tuple(out)

