from functools import wraps
import numpy as np


# TODO: document that Dataset can only load ndarray

def data_process_decorator(f, processor):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return processor(f(*args, **kwargs))
    return wrapper


class DataAccessor:
    def __init__(self, column, data_process_func=None):
        if data_process_func:
            self.get_by_name = data_process_decorator(self._get_by_name_impl,
                                                      data_process_func)
        else:
            self.get_by_name = self._get_by_name_impl
        # TODO: check the type of column
        self._be_fs = column._be_fs
        self._samples = column._samples
        self.schema_type = column.schema_type

    def _get_by_name_impl(self, name):
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
        all_keys = []
        all_remote_keys = []
        for col in columns:
            if isinstance(col, (tuple, list)):
                data_process_func = col[1]
                col = col[0]
                acc = DataAccessor(col, data_process_func)
            else:
                acc = DataAccessor(col)
            accessors.append(acc)
            all_keys.append(set(col.keys()))
            all_remote_keys.append(set(col.remote_reference_keys))
        common_keys = set.intersection(*all_keys)
        remote_keys = set.union(*all_remote_keys)
        common_local_keys = common_keys.difference(remote_keys)

        if keys:
            unique = set(keys)
            notCommon = unique.difference(common_keys)
            notLocal = unique.difference(common_local_keys)

            # TODO: These error message could eat up the whole terminal space if the size of
            #   non common and non local keys are huge
            if len(notCommon) > 0:
                raise KeyError(f'Keys: {notCommon} do not exist in all columns.')
            if len(notLocal) > 0:
                raise FileNotFoundError(
                    f'Keys: {notLocal} are remote data samples not downloaded locally.')
        else:
            keys = tuple(common_local_keys)
        # TODO: batch_size 1 means returning one sample with batch dimension, we don't need that
        batch_size = 1
        num_batches = len(keys)
        self._setup(accessors, keys, batch_size, num_batches)

    def _setup(self, accessors, keys, batch_size, num_batches):
        self.accessors = accessors
        self.batch_size = batch_size
        self.num_batches = num_batches
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
        return tuple([acc.get_by_name(item) for acc in self.accessors])

    def __iter__(self):
        start = 0
        end = self.batch_size
        for i in range(self.num_batches):
            batch = self.keys[start:end]
            start = end
            end = end + self.batch_size
            out = []
            for accessor in self.accessors:
                data = [accessor.get_by_name(name) for name in batch]
                # TODO: np stack or filling empty array
                out.append(np.stack(data))
            yield tuple(out)

