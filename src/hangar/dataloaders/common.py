import warnings
from typing import Iterable, Optional, Union

import numpy as np

DatasetsRef = Union['DatasetDataReader', Iterable['DatasetDataReader']]


class GroupedDsets(object):
    """Groups hangar datasets and validate suitability for usage in dataloaders.

    It can choose a subset of samples in the hangar datasets by checking the
    list of keys or an index range. :class:`GroupedDsets` expect all the input
    hangar datasets to have same length and same keys. Although it checks for
    the length of all the datasets, it doesn't do any explicit check for the
    sample names and hence the dataloaders while fetching the data will throw
    `KeyError` in case of non-similar keys.
    """

    def __init__(self,
                 hangar_datasets: DatasetsRef,
                 keys: Optional[Iterable[Union[int, str]]] = None,
                 index_range: Optional[slice] = None):

        if not isinstance(hangar_datasets, (list, tuple)):
            hangar_datasets = (hangar_datasets,)

        dset_keys, dset_lens = {}, []
        for dataset in hangar_datasets:
            kset = set(dataset.keys())
            dset_keys[dataset.name] = kset
            dset_lens.append(len(kset))

        if not np.allclose(dset_lens):
            warnings.warn('Datasets do not contain equal num samples', UserWarning)

        if (keys is not None) and (not isinstance(keys, (list, tuple, set))):
            keys = (keys,)
        userKeysSet = set(keys)
        commonDsetKeys = set.union(dset_keys.values())
        keysNotInCommon = userKeysSet.difference(commonDsetKeys)
        if len(keysNotInCommon) > 0:
            raise ValueError(
                f'Requested keys: {keysNotInCommon} do not exist in all datasets.')

        self.dataset_array = hangar_datasets
        self.dataset_names = list(dset_keys.keys())
        self._allowed_samples = None
        self.sample_subset(keys, index_range)

    def sample_subset(self, keys=None, index_range=None):
        """
        Based on `keys` or `index_range` (ignore `index_range` if `keys` is present)
        it makes a subset of sample names which is then used to fetch the data from
        hangar datasets. It is being called from the __init__ implicitly and almost
        always not required for users to call directly until a new set of `keys` or a
        new `index_range` is setup.
        """
        if keys:
            if not isinstance(keys, (list, tuple)):
                raise TypeError(f"`keys` must be a list/tuple, not {type(keys)}")
            sample_names = keys
        elif index_range:
            if not isinstance(index_range, slice):
                raise TypeError(f"`index_range` must be a slice, not {type(keys)}")
            sample_names = tuple(self.dataset_array[0].keys())[index_range]
        else:
            sample_names = tuple(self.dataset_array[0].keys())
        self._allowed_samples = sample_names
        return self._allowed_samples

    def get_types(self, converter=None):
        '''add docstring'''
        types = []
        for dset in self.dataset_array:
            if converter:
                types.append(converter(dset.dtype))
            else:
                types.append(dset.dtype)
        return tuple(types)

    def get_shapes(self, converter=None):
        '''add docstring'''
        if self.dataset_array[0].variable_shape:
            return None
        shapes = []
        for dset in self.dataset_array:
            if converter:
                shapes.append(converter(dset.shape))
            else:
                shapes.append(dset.shape)
        return tuple(shapes)

    @property
    def sample_names(self):
        return self._allowed_samples