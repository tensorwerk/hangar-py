import warnings
from typing import Iterable, Optional, Union


DatasetsRef = Union['DatasetDataReader', Iterable['DatasetDataReader']]


class GroupedDsets(object):
    """Groups hangar datasets and validate suitability for usage in dataloaders.

    It can choose a subset of samples in the hangar datasets by checking the
    list of keys or an index range. :class:`GroupedDsets` does not expect all the
    input hangar datasets to have same length and same keys. It takes a `set.union`
    of sample names from all the datasets and `keys` argument if passed and hence
    discard non-common keys while fetching. Based on `keys` or `index_range`
    (ignore `index_range` if `keys` is present) it makes a subset of sample names which
    is then used to fetch the data from hangar datasets.
    """

    def __init__(self,
                 hangar_datasets: DatasetsRef,
                 keys: Optional[Iterable[Union[int, str]]] = None,
                 index_range: Optional[slice] = None):

        if not isinstance(hangar_datasets, (list, tuple)):
            hangar_datasets = (hangar_datasets,)
        if len(hangar_datasets) == 0:
            raise ValueError("`hangar_datasets` cannot be empty")

        dataset_names, dset_lens, full_sample_names = [], set(), []
        for dataset in hangar_datasets:
            kset = set(dataset.keys())
            dataset_names.append(dataset.name)
            full_sample_names.append(kset)
            dset_lens.add(len(kset))
        sample_names = set.intersection(*full_sample_names)
        del full_sample_names
        if len(dset_lens) > 1:
            warnings.warn('Datasets do not contain equal num samples', UserWarning)

        if keys:
            if not isinstance(keys, (list, tuple, set)):
                keys = (keys,)
            keys = set(keys)
            noncommon_keys = keys.difference(sample_names)
            if len(noncommon_keys) > 0:
                raise ValueError(
                    f'Requested keys: {noncommon_keys} do not exist in all datasets.')
            self._allowed_samples = tuple(keys)
        elif index_range:
            if not isinstance(index_range, slice):
                raise TypeError(f"`index_range` must be a slice, not {type(keys)}")
            self._allowed_samples = tuple(sample_names)[index_range]
        else:
            self._allowed_samples = tuple(sample_names)

        self.dataset_array = hangar_datasets
        self.dataset_names = dataset_names

    def get_types(self, converter=None):
        """
        Get dtypes of the all the datasets in the `GroupedDsets`.

        Parameters
        ----------
        converter : Callable
            A function that takes default dtype (numpy) and convert it to another format

        Returns
        -------
        A tuple of types

        """
        types = []
        for dset in self.dataset_array:
            if converter:
                types.append(converter(dset.dtype))
            else:
                types.append(dset.dtype)
        return tuple(types)

    def get_shapes(self, converter=None):
        """
        Get shapes of the all the datasets in the `GroupedDsets`.

        Parameters
        ----------
        converter : Callable
            A function that takes default shape (numpy) and convert it to another format

        Returns
        -------
        A tuple of dataset shapes
        """
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
