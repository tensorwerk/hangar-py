
class GroupedDsets:
    """
    Groups all the hangar datasets and provide convenient functionalities to validate the
    datasets for downstream usage in the dataloaders. It can choose a subset of samples
    in the hangar datasets by checking the list of keys or an index range. :class:`GroupedDsets`
    expect all the input hangar datasets to have same length and same keys. Although it checks
    for the length of all the datasets, it doesn't do any explicit check for the sample names
    and hence the dataloaders while fetching the data will throw `KeyError` in case of
    non-similar keys.
    """
    def __init__(self, hangar_datasets, keys=None, index_range=None):
        if not isinstance(hangar_datasets, (list, tuple)):
            hangar_datasets = (hangar_datasets,)
        try:
            dataset_len = len(hangar_datasets[0])
        except IndexError:
            raise
        except Exception:
            raise TypeError("`hangar_datasets` has to be a valid hangar dataset or an"
                            "iterable that can hold other objects, like a list / tuple etc.")
        dataset_names = []
        for dset in hangar_datasets:
            if not (hasattr(dset, 'get') and hasattr(dset, 'name')):
                raise TypeError("`hangar_dataset` contains invalid hangar dataset(s)")
            if len(dset) != dataset_len:
                raise RuntimeError("Got datasets with different lengths")
            dataset_names.append(dset.name)
        self.dataset_array = hangar_datasets
        self.dataset_names = dataset_names
        self._allowed_samples = None
        self.sample_subset(keys, index_range)

    def sample_subset(self, keys=None, index_range=None):
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
        types = []
        for dset in self.dataset_array:
            if converter:
                types.append(converter(dset.dtype))
            else:
                types.append(dset.dtype)
        return tuple(types)

    def get_shapes(self, converter=None):
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
