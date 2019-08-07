import warnings
from typing import Iterable, Optional, Union


ArraysetsRef = Union['ArraysetDataReader', Iterable['ArraysetDataReader']]


class GroupedAsets(object):
    """Groups hangar arraysets and validate suitability for usage in dataloaders.

    It can choose a subset of samples in the hangar arraysets by checking the
    list of keys or an index range. :class:`GroupedAsets` does not expect all the
    input hangar arraysets to have same length and same keys. It takes a `set.union`
    of sample names from all the arraysets and `keys` argument if passed and hence
    discard non-common keys while fetching. Based on `keys` or `index_range`
    (ignore `index_range` if `keys` is present) it makes a subset of sample names which
    is then used to fetch the data from hangar arraysets.
    """

    def __init__(self,
                 arraysets: ArraysetsRef,
                 keys: Optional[Iterable[Union[int, str]]] = None,
                 index_range: Optional[slice] = None):

        if not isinstance(arraysets, (list, tuple)):
            arraysets = (arraysets,)
        if len(arraysets) == 0:
            raise ValueError("`hangar_arraysets` cannot be empty")

        arrayset_names, aset_lens, full_sample_names = [], set(), []
        for arrayset in arraysets:
            kset = set(arrayset.keys())
            arrayset_names.append(arrayset.name)
            full_sample_names.append(kset)
            aset_lens.add(len(kset))
        sample_names = set.intersection(*full_sample_names)
        del full_sample_names
        if len(aset_lens) > 1:
            warnings.warn('Arraysets do not contain equal num samples', UserWarning)

        if keys:
            if not isinstance(keys, (list, tuple, set)):
                keys = (keys,)
            keys = set(keys)
            noncommon_keys = keys.difference(sample_names)
            if len(noncommon_keys) > 0:
                raise ValueError(
                    f'Requested keys: {noncommon_keys} do not exist in all arraysets.')
            self._allowed_samples = tuple(keys)
        elif index_range:
            if not isinstance(index_range, slice):
                raise TypeError(f"`index_range` must be a slice, not {type(keys)}")
            self._allowed_samples = tuple(sample_names)[index_range]
        else:
            self._allowed_samples = tuple(sample_names)

        self.arrayset_array = arraysets
        self.arrayset_names = arrayset_names

    def get_types(self, converter=None):
        """
        Get dtypes of the all the arraysets in the `GroupedAsets`.

        Parameters
        ----------
        converter : Callable
            A function that takes default dtype (numpy) and convert it to another
            format

        Returns
        -------
        A tuple of types
        """
        types = []
        for aset in self.arrayset_array:
            if converter:
                types.append(converter(aset.dtype))
            else:
                types.append(aset.dtype)
        return tuple(types)

    def get_shapes(self, converter=None):
        """
        Get shapes of the all the arraysets in the `GroupedAsets`.

        Parameters
        ----------
        converter : Callable
            A function that takes default shape (numpy) and convert it to another
            format

        Returns
        -------
        A tuple of arrayset shapes
        """
        if self.arrayset_array[0].variable_shape:
            return None
        shapes = []
        for aset in self.arrayset_array:
            if converter:
                shapes.append(converter(aset.shape))
            else:
                shapes.append(aset.shape)
        return tuple(shapes)

    @property
    def sample_names(self):
        return self._allowed_samples
