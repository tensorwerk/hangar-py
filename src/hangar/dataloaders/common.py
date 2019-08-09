import warnings
from typing import Iterable, Optional, Union, Tuple, List


ArraysetsRef = Union['ArraysetDataReader', Iterable['ArraysetDataReader']]


class GroupedAsets(object):
    """Groups hangar arraysets and validate suitability for usage in dataloaders.

    It can choose a subset of samples in the hangar arraysets by checking the
    list of keys or an index range. :class:`GroupedAsets` does not expect all
    the input hangar arraysets to have same length and same keys. It takes a
    `set.union` of sample names from all the arraysets and `keys` argument if
    passed and hence discard non-common keys while fetching. Based on `keys` or
    `index_range` (ignore `index_range` if `keys` is present) it makes a subset
    of sample names which is then used to fetch the data from hangar arraysets.
    """

    def __init__(self,
                 arraysets: ArraysetsRef,
                 keys: Optional[Iterable[Union[int, str]]] = None,
                 index_range: Optional[slice] = None):

        self.arrayset_array = []
        self.arrayset_names = []
        self._allowed_samples: Tuple[Union[str, int]] = None

        if not isinstance(arraysets, (list, tuple, set)):
            arraysets = (arraysets,)
        if len(arraysets) == 0:
            raise ValueError('len(arraysets) cannot == 0')

        aset_lens = set()
        all_aset_keys = []
        for arrayset in arraysets:
            if arrayset.iswriteable is True:
                raise TypeError(f'Cannot load arraysets opened in `write-enabled` checkout.')
            self.arrayset_array.append(arrayset)
            self.arrayset_names.append(arrayset.name)
            aset_lens.add(len(arrayset))
            all_aset_keys.append(set(arrayset.keys()))
        common_aset_keys = set.intersection(*all_aset_keys)

        if len(aset_lens) > 1:
            warnings.warn('Arraysets do not contain equal num samples', UserWarning)

        if keys:
            keys = set(keys,)
            noncommon_keys = keys.difference(common_aset_keys)
            if len(noncommon_keys) > 0:
                raise ValueError(f'Keys: {noncommon_keys} do not exist in all arraysets.')
            self._allowed_samples = tuple(keys)
        elif index_range:
            if not isinstance(index_range, slice):
                raise TypeError(f'type(index_range): {type(index_range)} != slice')
            # need to sort before sliceing on index_range, but since sample
            # keys can be mixed int and str type, convert to common format and
            # sort on that first
            str_keys = [i if isinstance(i, str) else f'#{i}' for i in common_aset_keys]
            sorted_keys = sorted(str_keys)
            converted_keys = [int(i[1:]) if i.startswith('#') else i for i in sorted_keys]
            self._allowed_samples = tuple(converted_keys)[index_range]
        else:
            self._allowed_samples = tuple(common_aset_keys)

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
                print(aset)
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
