import warnings
from typing import Iterable, Optional, Union, Tuple


ArraysetsRef = Union['ArraysetDataReader', Iterable['ArraysetDataReader']]


class GroupedColumns(object):
    """Groups hangar columns and validate suitability for usage in dataloaders.

    It can choose a subset of samples in the hangar columns by checking the
    list of keys or an index range. :class:`GroupedColumns` does not expect all
    the input hangar columns to have same length and same keys. It takes a
    `set.intersection` of sample names from all the columns and `keys`
    argument if passed and hence discard non-common keys. This is then
    intersected with samples which are available locally; not existing as
    remote partial data). Based on `keys` or `index_range` (ignore
    `index_range` if `keys` is present) it makes a subset of sample names which
    is then used to fetch the data from hangar columns.
    """

    def __init__(self,
                 columns: ArraysetsRef,
                 keys: Optional[Iterable[Union[int, str]]] = None,
                 index_range: Optional[slice] = None):

        self.columns_col = []
        self.column_names = []
        self._allowed_samples: Tuple[Union[str, int]] = None

        if not isinstance(columns, (list, tuple, set)):
            columns = (columns,)
        if len(columns) == 0:
            raise ValueError('len(columns) cannot == 0')

        column_lens = set()
        all_keys = []
        all_remote_keys = []
        for col in columns:
            if col.iswriteable is True:
                raise TypeError(f'Cannot load columns opened in `write-enabled` checkout.')
            self.columns_col.append(col)
            self.column_names.append(col.column)
            column_lens.add(len(col))
            all_keys.append(set(col.keys()))
            all_remote_keys.append(set(col.remote_reference_keys))

        if len(column_lens) > 1:
            warnings.warn('Columns do not contain equal number of samples', UserWarning)

        common_keys = set.intersection(*all_keys)
        remote_keys = set.union(*all_remote_keys)
        common_local_keys = common_keys.difference(remote_keys)

        if keys:
            keys = set(keys,)
            notCommon = keys.difference(common_keys)
            notLocal = keys.difference(common_local_keys)
            if len(notCommon) > 0:
                raise KeyError(f'Keys: {notCommon} do not exist in all columns.')
            if len(notLocal) > 0:
                raise FileNotFoundError(
                    f'Keys: {notLocal} are remote data samples not downloaded locally.')
            self._allowed_samples = tuple(keys)
        elif index_range:
            if not isinstance(index_range, slice):
                raise TypeError(f'type(index_range): {type(index_range)} != slice')
            # need to sort before slicing on index_range, but since sample keys
            # can be mixed int and str type, convert to common format and sort
            # on that first
            str_keys = [i if isinstance(i, str) else f'#{i}' for i in common_local_keys]
            sorted_keys = sorted(str_keys)
            converted_keys = [int(i[1:]) if i.startswith('#') else i for i in sorted_keys]
            self._allowed_samples = tuple(converted_keys)[index_range]
        else:
            self._allowed_samples = tuple(common_local_keys)

        if len(self._allowed_samples) == 0:
            raise ValueError(
                f'No Samples available common to all columns and available locally.')

    def get_types(self, converter=None):
        """
        Get dtypes of the all the columns in the `GroupedColumns`.

        Parameters
        ----------
        converter : Callable
            A function that takes default dtype (numpy) and convert it to another
            format

        Returns
        -------
        Tuple[np.dtype]
        """
        dtypes = []
        for col in self.columns_col:
            if converter:
                dtypes.append(converter(col.dtype))
            else:
                dtypes.append(col.dtype)
        return tuple(dtypes)

    def get_shapes(self, converter=None):
        """
        Get shapes of the all the columns in the `GroupedColumns`.

        Parameters
        ----------
        converter : Callable
            A function that takes default shape (numpy) and convert it to another
            format

        Returns
        -------
        A tuple of column shapes
        """
        shapes = []
        for col in self.columns_col:
            if col.schema_type == 'variable_shape':
                aset_shape = (None,) * len(col.shape)
            else:
                aset_shape = col.shape
            if converter:
                shapes.append(converter(aset_shape))
            else:
                shapes.append(aset_shape)
        return tuple(shapes)

    @property
    def sample_names(self):
        return self._allowed_samples
