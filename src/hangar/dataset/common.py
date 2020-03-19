import typing
from typing import Union, Sequence, List, Set, Tuple

if typing.TYPE_CHECKING:
    import numpy as np
    from hangar.columns.column import ModifierTypes as Columns


class Dataset:
    """Dataset class that does the initial checks to verify whether the provided
    columns can be arranged together as a dataset. These verifications are done on the
    keys of each column. If ``keys`` argument is ``None``, initializer of this class
    makes the viable key list by checking the local and common keys across all columns.
    If ``keys`` argument is provided, then it verifies whether the provided keys are
    good candidates or not. After all the verification, It provides the ``__getitem__``
    method to the callee for consumption


    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or a Sequence
        A single column object of a sequence the column objects
    keys : Sequence[str]
        An sequence collection of sample names. If given only those samples will
        fetched from the column
    """

    def __init__(self, columns: Sequence['Columns'], keys: Sequence[str] = None):
        if not isinstance(columns, (list, tuple, set)):
            columns = (columns,)
        if len(columns) == 0:
            raise ValueError('len(columns) cannot == 0')
        all_keys: List[set] = []
        all_remote_keys: List[set] = []
        for col in columns:
            if col.iswriteable is True:
                raise TypeError(f'Cannot load columns opened in `write-enabled` checkout.')
            all_keys.append(set(col.keys()))
            all_remote_keys.append(set(col.remote_reference_keys))
        common_keys: Set[str] = set.intersection(*all_keys)
        remote_keys: Set[str] = set.union(*all_remote_keys)
        common_local_keys: Set[str] = common_keys.difference(remote_keys)
        if keys:
            if not isinstance(keys, (list, tuple, set)):
                raise TypeError('keys must be a list/tuple/set of hangar sample keys')
            unique = set(keys)
            not_common = unique.difference(common_keys)
            not_local = unique.difference(common_local_keys)
            if len(not_common) > 0:
                raise KeyError(f'{len(not_common)} keys do not exist in all columns.')
            if len(not_local) > 0:
                raise FileNotFoundError(
                    f'{len(not_local)} keys are remote data samples and are not downloaded'
                    f' locally.')
        else:
            keys = common_local_keys
        if len(keys) == 0:
            raise ValueError('No Samples available common to all '
                             'columns and available locally.')

        # TODO: May be we need light weight data accessors instead of columns
        self.columns = columns
        self.keys = list(keys)

    def __getitem__(self, item: Union[str, int]) -> Tuple['np.ndarray']:
        """It takes one sample name and returns a tuple of items from each column for
        the given sample name
        """
        return tuple([col[item] for col in self.columns])
