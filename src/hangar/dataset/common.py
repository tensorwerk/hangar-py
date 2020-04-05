import typing
from typing import Union, Sequence, Tuple, Any
from ..columns import is_column, is_writer_column
from ..optimized_utils import is_ordered_sequence, is_iterable

if typing.TYPE_CHECKING:
    from hangar.columns.column import ModifierTypes as Columns
    KeyType = Union[str, int]


class HangarDataset:
    """Dataset class that does the initial checks to verify whether the provided
    columns can be arranged together as a dataset. These verifications are done on the
    keys of each column. If ``keys`` argument is ``None``, initializer of this class
    makes the viable key list by checking the local and common keys across all columns.
    If ``keys`` argument is provided, then it verifies whether the provided keys are
    good candidates or not. After all the verification, It provides the ``__getitem__``
    method to the callee for consumption


    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or a Sequence['Columns']
        A single column object of a sequence the column objects
    keys : Sequence['KeyType']
        An sequence collection of sample names. If given only those samples will
        fetched from the column
    """

    def __init__(self,
                 columns: Union['Columns', Sequence['Columns']],
                 keys: Sequence['KeyType'] = None):

        # ------- verify user args are valid hangar column instance(s) --------

        if is_ordered_sequence(columns):
            if len(columns) == 0:
                raise TypeError(f'Atleast one element must exist in input sequence.')
            for obj in columns:
                if not is_column(obj):
                    raise TypeError(
                        f'All elements of input sequence must be hangar column objects.')
                elif is_writer_column(obj):
                    raise TypeError(
                        f'Columns cannot be used while accessed via a `write-enabled` '
                        f'checkout. Please close the checkout and reopen the column in '
                        f'via a new checkout opened in `read-only` mode.')
        else:
            if not is_column(columns):
                raise TypeError(
                    f'columns arguments must be a live Hangar column or '
                    f'sequenc of column instances, not {type(columns)}.')
            elif is_writer_column(columns):
                raise TypeError(
                    f'Columns cannot be used while accessed via a `write-enabled` '
                    f'checkout. Please close the checkout and reopen the column in '
                    f'via a new checkout opened in `read-only` mode.')
            columns = (columns,)

        # --------- inspect column keys / validate requested view ------------

        first_col = columns[0]
        common_local_keys = set(first_col.keys(local=True))
        remote_keys = set(first_col.remote_reference_keys)
        for col in columns[1:]:
            common_local_keys.intersection_update(col.keys(local=True))
            remote_keys.update(col.remote_reference_keys)

        if len(common_local_keys) == 0:
            raise KeyError(
                f'The intersection of common keys (whose data exists on the '
                f'local machine) between all specified columns is empty. No '
                f'data can be returned.')

        # -------- validate user requested sample keys exist in view ----------

        if keys is not None:
            if not is_iterable(keys):
                raise TypeError(
                    f'If `keys` argument is specified, an iterable sequence of key '
                    f'elements must be provided, not type {type(keys)}')
            # Note: The set of requested user keys is ONLY used for input validation.
            # We use the original sequence passed in by the user to specify which
            # samples and in which order data should be retrieved in.
            keys_set = set(keys)
            if not keys_set.issubset(common_local_keys):
                if not keys_set.isdisjoint(remote_keys):
                    raise FileNotFoundError(
                        f'Data corresponding to requested sample keys has not '
                        f'been downloaded to the local repo copy. Please fetch data '
                        f'for all requested samples and retry this operation.')
                else:
                    raise KeyError(
                        f'Not all requested sample keys exist in the specified '
                        f'columns, cannot continue with dataloader operation.')
        else:
            keys = common_local_keys

        self._keys = list(keys) if not isinstance(keys, list) else keys
        self._columns = columns

    @property
    def keys(self):
        return self._keys

    @property
    def columns(self):
        return self._columns

    def __getitem__(self, key: Union[str, int]) -> Tuple[Any]:
        """It takes one sample name and returns a tuple of items from each column for
        the given sample name
        """
        return tuple([col[key] for col in self._columns])
