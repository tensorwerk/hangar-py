from functools import reduce
from operator import getitem as op_getitem
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

        if is_ordered_sequence(columns):
            if len(columns) == 0:
                raise TypeError(f'Atleast one element must exist in input sequence.')
        else:
            columns = (columns,)
        for obj in columns:
            if not is_column(obj):
                raise TypeError(
                    f'All elements of input sequence must be hangar column objects.')
            elif is_writer_column(obj):
                raise TypeError(
                    f'Columns cannot be used while accessed via a `write-enabled` '
                    f'checkout. Please close the checkout and reopen the column in '
                    f'via a new checkout opened in `read-only` mode.')

        # ======= Process keys for efficient downstream handling (inefficient) =======
        if not keys:
            # If no keys, we expect all columns have same key names
            keys = tuple(columns[0].keys(local=True))
            for col in columns[1:]:
                next_keys = tuple(col.keys(local=True))
                if keys != next_keys:
                    raise RuntimeError("Keys from multiple columns couldn't be matched. "
                                       "Pass keys explicitly while creating dataset")
            if not keys:
                raise RuntimeError("No local data found")

        self._keys = keys
        self._columns = columns

    @property
    def columns(self):
        return self._columns

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, index: int) -> Tuple[Any]:
        """It takes one sample index and returns a tuple of items from each column for
        the given sample name for the given index.
        """
        # TODO should handle excpetion missing
        # TODO: explicit naming for getitem_from_index
        keys = self._keys[index]
        if isinstance(keys, (int, str)):
            ret = tuple((col[keys] for col in self._columns))
        elif isinstance(keys, (list, tuple)) and len(self._columns) == 1:
            ret = reduce(op_getitem, keys, self._columns)
        elif isinstance(keys, (list, tuple)):
            ret = tuple([reduce(op_getitem, keys, col)
                         for key, col in zip(keys, self._columns)])
        else:
            raise TypeError("Keys are not parsable. Try with a list/tuple of "
                            "int/str/list/tuple. Check the documentation for more"
                            " details")
        return ret
