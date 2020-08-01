import typing
from contextlib import ExitStack
from typing import Union, Sequence, Tuple, List, Any, Dict

from ..columns import is_column, is_writer_column
from ..mixins.datasetget import GetMixin
from ..optimized_utils import is_ordered_sequence

if typing.TYPE_CHECKING:
    from hangar.columns.column import ModifierTypes as Columns
    KeyType = Union[str, int, List, Tuple]


class HangarDataset(GetMixin):
    """Dataset class that does the initial checks to verify whether the provided
    columns can be arranged together as a dataset. These verifications are done on the
    keys of each column. If ``keys`` argument is ``None``, initializer of this class
    makes the key list by checking the local keys across all columns.
    If ``keys`` argument is provided, then it assumes the provided keys are valid and
    restrain from doing any more check on it.
    It provides the ``__getitem__`` accessor for downstream process to consume the
    grouped data


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
                 keys: 'KeyType' = None):

        self._columns: Dict[str, 'Columns'] = {}
        self._is_conman_counter = 0

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
                raise PermissionError(
                    f'Columns cannot be used while accessed via a `write-enabled` '
                    f'checkout. Please close the checkout and reopen the column in '
                    f'via a new checkout opened in `read-only` mode.')
            column_name = obj.column
            self._columns[column_name] = obj

        if not keys:
            if len(set((col.column_layout for col in self._columns.values()))) != 1:  # all same type
                raise ValueError(f"keys must be passed when all columns are not same type")

            keys = []
            for idx, col in enumerate(self._columns.values()):
                # only match top level keys, even for nested columns
                if idx == 0:
                    standard_keys = set(col.keys(local=True))
                    if len(standard_keys) == 0:
                        raise RuntimeError("No local data found")
                else:
                    keys = set(col.keys(local=True))
                    if len(standard_keys.symmetric_difference(keys)) != 0:
                        raise KeyError("Keys from multiple columns couldn't be matched. "
                                       "Pass keys explicitly while creating dataset")
                column_name = col.column
                if col.column_layout == 'flat':
                    column_keys = ((column_name, sample) for sample in col.keys(local=True))
                elif col.column_layout == 'nested':
                    column_keys = ((column_name, sample, ...) for sample in col.keys(local=True))
                else:
                    raise RuntimeError(f'unknown column layout: {col}')

                keys.extend(column_keys)

        self._keys = keys

    @property
    def columns(self):
        return self._columns

    @property
    def _is_conman(self):
        return bool(self._is_conman_counter)

    def __len__(self):
        return len(self._keys)

    def __enter__(self):
        with ExitStack() as stack:
            for asetN in list(self._columns.keys()):
                stack.enter_context(self._columns[asetN])
            self._is_conman_counter += 1
            self._stack = stack.pop_all()
        return self

    def __exit__(self, *exc):
        self._is_conman_counter -= 1
        self._stack.close()

    def index_get(self, index: int) -> Tuple[Any]:
        """It takes one sample index and returns a tuple of items from each column for
        the given sample name for the given index.
        """
        keys = self._keys[index]
        return self[keys]
