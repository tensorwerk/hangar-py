from typing import Sequence, TYPE_CHECKING
from collections import OrderedDict

from .common import HangarDataset
from ..utils import experimental

try:
    import torch
except (ImportError, ModuleNotFoundError):
    raise ImportError(
        'Could not import "pytorch" library. Ensure library is '
        'installed correctly to use pytorch dataloader functions') from None


if TYPE_CHECKING:
    from hangar.columns.column import ModifierTypes as Columns


class TorchDataset(torch.utils.data.Dataset):
    """TorchDataset inherits :class:`torch.utils.data.Dataset` and accepts few convenient
    arguments to wrap hangar columns to be used in :class:`torch.utils.data.DataLoaders`.
    It accepts a hangar Dataset object which exposes all the user requested columns and
    an array of keys to sample from. For more details, checkout
    `PyTorch Dataset <https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset>`_
    """

    def __init__(self, hangar_dataset: HangarDataset, as_dict: bool = False):
        self.dataset = hangar_dataset
        self.column_names = [col._column_name for col in hangar_dataset.columns]
        self._as_dict = as_dict

    def __len__(self) -> int:
        return len(self.dataset.keys)

    def __getitem__(self, index: int):
        key = self.dataset.keys[index]
        if self._as_dict:
            return OrderedDict(zip(self.column_names, self.dataset[key]))
        else:
            return self.dataset[key]


@experimental
def make_torch_dataset(columns: Sequence['Columns'], keys: Sequence[str] = None,
                       as_dict: bool = False) -> TorchDataset:
    """Returns a :class:`torch.utils.data.Dataset` object which can be loaded into
    a :class:`torch.utils.data.DataLoader`.

    .. Note::

        Column with layouts ``str`` or ``ndarray nested`` are not compatible with the
        dataset APIs in the current release. So making dataset is only possible for
        columns with layout ``ndarray flat``

    .. Note::

        PyTorch's :class:`torch.utils.data.DataLoader` can effectively do custom
        operations such as shuffling, batching, multiprocessed read etc and hence we
        limit the surface area of the dataset API here just to open the channel for
        reading. Use DataLoaders for such operations

    .. warning::

       On Windows systems, setting the parameter ``num_workers`` in the
       resulting :class:`torch.utils.data.DataLoader` method will result in a
       RuntimeError or deadlock. This is due to limitations of multiprocess
       start methods on Windows itself. Using the default argument value
       (``num_workers=0``) will let the DataLoader work in single process mode
       as expected.


    Parameters
    ----------
    columns : :class:`~hangar.columns.column.Columns` or Sequence
        A column object, a tuple of column object or a list of column
        objects.
    keys : Sequence[str]
        An sequence collection of sample names. If given only those samples will
        fetched from the column
    as_dict : bool
        Return the data as an OrderedDict with column names as keys. If False, it returns
        a tuple of arrays


    Returns
    -------
    dict or tuple

    Examples
    --------
    >>> from hangar import Repository
    >>> from torch.utils.data import DataLoader
    >>> from hangar import make_torch_dataset
    >>> from collections import namedtuple
    >>> repo = Repository('.')
    >>> co = repo.checkout()
    >>> imgcol = co.columns['images']
    >>> classcol = co.columns['classes']
    >>> dataset = make_torch_dataset((imgcol, classcol), as_dict=True)
    >>> loader = DataLoader(dataset, batch_size=16)
    >>> for batch in loader:
    ...     out = train_model(batch['images'])
    ...     loss = loss_fn(out, batch['classes'])

    Returns
    -------
    :class:`torch.utils.data.Dataset`
    """
    hangar_dataset = HangarDataset(columns, keys)
    return TorchDataset(hangar_dataset, as_dict)
