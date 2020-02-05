from typing import Union

from .specs import (
    HDF5_00_DataHashSpec,
    HDF5_01_DataHashSpec,
    NUMPY_10_DataHashSpec,
    LMDB_30_DataHashSpec,
    REMOTE_50_DataHashSpec,
)
from .specparse import backend_decoder

from .selection import (
    BACKEND_ACCESSOR_MAP,
    BACKEND_IS_LOCAL_MAP,
    AccessorMapType,
    backend_from_heuristics,
    backend_opts_from_heuristics,
    parse_user_backend_opts,
)

DataHashSpecsType = Union[
    HDF5_00_DataHashSpec,
    HDF5_01_DataHashSpec,
    NUMPY_10_DataHashSpec,
    LMDB_30_DataHashSpec,
    REMOTE_50_DataHashSpec,
]

__all__ = [
    'BACKEND_ACCESSOR_MAP', 'BACKEND_IS_LOCAL_MAP', 'AccessorMapType',
    'backend_decoder', 'backend_from_heuristics', 'backend_opts_from_heuristics',
    'parse_user_backend_opts', 'DataHashSpecsType'
]
