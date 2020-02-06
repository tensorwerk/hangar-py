from typing import Union

from .layout_flat import FlatSample
from .layout_nested import FlatSubsample, NestedSample
from .common import AsetTxn

ModifierTypes = Union[NestedSample, FlatSubsample]

from .arrayset import Arraysets
from .metadata import MetadataReader, MetadataWriter


__all__ = (
    'Arraysets', 'AsetTxn', 'NestedSample', 'FlatSubsample',
    'FlatSample', 'ModifierTypes', 'MetadataReader', 'MetadataWriter',
)
