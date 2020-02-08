from typing import Union

ModifierTypes = Union['NestedSample', 'FlatSubsample']

from .layout_flat import FlatSample
from .layout_nested import FlatSubsample, NestedSample
from .metadata import MetadataReader, MetadataWriter
from .column import Columns


__all__ = (
    'Columns', 'NestedSample', 'FlatSubsample',
    'FlatSample', 'ModifierTypes', 'MetadataReader', 'MetadataWriter',
)
