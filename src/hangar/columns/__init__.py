from typing import Union

ModifierTypes = Union['NestedSampleReader', 'FlatSubsampleReader']

from .metadata import MetadataReader, MetadataWriter
from .column import Columns


__all__ = ('Columns', 'ModifierTypes', 'MetadataReader', 'MetadataWriter')
