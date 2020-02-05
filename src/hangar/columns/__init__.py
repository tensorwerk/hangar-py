from typing import Union

from .flat import FlatSample
from .nested import FlatSubsample, NestedSample
from .constructors import Sample, Subsample, AsetTxn

ModifierTypes = Union[NestedSample, FlatSubsample]

from .arrayset import Arraysets
from .metadata import MetadataReader, MetadataWriter


__all__ = ('Arraysets', 'Sample', 'Subsample', 'AsetTxn', 'NestedSample',
           'FlatSubsample', 'FlatSample', 'ModifierTypes',
           'MetadataReader',
           'MetadataWriter')
