from typing import Union

from .arrayset_nested import (
    SubsampleReaderModifier,
    SubsampleWriterModifier,
    SubsampleReader,
    SubsampleWriter
)
from .arrayset_flat import SampleReaderModifier, SampleWriterModifier
from .constructors import Sample, Subsample, AsetTxn

ModifierTypes = Union[
    SubsampleReaderModifier,
    SubsampleWriterModifier,
    SampleReaderModifier,
    SampleWriterModifier]

WriterModifierTypes = Union[SubsampleWriterModifier, SampleWriterModifier]

from .arrayset import Arraysets
from .metadata import MetadataReader, MetadataWriter

__all__ = ('Arraysets', 'Sample', 'Subsample', 'AsetTxn',
           'SubsampleReaderModifier', 'SubsampleWriterModifier',
           'SubsampleReader', 'SubsampleWriter',
           'SampleReaderModifier', 'SampleWriterModifier',
           'ModifierTypes', 'WriterModifierTypes',
           'MetadataReader', 'MetadataWriter')
