from typing import Union

from .aset_nested import (
    SubsampleReaderModifier,
    SubsampleWriterModifier,
    SubsampleReader,
    SubsampleWriter
)
from .aset_flat import SampleReaderModifier, SampleWriterModifier
from .constructors import Sample, Subsample, AsetTxn

ModifierTypes = Union[
    SubsampleReaderModifier,
    SubsampleWriterModifier,
    SampleReaderModifier,
    SampleWriterModifier]

__all__ = ('Sample', 'Subsample', 'AsetTxn',
           'SubsampleReaderModifier', 'SubsampleWriterModifier',
           'SubsampleReader', 'SubsampleWriter',
           'SampleReaderModifier', 'SampleWriterModifier',
           'ModifierTypes')
