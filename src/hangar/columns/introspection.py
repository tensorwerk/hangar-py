from .layout_flat import FlatSampleReader, FlatSampleWriter
from .layout_nested import (
    FlatSubsampleReader,
    FlatSubsampleWriter,
    NestedSampleReader,
    NestedSampleWriter
)


def is_column(obj) -> bool:
    """Determine if arbitrary input is an instance of a column layout.

    Returns
    -------
    bool: True if input is an column, otherwise False.
    """
    return isinstance(obj, (FlatSampleReader, FlatSubsampleReader, NestedSampleReader))


def is_writer_column(obj) -> bool:
    """Determine if arbitrary input is an instance of a write-enabled column layout.

    Returns
    -------
    bool: True if input is write-enabled column, otherwise False.
    """
    return isinstance(obj, (FlatSampleWriter, FlatSubsampleWriter, NestedSampleWriter))
