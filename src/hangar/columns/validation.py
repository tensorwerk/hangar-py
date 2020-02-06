from typing import NamedTuple
from ..utils import is_ascii

import numpy as np


class CompatibleData(NamedTuple):
    """Bool describing if data is compatible and if False, the reason it is rejected.
    """
    compatible: bool
    reason: str


def _ndarray_column_variable_shape(data, dtype_num, shape) -> CompatibleData:
    """Determine if an array is compatible with the arraysets schema

    Parameters
    ----------
    data : :class:`np.ndarray`
        input user data to check that it is a numpy array and is compatible
        with the current schema.
    dtype_num : int
        numpy numeric code for the array dtype
    shape : Tuple[int]
        maximum dimension sizes of a single sample

    Returns
    -------
    CompatibleData
        compatible and reason field
    """
    compatible = True
    reason = ''

    if not isinstance(data, np.ndarray):
        compatible = False
        reason = f'`data` argument type: {type(data)} != `np.ndarray`'
    elif data.dtype.num != dtype_num:
        compatible = False
        reason = f'dtype: {data.dtype} != aset: {np.typeDict[dtype_num]}.'
    elif not data.flags.c_contiguous:
        compatible = False
        reason = f'`data` must be "C" contiguous array.'
    elif data.ndim != len(shape):
        compatible = False
        reason = f'data rank {data.ndim} != aset rank {len(shape)}'
    elif not all([(dim > maxdim) for dim, maxdim in zip(data.shape, shape)]):
        compatible = False
        reason = f'shape {data.shape} exceeds schema max {shape}'

    res = CompatibleData(compatible, reason)
    return res

_ndarray_column_variable_shape._schema_fields_ = ('dtype_num', 'shape')


def _ndarray_column_fixed_shape(data, dtype_num, shape) -> CompatibleData:
    """Determine if an array is compatible with the arraysets schema

    Parameters
    ----------
    data : :class:`np.ndarray`
        input user data to check that it is a numpy array and is compatible
        with the current schema.
    dtype_num : int
        numpy numeric code for the array dtype
    shape : Tuple[int]
        required shape of the array

    Returns
    -------
    CompatibleData
        compatible and reason field
    """
    compatible = True
    reason = ''

    if not isinstance(data, np.ndarray):
        compatible = False
        reason = f'`data` argument type: {type(data)} != `np.ndarray`'
    elif data.dtype.num != dtype_num:
        compatible = False
        reason = f'dtype: {data.dtype} != aset: {np.typeDict[dtype_num]}.'
    elif not data.flags.c_contiguous:
        compatible = False
        reason = f'`data` must be "C" contiguous array.'
    elif data.shape != shape:
        compatible = False
        reason = f'data shape {data.shape} != fixed schema {shape}'

    res = CompatibleData(compatible, reason)
    return res

_ndarray_column_fixed_shape._schema_fields_ = ('dtype_num', 'shape')


def _str_column_variable_shape(data) -> CompatibleData:
    """Determine if an array is compatible with the arraysets schema

    Parameters
    ----------
    data : str
        input user data to check that it is string data and is compatible
        with the current schema.

    Returns
    -------
    CompatibleData
        compatible and reason field
    """
    compatible = True
    reason = ''

    if not isinstance(data, str) or not is_ascii(data):
        compatible = False
        reason = f'data {data} not valid. Must be ascii-only str'

    res = CompatibleData(compatible, reason)
    return res

_str_column_variable_shape._schema_fields_ = ()


class DataValidator:

    __slots__ = ('_schema', '_dispatcher', '_schema_fields')

    def __init__(self):
        self._schema = None
        self._dispatcher = None
        self._schema_fields = None

    @property
    def schema(self):
        return self._schema

    @schema.setter
    def schema(self, value):
        self._schema = value

        if self._schema['column_type'] == 'ndarray':
            if self._schema['schema_type'] == 'fixed_shape':
                self._dispatcher = _ndarray_column_fixed_shape
                self._schema_fields = [
                    self._schema[f] for f in _ndarray_column_fixed_shape._schema_fields_]
            elif self._schema['schema_type'] == 'variable_shape':
                self._dispatcher = _ndarray_column_variable_shape
                self._schema_fields = [
                    self._schema[f] for f in _ndarray_column_variable_shape._schema_fields_]
            else:
                raise ValueError(f'missing parser method {self._schema["schema_type"]}')
        elif self._schema['column_type'] == 'str':
            if self._schema['schema_type'] == 'variable_shape':
                self._dispatcher = _str_column_variable_shape
                self._schema_fields = [
                    self._schema[f] for f in _str_column_variable_shape._schema_fields_]
            else:
                raise ValueError(f'missing parser method {self._schema["schema_type"]}')
        else:
            raise ValueError(f'missing parser method {self._schema["schema_type"]}')

    @schema.deleter
    def schema(self):
        self._schema = None
        self._dispatcher = None
        self._schema_fields = None

    def verify_data_compatible(self, data):
        return self._dispatcher(data, *self._schema_fields)
