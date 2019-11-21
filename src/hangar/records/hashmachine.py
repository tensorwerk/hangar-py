import struct
import json
from typing import Tuple
from hashlib import blake2b

import numpy as np


# ---------------------------- Arrayset Data ----------------------------------


def array_hash_digest(array: np.ndarray, *, tcode='0') -> str:
    """Generate the hex digest of some array data.

    This method hashes the concatenation of both array data bytes as well as a
    binary struct with the array shape and dtype num packed in. This is in
    order to avoid hash collisions where an array can have the same bytes, but
    different shape. an example of a collision is: np.zeros((10, 10, 10)) and
    np.zeros((1000,))

    Parameters
    ----------
    array : np.ndarray
        array data to take the hash of
    tcode : str, optional, kwarg-only
        hash calculation type code. Included to allow future updates to change
        hashing algorithm, kwarg-only, by default '0'

    Returns
    -------
    str
        hex digest of the array data with typecode prepended by '{tcode}='.
    """
    if tcode == '0':
        hasher = blake2b(array, digest_size=20)
        other_info = struct.pack(f'<{len(array.shape)}QB', *array.shape, array.dtype.num)
        hasher.update(other_info)
        res = f'0={hasher.hexdigest()}'
    else:
        raise ValueError(
            f'Invalid Array Hash Digest Type Code {tcode}. If encountered during '
            f'normal operation, please report to hangar development team.')
    return res


# ------------------------------ Schema ---------------------------------------


def schema_hash_digest(shape: Tuple[int], size: int, dtype_num: int,
                       named_samples: bool, variable_shape: bool,
                       backend_code: str, backend_opts: dict,
                       *, tcode: str = '1') -> str:
    """Generate the schema hash for some schema specification

    Parameters
    ----------
    shape : Tuple[int]
        shape of the array data
    size : int
        number of elements in the array data
    dtype_num : int
        datatype numeric code of array data
    named_samples : bool
        do samples go by user provided, or generated names?
    variable_shape : bool
        can samples contain dimensions with lower length then the a dimension's
        max size?
    backend_code : str
        backend format code which specified backend new samples writes to this
        schema are stored in.
    backend_opts : dict
        backend options applied to new writes of samples to this schema.
    tcode : str, optional, kwarg-only
        hash calculation type code. Included to allow future updates to change
        hashing algorithm, kwarg-only, by default '1'

    Returns
    -------
    str
        hex digest of this information with typecode prepended by '{tcode}='.
    """
    if tcode == '1':
        optsHsh = json.dumps(backend_opts, separators=(',', ':')).encode()
        schema_pack = struct.pack(f'<{len(shape)}QQB??2s{len(optsHsh)}s', *shape,
                                  size, dtype_num, named_samples, variable_shape,
                                  backend_code.encode(), optsHsh)
        schemaHsh = blake2b(schema_pack, digest_size=6)
        res = f'1={schemaHsh.hexdigest()}'
    else:
        raise ValueError(
            f'Invalid Schema Hash Type Code {tcode}. If encountered during '
            f'normal operation, please report to hangar development team.')
    return res


# --------------------------- Metadata ----------------------------------------


def metadata_hash_digest(value: str, *, tcode: str = '2') -> str:
    """Generate the hash digest of some metadata value

    Parameters
    ----------
    value : str
        data to set as the value of the metadata key.
    tcode : str, optional, kwarg-only
        hash calculation type code. Included to allow future updates to change
        hashing algorithm, kwarg-only, by default '2'

    Returns
    -------
    str
        hex digest of the metadata value with typecode prepended by '{tcode}='.
    """
    if tcode == '2':
        hasher = blake2b(value.encode(), digest_size=20)
        res = f'2={hasher.hexdigest()}'
    else:
        raise ValueError(
            f'Invalid Metadata Hash Type Code {tcode}. If encountered during '
            f'normal operation, please report to hangar development team.')
    return res