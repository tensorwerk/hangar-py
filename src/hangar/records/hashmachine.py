import json
import struct
from hashlib import blake2b
from typing import Tuple

import numpy as np


def hash_type_code_from_digest(digest: str) -> str:
    type_code, digest = digest.split('=')
    return type_code


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


def _make_hashable(o):
    """Sort container object and deterministically output frozen representation"""
    if isinstance(o, (tuple, list)):
        return tuple((_make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(_make_hashable(e) for e in o))

    return o


def schema_hash_digest(schema: dict, *, tcode='1') -> str:
    """Generate the schema hash for some schema specification

    Returns
    -------
    str
        hex digest of this information with typecode prepended by '{tcode}='.
    """
    if tcode == '1':
        frozenschema = _make_hashable(schema)
        serialized = repr(frozenschema).encode()
        digest = blake2b(serialized, digest_size=6).hexdigest()
        res = f'1={digest}'
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
