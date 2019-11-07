import struct
import json
from typing import Tuple
from hashlib import blake2b

import numpy as np


# ---------------------------- Arrayset Data ----------------------------------


def array_hash_digest(array: np.ndarray) -> str:
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

    Returns
    -------
    str
        hex digest of the array data.
    """
    hasher = blake2b(array, digest_size=20)
    other_info = struct.pack(f'<{len(array.shape)}QB', *array.shape, array.dtype.num)
    hasher.update(other_info)
    return hasher.hexdigest()


# ------------------------------ Schema ---------------------------------------


def schema_hash_digest(shape: Tuple[int], size: int, dtype_num: int,
                       named_samples: bool, variable_shape: bool,
                       backend_code: str, backend_opts: dict) -> str:
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

    Returns
    -------
    str
        hex digest of this information packed into a binary structure.
    """

    optsHsh = json.dumps(backend_opts, separators=(',', ':')).encode()
    schema_pack = struct.pack(f'<{len(shape)}QQB??2s{len(optsHsh)}s', *shape,
                              size, dtype_num, named_samples, variable_shape,
                              backend_code.encode(), optsHsh)
    schemaHsh = blake2b(schema_pack, digest_size=6)
    return schemaHsh.hexdigest()


# --------------------------- Metadata ----------------------------------------


def metadata_hash_digest(value: str) -> str:
    """Generate the hash digest of some metadata value

    Parameters
    ----------
    value : str
        data to set as the value of the metadata key.

    Returns
    -------
    str
        hex digest of the encoded bytes of this string value.
    """
    hasher = blake2b(value.encode(), digest_size=20)
    return hasher.hexdigest()