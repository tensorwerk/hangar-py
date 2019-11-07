import struct
import json
from typing import Tuple
from hashlib import blake2b

import numpy as np


# ---------------------------- Arrayset Data ----------------------------------


def array_hash_digest(array: np.ndarray) -> str:

    hasher = blake2b(array, digest_size=20)
    other_info = struct.pack(f'<{len(array.shape)}QB', *array.shape, array.dtype.num)
    hasher.update(other_info)
    return hasher.hexdigest()


# ------------------------------ Schema ---------------------------------------


def schema_hash_digest(shape: Tuple[int], size: int, dtype_num: int,
                       named_samples: bool, variable_shape: bool,
                       backend_code: str, backend_opts: dict) -> str:

    optsHsh = json.dumps(backend_opts, separators=(',', ':')).encode()
    schema_pack = struct.pack(f'<{len(shape)}QQB??2s{len(optsHsh)}s', *shape,
                              size, dtype_num, named_samples, variable_shape,
                              backend_code.encode(), optsHsh)
    schemaHsh = blake2b(schema_pack, digest_size=6)
    return schemaHsh.hexdigest()


# --------------------------- Metadata ----------------------------------------


def metadata_hash_digest(value: str) -> str:

    hasher = blake2b(value.encode(), digest_size=20)
    return hasher.hexdigest()