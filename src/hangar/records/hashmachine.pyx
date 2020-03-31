import array
from cpython cimport array

import numpy as np
from hashlib import blake2b


cpdef str hash_type_code_from_digest(str digest):
    return digest[0]


cpdef object hash_func_from_tcode(str tcode):
    if tcode == '0':
        return ndarray_hasher_tcode_0
    elif tcode == '1':
        return schema_hasher_tcode_1
    elif tcode == '2':
        return pystr_hasher_tcode_2
    else:
        raise ValueError(f'unknown hash function type code. tcode: {tcode}')


# ---------------------------- numpy ndarray data ------------------------------


cdef bytes ser_int_list(list lst):
    cdef Py_ssize_t n=len(lst)
    cdef array.array res=array.array('i')

    array.resize(res, n)  #preallocate memory
    for i in range(n):
        # lst.__get__() needs Python-Integer, so let i
        # be a python-integer (not cdef)
        res.data.as_ints[i] = lst[i]
    return res.data.as_chars[:n*sizeof(int)]


def ndarray_hasher_tcode_0(array not None):
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
        hex digest of the array data with typecode prepended by '{tcode}='.
    """
    cdef str digest
    cdef bytes other_info
    cdef list shape = []

    shape = list(array.shape)
    shape.append(array.dtype.num)
    other_info = ser_int_list(shape)

    hasher = blake2b(array, digest_size=20)
    hasher.update(other_info)
    digest = hasher.hexdigest()
    return f'0={digest}'


# ------------------------------ Schema ---------------------------------------


def _make_hashable(o):
    """Sort container object and deterministically output frozen representation
    """
    if isinstance(o, (tuple, list)):
        return tuple((_make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(_make_hashable(e) for e in o))
    return o


cpdef str schema_hasher_tcode_1(dict schema):
    """Generate the schema hash for some schema specification

    Parameters
    ----------
    schema : dict
        dict representation of the schema spec.

    Returns
    -------
    str
        hex digest of this information with typecode prepended by '{tcode}='.
    """
    cdef bytes serialized
    cdef str digest, res

    frozenschema = _make_hashable(schema)
    serialized = repr(frozenschema).encode()
    digest = blake2b(serialized, digest_size=6).hexdigest()
    res = f'1={digest}'
    return res


# --------------------------- string type data ----------------------------------------


cpdef str pystr_hasher_tcode_2(str value):
    """Generate the hash digest of some metadata value

    Parameters
    ----------
    value : str
        data to set as the value of the metadata key.

    Returns
    -------
    str
        hex digest of the metadata value with typecode prepended by '{tcode}='.
    """
    cdef bytes raw
    cdef str digest, res

    raw = value.encode()
    digest = blake2b(raw, digest_size=20).hexdigest()
    res = f'2={digest}'
    return res
