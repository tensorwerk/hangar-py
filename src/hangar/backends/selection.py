from collections import namedtuple

from .hdf5 import HDF5_00_Parser, HDF5_00_FileHandles
from .np_raw import NUMPY_00_Parser, NUMPY_00_FileHandles

CODE_BACKEND_MAP = {
    # LOCALS -> [0:50]
    b'00': HDF5_00_Parser(),   # hdf5_00
    b'01': NUMPY_00_Parser(),  # numpy_00
    b'02': None,               # tiledb_00
    # REMOTES -> [50:100]
    b'50': None,               # remote_00
    b'51': None,               # url_00
}

BACKEND_CODE_MAP = {
    # LOCALS -> [0:50]
    'hdf5_00': HDF5_00_Parser(),    # 00
    'numpy_00': NUMPY_00_Parser(),  # 01
    'tiledb_00': None,              # 02
    # REMOTES -> [50:100]
    'remote_00': None,              # 50
    'tiledb_01': None,              # 51
}

BACKEND_ACCESSOR_MAP = {
    # LOCALS -> [0:50]
    'hdf5_00': HDF5_00_FileHandles,
    'numpy_00': NUMPY_00_FileHandles,
    'tiledb_00': None,
    # REMOTES -> [50:100]
    'remote_00': None,
    'url_00': None,
}


def backend_decoder(db_val: bytes) -> namedtuple:

    parser = CODE_BACKEND_MAP[db_val[:2]]
    decoded = parser.decode(db_val)
    return decoded


def backend_encoder(backend, *args, **kwargs):

    parser = BACKEND_CODE_MAP[backend]
    encoded = parser.encode(*args, **kwargs)
    return encoded