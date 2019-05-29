from collections import namedtuple

from .hdf5 import HDF5_00_Parser

CODE_BACKEND_MAP = {
    # LOCALS -> [0:50]
    b'00': HDF5_00_Parser(),  # hdf5_00
    b'01': None,              # numpy_00
    b'02': None,              # tiledb_00
    # REMOTES -> [50:100]
    b'50': None,              # remote_00
    b'51': None,              # url_00
}

BACKEND_CODE_MAP = {
    # LOCALS -> [0:50]
    'hdf5_00': HDF5_00_Parser(),  # 00
    'numpy_00': None,             # 01
    'tiledb_00': None,            # 02
    # REMOTES -> [50:100]
    'remote_00': None,            # 50
    'url_00': None,               # 51
}

LOCATION_CODE_BACKEND_MAP = {
    # LOCALS -> [0:50]
    '00': 'hdf5_00',
    '01': 'numpy_00',
    '02': 'tiledb_00',
    # REMOTES -> [50:100]
    '50': 'remote_00',
    '51': 'url_00',
}
LOCATION_BACKEND_CODE_MAP = dict([[v, k] for k, v in LOCATION_CODE_BACKEND_MAP.items()])


def backend_decoder(db_val: bytes) -> namedtuple:

    parser = CODE_BACKEND_MAP[db_val[:2]]
    decoded = parser.decode(db_val)
    return decoded


def backend_decoder_name(db_val: bytes) -> str:
    val = db_val.decode()[:2]
    name = LOCATION_CODE_BACKEND_MAP[val]
    return name


def backend_encoder(backend, *args, **kwargs):

    parser = BACKEND_CODE_MAP[backend]
    encoded = parser.encode(*args, **kwargs)
    return encoded