'''Definition and dynamic routing to Hangar backend implementations.

This module defines the available backends for a Hangar installation & provides
dynamic routing of method calls to the appropriate backend from a stored record
specification.

Identifiation
-------------

A two digit code identifies which backend some record belongs to. In general,
the codes are split with the following pattern:

    * [00 : 49] -> backends handeling data residing on the local disk
    * [50 : 99] -> backends refereing to data residing on a remote server

Process & Guarrentees
---------------------

In order to maintain backwards compatibility across versions of Hangar into the
future the following ruleset is specified and MUST BE HONORED:

    * When a new backend is proposed, the contributor(s) provide the class with a
      cannonical name (HDF5_00, TILEDB_01, etc) for developer consumption in the
      backend. The review team will provide an available two-digit code which all
      records corresponding to that backend must identify themselves with.

    * Once a new backend is accepted, the code assigned to it is PERMENANT &
      UNCHANGING. The same code cannot be used in the future for other backends.

    * Each backend independently determines the information it needs to log/store to
      uniquely identify and retrieve a sample stored by it. There is no standard
      format, each is free to define whatever fields they find most convenient.
      Unique encode/decode methods are defined in order to serialize this
      information to bytes and then reconstruct the information later. These bytes
      are what are passed in when a retrieval request is made, and returned when a
      storage request for some piece of data is performed.

    * Once accepted, The record format specified (ie. the byte representation
     described above) cannot be modified in any way. This must remain permenant!

    * Backend (internal) methods can be updated, optimized, and/or changed at any
     time so long as:

        * No changes to the record format specification are introduced
        * Data stored via any previous iteration of the backend's accessor methods
          can be retrieved bitwise exactally by the "updated" version.

Before proposing a new backend or making changes to this file, please consider
reaching out to the Hangar core development team so we can guide you through the
process.
'''

from collections import namedtuple

import numpy as np

from .hdf5 import HDF5_00_Parser, HDF5_00_FileHandles
from .np_mmap import NUMPY_00_Parser, NUMPY_00_FileHandles
from .remote_unknown import REMOTE_UNKNOWN_00_Parser, REMOTE_UNKNOWN_00_Handler

BACKEND_PARSER_MAP = {
    # LOCALS -> [00:50]
    b'00': HDF5_00_Parser(),
    b'01': NUMPY_00_Parser(),
    b'02': None,               # tiledb_00 - Reserved
    # b'03': NUMPY_01_Parser(),
    # REMOTES -> [50:100]
    b'50': REMOTE_UNKNOWN_00_Parser(),
    b'51': None,               # url_00 - Reserved
}

BACKEND_ACCESSOR_MAP = {
    # LOCALS -> [0:50]
    '00': HDF5_00_FileHandles,
    '01': NUMPY_00_FileHandles,
    '02': None,               # tiledb_00 - Reserved
    # '03': NUMPY_01_FileHandles,
    # REMOTES -> [50:100]
    '50': REMOTE_UNKNOWN_00_Handler,
    '51': None,               # url_00 - Reserved
}


def backend_decoder(db_val: bytes) -> namedtuple:
    '''Determine backend and decode specification for a raw hash record value.

    Parameters
    ----------
    db_val : bytes
        unmodified record specification bytes retrieved for a particular hash

    Returns
    -------
    namedtuple
        decoded specification with fields filled out uniquely for each backend.
        The only field common to all backends is located at index [0] with the
        field name `backend`.
    '''
    parser = BACKEND_PARSER_MAP[db_val[:2]]
    decoded = parser.decode(db_val)
    return decoded


def backend_from_heuristics(array: np.ndarray) -> str:
    '''Given a prototype array, attempt to select the appropriate backend.

    Parameters
    ----------
    array : np.ndarray
        prototype array to determine the appropriate backend for.

    Returns
    -------
    str
        Backend code specification for the selected backend.

    TODO
    ----
    Configuration of this entire module as the available backends fill out.
    '''

    # uncompressed numpy memmap data is most appropriate for data whose shape is
    # likely small tabular row data (CSV or such...)
    if (array.ndim == 1) and (array.size < 400):
        backend = '01'
    # hdf5 is the default backend for larger array sizes.
    else:
        backend = '00'

    return backend