"""Definition and dynamic routing to Hangar backend implementations.

This module defines the available backends for a Hangar installation & provides
dynamic routing of method calls to the appropriate backend from a stored record
specification.

Identification
--------------

A two character ascii code identifies which backend/version some record belongs
to. Valid characters are the union of ``ascii_lowercase``, ``ascii_uppercase``,
and ``ascii_digits``:

.. centered:: ``abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789``

Though stored as bytes in the backend, we use human readable characters (and not
unprintable bytes) to aid in human tasks like developer database dumps and
debugging. The characters making up the two digit code have the following
symantic meanings:

   *  First Character (element 0) indicates the ``backend type`` used.

   *  Second character (element 1) indicates the ``version`` of the backend type
      which should be used to parse the specification & accesss data (more on
      this later)

The number of codes possible (a 2-choice permutation with repetition) is: 3844
which we anticipate to be more then sufficient long into the future. As a
convention, the range of values in which the first digit of the code falls into
can be used to identify the storage medium location:

   *  Lowercase ``ascii_letters`` & digits ``[0, 1, 2, 3, 4]`` -> reserved for
      backends handling data on the local disk.

   *  Uppercase ``ascii_letters`` & digits ``[5, 6, 7, 8, 9]`` -> reserved for
      backends referring to data residing on a remote server.

This is not a hard and fast rule though, and can be changed in the future if the
need arises.

Process & Guarantees
--------------------

In order to maintain backwards compatibility across versions of Hangar into the
future the following ruleset is specified and MUST BE HONORED:

*  When a new backend is proposed, the contributor(s) provide the class with a
   meaningful name (``HDF5``, ``NUMPY``, ``TILEDB``, etc) identifying the
   backend to Hangar developers. The review team will provide:

   -  ``backend type`` code
   -  ``version`` code

   which all records related to that implementation identify themselves with. In
   addition, Externally facing classes / methods go by a canonical name which is
   the concatenation of the ``meaningful name`` and assigned ``"format code"``
   ie. for ``backend name: 'NUMPY'`` assigned ``type code: '1'`` and ``version
   code: '0'`` must start external method/class names with: ``NUMPY_10_foo``

*  Once a new backend is accepted, the code assigned to it is PERMANENT &
   UNCHANGING. The same code cannot be used in the future for other backends.

*  Each backend independently determines the information it needs to log/store
   to uniquely identify and retrieve a sample stored by it. There is no standard
   format, each is free to define whatever fields they find most convenient.
   Unique encode/decode methods are defined in order to serialize this
   information to bytes and then reconstruct the information later. These bytes
   are what are passed in when a retrieval request is made, and returned when a
   storage request for some piece of data is performed.

*  Once accepted, The record format specified (ie. the byte representation
   described above) cannot be modified in any way. This must remain permanent!

*  Backend (internal) methods can be updated, optimized, and/or changed at any
   time so long as:

   *  No changes to the record format specification are introduced

   *  Data stored via any previous iteration of the backend's accessor methods
      can be retrieved bitwise exactly by the "updated" version.

Before proposing a new backend or making changes to this file, please consider
reaching out to the Hangar core development team so we can guide you through the
process.
"""
import string
from typing import Dict

from .specs import (
    HDF5_00_DataHashSpec,
    HDF5_01_DataHashSpec,
    NUMPY_10_DataHashSpec,
    LMDB_30_DataHashSpec,
    REMOTE_50_DataHashSpec,
)
from .specparse import backend_decoder

from .hdf5_00 import HDF5_00_FileHandles, HDF5_00_Options
from .hdf5_01 import HDF5_01_FileHandles, HDF5_01_Options
from .lmdb_30 import LMDB_30_FileHandles, LMDB_30_Options
from .numpy_10 import NUMPY_10_FileHandles, NUMPY_10_Options
from .remote_50 import REMOTE_50_Handler, REMOTE_50_Options


BACKEND_ACCESSOR_MAP = {
    # LOCALS -> [00:50] + ['aa':'zz']
    '00': HDF5_00_FileHandles,
    '01': HDF5_01_FileHandles,
    '10': NUMPY_10_FileHandles,
    '30': LMDB_30_FileHandles,
    # REMOTES -> [50:99] + ['AA':'ZZ']
    '50': REMOTE_50_Handler,
}

BACKEND_OPTIONS_MAP = {
    '00': HDF5_00_Options,
    '01': HDF5_01_Options,
    '10': NUMPY_10_Options,
    '30': LMDB_30_Options,
    '50': REMOTE_50_Options,
}

_local_prefixes = string.digits[0:5] + string.ascii_lowercase

BACKEND_IS_LOCAL_MAP: Dict[str, bool] = {
    k: bool(k[0] in _local_prefixes) for k in BACKEND_ACCESSOR_MAP.keys()
}

__all__ = [
    'backend_decoder', 'HDF5_00_DataHashSpec', 'HDF5_01_DataHashSpec',
    'NUMPY_10_DataHashSpec', 'LMDB_30_DataHashSpec', 'REMOTE_50_DataHashSpec',
    'BACKEND_OPTIONS_MAP', 'BACKEND_ACCESSOR_MAP', 'BACKEND_IS_LOCAL_MAP',
]
