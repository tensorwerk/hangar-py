"""Remote server location unknown backend, Identifier: ``REMOTE_50``

Backend Identifiers
===================

*  Backend: ``5``
*  Version: ``0``
*  Format Code: ``50``
*  Canonical Name: ``REMOTE_50``

Storage Method
==============

*  This backend merely acts to record that there is some data sample with some
   ``hash`` and ``schema_shape`` present in the repository. It does not store the
   actual data on the local disk, but indicates that if it should be retrieved,
   you need to ask the remote hangar server for it. Once present on the local
   disk, the backend locating info will be updated with one of the `local` data
   backend specifications.

Record Format
=============

Fields Recorded for Each Array
------------------------------

*  Format Code
*  Schema Hash

Separators used
---------------

* ``SEP_KEY: ":"``

Examples
--------

1)  Adding the first piece of data to a file:

    *  Schema Hash: "ae43A21a"

    ``Record Data => '50:ae43A21a'``

1)  Adding to a piece of data to a the middle of a file:

    *  Schema Hash: "ae43A21a"

    ``Record Data => '50:ae43A21a'``

Technical Notes
===============

*  The schema_hash field is required in order to allow effective placement of
   actual retrieved data into suitable sized collections on a ``fetch-data()``
   operation
"""
import os
import re
from typing import NamedTuple, Optional

import numpy as np

from .. import constants as c

# -------------------------------- Parser Implementation ----------------------

_FmtCode = '50'
# split up a formated parsed string into unique
_patern = fr'\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}'
_SplitDecoderRE = re.compile(fr'[{_patern}]')

REMOTE_50_DataHashSpec = NamedTuple('REMOTE_50_DataHashSpec',
                                    [('backend', str), ('schema_hash', str)])


def remote_50_encode(schema_hash: str = '') -> bytes:
    """returns an db value saying that this hash exists somewhere on a remote

    Returns
    -------
    bytes
        hash data db value
    """
    return f'{_FmtCode}{c.SEP_KEY}{schema_hash}'.encode()


def remote_50_decode(db_val: bytes) -> REMOTE_50_DataHashSpec:
    """converts a numpy data hash db val into a numpy data python spec

    Parameters
    ----------
    db_val : bytes
        data hash db val

    Returns
    -------
    REMOTE_50_DataHashSpec
        hash specification containing an identifies: `backend`, `schema_hash`
    """
    db_str = db_val.decode()
    _, schema_hash = _SplitDecoderRE.split(db_str)
    raw_val = REMOTE_50_DataHashSpec(backend=_FmtCode, schema_hash=schema_hash)
    return raw_val


# ------------------------- Accessor Object -----------------------------------


class REMOTE_50_Handler(object):

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.repo_path = repo_path
        self.schema_shape = schema_shape
        self.schema_dtype = schema_dtype
        self._dflt_backend_opts: Optional[dict] = None
        self.mode: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    @property
    def backend_opts(self):
        return self._dflt_backend_opts

    @backend_opts.setter
    def backend_opts(self, val):
        if self.mode == 'a':
            self._dflt_backend_opts = val
            return
        else:
            raise AttributeError(f"can't set property in read only mode")

    def open(self, mode, *args, **kwargs):
        self.mode = mode
        return

    def close(self, *args, **kwargs):
        return

    @staticmethod
    def delete_in_process_data(*args, **kwargs) -> None:
        """mockup of clearing staged directory for upstream calls.
        """
        return

    def read_data(self, hashVal: REMOTE_50_DataHashSpec) -> None:
        raise FileNotFoundError(
            f'data hash spec: {REMOTE_50_DataHashSpec} does not exist on this machine. '
            f'Perform a `data-fetch` operation to retrieve it from the remote server.')

    def write_data(self, schema_hash: str, *args, **kwargs) -> bytes:
        """Provide a formatted byte representation for storage as a remote reference

        Parameters
        ----------
        schema_hash : str
            schema hash which the referenced data sample should be accessed under

        Returns
        -------
        bytes
            formated raw values encoding lookup information
        """
        return remote_50_encode(schema_hash=schema_hash)
