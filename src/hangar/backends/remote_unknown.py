'''Remote server location unknown backend, Identifier: ``REMOTE_UNKNOWN_00``

Backend Identifiers
===================

*  Format Code: ``50``
*  Canonical Name: ``REMOTE_UNKNOWN_00``

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

* ``SEP_KEY``

Examples
--------

Note: all examples use ``SEP_KEY: ":"``

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
'''
import logging
import os
from collections import namedtuple
import re

import numpy as np

from .. import constants as c

logger = logging.getLogger(__name__)


DataHashSpec = namedtuple('DataHashSpec', field_names=['backend', 'schema_hash'])


class REMOTE_UNKNOWN_00_Parser(object):

    __slots__ = ['FmtCode', 'SplitDecoderRE']

    def __init__(self):
        self.FmtCode = '50'
        self.SplitDecoderRE = re.compile(fr'[\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}]')

    def encode(self, schema_hash: str = '') -> bytes:
        '''returns an db value saying that this hash exists somewhere on a remote

        Returns
        -------
        bytes
            hash data db value
        '''
        return f'{self.FmtCode}{c.SEP_KEY}{schema_hash}'.encode()

    def decode(self, db_val: bytes) -> namedtuple:
        '''converts a numpy data hash db val into a numpy data python spec

        Parameters
        ----------
        db_val : bytes
            data hash db val

        Returns
        -------
        namedtuple
            hash specification containing an identifies: `backend`
        '''
        db_str = db_val.decode()
        _, schema_hash = self.SplitDecoderRE.split(db_str)
        raw_val = DataHashSpec(backend=self.FmtCode, schema_hash=schema_hash)
        return raw_val


class REMOTE_UNKNOWN_00_Handler(object):

    def __init__(self, repo_path: os.PathLike, schema_shape: tuple, schema_dtype: np.dtype):
        self.repo_path = repo_path
        self.schema_shape = schema_shape
        self.schema_dtype = schema_dtype
        self.fmtParser = REMOTE_UNKNOWN_00_Parser()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    def open(self, *args, **kwargs):
        return

    def close(self, *args, **kwargs):
        return

    @staticmethod
    def delete_in_process_data(*args, **kwargs):
        '''mockup of clearing staged directory for upstream calls.
        '''
        logger.debug(f'delete_in_process_data for REMOTE_UNKNOWN_00_Handler called.')
        return

    def read_data(self, hashVal: namedtuple) -> None:
        raise FileNotFoundError(
            f'data sample with digest: {hashVal} does not exist on this machine. '
            f'Perform a `data-fetch` operation to retrieve it from the remote server.')

    def write_data(self, schema_hash: str = '', *args, **kwargs):
        return self.fmtParser.encode(schema_hash=schema_hash)