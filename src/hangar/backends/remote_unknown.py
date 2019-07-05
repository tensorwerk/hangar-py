import logging
import os
from collections import namedtuple
import re

import numpy as np

from .. import constants as c

logger = logging.getLogger(__name__)


DataHashSpec = namedtuple('DataHashSpec', field_names=['backend'])


class REMOTE_UNKNOWN_00_Parser(object):

    __slots__ = ['FmtCode', 'RemoteHashSpec', 'SplitDecoderRE']

    def __init__(self):
        self.FmtCode = '50'
        self.SplitDecoderRE = re.compile(fr'[\{c.SEP_KEY}\{c.SEP_HSH}\{c.SEP_SLC}]')
        self.RemoteHashSpec = namedtuple('DataHashSpec', field_names=['backend', 'schema_hash'])

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
        raw_val = self.RemoteHashSpec(backend=self.FmtCode, schema_hash=schema_hash)
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
        logger.warning(f'delete_in_process_data for REMOTE_UNKNOWN_00_Handler called.')
        return

    def read_data(self, hashVal: namedtuple) -> None:
        raise FileNotFoundError(
            f'data sample with digest: {hashVal} does not exist on this machine. '
            f'Perform a `data-fetch` operation to retrieve it from the remote server.')

    def write_data(self, schema_hash: str = '', *args, **kwargs):
        # logger.warning(
        #     f'Cannot write data directly to {self.__class__.__name__} backend. '
        #     f'Function called with args: {args} and kwargs: {kwargs}.')
        return self.fmtParser.encode(schema_hash=schema_hash)