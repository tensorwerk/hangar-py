import logging
import os
from collections import namedtuple

import numpy as np

logger = logging.getLogger(__name__)


class REMOTE_UNKNOWN_00_Parser(object):

    __slots__ = ['FmtCode', 'RemoteHashSpec']

    def __init__(self):

        self.FmtCode = '50'
        RemoteHashSpec = namedtuple('DataHashSpec', field_names=['backend'])
        self.RemoteHashSpec = RemoteHashSpec(backend=self.FmtCode)

    def encode(self) -> bytes:
        '''returns an db value saying that this hash exists somewhere on a remote

        Returns
        -------
        bytes
            hash data db value
        '''
        return self.FmtCode.encode()

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
        return self.RemoteHashSpec


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
        logger.warning(f'read called for {self.__class__.__name__} backend. None returning.')
        return None

    def write_data(self, *args, **kwargs):
        raise RuntimeError(
            f'write called for {self.__class__.__name__} backend. with '
            f'args: {args} and kwargs: {kwargs}. not allowed.')
        return self.fmtParser.encode()