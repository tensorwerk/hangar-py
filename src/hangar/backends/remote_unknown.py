import logging
import os
import warnings
from collections import ChainMap, namedtuple
from functools import partial
from os.path import join as pjoin
from os.path import splitext as psplitext

import numpy as np

from .. import constants as c

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

        # self.STAGEDIR = pjoin(self.repo_path, c.DIR_DATA_STAGE, self.fmtParser.FmtCode)
        # self.REMOTEDIR = pjoin(self.repo_path, c.DIR_DATA_REMOTE, self.fmtParser.FmtCode)
        # self.DATADIR = pjoin(self.repo_path, c.DIR_DATA, self.fmtParser.FmtCode)
        # self.STOREDIR = pjoin(self.repo_path, c.DIR_DATA_STORE, self.fmtParser.FmtCode)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    def open(self, *args, **kwargs):
        return

    def close(self, *args, **kwargs):
        return

    @staticmethod
    def remove_unstored_changes(*args, **kwargs):
        '''mockup of clearing staged directory for upstream calls.
        '''
        logger.warning(f'remove_unstored_changes method of REMOTE_UNKNOWN_00_Handler called')
        return

    def read_data(self, hashVal: namedtuple) -> None:
        logger.warning(f'read called for {self.__class__.__name__} backend. None returning.')
        return None

    def write_data(self, *args, **kwargs):
        e = RuntimeError(
            f'write called for {self.__class__.__name__} backend. with '
            f'args: {args} and kwargs: {kwargs}. not allowed.')
        return self.fmtParser.encode()