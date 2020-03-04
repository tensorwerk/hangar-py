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
from pathlib import Path
from typing import Optional

from .specs import REMOTE_50_DataHashSpec
from ..op_state import writer_checkout_only, reader_checkout_only
from ..typesystem import EmptyDict, checkedmeta


# -------------------------------- Parser Implementation ----------------------

_FmtCode = '50'


def remote_50_encode(schema_hash: str = '') -> bytes:
    """returns an db value saying that this hash exists somewhere on a remote

    Returns
    -------
    bytes
        hash data db value
    """
    return f'50:{schema_hash}'.encode()


# ------------------------- Accessor Object -----------------------------------


class REMOTE_50_Options(metaclass=checkedmeta):
    _backend_options = EmptyDict()

    def __init__(self, backend_options, *args, **kwargs):
        if backend_options is None:
            backend_options = self.default_options
        self._backend_options = backend_options

    @property
    def default_options(self):
        return {}

    @property
    def backend_options(self):
        return self._backend_options

    @property
    def init_requires(self):
        return ('repo_path',)


class REMOTE_50_Handler(object):

    def __init__(self, repo_path: Path, *args, **kwargs):
        self.repo_path = repo_path
        self._dflt_backend_opts: Optional[dict] = None
        self._mode: Optional[str] = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    @reader_checkout_only
    def __getstate__(self) -> dict:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.close()
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)
        self.open(mode=self._mode)

    @property
    def backend_opts(self):
        return self._dflt_backend_opts

    @writer_checkout_only
    def _backend_opts_set(self, val):
        """Nonstandard descriptor method. See notes in ``backend_opts.setter``.
        """
        self._dflt_backend_opts = val
        return

    @backend_opts.setter
    def backend_opts(self, value):
        """
        Using seperate setter method (with ``@writer_checkout_only`` decorator
        applied) due to bug in python <3.8.

        From: https://bugs.python.org/issue19072
            > The classmethod decorator when applied to a function of a class,
            > does not honour the descriptor binding protocol for whatever it
            > wraps. This means it will fail when applied around a function which
            > has a decorator already applied to it and where that decorator
            > expects that the descriptor binding protocol is executed in order
            > to properly bind the function to the class.
        """
        return self._backend_opts_set(value)

    def open(self, mode, *args, **kwargs):
        self._mode = mode
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
