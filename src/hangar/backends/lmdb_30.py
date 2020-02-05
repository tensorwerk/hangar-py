import os
from contextlib import suppress
from collections import ChainMap
from itertools import permutations
from functools import partial
from pathlib import Path
from typing import Optional
import string
import shutil

from xxhash import xxh64_hexdigest
import lmdb

from . import LMDB_30_DataHashSpec
from ..constants import DIR_DATA_REMOTE, DIR_DATA_STAGE, DIR_DATA_STORE, DIR_DATA
from ..utils import random_string
from ..op_state import reader_checkout_only, writer_checkout_only


LMDB_SETTINGS = {
    'map_size': 300_000_000,
    'meminit': False,
    'subdir': True,
    'lock': True,
    'max_spare_txns': 4,
}
_FmtCode = '30'


def lmdb_30_encode(uid, row_idx, checksum):
    res = f'{_FmtCode}:{uid}:{row_idx}:{checksum}'
    return res.encode()


def lexicographic_keys():
    lexicographic_ids = ''.join([
        string.digits,
        string.ascii_uppercase,
        string.ascii_lowercase,
    ])
    # permutations generates results in lexicographic order
    # total of 14_776_336 total ids can be generated with
    # a row_id consiting of 4 characters. This is more keys than
    # we will ever allow in a single LMDB database
    p = permutations(lexicographic_ids, 4)

    for perm in p:
        res = ''.join(perm)
        yield res


class LMDB_30_FileHandles:

    def __init__(self, repo_path: Path, *args, **kwargs):

        self.path: Path = repo_path

        self.rFp = {}
        self.wFp = {}
        self.Fp = ChainMap(self.rFp, self.wFp)

        self.mode: Optional[str] = None
        self.w_uid: Optional[str] = None
        self.row_idx: Optional[str] = None
        self._dflt_backend_opts: Optional[dict] = None

        self.STAGEDIR: Path = Path(self.path, DIR_DATA_STAGE, _FmtCode)
        self.REMOTEDIR: Path = Path(self.path, DIR_DATA_REMOTE, _FmtCode)
        self.STOREDIR: Path = Path(self.path, DIR_DATA_STORE, _FmtCode)
        self.DATADIR: Path = Path(self.path, DIR_DATA, _FmtCode)
        self.DATADIR.mkdir(exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return

    @reader_checkout_only
    def __getstate__(self) -> dict:
        """ensure multiprocess operations can pickle relevant data.
        """
        self.close()
        state = self.__dict__.copy()
        del state['rFp']
        del state['wFp']
        del state['Fp']
        return state

    def __setstate__(self, state: dict) -> None:  # pragma: no cover
        """ensure multiprocess operations can pickle relevant data.
        """
        self.__dict__.update(state)
        self.rFp = {}
        self.wFp = {}
        self.Fp = ChainMap(self.rFp, self.wFp)

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

    def open(self, mode: str, *, remote_operation: bool = False):
        """Open an lmdb file handle.

        Parameters
        ----------
        mode : str
            one of `r` or `a` for read only / read-write.
        remote_operation : optional, kwarg only, bool
            if this lmdb data is being created from a remote fetch operation, then
            we don't open any files for reading, and only open files for writing
            which exist in the remote data dir. (default is false, which means that
            write operations use the stage data dir and read operations use data store
            dir)
        """
        self.mode = mode
        if self.mode == 'a':
            process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
            process_dir.mkdir(exist_ok=True)
            for uidpth in process_dir.iterdir():
                if uidpth.suffix == '.lmdbdir':
                    file_pth = self.DATADIR.joinpath(uidpth.stem)
                    self.rFp[uidpth.stem] = partial(lmdb.open, str(file_pth), readonly=True,
                                                    **LMDB_SETTINGS)

        if not remote_operation:
            if not self.STOREDIR.is_dir():
                return
            for uidpth in self.STOREDIR.iterdir():
                if uidpth.suffix == '.lmdbdir':
                    file_pth = self.DATADIR.joinpath(uidpth.stem)
                    self.rFp[uidpth.stem] = partial(lmdb.open, str(file_pth), readonly=True,
                                                    **LMDB_SETTINGS)

    def close(self):
        """Close a file handle after writes have been completed

        behavior changes depending on write-enable or read-only file

        Returns
        -------
        bool
            True if success, otherwise False.
        """
        if self.mode == 'a':
            for uid in list(self.wFp.keys()):
                with suppress(AttributeError):
                    self.wFp[uid].close()
                del self.wFp[uid]
            self.w_uid = None
            self.row_idx = None

        for uid in list(self.rFp.keys()):
            with suppress(AttributeError):
                self.rFp[uid].close()
            del self.rFp[uid]

    @staticmethod
    def delete_in_process_data(repo_path: Path, *, remote_operation=False) -> None:
        """Removes some set of files entirely from the stage/remote directory.

        DANGER ZONE. This should essentially only be used to perform hard resets
        of the repository state.

        Parameters
        ----------
        repo_path : Path
            path to the repository on disk
        remote_operation : optional, kwarg only, bool
            If true, modify contents of the remote_dir, if false (default) modify
            contents of the staging directory.
        """
        data_dir = Path(repo_path, DIR_DATA, _FmtCode)
        PDIR = DIR_DATA_STAGE if not remote_operation else DIR_DATA_REMOTE
        process_dir = Path(repo_path, PDIR, _FmtCode)
        if not process_dir.is_dir():
            return

        for uidpth in process_dir.iterdir():
            if uidpth.suffix == '.lmdbdir':
                os.remove(process_dir.joinpath(uidpth.name))
                db_dir = data_dir.joinpath(uidpth.suffix)
                shutil.rmtree(str(db_dir))
        os.rmdir(process_dir)

    def _create_schema(self, *, remote_operation: bool = False):

        uid = random_string()
        db_dir_path = self.DATADIR.joinpath(f'{uid}')
        self.wFp[uid] = lmdb.open(str(db_dir_path), **LMDB_SETTINGS)

        self.w_uid = uid
        self.row_idx = lexicographic_keys()

        process_dir = self.REMOTEDIR if remote_operation else self.STAGEDIR
        Path(process_dir, f'{uid}.lmdbdir').touch()

    def read_data(self, hashVal: LMDB_30_DataHashSpec) -> str:
        """Read data from an hdf5 file handle at the specified locations

        Parameters
        ----------
        hashVal : LMDB_30_DataHashSpec
            record specification parsed from its serialized store val in lmdb.

        Returns
        -------
        str
            requested data.
        """
        try:
            with self.Fp[hashVal.uid].begin(write=False) as txn:
                res = txn.get(hashVal.row_idx.encode(), default=False)
                if res is False:
                    raise RuntimeError(hashVal)
        except AttributeError:
            self.Fp[hashVal.uid] = self.Fp[hashVal.uid]()
            return self.read_data(hashVal)
        except KeyError:
            process_dir = self.STAGEDIR if self.mode == 'a' else self.STOREDIR
            if Path(process_dir, f'{hashVal.uid}.lmdbdir').is_file():
                file_pth = self.DATADIR.joinpath(hashVal.uid)
                self.rFp[hashVal.uid] = lmdb.open(str(file_pth), readonly=True, **LMDB_SETTINGS)
                return self.read_data(hashVal)
            else:
                raise

        out = res.decode()
        if xxh64_hexdigest(res) != hashVal.checksum:
            raise RuntimeError(
                f'DATA CORRUPTION Checksum {xxh64_hexdigest(res)} != recorded {hashVal}')
        return out

    def write_data(self, data: str, *, remote_operation: bool = False) -> bytes:
        """verifies correctness of array data and performs write operation.

        Parameters
        ----------
        data: str
            data to write to group.
        remote_operation : optional, kwarg only, bool
            If this is a remote process which is adding data, any necessary
            hdf5 dataset files will be created in the remote data dir instead
            of the stage directory. (default is False, which is for a regular
            access process)

        Returns
        -------
        bytes
            string identifying the collection dataset and collection dim-0 index
            which the array can be accessed at.
        """
        encoded_data = data.encode()
        checksum = xxh64_hexdigest(encoded_data)

        if self.w_uid in self.wFp:
            try:
                row_idx = next(self.row_idx)
            except StopIteration:
                self._create_schema(remote_operation=remote_operation)
                return self.write_data(data, remote_operation=remote_operation)
        else:
            self._create_schema(remote_operation=remote_operation)
            return self.write_data(data, remote_operation=remote_operation)

        encoded_row_idx = row_idx.encode()

        try:
            with self.wFp[self.w_uid].begin(write=True) as txn:
                txn.put(encoded_row_idx, encoded_data, append=True)
        except lmdb.MapFullError:
            self._create_schema(remote_operation=remote_operation)
            return self.write_data(data, remote_operation=remote_operation)

        return lmdb_30_encode(self.w_uid, row_idx, checksum)
