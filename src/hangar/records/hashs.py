from pathlib import Path
from typing import Iterable, List, Tuple, Union

import lmdb

from .column_parsers import (
    hash_record_count_start_range_key,
    hash_schema_raw_key_from_db_key,
    hash_data_raw_key_from_db_key,
    schema_hash_db_key_from_digest,
    schema_spec_from_db_val,
    schema_record_count_start_range_key
)
from ..backends import BACKEND_ACCESSOR_MAP, backend_decoder
from ..txnctx import TxnRegister
from ..mixins import CursorRangeIterator
from ..utils import ilen


class HashQuery(CursorRangeIterator):
    """Traverse and query contents contained in ``hashenv`` db

    These methods operate on the database which store the mapping of some data
    digest to it's location on disk (or value in the case of metadata and
    schemas). These databases are not specific to a particular commit; the
    records are for every piece of data stored in every commit across history.

    There are relatively few procedures which require traversal and mapping
    across data records in this manner. The two most notable use cases are:

        1. Remote client-server negotiation operations
        2. Verifying the integrity of a repositories historical provenance, commit
        contents, and data stored on disk.
    """

    def __init__(self, hashenv: lmdb.Environment):
        self._hashenv = hashenv

    # ------------------ traversing the unpacked records ----------------------

    def _traverse_all_hash_records(self, keys: bool = True, values: bool = True
                                   ) -> Iterable[Union[bytes, Tuple[bytes, bytes]]]:
        """PUll out all binary encoded data hash records.

        Parameters
        ----------
        keys : bool, optional
            if True, returns keys, by default True
        values : bool, optional
            if True, return values, by default True

        Yields
        -------
        Union[bytes, Tuple[bytes, bytes]]
            Iterable of schema record keys, values, or items tuple
        """
        startHashRangeKey = hash_record_count_start_range_key()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            yield from self.cursor_range_iterator(hashtxn, startHashRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def _traverse_all_schema_records(self, keys: bool = True, values: bool = True
                                     ) -> Iterable[Union[bytes, Tuple[bytes, bytes]]]:
        """Pull out all binary encoded schema hash records.

        Parameters
        ----------
        keys : bool, optional
            if True, returns keys, by default True
        values : bool, optional
            if True, return values, by default True

        Yields
        -------
        Union[bytes, Tuple[bytes, bytes]]
            Iterable of schema record keys, values, or items tuple
        """
        startSchemaRangeKey = schema_record_count_start_range_key()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            yield from self.cursor_range_iterator(hashtxn, startSchemaRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def list_all_hash_keys_raw(self) -> List[str]:
        recs = self._traverse_all_hash_records(keys=True, values=False)
        return list(map(hash_data_raw_key_from_db_key, recs))

    def gen_all_hash_keys_db(self) -> Iterable[bytes]:
        return self._traverse_all_hash_records(keys=True, values=False)

    def list_all_schema_digests(self) -> List[str]:
        recs = self._traverse_all_schema_records(keys=True, values=False)
        return list(map(hash_schema_raw_key_from_db_key, recs))

    def gen_all_schema_keys_db(self) -> Iterable[bytes]:
        return self._traverse_all_schema_records(keys=True, values=False)

    def num_data_records(self) -> int:
        """Total count of all data digests / backends specs stored over all repo history.
        """
        num_total = self._hashenv.stat()['entries']
        remaining = num_total - self.num_schema_records()
        return remaining

    def num_schema_records(self) -> int:
        """Total count of schema digests / spec defs stored over all repo history.
        """
        return ilen(self._traverse_all_schema_records(keys=True, values=False))

    def gen_all_data_digests_and_parsed_backend_specs(self):
        for dbk, dbv in self._traverse_all_hash_records(keys=True, values=True):
            rawk = hash_data_raw_key_from_db_key(dbk)
            rawv = backend_decoder(dbv)
            yield (rawk, rawv)

    def gen_all_schema_digests_and_parsed_specs(self) -> Iterable[Tuple[str, dict]]:
        for dbk, dbv in self._traverse_all_schema_records(keys=True, values=True):
            rawk = hash_schema_raw_key_from_db_key(dbk)
            rawv = schema_spec_from_db_val(dbv)
            yield (rawk, rawv)

    def get_schema_digest_spec(self, digest) -> dict:
        schemaHashKey = schema_hash_db_key_from_digest(digest)
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            schemaSpecVal = hashtxn.get(schemaHashKey)
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

        schema_spec = schema_spec_from_db_val(schemaSpecVal)
        return schema_spec


def backends_remove_in_process_data(repo_path: Path, *, remote_operation: bool = False):
    """DANGER! Permanently delete uncommitted data files/links for stage or remote area.

    This searches each backend accessors staged (or remote) folder structure for
    files, and if any are present the symlinks in stagedir and backing data
    files in datadir are removed.

    Parameters
    ----------
    repo_path : Path
        path to the repository on disk
    remote_operation : optional, kwarg only, bool
        If true, modify contents of the remote_dir, if false (default) modify
        contents of the staging directory.
    """
    for backend, accesor in BACKEND_ACCESSOR_MAP.items():
        if accesor is not None:
            accesor.delete_in_process_data(repo_path=repo_path,
                                           remote_operation=remote_operation)


def clear_stage_hash_records(stagehashenv):
    """Drop all records in the stagehashenv db

    This operation should be performed anytime a reset of the staging area is
    performed (including for commits, merges, and checkouts)

    Parameters
    ----------
    stagehashenv : lmdb.Environment
        db where staged data hash additions are recorded
    """
    stagehashtxn = TxnRegister().begin_writer_txn(stagehashenv)
    with stagehashtxn.cursor() as cursor:
        positionExists = cursor.first()
        while positionExists:
            positionExists = cursor.delete()
    cursor.close()
    TxnRegister().commit_writer_txn(stagehashenv)


def remove_stage_hash_records_from_hashenv(hashenv, stagehashenv):
    """Remove references to data additions during a hard reset

    For every hash record in stagehashenv, remove the corresponding k/v pair
    from the hashenv db. This is a dangerous operation if the stagehashenv was
    not appropriately constructed!!!

    Parameters
    ----------
    hashenv : lmdb.Environment
        db where all the permanent hash records are stored
    stagehashenv : lmdb.Environment
        db where all the staged hash records to be removed are stored.
    """
    stageHashKeys = HashQuery(stagehashenv).gen_all_hash_keys_db()
    hashtxn = TxnRegister().begin_writer_txn(hashenv)
    for hashKey in stageHashKeys:
        hashtxn.delete(hashKey)
    TxnRegister().commit_writer_txn(hashenv)
