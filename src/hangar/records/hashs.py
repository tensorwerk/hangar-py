import lmdb
from typing import Iterable, List, Tuple

from .parsing import (
    arrayset_record_schema_raw_key_from_db_key,
    arrayset_record_schema_raw_val_from_db_val,
    hash_data_raw_key_from_db_key,
    hash_meta_raw_val_from_db_val,
    hash_schema_raw_key_from_db_key,
    RawArraysetSchemaVal,
)
from ..backends.selection import BACKEND_ACCESSOR_MAP, backend_decoder, _DataHashSpecs
from ..constants import K_HASH, K_SCHEMA
from ..txnctx import TxnRegister


class HashQuery(object):
    """Traverse and query contents contained in ``hashenv`` and ``labelenv`` dbs

    These methods operate on the databases which store the mapping of some data
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

    def _traverse_all_hash_records(self, keys: bool = True, values: bool = True):
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
            bytes of keys if ``keys=True`` and ``values=False``

            bytes of values if ``keys=False`` and ``values=True``

            Tuple of bytes corresponding to ``(keys, values)`` if
            ``keys=True`` and ``values=True``.
        """
        startHashRangeKey = f'{K_HASH}'.encode()
        len_RangeKey = len(startHashRangeKey)
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                hashsExist = cursor.set_range(startHashRangeKey)
                # divide loop into returned type sections as perf optimization
                # (rather then if/else checking on every iteration of loop)
                if keys and not values:
                    while hashsExist:
                        hashRecKey = cursor.key()
                        if hashRecKey[:len_RangeKey] == startHashRangeKey:
                            yield hashRecKey
                            hashsExist = cursor.next()
                            continue
                        else:
                            hashsExist = False
                elif values and not keys:  # pragma: no cover
                    while hashsExist:
                        hashRecKey, hashRecVal = cursor.item()
                        if hashRecKey[:len_RangeKey] == startHashRangeKey:
                            yield hashRecVal
                            hashsExist = cursor.next()
                            continue
                        else:
                            hashsExist = False
                elif keys and values:
                    while hashsExist:
                        hashRecKey, hashRecVal = cursor.item()
                        if hashRecKey[:len_RangeKey] == startHashRangeKey:
                            yield (hashRecKey, hashRecVal)
                            hashsExist = cursor.next()
                            continue
                        else:
                            hashsExist = False
                else:  # pragma: no cover
                    raise ValueError(f'Both keys and values argument cannot be False')
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def _traverse_all_schema_records(self, keys: bool = True, values: bool = True):
        """PUll out all binary encoded schema hash records.

        Parameters
        ----------
        keys : bool, optional
            if True, returns keys, by default True
        values : bool, optional
            if True, return values, by default True

        Yields
        -------
        Union[bytes, Tuple[bytes, bytes]]
            bytes of keys if ``keys=True`` and ``values=False``

            bytes of values if ``keys=False`` and ``values=True``

            Tuple of bytes corresponding to ``(keys, values)`` if
            ``keys=True`` and ``values=True``.
        """
        startSchemaRangeKey = f'{K_SCHEMA}'.encode()
        len_RangeKey = len(startSchemaRangeKey)
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                schemasExist = cursor.set_range(startSchemaRangeKey)
                # divide loop into returned type sections as perf optimization
                # (rather then if/else checking on every iteration of loop)
                if keys and not values:
                    while schemasExist:
                        schemaRecKey = cursor.key()
                        if schemaRecKey[:len_RangeKey] == startSchemaRangeKey:
                            yield schemaRecKey
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                elif values and not keys:  # pragma: no cover
                    while schemasExist:
                        schemaRecKey, schemaRecVal = cursor.item()
                        if schemaRecKey[:len_RangeKey] == startSchemaRangeKey:
                            yield schemaRecVal
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                elif keys and values:
                    while schemasExist:
                        schemaRecKey, schemaRecVal = cursor.item()
                        if schemaRecKey[:len_RangeKey] == startSchemaRangeKey:
                            yield (schemaRecKey, schemaRecVal)
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                else:  # pragma: no cover
                    raise ValueError(f'Both keys and values argument cannot be False')
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def list_all_hash_keys_raw(self) -> List[str]:
        recs = self._traverse_all_hash_records(keys=True, values=False)
        out = list(map(hash_data_raw_key_from_db_key, recs))
        return out

    def gen_all_hash_keys_db(self) -> Iterable[bytes]:
        recs = self._traverse_all_hash_records(keys=True, values=False)
        return recs

    def num_arrays(self) -> int:
        num_total = self._hashenv.stat()['entries']
        remaining = num_total - self.num_schemas()
        return remaining

    def list_all_schema_keys_raw(self) -> List[str]:
        recs = self._traverse_all_schema_records(keys=True, values=False)
        out = list(map(hash_schema_raw_key_from_db_key, recs))
        return out

    def gen_all_schema_keys_db(self) -> Iterable[bytes]:
        recs = self._traverse_all_schema_records(keys=True, values=False)
        return recs

    def num_schemas(self) -> int:
        recs = tuple(self._traverse_all_schema_records(keys=True, values=False))
        return len(recs)

    def num_meta(self) -> int:
        n = self._hashenv.stat()['entries']
        return n

    def gen_all_hash_keys_raw_array_vals_parsed(self) -> Iterable[Tuple[str, _DataHashSpecs]]:
        recs = self._traverse_all_hash_records(keys=True, values=True)
        for dbk, dbv in recs:
            rawk = hash_data_raw_key_from_db_key(dbk)
            rawv = backend_decoder(dbv)
            yield (rawk, rawv)

    def gen_all_hash_keys_raw_meta_vals_parsed(self) -> Iterable[Tuple[str, str]]:
        recs = self._traverse_all_hash_records(keys=True, values=True)
        for dbk, dbv in recs:
            rawk = hash_data_raw_key_from_db_key(dbk)
            rawv = hash_meta_raw_val_from_db_val(dbv)
            yield (rawk, rawv)

    def gen_all_schema_keys_raw_vals_parsed(self) -> Iterable[Tuple[str, RawArraysetSchemaVal]]:
        recs = self._traverse_all_schema_records(keys=True, values=True)
        for dbk, dbv in recs:
            rawk = arrayset_record_schema_raw_key_from_db_key(dbk)
            rawv = arrayset_record_schema_raw_val_from_db_val(dbv)
            yield (rawk, rawv)


def delete_in_process_data(repo_path, *, remote_operation=False):
    """DANGER! Permanently delete uncommitted data files/links for stage or remote area.

    This searches each backend accessors staged (or remote) folder structure for
    files, and if any are present the symlinks in stagedir and backing data
    files in datadir are removed.

    Parameters
    ----------
    repo_path : str
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
