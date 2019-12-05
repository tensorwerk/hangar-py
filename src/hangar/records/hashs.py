import lmdb

from . import parsing
from .. import constants as c
from ..context import TxnRegister
from ..backends.selection import BACKEND_ACCESSOR_MAP, backend_decoder


class HashQuery(object):

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
        startHashRangeKey = f'{c.K_HASH}'.encode()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                hashsExist = cursor.set_range(startHashRangeKey)
                # divide loop into returned type sections as perf optimization
                # (rather then if/else checking on every iteration of loop)
                if keys and not values:
                    while hashsExist:
                        hashRecKey, hashRecVal = cursor.item()
                        if hashRecKey.startswith(startHashRangeKey):
                            yield hashRecKey
                            hashsExist = cursor.next()
                            continue
                        else:
                            hashsExist = False
                elif values and not keys:  # pragma: no cover
                    while hashsExist:
                        hashRecKey, hashRecVal = cursor.item()
                        if hashRecKey.startswith(startHashRangeKey):
                            yield hashRecVal
                            hashsExist = cursor.next()
                            continue
                        else:
                            hashsExist = False
                elif keys and values:
                    while hashsExist:
                        hashRecKey, hashRecVal = cursor.item()
                        if hashRecKey.startswith(startHashRangeKey):
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
        startSchemaRangeKey = f'{c.K_SCHEMA}'.encode()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                schemasExist = cursor.set_range(startSchemaRangeKey)
                # divide loop into returned type sections as perf optimization
                # (rather then if/else checking on every iteration of loop)
                if keys and not values:
                    while schemasExist:
                        schemaRecKey, schemaRecVal = cursor.item()
                        if schemaRecKey.startswith(startSchemaRangeKey):
                            yield schemaRecKey
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                elif values and not keys:  # pragma: no cover
                    while schemasExist:
                        schemaRecKey, schemaRecVal = cursor.item()
                        if schemaRecKey.startswith(startSchemaRangeKey):
                            yield schemaRecVal
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                elif keys and values:
                    while schemasExist:
                        schemaRecKey, schemaRecVal = cursor.item()
                        if schemaRecKey.startswith(startSchemaRangeKey):
                            yield (schemaRecKey, schemaRecVal)
                            schemasExist = cursor.next()
                            continue
                        else:
                            schemasExist = False
                else:  # pragma: no cover
                    raise ValueError(f'Both keys and values argument cannot be False')
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def list_all_hash_keys_raw(self):
        recs = self._traverse_all_hash_records(keys=True, values=False)
        out = list(map(parsing.hash_data_raw_key_from_db_key, recs))
        return out

    def gen_all_hash_keys_db(self):
        recs = self._traverse_all_hash_records(keys=True, values=False)
        return recs

    def num_arrays(self):
        num_total = self._hashenv.stat()['entries']
        remaining = num_total - self.num_schemas()
        return remaining

    def list_all_schema_keys_raw(self):
        recs = self._traverse_all_schema_records(keys=True, values=False)
        out = list(map(parsing.hash_schema_raw_key_from_db_key, recs))
        return out

    def gen_all_schema_keys_db(self):
        recs = self._traverse_all_schema_records(keys=True, values=False)
        return recs

    def num_schemas(self):
        recs = tuple(self._traverse_all_schema_records(keys=True, values=False))
        return len(recs)

    def num_meta(self):
        n = self._hashenv.stat()['entries']
        return n

    def gen_all_hash_keys_raw_array_vals_parsed(self):
        recs = self._traverse_all_hash_records(keys=True, values=True)
        func_raw_key = parsing.hash_data_raw_key_from_db_key
        func_parsed_val = backend_decoder
        for dbk, dbv in recs:
            rawk = func_raw_key(dbk)
            rawv = func_parsed_val(dbv)
            yield (rawk, rawv)

    def gen_all_hash_keys_raw_meta_vals_parsed(self):
        recs = self._traverse_all_hash_records(keys=True, values=True)
        func_raw_key = parsing.hash_data_raw_key_from_db_key
        func_parsed_val = parsing.hash_meta_raw_val_from_db_val
        for dbk, dbv in recs:
            rawk = func_raw_key(dbk)
            rawv = func_parsed_val(dbv)
            yield (rawk, rawv)

    def gen_all_schema_keys_raw_vals_parsed(self):
        recs = self._traverse_all_schema_records(keys=True, values=True)
        func_raw_key = parsing.arrayset_record_schema_raw_key_from_db_key
        func_parsed_val = parsing.arrayset_record_schema_raw_val_from_db_val
        for dbk, dbv in recs:
            rawk = func_raw_key(dbk)
            rawv = func_parsed_val(dbv)
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
