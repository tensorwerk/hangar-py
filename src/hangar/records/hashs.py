import logging

from . import parsing
from .. import config
from ..context import TxnRegister
from ..backends.selection import backend_decoder, BACKEND_ACCESSOR_MAP

logger = logging.getLogger(__name__)


class HashQuery(object):

    def __init__(self, hashenv):
        self._hashenv = hashenv

    # ------------------ traversing the unpacked records ----------------------

    def _traverse_all_records(self, keys=True, vals=True):
        '''Pull out all records in the database as a tuple of binary encoded

        Returns
        -------
        list of tuples of bytes
            list type stack of tuples with each db_key, db_val pair
        '''
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                cursor.first()
                for db_kv in cursor.iternext(keys=keys, values=vals):
                    yield db_kv
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def _traverse_all_hash_records(self, keys=True, vals=True):
        '''Pull out all records in the database as a tuple of binary encoded

        Returns
        -------
        list of tuples of bytes
            list type stack of tuples with each db_key, db_val pair
        '''
        HASH = config.get('hangar.keys.hash')
        startHashRangeKey = f'{HASH}'.encode()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                hashsExist = cursor.set_range(startHashRangeKey)
                while hashsExist:
                    hashRecKey, hashRecVal = cursor.item()
                    if hashRecKey.startswith(startHashRangeKey):
                        if keys and vals:
                            yield (hashRecKey, hashRecVal)
                        elif keys:
                            yield hashRecKey
                        else:
                            yield hashRecVal
                        hashsExist = cursor.next()
                        continue
                    else:
                        hashsExist = False
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def _traverse_all_schema_records(self, keys=True, vals=True):
        '''Pull out all records in the database as a tuple of binary encoded

        Returns
        -------
        list of tuples of bytes
            list type stack of tuples with each db_key, db_val pair
        '''
        SCHEMA = config.get('hangar.keys.schema')
        startSchemaRangeKey = f'{SCHEMA}'.encode()
        try:
            hashtxn = TxnRegister().begin_reader_txn(self._hashenv)
            with hashtxn.cursor() as cursor:
                schemasExist = cursor.set_range(startSchemaRangeKey)
                while schemasExist:
                    schemaRecKey, schemaRecVal = cursor.item()
                    if schemaRecKey.startswith(startSchemaRangeKey):
                        if keys and vals:
                            yield (schemaRecKey, schemaRecVal)
                        elif keys:
                            yield schemaRecKey
                        else:
                            yield schemaRecVal
                        schemasExist = cursor.next()
                        continue
                    else:
                        schemasExist = False
        finally:
            TxnRegister().abort_reader_txn(self._hashenv)

    def list_all_hash_keys_raw(self):
        recs = self._traverse_all_hash_records(keys=True, vals=False)
        out = list(map(parsing.hash_data_raw_key_from_db_key, recs))
        return out

    def list_all_hash_keys_db(self):
        recs = self._traverse_all_hash_records(keys=True, vals=False)
        return recs

    def list_all_hash_values(self):
        recs = self._traverse_all_hash_records(keys=False, vals=True)
        formatted = map(backend_decoder, recs)
        return formatted

    def list_all_schema_keys_raw(self):
        recs = self._traverse_all_schema_records(keys=True, vals=False)
        out = list(map(parsing.hash_schema_raw_key_from_db_key, recs))
        return out

    def list_all_schema_keys_db(self):
        recs = self._traverse_all_schema_records(keys=True, vals=False)
        return recs


def remove_unused(repo_path, stagehashenv):
    '''If no changes made to staged files, remove and unlik them from stagedir

    This searchs the stagehashenv file for all schemas & instances, and if any
    files are present in the stagedir without references in stagehashenv, the
    symlinks in stagedir and backing data files in datadir are removed.

    Parameters
    ----------
    repo_path : str
        path to the repository on disk
    stagehashenv : `lmdb.Environment`
        db where all stage hash additions are recorded

    '''
    for backend, accesor in BACKEND_ACCESSOR_MAP.items():
        if accesor is not None:
            acc = accesor(repo_path=repo_path)
            acc.remove_unused(repo_path=repo_path, stagehashenv=stagehashenv)


def clear_stage_hash_records(stagehashenv):
    '''Drop all records in the stagehashenv db

    This operation should be performed anytime a reset of the staging area is
    performed (including for commits, merges, and checkouts)

    Parameters
    ----------
    stagehashenv : lmdb.Environment
        db where staged data hash additions are recorded
    '''
    print(f'removing all stage hash records')
    stagehashtxn = TxnRegister().begin_writer_txn(stagehashenv)
    with stagehashtxn.cursor() as cursor:
        positionExists = cursor.first()
        while positionExists:
            positionExists = cursor.delete()
    cursor.close()
    TxnRegister().commit_writer_txn(stagehashenv)


def remove_stage_hash_records_from_hashenv(hashenv, stagehashenv):
    '''Remove references to data additions during a hard reset

    For every hash record in stagehashenv, remove the corresponding k/v pair
    from the hashenv db. This is a dangerous operation if the stagehashenv was
    not appropriatly constructed!!!

    Parameters
    ----------
    hashenv : lmdb.Environment
        db where all the permanant hash records are stored
    stagehashenv : lmdb.Environment
        db where all the staged hash records to be removed are stored.

    '''
    stageHashKeys = HashQuery(stagehashenv).list_all_hash_keys_db()

    hashtxn = TxnRegister().begin_writer_txn(hashenv)
    for hashKey in stageHashKeys:
        logger.info(f'deleting: {hashKey}')
        hashtxn.delete(hashKey)
    TxnRegister().commit_writer_txn(hashenv)
