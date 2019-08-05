from typing import Tuple, List, Iterator, Iterable, Set, Dict

import lmdb

from .. import constants as c
from . import parsing
from .parsing import RawDataRecordKey, RawDataRecordVal
from .parsing import MetadataRecordKey, MetadataRecordVal
from .parsing import RawCellstoreSchemaVal
from ..context import TxnRegister

RawDataTuple = Tuple[RawDataRecordKey, RawDataRecordVal]
RawMetaTuple = Tuple[MetadataRecordKey, MetadataRecordVal]

'''
Data record queries
-------------------
'''


class RecordQuery(object):

    def __init__(self, dataenv: lmdb.Environment):
        self._dataenv = dataenv

# ------------------ traversing the unpacked records ------------------------------------

    def _traverse_all_records(self) -> Iterator[Tuple[bytes, bytes]]:
        '''Pull out all records in the database as a tuple of binary encoded

        Returns
        -------
        list of tuples of bytes
            list type stack of tuples with each db_key, db_val pair
        '''
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                for db_kv in cursor.iternext(keys=True, values=True):
                    yield db_kv
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

    def _traverse_metadata_records(self) -> Dict[bytes, bytes]:
        '''Internal method to traverse all metadata records and pull out keys/db_values

        Returns
        -------
        Dict[bytes, bytes]
            dictionary of metadata db keys and db_values
        '''
        metadataRecords = {}
        metadataCountKey = parsing.metadata_count_db_key()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                metadataExists = cursor.set_key(metadataCountKey)
                if metadataExists is True:
                    numMetadata = parsing.metadata_count_raw_val_from_db_val(cursor.value())
                    for i in range(numMetadata):
                        cursor.next()
                        metaRecKey, metaRecValue = cursor.item()
                        metadataRecords[metaRecKey] = metaRecValue
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

        return metadataRecords

    def _traverse_cellstore_schema_records(self) -> Dict[bytes, bytes]:
        '''Internal method to traverse all schema records and pull out k/v db pairs.

        Returns
        -------
        Dict[bytes, bytes]
            dictionary of db schema keys and db_values
        '''
        schemaRecords = {}
        startSchemaRangeKey = f'{c.K_SCHEMA}'.encode()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                schemas_exist = cursor.set_range(startSchemaRangeKey)
                while schemas_exist:
                    schemaRecKey, schemaRecVal = cursor.item()
                    if schemaRecKey.startswith(startSchemaRangeKey):
                        schemaRecords[schemaRecKey] = schemaRecVal
                        schemas_exist = cursor.next()
                        continue
                    else:
                        schemas_exist = False
        finally:
            datatxn = TxnRegister().abort_reader_txn(self._dataenv)

        return schemaRecords

    def _traverse_cellstore_data_records(self, cellstore_name) -> Dict[bytes, bytes]:
        '''Internal method to traverse cellstore data records and get keys/db_values

        The cellstore name is required because this method controls the cursor
        movement by first setting it's position on the cellstore record count
        key, reading it's value "N" and then sequentially pulling records out of
        the db for N loops.

        Parameters
        ----------
        cellstore_name : str
            name of the cellstore to traverse records for.

        Returns
        -------
        Dict[bytes, bytes]
            dict of db_key/db_values for each record traversed
        '''
        data_records = {}
        startDsetRecCountRngK = parsing.cellstore_record_count_db_key_from_raw_key(cellstore_name)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                dataRecordsExist = cursor.set_range(startDsetRecCountRngK)
                dataRecKeySubString = f'{startDsetRecCountRngK.decode()}{c.SEP_KEY}'.encode()
                cursor.next()
                while dataRecordsExist:
                    dataRecKey, dataRecVal = cursor.item()
                    if dataRecKey.startswith(dataRecKeySubString):
                        data_records[dataRecKey] = dataRecVal
                        dataRecordsExist = cursor.next()
                        continue
                    else:
                        dataRecordsExist = False

        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

        return data_records

# ------------------------- process cellstores --------------------------------------------

    def cellstore_names(self) -> List[str]:
        '''Find all named cellstores in the checkout

        Returns
        -------
        List[str]
            list of all cellstore names
        '''
        recs = self._traverse_cellstore_schema_records()
        cellstore_names = map(parsing.cellstore_record_schema_raw_key_from_db_key, recs.keys())
        return list(cellstore_names)

    def cellstore_data_records(self, cellstore_name: str) -> Iterable[RawDataTuple]:
        '''Returns the raw data record key and record values for a specific cellstore.

        Parameters
        ----------
        cellstore_name : str
            name of the cellstore to pull records for

        Yields
        ------
        tuple
            generator of key and value data record specs
        '''
        recs = self._traverse_cellstore_data_records(cellstore_name)
        if len(recs) > 0:
            data_rec_keys = map(parsing.data_record_raw_key_from_db_key, recs.keys())
            data_rec_vals = map(parsing.data_record_raw_val_from_db_val, recs.values())
            recs = zip(data_rec_keys, data_rec_vals)
        return recs

    def cellstore_data_names(self, cellstore_name):
        '''Find all data names contained within a cellstore.

        If you need both names, and hash values, call the `cellstore_data_records`
        function. The underlying cost of searching the db is identical, this just provides
        a pretty filter on top.

        Parameters
        ----------
        cellstore_name : str
            name of the cellstore to retrieve names for

        Returns
        -------
        list of str
            list of data names contained in the cellstore
        '''
        recs = self._traverse_cellstore_data_records(cellstore_name)
        data_key_rec = map(parsing.data_record_raw_key_from_db_key, recs.keys())
        data_names = list(map(lambda x: x.data_name, data_key_rec))
        return data_names

    def cellstore_data_hashes(self, cellstore_name: str) -> Set[RawDataRecordVal]:
        '''Find all data hashes contained within a particular cellstore

        Note: this method does not remove any duplicates which may be present,
        if dedup is required, process it downstream

        Parameters
        ----------
        cellstore_name : str
            name of the cellstore to find the hashes contained in

        Returns
        -------
        list
            all hash values for all data pieces in the cellstore
        '''
        recs = self._traverse_cellstore_data_records(cellstore_name)
        data_val_rec = map(parsing.data_record_raw_val_from_db_val, recs.values())
        all_hashes = set(data_val_rec)
        return all_hashes

    def data_hashes(self) -> Set[RawDataRecordVal]:
        '''Find all data hashes contained within all cellstores

        Note: this method does not remove any duplicates which may be present,
        if dedup is required, process it downstream

        Returns
        -------
        list
            all hash values for all data pieces in the commit
        '''
        cellstores = self.cellstore_names()
        all_hashes = set()
        for cellstore in cellstores:
            recs = self._traverse_cellstore_data_records(cellstore)
            data_val_rec = set(map(parsing.data_record_raw_val_from_db_val, recs.values()))
            all_hashes.update(data_val_rec)
        return all_hashes

# ------------------------- process schema ----------------------------------------------

    def cellstore_schema_spec(self, cellstore_name) -> RawCellstoreSchemaVal:
        '''Return the schema spec for a specific cellstore name.

        If you need both names, and schema spec values, use the `schema_specs` method. The
        underlying cost of searching the db is identical, and this is just a useful filter
        on top.

        Parameters
        ----------
        cellstore_name : str
            name of the cellstore to get the schema for

        Returns
        -------
        RawCellstoreSchemaVal
            raw schema spec for the cellstore requested
        '''
        recs = self._traverse_cellstore_schema_records()
        dsetSchemaKey = parsing.cellstore_record_schema_db_key_from_raw_key(cellstore_name)
        schemaRecVal = recs[dsetSchemaKey]
        schemaRec = parsing.cellstore_record_schema_raw_val_from_db_val(schemaRecVal)
        return schemaRec

    def schema_specs(self) -> Dict[str, RawCellstoreSchemaVal]:
        '''Return the all schema specs defined by all cellstores.

        Returns
        -------
        Dict[str, RawDataSchemaVal]
            dict of cellstore names: raw schema spec for each cellstore schema
        '''
        recs = self._traverse_cellstore_schema_records()
        if len(recs) > 0:
            schKeys = map(parsing.cellstore_record_schema_raw_key_from_db_key, recs.keys())
            schVals = map(parsing.cellstore_record_schema_raw_val_from_db_val, recs.values())
            recs = dict(zip(schKeys, schVals))
        return recs

    def schema_hashes(self) -> List[str]:
        '''Find all schema hashes inside of a commit

        Returns
        -------
        List[str]
            list of all schema hash digests
        '''
        recs = self._traverse_cellstore_schema_records()
        all_schema_hashes = []
        if len(recs) > 0:
            schema_rec_vals = map(parsing.cellstore_record_schema_raw_val_from_db_val, recs.values())
            schema_hashs = map(lambda x: x.schema_hash, schema_rec_vals)
            all_schema_hashes.extend(schema_hashs)
        return all_schema_hashes

    def data_hash_to_schema_hash(self) -> Dict[str, str]:
        '''For all hashs in the commit, map sample hash to schema hash.

        Returns
        -------
        Dict[str, str]
            mapping of sample hash to dset_schema_hash
        '''
        dsetns = self.cellstore_names()
        odict = {}
        for dsetn in dsetns:
            dset_hash_vals = self.cellstore_data_hashes(dsetn)
            dset_schema_spec = self.cellstore_schema_spec(dsetn)
            dset_schema_hash = dset_schema_spec.schema_hash
            for dset_hash_val in dset_hash_vals:
                odict[dset_hash_val.data_hash] = dset_schema_hash

        return odict

# --------------------------- process metadata ------------------------------------------

    def metadata_names(self) -> List[str]:
        '''Find all metadata names contained within checkout

        If you need both names, and hash values, call the `metadata_records` function. The
        underlying cost of searching the db is identical, this just provides a pretty
        filter on top.

        Returns
        -------
        List[str]
            list of metadata names contained in the checkout
        '''
        recs = self._traverse_metadata_records()
        if len(recs) > 0:
            meta_key_rec = map(parsing.metadata_record_raw_key_from_db_key, recs.keys())
            meta_names = list(map(lambda x: x.meta_name, meta_key_rec))
        else:
            meta_names = []
        return meta_names

    def metadata_records(self) -> Iterable[RawMetaTuple]:
        '''returns all the metadata record specs for all metadata keys

        Returns
        -------
        Iterable[RawMetaTuple]
            dict of metadata names: metadata record spec for all metadata pieces
        '''
        recs = self._traverse_metadata_records()
        if len(recs) > 0:
            meta_rec_keys = map(parsing.metadata_record_raw_key_from_db_key, recs.keys())
            meta_rec_vals = map(parsing.metadata_record_raw_val_from_db_val, recs.values())
            recs = zip(meta_rec_keys, meta_rec_vals)
        return recs

    def metadata_hashes(self) -> List[str]:
        '''Find all hashs for all metadata in a commit

        This method does not deduplicate identical hash records. if needed, postprocess
        downstream

        Returns
        -------
        List[str]
            list of all hashes in the commit
        '''
        recs = self._traverse_metadata_records()
        all_hashes = []
        if len(recs) > 0:
            meta_rec_vals = map(parsing.metadata_record_raw_val_from_db_val, recs.values())
            meta_hashs = map(lambda x: x.meta_hash, meta_rec_vals)
            all_hashes.extend(meta_hashs)
        return all_hashes

# ---------------------------------- python access to all records at once ---------------

    def all_records(self):
        '''Get a nested dict of all metadata and data records out from an unpacked commit.

        .. todo:: Better documentation of this.

        Returns
        -------
        dict
            dict with primary keys: 'cellstores', 'metadata'; with cellstores nesting
            'schema' and 'data' keys/values inside
        '''
        dset_names = self.cellstore_names()
        schema_records = self.schema_specs()
        dsetRecs = {}
        for dsetName in dset_names:
            dsetRecs[dsetName] = {
                'schema': schema_records[dsetName],
                'data': dict(self.cellstore_data_records(dsetName)),
            }

        res = {
            'cellstores': dsetRecs,
            'metadata': dict(self.metadata_records()),
        }
        return res
