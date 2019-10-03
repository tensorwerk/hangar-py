from typing import Tuple, List, Iterator, Iterable, Set, Dict

import lmdb

from .. import constants as c
from . import parsing
from .parsing import RawDataRecordKey, RawDataRecordVal
from .parsing import MetadataRecordKey, MetadataRecordVal
from .parsing import RawArraysetSchemaVal
from ..context import TxnRegister

RawDataTuple = Tuple[RawDataRecordKey, RawDataRecordVal]
RawMetaTuple = Tuple[MetadataRecordKey, MetadataRecordVal]

"""
Data record queries
-------------------
"""


class RecordQuery(object):

    def __init__(self, dataenv: lmdb.Environment):
        self._dataenv = dataenv

# ------------------ traversing the unpacked records ------------------------------------

    def _traverse_all_records(self) -> Iterator[Tuple[bytes, bytes]]:
        """Pull out all records in the database as a tuple of binary encoded

        Returns
        -------
        list of tuples of bytes
            list type stack of tuples with each db_key, db_val pair
        """
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                for db_kv in cursor.iternext(keys=True, values=True):
                    yield db_kv
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

    def _traverse_metadata_records(self) -> Dict[bytes, bytes]:
        """Internal method to traverse all metadata records and pull out keys/db_values

        Returns
        -------
        Dict[bytes, bytes]
            dictionary of metadata db keys and db_values
        """
        metadataRecords = {}
        metadataRangeKey = parsing.metadata_range_key()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                if cursor.set_range(metadataRangeKey):
                    for k, v in cursor.iternext(keys=True, values=True):
                        if k.startswith(metadataRangeKey):
                            metadataRecords[k] = v
                        else:
                            break

        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

        return metadataRecords

    def _traverse_arrayset_schema_records(self) -> Dict[bytes, bytes]:
        """Internal method to traverse all schema records and pull out k/v db pairs.

        Returns
        -------
        Dict[bytes, bytes]
            dictionary of db schema keys and db_values
        """
        schemaRecords = {}
        startSchemaRangeKey = f'{c.K_SCHEMA}'.encode()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
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

    def _traverse_arrayset_data_records(self, arrayset_name) -> Dict[bytes, bytes]:
        """Internal method to traverse arrayset data records and get keys/db_values

        The arrayset name is required because this method controls the cursor
        movement by first setting it's position on the arrayset record count
        key, reading it's value "N" and then sequentially pulling records out of
        the db for N loops.

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to traverse records for.

        Returns
        -------
        Dict[bytes, bytes]
            dict of db_key/db_values for each record traversed
        """
        data_records = {}
        startAsetRecCountRngK = parsing.arrayset_record_count_range_key(arrayset_name)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                dataRecordsExist = cursor.set_range(startAsetRecCountRngK)
                while dataRecordsExist:
                    dataRecKey, dataRecVal = cursor.item()
                    if dataRecKey.startswith(startAsetRecCountRngK):
                        data_records[dataRecKey] = dataRecVal
                        dataRecordsExist = cursor.next()
                        continue
                    else:
                        dataRecordsExist = False

        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

        return data_records

# ------------------------- process arraysets --------------------------------------------

    def arrayset_names(self) -> List[str]:
        """Find all named arraysets in the checkout

        Returns
        -------
        List[str]
            list of all arrayset names
        """
        recs = self._traverse_arrayset_schema_records()
        arrayset_names = map(parsing.arrayset_record_schema_raw_key_from_db_key, recs.keys())
        return list(arrayset_names)

    def arrayset_count(self) -> int:
        """Return number of arraysets/schemas in the commit

        Returns
        -------
        int
            len of arraysets
        """
        nrecs = len(self._traverse_arrayset_schema_records())
        return nrecs

    def data_hashes(self) -> Set[RawDataRecordVal]:
        """Find all data hashes contained within all arraysets

        Note: this method does not remove any duplicates which may be present,
        if de-dup is required, process it downstream

        Returns
        -------
        list
            all hash values for all data pieces in the commit
        """
        arraysets = self.arrayset_names()
        all_hashes = set()
        for arrayset in arraysets:
            recs = self._traverse_arrayset_data_records(arrayset)
            data_val_rec = set(map(parsing.data_record_raw_val_from_db_val, recs.values()))
            all_hashes.update(data_val_rec)
        return all_hashes

# ------------------------ process arrayset data records ----------------------

    def arrayset_data_records(self, arrayset_name: str) -> Iterable[RawDataTuple]:
        """Returns the raw data record key and record values for a specific arrayset.

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to pull records for

        Yields
        ------
        tuple
            generator of key and value data record specs
        """
        recs = self._traverse_arrayset_data_records(arrayset_name)
        if len(recs) > 0:
            data_rec_keys = map(parsing.data_record_raw_key_from_db_key, recs.keys())
            data_rec_vals = map(parsing.data_record_raw_val_from_db_val, recs.values())
            recs = zip(data_rec_keys, data_rec_vals)
        return recs

    def arrayset_data_names(self, arrayset_name):
        """Find all data names contained within a arrayset.

        If you need both names, and hash values, call the `arrayset_data_records`
        function. The underlying cost of searching the db is identical, this just provides
        a pretty filter on top.

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to retrieve names for

        Returns
        -------
        list of str
            list of data names contained in the arrayset
        """
        recs = self._traverse_arrayset_data_records(arrayset_name)
        data_key_rec = map(parsing.data_record_raw_key_from_db_key, recs.keys())
        data_names = list(map(lambda x: x.data_name, data_key_rec))
        return data_names

    def arrayset_data_hashes(self, arrayset_name: str) -> Set[RawDataRecordVal]:
        """Find all data hashes contained within a particular arrayset

        Note: this method does not remove any duplicates which may be present,
        if dedup is required, process it downstream

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to find the hashes contained in

        Returns
        -------
        list
            all hash values for all data pieces in the arrayset
        """
        recs = self._traverse_arrayset_data_records(arrayset_name)
        data_val_rec = map(parsing.data_record_raw_val_from_db_val, recs.values())
        all_hashes = set(data_val_rec)
        return all_hashes

    def arrayset_data_count(self, arrayset_name: str) -> int:
        """Return the number of samples in an arrayset with the provided name

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to query

        Returns
        -------
        int
            number of samples in the arrayset with given name
        """
        nrecs = len(self._traverse_arrayset_data_records(arrayset_name))
        return nrecs

# ------------------------- process schema ----------------------------------------------

    def arrayset_schema_spec(self, arrayset_name) -> RawArraysetSchemaVal:
        """Return the schema spec for a specific arrayset name.

        If you need both names, and schema spec values, use the `schema_specs` method. The
        underlying cost of searching the db is identical, and this is just a useful filter
        on top.

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to get the schema for

        Returns
        -------
        RawArraysetSchemaVal
            raw schema spec for the arrayset requested
        """
        recs = self._traverse_arrayset_schema_records()
        asetSchemaKey = parsing.arrayset_record_schema_db_key_from_raw_key(arrayset_name)
        schemaRecVal = recs[asetSchemaKey]
        schemaRec = parsing.arrayset_record_schema_raw_val_from_db_val(schemaRecVal)
        return schemaRec

    def schema_specs(self) -> Dict[str, RawArraysetSchemaVal]:
        """Return the all schema specs defined by all arraysets.

        Returns
        -------
        Dict[str, RawDataSchemaVal]
            dict of arrayset names: raw schema spec for each arrayset schema
        """
        recs = self._traverse_arrayset_schema_records()
        if len(recs) > 0:
            schKeys = map(parsing.arrayset_record_schema_raw_key_from_db_key, recs.keys())
            schVals = map(parsing.arrayset_record_schema_raw_val_from_db_val, recs.values())
            recs = dict(zip(schKeys, schVals))
        return recs

    def schema_hashes(self) -> List[str]:
        """Find all schema hashes inside of a commit

        Returns
        -------
        List[str]
            list of all schema hash digests
        """
        recs = self._traverse_arrayset_schema_records()
        all_schema_hashes = []
        if len(recs) > 0:
            schema_rec_vals = map(parsing.arrayset_record_schema_raw_val_from_db_val, recs.values())
            schema_hashs = map(lambda x: x.schema_hash, schema_rec_vals)
            all_schema_hashes.extend(schema_hashs)
        return all_schema_hashes

    def data_hash_to_schema_hash(self) -> Dict[str, str]:
        """For all hashes in the commit, map sample hash to schema hash.

        Returns
        -------
        Dict[str, str]
            mapping of sample hash to aset_schema_hash
        """
        asetns = self.arrayset_names()
        odict = {}
        for asetn in asetns:
            aset_hash_vals = self.arrayset_data_hashes(asetn)
            aset_schema_spec = self.arrayset_schema_spec(asetn)
            aset_schema_hash = aset_schema_spec.schema_hash
            for aset_hash_val in aset_hash_vals:
                odict[aset_hash_val.data_hash] = aset_schema_hash

        return odict

# --------------------------- process metadata ------------------------------------------

    def metadata_names(self) -> List[str]:
        """Find all metadata names contained within checkout

        If you need both names, and hash values, call the `metadata_records` function. The
        underlying cost of searching the db is identical, this just provides a pretty
        filter on top.

        Returns
        -------
        List[str]
            list of metadata names contained in the checkout
        """
        recs = self._traverse_metadata_records()
        if len(recs) > 0:
            meta_key_rec = map(parsing.metadata_record_raw_key_from_db_key, recs.keys())
            meta_names = list(map(lambda x: x.meta_name, meta_key_rec))
        else:
            meta_names = []
        return meta_names

    def metadata_records(self) -> Iterable[RawMetaTuple]:
        """returns all the metadata record specs for all metadata keys

        Returns
        -------
        Iterable[RawMetaTuple]
            dict of metadata names: metadata record spec for all metadata pieces
        """
        recs = self._traverse_metadata_records()
        if len(recs) > 0:
            meta_rec_keys = map(parsing.metadata_record_raw_key_from_db_key, recs.keys())
            meta_rec_vals = map(parsing.metadata_record_raw_val_from_db_val, recs.values())
            recs = zip(meta_rec_keys, meta_rec_vals)
        return recs

    def metadata_hashes(self) -> List[str]:
        """Find all hashs for all metadata in a commit

        This method does not deduplicate identical hash records. if needed, postprocess
        downstream

        Returns
        -------
        List[str]
            list of all hashes in the commit
        """
        recs = self._traverse_metadata_records()
        all_hashes = []
        if len(recs) > 0:
            meta_rec_vals = map(parsing.metadata_record_raw_val_from_db_val, recs.values())
            meta_hashs = map(lambda x: x.meta_hash, meta_rec_vals)
            all_hashes.extend(meta_hashs)
        return all_hashes

    def metadata_count(self) -> int:
        """Find the number of metadata samples in the commit

        Returns
        -------
        int
            number of metadata samples
        """
        nrecs = len(self._traverse_metadata_records())
        return nrecs