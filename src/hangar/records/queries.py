from typing import Dict, Iterable, Iterator, List, Set, Tuple

import lmdb

from .parsing import (
    arrayset_record_count_range_key,
    arrayset_record_schema_db_key_from_raw_key,
    arrayset_record_schema_raw_key_from_db_key,
    arrayset_record_schema_raw_val_from_db_val,
    data_record_raw_key_from_db_key,
    data_record_raw_val_from_db_val,
    metadata_range_key,
    metadata_record_raw_key_from_db_key,
    metadata_record_raw_val_from_db_val,
    MetadataRecordKey, MetadataRecordVal,
    RawArraysetSchemaVal, RawDataRecordKey, RawDataRecordVal,
)
from ..constants import K_SCHEMA
from ..txnctx import TxnRegister

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
        res = {}
        metadataRangeKey = metadata_range_key()
        len_RangeKey = len(metadataRangeKey)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                if cursor.set_range(metadataRangeKey):
                    for k, v in cursor.iternext(keys=True, values=True):
                        if k[:len_RangeKey] == metadataRangeKey:
                            res[k] = v
                            continue
                        else:
                            break
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)
        return res

    def _traverse_arrayset_schema_records(self) -> Dict[bytes, bytes]:
        """Internal method to traverse all schema records and pull out k/v db pairs.

        Returns
        -------
        Dict[bytes, bytes]
            dictionary of db schema keys and db_values
        """
        res = {}
        startSchemaRangeKey = f'{K_SCHEMA}'.encode()
        len_RangeKey = len(startSchemaRangeKey)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                if cursor.set_range(startSchemaRangeKey):
                    for k, v in cursor.iternext(keys=True, values=True):
                        if k[:len_RangeKey] == startSchemaRangeKey:
                            res[k] = v
                            continue
                        else:
                            break
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)
        return res

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
        res = {}
        startAsetRecCountRngK = arrayset_record_count_range_key(arrayset_name)
        len_RangeKey = len(startAsetRecCountRngK)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            with datatxn.cursor() as cursor:
                cursor.first()
                if cursor.set_range(startAsetRecCountRngK):
                    for k, v in cursor.iternext(keys=True, values=True):
                        if k[:len_RangeKey] == startAsetRecCountRngK:
                            res[k] = v
                            continue
                        else:
                            break
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)
        return res

# ------------------------- process arraysets --------------------------------------------

    def arrayset_names(self) -> List[str]:
        """Find all named arraysets in the checkout

        Returns
        -------
        List[str]
            list of all arrayset names
        """
        recs = self._traverse_arrayset_schema_records()
        return [arrayset_record_schema_raw_key_from_db_key(_) for _ in recs.keys()]

    def arrayset_count(self) -> int:
        """Return number of arraysets/schemas in the commit

        Returns
        -------
        int
            len of arraysets
        """
        return len(self._traverse_arrayset_schema_records())

    def data_hashes(self) -> List[str]:
        """Find all data hashes contained within all arraysets

        Note: this method does not deduplicate values

        Returns
        -------
        List[str]
            all hash values for all data pieces in the commit
        """
        arraysets = self.arrayset_names()
        all_hashes = []
        for arrayset in arraysets:
            recs = self._traverse_arrayset_data_records(arrayset)
            data_rec = map(data_record_raw_val_from_db_val, recs.values())
            data_val_rec = [x.data_hash for x in data_rec]
            all_hashes.extend(data_val_rec)
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
            data_rec_keys = map(data_record_raw_key_from_db_key, recs.keys())
            data_rec_vals = map(data_record_raw_val_from_db_val, recs.values())
            recs = zip(data_rec_keys, data_rec_vals)
        return recs

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
        return set(map(data_record_raw_val_from_db_val, recs.values()))

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
        return len(self._traverse_arrayset_data_records(arrayset_name))

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
        asetSchemaKey = arrayset_record_schema_db_key_from_raw_key(arrayset_name)
        schemaRecVal = recs[asetSchemaKey]
        return arrayset_record_schema_raw_val_from_db_val(schemaRecVal)

    def schema_specs(self) -> Dict[str, RawArraysetSchemaVal]:
        """Return the all schema specs defined by all arraysets.

        Returns
        -------
        Dict[str, RawDataSchemaVal]
            dict of arrayset names: raw schema spec for each arrayset schema
        """
        recs = self._traverse_arrayset_schema_records()
        if len(recs) > 0:
            schKeys = map(arrayset_record_schema_raw_key_from_db_key, recs.keys())
            schVals = map(arrayset_record_schema_raw_val_from_db_val, recs.values())
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
            schema_rec_vals = map(arrayset_record_schema_raw_val_from_db_val, recs.values())
            schema_hashs = [x.schema_hash for x in schema_rec_vals]
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

    def metadata_records(self) -> Iterable[RawMetaTuple]:
        """returns all the metadata record specs for all metadata keys

        Returns
        -------
        Iterable[RawMetaTuple]
            dict of metadata names: metadata record spec for all metadata pieces
        """
        recs = self._traverse_metadata_records()
        if len(recs) > 0:
            meta_rec_keys = map(metadata_record_raw_key_from_db_key, recs.keys())
            meta_rec_vals = map(metadata_record_raw_val_from_db_val, recs.values())
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
            meta_rec_vals = map(metadata_record_raw_val_from_db_val, recs.values())
            meta_hashs = [x.meta_hash for x in meta_rec_vals]
            all_hashes.extend(meta_hashs)
        return all_hashes

    def metadata_count(self) -> int:
        """Find the number of metadata samples in the commit

        Returns
        -------
        int
            number of metadata samples
        """
        return len(self._traverse_metadata_records())
