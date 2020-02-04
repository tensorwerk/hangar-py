from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union

import lmdb

from .parsing import (
    arrayset_record_count_range_key,
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
from ..utils import ilen
from ..mixins import CursorRangeIterator

RawDataTuple = Tuple[RawDataRecordKey, RawDataRecordVal]
RawMetaTuple = Tuple[MetadataRecordKey, MetadataRecordVal]


class RecordQuery(CursorRangeIterator):

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

    def _traverse_metadata_records(self, keys: bool = True, values: bool = True
                                   ) -> Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]:
        """Internal method to traverse all metadata records and pull out keys/db_values

        Both `keys` and `values` parameters cannot be simultaneously set to False.

        Parameters
        ----------
        keys : bool, optional
            If True, yield metadata keys encountered, if False only values are returned.
            By default, True.
        values : bool, optional
            If True, yield metadata hash values encountered, if False only keys are returned.
            By default, True.

        Yields
        ------
        Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]
            Iterable of metadata keys, values, or items tuple
        """
        metadataRangeKey = metadata_range_key()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            yield from self.cursor_range_iterator(datatxn, metadataRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

    def _traverse_arrayset_schema_records(self, keys: bool = True, values: bool = True
                                          ) -> Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]:
        """Internal method to traverse all schema records and pull out k/v db pairs.

        Parameters
        ----------
        keys : bool, optional
            If True, yield metadata keys encountered, if False only values are returned.
            By default, True.
        values : bool, optional
            If True, yield metadata hash values encountered, if False only keys are returned.
            By default, True.

        Yields
        ------
        Iterable[Union[Tuple[bytes], Tuple[bytes, bytes]]]:
            db schema keys and db_values
        """
        startSchemaRangeKey = f'{K_SCHEMA}'.encode()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            yield from self.cursor_range_iterator(datatxn, startSchemaRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

    def _traverse_arrayset_data_records(self, arrayset_name, *,
                                        keys: bool = True, values: bool = True
                                        ) -> Iterable[Union[bytes, Tuple[bytes, bytes]]]:
        """Internal method to traverse arrayset data records and get keys/db_values

        The arrayset name is required because this method controls the cursor
        movement by first setting it's position on the arrayset record count
        key, reading it's value "N" and then sequentially pulling records out of
        the db for N loops.

        Parameters
        ----------
        arrayset_name : str
            name of the arrayset to traverse records for.
        keys : bool, optional
            If True, yield metadata keys encountered, if False only values are returned.
            By default, True.
        values : bool, optional
            If True, yield metadata hash values encountered, if False only keys are returned.
            By default, True.

        Yields
        ------
        Iterable[Union[bytes, Tuple[bytes, bytes]]]:
            dict of db_key/db_values for each record traversed
        """
        startAsetRecCountRngK = arrayset_record_count_range_key(arrayset_name)
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            yield from self.cursor_range_iterator(datatxn, startAsetRecCountRngK, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

# ------------------------- process arraysets --------------------------------------------

    def arrayset_names(self) -> List[str]:
        """Find all named arraysets in the checkout

        Returns
        -------
        List[str]
            list of all arrayset names
        """
        recs = self._traverse_arrayset_schema_records(keys=True, values=False)
        return [arrayset_record_schema_raw_key_from_db_key(rec) for rec in recs]

    def arrayset_count(self) -> int:
        """Return number of arraysets/schemas in the commit

        Returns
        -------
        int
            len of arraysets
        """
        return ilen(self._traverse_arrayset_schema_records(keys=True, values=False))

    def data_hashes(self) -> List[str]:
        """Find all data hashes contained within all arraysets

        Note: this method does not deduplicate values

        Returns
        -------
        List[str]
            all hash values for all data pieces in the commit
        """
        all_hashes = []
        arraysets = self.arrayset_names()
        for arrayset in arraysets:
            recs = self._traverse_arrayset_data_records(arrayset, keys=False, values=True)
            data_rec = map(data_record_raw_val_from_db_val, recs)
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
        Iterable[RawDataTuple]
            generator of key and value data record specs
        """
        for data_key, data_val in self._traverse_arrayset_data_records(arrayset_name):
            data_rec_key = data_record_raw_key_from_db_key(data_key)
            data_rec_val = data_record_raw_val_from_db_val(data_val)
            yield (data_rec_key, data_rec_val)

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
        Set[RawArraysetSchemaVal]
            all hash values for all data pieces in the arrayset
        """
        recs = self._traverse_arrayset_data_records(arrayset_name, keys=False, values=True)
        return set(map(data_record_raw_val_from_db_val, recs))

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
        recs = self._traverse_arrayset_data_records(arrayset_name, keys=True, values=False)
        return ilen(recs)  # regular len method not defined for generator iterable

# ------------------------- process schema ----------------------------------------------

    def schema_specs(self) -> Dict[str, RawArraysetSchemaVal]:
        """Return the all schema specs defined by all arraysets.

        Returns
        -------
        Dict[str, RawDataSchemaVal]
            dict of arrayset names: raw schema spec for each arrayset schema
        """
        recs = {}
        for schema_key, schema_val in self._traverse_arrayset_schema_records():
            schKey = arrayset_record_schema_raw_key_from_db_key(schema_key)
            schVal = arrayset_record_schema_raw_val_from_db_val(schema_val)
            recs[schKey] = schVal
        return recs

    def schema_hashes(self) -> List[str]:
        """Find all schema hashes inside of a commit

        Returns
        -------
        List[str]
            list of all schema hash digests
        """
        all_schema_hashes = []
        for schema_rec_val in self._traverse_arrayset_schema_records(keys=False, values=True):
            schema_rec = arrayset_record_schema_raw_val_from_db_val(schema_rec_val)
            all_schema_hashes.append(schema_rec.schema_hash)
        return all_schema_hashes

    def data_hash_to_schema_hash(self) -> Dict[str, str]:
        """For all hashes in the commit, map sample hash to schema hash.

        Returns
        -------
        Dict[str, str]
            mapping of sample hash to aset_schema_hash
        """
        odict = {}
        aset_names = self.arrayset_names()
        aset_schema_specs = self.schema_specs()
        for asetn in aset_names:
            aset_hash_vals = self.arrayset_data_hashes(asetn)
            aset_schema_hash = aset_schema_specs[asetn].schema_hash
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
        for meta_key, meta_hash in self._traverse_metadata_records(keys=True, values=True):
            meta_rec_key = metadata_record_raw_key_from_db_key(meta_key)
            meta_rec_val = metadata_record_raw_val_from_db_val(meta_hash)
            yield (meta_rec_key, meta_rec_val)

    def metadata_hashes(self) -> List[str]:
        """Find all hashs for all metadata in a commit

        This method does not deduplicate identical hash records. if needed, postprocess
        downstream

        Returns
        -------
        List[str]
            list of all hashes in the commit
        """
        recs = self._traverse_metadata_records(keys=False, values=True)
        meta_rec_vals = map(metadata_record_raw_val_from_db_val, recs)
        meta_hashs = [x.meta_hash for x in meta_rec_vals]
        return meta_hashs

    def metadata_count(self) -> int:
        """Find the number of metadata samples in the commit

        Returns
        -------
        int
            number of metadata samples
        """
        # regular len method not defined on generator
        return ilen(self._traverse_metadata_records(keys=True, values=False))
