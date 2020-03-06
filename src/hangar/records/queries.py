from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union

import lmdb

from .column_parsers import (
    data_record_digest_val_from_db_val,
    dynamic_layout_data_record_db_start_range_key,
    dynamic_layout_data_record_from_db_key,
    schema_column_record_from_db_key,
    schema_db_range_key_from_column_unknown_layout,
    schema_record_count_start_range_key,
)
from .recordstructs import (
    FlatColumnDataKey,
    NestedColumnDataKey,
    DataRecordVal,
)
from ..txnctx import TxnRegister
from ..utils import ilen
from ..mixins import CursorRangeIterator

RawDataTuple = Tuple[Union[FlatColumnDataKey, NestedColumnDataKey], DataRecordVal]


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

    def _traverse_column_schema_records(self, keys: bool = True, values: bool = True
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
        startSchemaRangeKey = schema_record_count_start_range_key()
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            yield from self.cursor_range_iterator(datatxn, startSchemaRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

    def _traverse_column_data_records(self,
                                      column_name: str,
                                      *,
                                      keys: bool = True,
                                      values: bool = True) -> Iterable[Union[bytes, Tuple[bytes, bytes]]]:
        """Internal method to traverse column data records and get keys/db_values

        The column name is required because this method controls the cursor
        movement by first setting it's position on the column record count
        key, reading it's value "N" and then sequentially pulling records out of
        the db for N loops.

        Parameters
        ----------
        column_name : str
            name of the column to traverse records for.
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
        try:
            datatxn = TxnRegister().begin_reader_txn(self._dataenv)
            schemaColumnRangeKey = schema_db_range_key_from_column_unknown_layout(column_name)
            with datatxn.cursor() as cur:
                cur.set_range(schemaColumnRangeKey)
                schemaColumnKey = cur.key()
            column_record = schema_column_record_from_db_key(schemaColumnKey)
            startRangeKey = dynamic_layout_data_record_db_start_range_key(column_record)
            yield from self.cursor_range_iterator(datatxn, startRangeKey, keys, values)
        finally:
            TxnRegister().abort_reader_txn(self._dataenv)

# ------------------------- process columns --------------------------------------------

    def column_names(self) -> List[str]:
        """Find all named columns in the checkout

        Returns
        -------
        List[str]
            list of all column names
        """
        recs = self._traverse_column_schema_records(keys=True, values=False)
        column_recs = map(schema_column_record_from_db_key, recs)
        return [x.column for x in column_recs]

    def column_count(self) -> int:
        """Return number of columns/schemas in the commit

        Returns
        -------
        int
            len of columns
        """
        return ilen(self._traverse_column_schema_records(keys=True, values=False))

    def data_hashes(self) -> List[str]:
        """Find all data hashes contained within all columns

        Note: this method does not deduplicate values

        Returns
        -------
        List[str]
            all hash values for all data pieces in the commit
        """
        all_hashes = []
        columns = self.column_names()
        for column in columns:
            recs = self._traverse_column_data_records(column, keys=False, values=True)
            data_rec = map(data_record_digest_val_from_db_val, recs)
            data_val_rec = [x.digest for x in data_rec]
            all_hashes.extend(data_val_rec)
        return all_hashes

# ------------------------ process column data records ----------------------

    def column_data_records(self, column_name: str) -> Iterable[RawDataTuple]:
        """Returns the raw data record key and record values for a specific column.

        Parameters
        ----------
        column_name : str
            name of the column to pull records for

        Yields
        ------
        Iterable[RawDataTuple]
            generator of key and value data record specs
        """
        for data_key, data_val in self._traverse_column_data_records(column_name):
            data_rec_key = dynamic_layout_data_record_from_db_key(data_key)
            data_rec_val = data_record_digest_val_from_db_val(data_val)
            yield (data_rec_key, data_rec_val)

    def column_data_hashes(self, column_name: str) -> Set[DataRecordVal]:
        """Find all data hashes contained within a particular column

        Note: this method does not remove any duplicates which may be present,
        if dedup is required, process it downstream

        Parameters
        ----------
        column_name : str
            name of the column to find the hashes contained in

        Returns
        -------
        Set[DataRecordVal]
            all hash values for all data pieces in the column
        """
        recs = self._traverse_column_data_records(column_name, keys=False, values=True)
        return set(map(data_record_digest_val_from_db_val, recs))

    def column_data_count(self, column_name: str) -> int:
        """Return the number of samples in an column with the provided name

        Parameters
        ----------
        column_name : str
            name of the column to query

        Returns
        -------
        int
            number of samples in the column with given name
        """
        recs = self._traverse_column_data_records(column_name, keys=True, values=False)
        return ilen(recs)  # regular len method not defined for generator iterable

# ------------------------- process schema ----------------------------------------------

    def schema_specs(self):
        """Return the all schema specs defined by all columns.

        Returns
        -------
        dict
            dict of column spec key and digest for each column schema
        """
        recs = {}
        for schema_key, schema_val in self._traverse_column_schema_records():
            schema_record = schema_column_record_from_db_key(schema_key)
            schema_val = data_record_digest_val_from_db_val(schema_val)
            recs[schema_record] = schema_val
        return recs

    def schema_hashes(self) -> List[str]:
        """Find all schema hashes inside of a commit

        Returns
        -------
        List[str]
            list of all schema hash digests in the commit
        """
        all_schema_hashes = []
        for schema_rec_val in self._traverse_column_schema_records(keys=False, values=True):
            digest = data_record_digest_val_from_db_val(schema_rec_val)
            all_schema_hashes.append(digest.digest)
        return all_schema_hashes

    def data_hash_to_schema_hash(self) -> Dict[str, str]:
        """For all hashes in the commit, map sample hash to schema hash.

        Returns
        -------
        Dict[str, str]
            mapping of sample hash to aset_schema_hash
        """
        odict = {}
        aset_names = self.column_names()
        aset_schema_specs = self.schema_specs()
        col_names_schema_digests = {k.column: v.digest for k, v in aset_schema_specs.items()}
        for asetn in aset_names:
            aset_hash_vals = self.column_data_hashes(asetn)
            aset_schema_hash = col_names_schema_digests[asetn]
            for aset_hash_val in aset_hash_vals:
                odict[aset_hash_val.digest] = aset_schema_hash
        return odict
