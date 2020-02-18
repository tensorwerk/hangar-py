"""
Data Hash parsing functions used to convert db key/val to raw pyhon obj
-----------------------------------------------------------------------
"""

# -------------------- raw (python) -> db ----------------------------------------

cpdef bytes hash_schema_db_key_from_raw_key(str schema_hash):
    return f's:{schema_hash}'.encode()


cpdef bytes hash_data_db_key_from_raw_key(str data_hash):
    return f'h:{data_hash}'.encode()

# ----------------------------- db -> raw (python) ----------------------------

cpdef str hash_schema_raw_key_from_db_key(bytes db_key):
    return db_key.decode()[2:]


cpdef str hash_data_raw_key_from_db_key(bytes db_key):
    return db_key.decode()[2:]


"""
Metadata/Label Hash parsing functions used to convert db key/val to raw pyhon obj
---------------------------------------------------------------------------------
"""

# -------------------- raw (python) -> db ----------------------------------------

cpdef bytes hash_meta_db_key_from_raw_key(str meta_hash):
    return f'h:{meta_hash}'.encode()


cpdef bytes hash_meta_db_val_from_raw_val(str meta_val):
    return meta_val.encode()

# ----------------------------- db -> raw (python) ----------------------------

cpdef str hash_meta_raw_key_from_db_key(bytes db_key):
    return db_key.decode()[2:]


cpdef str hash_meta_raw_val_from_db_val(bytes db_val):
    return db_val.decode()
