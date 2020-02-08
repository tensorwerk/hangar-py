import ast

from typing import NamedTuple


class CompatibleData(NamedTuple):
    """Bool describing if data is compatible and if False, the reason it is rejected.
    """
    compatible: bool
    reason: str


def schema_db_key_from_column_name(column: str) -> bytes:
    return f's:{column}'.encode()


def column_name_from_schema_db_key(raw: bytes) -> str:
    return raw.decode()[2:]


def schema_db_val_from_spec(schema: dict) -> bytes:
    serialized = repr(schema).replace(' ', '')
    raw = serialized.encode()
    return raw


def schema_spec_from_db_val(raw: bytes) -> dict:
    serialized = raw.decode()
    schema = ast.literal_eval(serialized)
    return schema


def schema_hash_db_key_from_digest(digest: str) -> bytes:
    return f's:{digest}'.encode()



