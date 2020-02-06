import ast
import hashlib


def _make_hashable(o):
    """Sort container object and deterministically output frozen representation"""
    if isinstance(o, (tuple, list)):
        return tuple((_make_hashable(e) for e in o))

    if isinstance(o, dict):
        return tuple(sorted((k, _make_hashable(v)) for k, v in o.items()))

    if isinstance(o, (set, frozenset)):
        return tuple(sorted(_make_hashable(e) for e in o))

    return o


def schema_hash_digest(schema: dict, *, tcode='1') -> str:
    """Generate the schema hash for some schema specification

    Returns
    -------
    str
        hex digest of this information with typecode prepended by '{tcode}='.
    """
    if tcode == '1':
        frozenschema = _make_hashable(schema)
        serialized = repr(frozenschema).encode()
        digest = hashlib.blake2b(serialized, digest_size=6).hexdigest()
        res = f'1={digest}'
    else:
        raise ValueError(
            f'Invalid Schema Hash Type Code {tcode}. If encountered during '
            f'normal operation, please report to hangar development team.')
    return res


def schema_db_key_from_column_name(column: str) -> bytes:
    return f's:{column}'.encode()


def column_name_from_schema_db_key(raw: bytes) -> str:
    return raw.decode()[2:]


def schema_db_val_from_spec(schema: dict) -> bytes:
    raw = repr(schema).encode()
    return raw


def schema_spec_from_db_val(raw: bytes) -> dict:
    serialized = raw.decode()
    schema = ast.literal_eval(serialized)
    return schema


def schema_hash_db_key_from_digest(digest: str) -> bytes:
    return f's:{digest}'.encode()
