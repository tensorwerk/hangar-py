import ast

from typing import NamedTuple, Union

KeyType = Union[str, int]


class CompatibleData(NamedTuple):
    """Bool describing if data is compatible and if False, the reason it is rejected.
    """
    compatible: bool
    reason: str


class ColumnSchemaKey(NamedTuple):
    column: str
    layout: str


def schema_record_count_start_range_key():
    return 's:'.encode()


def schema_db_key_from_column(column: str, layout: str) -> bytes:
    """column schema db formated key from name and layout.

    Parameters
    ----------
    column: str
        name of the column
    layout: str
        layout of the column schema ('flat', 'nested', etc.)
    """
    if layout == 'flat':
        serial = f's:{column}:f'
    elif layout == 'nested':
        serial = f's:{column}:n'
    else:
        raise ValueError(f'layout {layout} not valid')
    return serial.encode()


def schema_db_range_key_from_column_unknown_layout(column: str) -> bytes:
    serial = f's:{column}:'
    return serial.encode()


def schema_column_record_from_db_key(raw: bytes) -> ColumnSchemaKey:
    serial = raw.decode()
    _, column, layout = serial.split(':')
    if layout == 'f':
        layout = 'flat'
    elif layout == 'n':
        layout = 'nested'
    else:
        raise ValueError(f'layout unknown for serial key {serial}')
    return ColumnSchemaKey(column, layout)


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


# -------------------------- flat parser --------------------------------------


class FlatColumnDataKey(NamedTuple):
    column: str
    sample: KeyType

    @property
    def layout(self):
        return 'flat'


def flat_data_column_record_start_range_key(column: str) -> bytes:
    serial = f'f:{column}:'
    return serial.encode()


def flat_data_db_key_from_names(column: str, sample: KeyType) -> bytes:
    if isinstance(sample, int):
        serial = f'f:{column}:#{sample}'
    else:
        serial = f'f:{column}:{sample}'
    return serial.encode()


def flat_data_record_from_db_key(raw: bytes) -> FlatColumnDataKey:
    serial = raw.decode()
    _, column, sample = serial.split(':')
    if sample[0] == '#':
        sample = int(sample[1:])
    return FlatColumnDataKey(column, sample)


# -------------------------- nested parser ------------------------------------


class NestedColumnDataKey(NamedTuple):
    column: str
    sample: KeyType
    subsample: KeyType

    @property
    def layout(self):
        return 'nested'


def nested_data_column_record_start_range_key(column: str) -> bytes:
    serial = f'n:{column}:'
    return serial.encode()


def nested_data_db_key_from_names(column: str,
                                  sample: KeyType,
                                  subsample: KeyType) -> bytes:
    if isinstance(sample, int):
        sample = f'#{sample}'
    if isinstance(subsample, int):
        subsample = f'#{subsample}'
    serial = f'n:{column}:{sample}:{subsample}'
    return serial.encode()


def nested_data_record_from_db_key(raw: bytes) -> NestedColumnDataKey:
    serial = raw.decode()
    _, column, sample, subsample = serial.split(':')
    if sample[0] == '#':
        sample = int(sample[1:])
    if subsample[0] == '#':
        subsample = int(subsample[1:])
    return NestedColumnDataKey(column, sample, subsample)


# ----------------------- dynamic parser selection ----------------------------


_LAYOUT_RECORD_PARSER_MAP = {
    b'f:': flat_data_record_from_db_key,
    b'n:': nested_data_record_from_db_key,
    b's:': schema_column_record_from_db_key,
}


def dynamic_layout_data_record_from_db_key(raw: bytes):
    func = _LAYOUT_RECORD_PARSER_MAP[raw[:2]]
    res = func(raw)
    return res


_LAYOUT_DATA_COLUMN_RANGE_KEY_MAP = {
    'flat': flat_data_column_record_start_range_key,
    'nested': nested_data_column_record_start_range_key,
}


def dynamic_layout_data_record_db_start_range_key(column_record: ColumnSchemaKey) -> bytes:
    func = _LAYOUT_DATA_COLUMN_RANGE_KEY_MAP[column_record.layout]
    res = func(column_record.column)
    return res
