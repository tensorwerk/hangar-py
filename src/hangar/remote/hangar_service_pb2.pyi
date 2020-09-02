# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
    EnumDescriptor as google___protobuf___descriptor___EnumDescriptor,
    FileDescriptor as google___protobuf___descriptor___FileDescriptor,
)

from google.protobuf.internal.containers import (
    RepeatedScalarFieldContainer as google___protobuf___internal___containers___RepeatedScalarFieldContainer,
)

from google.protobuf.internal.enum_type_wrapper import (
    _EnumTypeWrapper as google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper,
)

from google.protobuf.message import (
    Message as google___protobuf___message___Message,
)

from typing import (
    Iterable as typing___Iterable,
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    NewType as typing___NewType,
    Optional as typing___Optional,
    Text as typing___Text,
    cast as typing___cast,
)

from typing_extensions import (
    Literal as typing_extensions___Literal,
)


builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int


DESCRIPTOR: google___protobuf___descriptor___FileDescriptor = ...

DataLocationValue = typing___NewType('DataLocationValue', builtin___int)
type___DataLocationValue = DataLocationValue
DataLocation: _DataLocation
class _DataLocation(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[DataLocationValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    REMOTE_SERVER = typing___cast(DataLocationValue, 0)
    MINIO = typing___cast(DataLocationValue, 1)
    S3 = typing___cast(DataLocationValue, 2)
    GCS = typing___cast(DataLocationValue, 3)
    ABS = typing___cast(DataLocationValue, 4)
REMOTE_SERVER = typing___cast(DataLocationValue, 0)
MINIO = typing___cast(DataLocationValue, 1)
S3 = typing___cast(DataLocationValue, 2)
GCS = typing___cast(DataLocationValue, 3)
ABS = typing___cast(DataLocationValue, 4)
type___DataLocation = DataLocation

DataTypeValue = typing___NewType('DataTypeValue', builtin___int)
type___DataTypeValue = DataTypeValue
DataType: _DataType
class _DataType(google___protobuf___internal___enum_type_wrapper____EnumTypeWrapper[DataTypeValue]):
    DESCRIPTOR: google___protobuf___descriptor___EnumDescriptor = ...
    NP_ARRAY = typing___cast(DataTypeValue, 0)
    SCHEMA = typing___cast(DataTypeValue, 1)
    STR = typing___cast(DataTypeValue, 2)
    BYTES = typing___cast(DataTypeValue, 3)
NP_ARRAY = typing___cast(DataTypeValue, 0)
SCHEMA = typing___cast(DataTypeValue, 1)
STR = typing___cast(DataTypeValue, 2)
BYTES = typing___cast(DataTypeValue, 3)
type___DataType = DataType

class PushBeginContextRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    client_uuid: typing___Text = ...

    def __init__(self,
        *,
        client_uuid : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"client_uuid",b"client_uuid"]) -> None: ...
type___PushBeginContextRequest = PushBeginContextRequest

class PushBeginContextReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def err(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        err : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"err",b"err"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"err",b"err"]) -> None: ...
type___PushBeginContextReply = PushBeginContextReply

class PushEndContextRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    client_uuid: typing___Text = ...

    def __init__(self,
        *,
        client_uuid : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"client_uuid",b"client_uuid"]) -> None: ...
type___PushEndContextRequest = PushEndContextRequest

class PushEndContextReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def err(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        err : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"err",b"err"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"err",b"err"]) -> None: ...
type___PushEndContextReply = PushEndContextReply

class ErrorProto(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    code: builtin___int = ...
    message: typing___Text = ...

    def __init__(self,
        *,
        code : typing___Optional[builtin___int] = None,
        message : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"code",b"code",u"message",b"message"]) -> None: ...
type___ErrorProto = ErrorProto

class BranchRecord(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    name: typing___Text = ...
    commit: typing___Text = ...

    def __init__(self,
        *,
        name : typing___Optional[typing___Text] = None,
        commit : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"name",b"name"]) -> None: ...
type___BranchRecord = BranchRecord

class HashRecord(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    type: typing___Text = ...
    digest: typing___Text = ...

    def __init__(self,
        *,
        type : typing___Optional[typing___Text] = None,
        digest : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"digest",b"digest",u"type",b"type"]) -> None: ...
type___HashRecord = HashRecord

class CommitRecord(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    parent: builtin___bytes = ...
    ref: builtin___bytes = ...
    spec: builtin___bytes = ...

    def __init__(self,
        *,
        parent : typing___Optional[builtin___bytes] = None,
        ref : typing___Optional[builtin___bytes] = None,
        spec : typing___Optional[builtin___bytes] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"parent",b"parent",u"ref",b"ref",u"spec",b"spec"]) -> None: ...
type___CommitRecord = CommitRecord

class SchemaRecord(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    digest: typing___Text = ...
    blob: builtin___bytes = ...

    def __init__(self,
        *,
        digest : typing___Optional[typing___Text] = None,
        blob : typing___Optional[builtin___bytes] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"blob",b"blob",u"digest",b"digest"]) -> None: ...
type___SchemaRecord = SchemaRecord

class DataOriginRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    digest: typing___Text = ...

    def __init__(self,
        *,
        digest : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"digest",b"digest"]) -> None: ...
type___DataOriginRequest = DataOriginRequest

class DataOriginReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class CompressionOptsEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___CompressionOptsEntry = CompressionOptsEntry

    location: type___DataLocationValue = ...
    data_type: type___DataTypeValue = ...
    digest: typing___Text = ...
    uri: typing___Text = ...
    compression: builtin___bool = ...

    @property
    def compression_opts(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        *,
        location : typing___Optional[type___DataLocationValue] = None,
        data_type : typing___Optional[type___DataTypeValue] = None,
        digest : typing___Optional[typing___Text] = None,
        uri : typing___Optional[typing___Text] = None,
        compression : typing___Optional[builtin___bool] = None,
        compression_opts : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"compression",b"compression",u"compression_opts",b"compression_opts",u"data_type",b"data_type",u"digest",b"digest",u"location",b"location",u"uri",b"uri"]) -> None: ...
type___DataOriginReply = DataOriginReply

class PushFindDataOriginRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    data_type: type___DataTypeValue = ...
    digest: typing___Text = ...
    compression_is_desired: builtin___bool = ...

    def __init__(self,
        *,
        data_type : typing___Optional[type___DataTypeValue] = None,
        digest : typing___Optional[typing___Text] = None,
        compression_is_desired : typing___Optional[builtin___bool] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"compression_is_desired",b"compression_is_desired",u"data_type",b"data_type",u"digest",b"digest"]) -> None: ...
type___PushFindDataOriginRequest = PushFindDataOriginRequest

class PushFindDataOriginReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class CompressionOptsExpectedEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___CompressionOptsExpectedEntry = CompressionOptsExpectedEntry

    digest: typing___Text = ...
    location: type___DataLocationValue = ...
    uri: typing___Text = ...
    compression_expected: builtin___bool = ...

    @property
    def compression_opts_expected(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    def __init__(self,
        *,
        digest : typing___Optional[typing___Text] = None,
        location : typing___Optional[type___DataLocationValue] = None,
        uri : typing___Optional[typing___Text] = None,
        compression_expected : typing___Optional[builtin___bool] = None,
        compression_opts_expected : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"compression_expected",b"compression_expected",u"compression_opts_expected",b"compression_opts_expected",u"digest",b"digest",u"location",b"location",u"uri",b"uri"]) -> None: ...
type___PushFindDataOriginReply = PushFindDataOriginReply

class PingRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
type___PingRequest = PingRequest

class PingReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    result: typing___Text = ...

    def __init__(self,
        *,
        result : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"result",b"result"]) -> None: ...
type___PingReply = PingReply

class GetClientConfigRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    def __init__(self,
        ) -> None: ...
type___GetClientConfigRequest = GetClientConfigRequest

class GetClientConfigReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class ConfigEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key: typing___Text = ...
        value: typing___Text = ...

        def __init__(self,
            *,
            key : typing___Optional[typing___Text] = None,
            value : typing___Optional[typing___Text] = None,
            ) -> None: ...
        def ClearField(self, field_name: typing_extensions___Literal[u"key",b"key",u"value",b"value"]) -> None: ...
    type___ConfigEntry = ConfigEntry


    @property
    def config(self) -> typing___MutableMapping[typing___Text, typing___Text]: ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        config : typing___Optional[typing___Mapping[typing___Text, typing___Text]] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"config",b"config",u"error",b"error"]) -> None: ...
type___GetClientConfigReply = GetClientConfigReply

class FetchBranchRecordRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___BranchRecord: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___BranchRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> None: ...
type___FetchBranchRecordRequest = FetchBranchRecordRequest

class FetchBranchRecordReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___BranchRecord: ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___BranchRecord] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error",u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error",u"rec",b"rec"]) -> None: ...
type___FetchBranchRecordReply = FetchBranchRecordReply

class FetchDataRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    uri: typing___Text = ...

    def __init__(self,
        *,
        uri : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"uri",b"uri"]) -> None: ...
type___FetchDataRequest = FetchDataRequest

class FetchDataReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    uri: typing___Text = ...
    raw_data: builtin___bytes = ...
    nbytes: builtin___int = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        uri : typing___Optional[typing___Text] = None,
        raw_data : typing___Optional[builtin___bytes] = None,
        nbytes : typing___Optional[builtin___int] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error",u"nbytes",b"nbytes",u"raw_data",b"raw_data",u"uri",b"uri"]) -> None: ...
type___FetchDataReply = FetchDataReply

class FetchCommitRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit"]) -> None: ...
type___FetchCommitRequest = FetchCommitRequest

class FetchCommitReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    total_byte_size: builtin___int = ...

    @property
    def record(self) -> type___CommitRecord: ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        total_byte_size : typing___Optional[builtin___int] = None,
        record : typing___Optional[type___CommitRecord] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error",u"record",b"record"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"error",b"error",u"record",b"record",u"total_byte_size",b"total_byte_size"]) -> None: ...
type___FetchCommitReply = FetchCommitReply

class FetchSchemaRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___SchemaRecord: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___SchemaRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> None: ...
type___FetchSchemaRequest = FetchSchemaRequest

class FetchSchemaReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___SchemaRecord: ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___SchemaRecord] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error",u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error",u"rec",b"rec"]) -> None: ...
type___FetchSchemaReply = FetchSchemaReply

class PushBranchRecordRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___BranchRecord: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___BranchRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> None: ...
type___PushBranchRecordRequest = PushBranchRecordRequest

class PushBranchRecordReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> None: ...
type___PushBranchRecordReply = PushBranchRecordReply

class PushDataRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    uri: typing___Text = ...
    raw_data: builtin___bytes = ...
    nbytes: builtin___int = ...
    data_type: type___DataTypeValue = ...
    schema_hash: typing___Text = ...

    def __init__(self,
        *,
        uri : typing___Optional[typing___Text] = None,
        raw_data : typing___Optional[builtin___bytes] = None,
        nbytes : typing___Optional[builtin___int] = None,
        data_type : typing___Optional[type___DataTypeValue] = None,
        schema_hash : typing___Optional[typing___Text] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"data_type",b"data_type",u"nbytes",b"nbytes",u"raw_data",b"raw_data",u"schema_hash",b"schema_hash",u"uri",b"uri"]) -> None: ...
type___PushDataRequest = PushDataRequest

class PushDataReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> None: ...
type___PushDataReply = PushDataReply

class PushCommitRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    total_byte_size: builtin___int = ...

    @property
    def record(self) -> type___CommitRecord: ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        total_byte_size : typing___Optional[builtin___int] = None,
        record : typing___Optional[type___CommitRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"record",b"record"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"record",b"record",u"total_byte_size",b"total_byte_size"]) -> None: ...
type___PushCommitRequest = PushCommitRequest

class PushCommitReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> None: ...
type___PushCommitReply = PushCommitReply

class PushSchemaRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def rec(self) -> type___SchemaRecord: ...

    def __init__(self,
        *,
        rec : typing___Optional[type___SchemaRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"rec",b"rec"]) -> None: ...
type___PushSchemaRequest = PushSchemaRequest

class PushSchemaReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> None: ...
type___PushSchemaReply = PushSchemaReply

class FindMissingCommitsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commits: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...

    @property
    def branch(self) -> type___BranchRecord: ...

    def __init__(self,
        *,
        commits : typing___Optional[typing___Iterable[typing___Text]] = None,
        branch : typing___Optional[type___BranchRecord] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"branch",b"branch"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"branch",b"branch",u"commits",b"commits"]) -> None: ...
type___FindMissingCommitsRequest = FindMissingCommitsRequest

class FindMissingCommitsReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commits: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...

    @property
    def branch(self) -> type___BranchRecord: ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        commits : typing___Optional[typing___Iterable[typing___Text]] = None,
        branch : typing___Optional[type___BranchRecord] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"branch",b"branch",u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"branch",b"branch",u"commits",b"commits",u"error",b"error"]) -> None: ...
type___FindMissingCommitsReply = FindMissingCommitsReply

class FindMissingHashRecordsRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    hashs: builtin___bytes = ...
    total_byte_size: builtin___int = ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        hashs : typing___Optional[builtin___bytes] = None,
        total_byte_size : typing___Optional[builtin___int] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"hashs",b"hashs",u"total_byte_size",b"total_byte_size"]) -> None: ...
type___FindMissingHashRecordsRequest = FindMissingHashRecordsRequest

class FindMissingHashRecordsReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    hashs: builtin___bytes = ...
    total_byte_size: builtin___int = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        hashs : typing___Optional[builtin___bytes] = None,
        total_byte_size : typing___Optional[builtin___int] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"error",b"error",u"hashs",b"hashs",u"total_byte_size",b"total_byte_size"]) -> None: ...
type___FindMissingHashRecordsReply = FindMissingHashRecordsReply

class FindMissingSchemasRequest(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    schema_digests: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        schema_digests : typing___Optional[typing___Iterable[typing___Text]] = None,
        ) -> None: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"schema_digests",b"schema_digests"]) -> None: ...
type___FindMissingSchemasRequest = FindMissingSchemasRequest

class FindMissingSchemasReply(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    commit: typing___Text = ...
    schema_digests: google___protobuf___internal___containers___RepeatedScalarFieldContainer[typing___Text] = ...

    @property
    def error(self) -> type___ErrorProto: ...

    def __init__(self,
        *,
        commit : typing___Optional[typing___Text] = None,
        schema_digests : typing___Optional[typing___Iterable[typing___Text]] = None,
        error : typing___Optional[type___ErrorProto] = None,
        ) -> None: ...
    def HasField(self, field_name: typing_extensions___Literal[u"error",b"error"]) -> builtin___bool: ...
    def ClearField(self, field_name: typing_extensions___Literal[u"commit",b"commit",u"error",b"error",u"schema_digests",b"schema_digests"]) -> None: ...
type___FindMissingSchemasReply = FindMissingSchemasReply
