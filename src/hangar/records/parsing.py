import json
from hashlib import blake2b
from itertools import cycle
from random import randint
from time import perf_counter, sleep
from typing import Union, NamedTuple, Tuple, Iterable

import blosc

from ..constants import (
    CMT_DIGEST_JOIN_KEY,
    CMT_KV_JOIN_KEY,
    CMT_REC_JOIN_KEY,
    K_BRANCH,
    K_HASH,
    K_HEAD,
    K_INT,
    K_REMOTES,
    K_SCHEMA,
    K_STGARR,
    K_STGMETA,
    K_VERSION,
    K_WLOCK,
    SEP_CMT,
    SEP_KEY,
    WLOCK_SENTINAL,
)
from .._version import parse as version_parse
from .._version import Version


cycle_list = (str(c).rjust(4, '0') for c in range(9_999))
NAME_CYCLER = cycle(cycle_list)
RANDOM_NAME_SEED = str(randint(0, 999_999_999)).rjust(0, '0')
perf_counter()  # call to init monotonic start point


def generate_sample_name() -> str:
    ncycle = next(NAME_CYCLER)
    if ncycle == '0000':
        sleep(0.001)

    sec, subsec = str(perf_counter()).split('.')
    name = f'{RANDOM_NAME_SEED}{sec.rjust(6, "0")}{subsec.ljust(9, "0")}{ncycle}'
    return name


"""
Parsing functions used to deal with repository state The parsers defined in this
section handle repo/branch records.

Methods working with repository version specifiers
--------------------------------------------------
"""


def repo_version_raw_spec_from_raw_string(v_str: str) -> Version:
    """Convert from user facing string representation to Version object
    """
    return version_parse(v_str)


# ------------------------- db version key is fixed -----------------


def repo_version_db_key() -> bytes:
    """The db formated key which version information can be accessed at

    Returns
    -------
    bytes
        db formatted key to use to get/set the repository software version.
    """
    return K_VERSION.encode()


# ------------------------ raw -> db --------------------------------


def repo_version_db_val_from_raw_val(v_spec: Version) -> bytes:
    """determine repository version db specifier from version spec.

    Parameters
    ----------
    v_spec : Version
        This class abstracts handling of a project’s versions. A Version
        instance is comparison aware and can be compared and sorted using the
        standard Python interfaces.

    Returns
    -------
    bytes
        db formatted specification of version
    """
    return str(v_spec).encode()


# ---------------------------- db -> raw ----------------------------


def repo_version_raw_val_from_db_val(db_val: bytes) -> Version:
    """determine software version of hangar repository is written for.

    Parameters
    ----------
    db_val : bytes
        db formatted specification of version string

    Returns
    -------
    Version
        This class abstracts handling of a project’s versions. A Version
        instance is comparison aware and can be compared and sorted using the
        standard Python interfaces.
    """
    db_str = db_val.decode()
    return version_parse(db_str)


"""
Methods working with writer HEAD branch name
--------------------------------------------
"""

# --------------------- db HEAD key is fixed ------------------------


def repo_head_db_key() -> bytes:
    """db_key of the head staging branch name.

    Returns
    -------
    bytestring
        lmdb key to query while looking up the head staging branch name
    """
    return K_HEAD.encode()


# --------------------- raw -> db -----------------------------------


def repo_head_db_val_from_raw_val(branch_name: str) -> bytes:
    return f'{K_BRANCH}{branch_name}'.encode()


# --------------------- db -> raw -----------------------------------

def repo_head_raw_val_from_db_val(db_val: bytes) -> str:
    return db_val.decode()[len(K_BRANCH):]


"""
Methods working with branch names / head commit values
------------------------------------------------------
"""

# ---------------------- raw -> db --------------------------------


def repo_branch_head_db_key_from_raw_key(branch_name: str) -> bytes:
    return f'{K_BRANCH}{branch_name}'.encode()


def repo_branch_head_db_val_from_raw_val(commit_hash: str) -> bytes:
    return f'{commit_hash}'.encode()


# ---------------- db -> raw -----------------------------------


def repo_branch_head_raw_key_from_db_key(db_key: bytes) -> str:
    return db_key.decode()[len(K_BRANCH):]


def repo_branch_head_raw_val_from_db_val(db_val: bytes) -> str:
    try:
        commit_hash = db_val.decode()
    except AttributeError:
        commit_hash = ''
    return commit_hash


"""
Methods working with writer lock key/values
-------------------------------------------
"""

# ------------------- db key for lock is fixed -------------------


def repo_writer_lock_db_key() -> bytes:
    return K_WLOCK.encode()


def repo_writer_lock_sentinal_db_val() -> bytes:
    return WLOCK_SENTINAL.encode()


def repo_writer_lock_force_release_sentinal() -> str:
    return 'FORCE_RELEASE'


# ------------------------- raw -> db ------------------------------


def repo_writer_lock_db_val_from_raw_val(lock_uuid: str) -> bytes:
    return f'{lock_uuid}'.encode()


# -------------------------- db -> raw ------------------------------


def repo_writer_lock_raw_val_from_db_val(db_val: bytes) -> str:
    return db_val.decode()


"""
Parsing functions used to deal with unpacked records.
----------------------------------------------------

The parsers defined in this section handle unpacked data records.
Note that the unpacked structure for the staging area, and for a
commit checked out for reading is identical, with the only difference
being the checkout objects ability to perform write operations.

The following records can be parsed:
    * data records
    * arrayset count records
    * arrayset schema records
    * metadata records
    * metadata count records
"""

# ------------------- named tuple classes used ----------------------


class RawDataRecordKey(NamedTuple):
    """Represents a Data Sample Record Key"""
    aset_name: str = None
    data_name: Union[str, int] = None
    subsample: Union[str, int] = None


class RawDataRecordVal(NamedTuple):
    """Represents a Data Sample Record Hash Value"""
    data_hash: str


class RawArraysetSchemaVal(NamedTuple):
    """Information Specifying a Arrayset Schema"""
    schema_hash: str
    schema_dtype: int
    schema_is_var: bool
    schema_max_shape: tuple
    schema_default_backend: str
    schema_default_backend_opts: dict
    schema_contains_subsamples: bool


"""
Parsing functions to convert lmdb data record keys/vals to/from python vars
----------------------------------------------------------------------------
"""

# -------------------- db -> raw (python) -----------------------------


def data_record_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_STGARR)) -> RawDataRecordKey:
    """Convert and split a lmdb record key & value into python objects

    Parameters
    ----------
    db_key : bytes
        full lmdb record key

    Returns
    -------
    RawDataRecordKey
        Tuple containing the record aset_name, data_name
    """
    key = db_key.decode()
    key_items = key[_SPLT:].split(SEP_KEY)

    if len(key_items) == 2:
        aset, sample = key_items
        if sample[0] == K_INT:
            sample = int(sample[1:])
        return RawDataRecordKey(aset, sample)
    else:
        aset, sample, subsample = key_items
        if sample[0] == K_INT:
            sample = int(sample[1:])
        if subsample[0] == K_INT:
            subsample = int(subsample[1:])
        return RawDataRecordKey(aset, sample, subsample)


def data_record_raw_val_from_db_val(db_val: bytes) -> RawDataRecordVal:
    """Convert and split a lmdb record value into python objects

    Parameters
    ----------
    db_val : bytes
        full lmdb record value

    Returns
    -------
    RawDataRecordVal
        Tuple containing the record data_hash
    """
    return RawDataRecordVal(db_val.decode())


# -------------------- raw (python) -> db -----------------------------


def data_record_db_key_from_raw_key(aset_name: str, data_name: Union[str, int], *,
                                    subsample: Union[str, int] = None) -> bytes:
    """converts a python record spec into the appropriate lmdb key

    Parameters
    ----------
    aset_name : string
        name of the arrayset for the record
    data_name : Union[str, int]
        name of the data sample for the record
    subsample : Union[str, int], optional
        name of the subsample for the record, default = None

    Returns
    -------
    bytes
        Byte encoded db record key
    """
    if isinstance(data_name, int):
        data_part = f'{K_STGARR}{aset_name}{SEP_KEY}{K_INT}{data_name}'
    else:
        data_part = f'{K_STGARR}{aset_name}{SEP_KEY}{data_name}'

    if isinstance(subsample, int):
        return f'{data_part}{SEP_KEY}{K_INT}{subsample}'.encode()
    elif isinstance(subsample, str):
        return f'{data_part}{SEP_KEY}{subsample}'.encode()
    else:
        return data_part.encode()


def data_record_db_val_from_raw_val(data_hash: str) -> bytes:
    """convert a python record spec into the appropriate lmdb value

    Parameters
    ----------
    data_hash : string
        hash of the data sample

    Returns
    -------
    bytestring
        Byte encoded db record val.
    """
    return f'{data_hash}'.encode()


"""
Functions to convert arrayset schema records to/from python objects.
--------------------------------------------------------------------
"""

# ----------------- raw schema -> db schema -----------------------------


def arrayset_record_schema_db_key_from_raw_key(aset_name):
    """Get the db schema key for a named arrayset

    Parameters
    ----------
    aset_name : string
        the name of the arrayset whose schema is found.

    Returns
    -------
    bytestring
        the db_key which can be used to query the schema
    """
    return f'{K_SCHEMA}{aset_name}'.encode()


def arrayset_record_schema_db_val_from_raw_val(schema_hash,
                                               schema_is_var,
                                               schema_max_shape,
                                               schema_dtype,
                                               schema_default_backend,
                                               schema_default_backend_opts,
                                               schema_contains_subsamples):
    """Format the db_value which includes all details of the arrayset schema.

    Parameters
    ----------
    schema_hash : string
        The hash of the schema calculated at initialization.
    schema_is_var : bool
        Are samples in the arrayset variable shape or not?
    schema_max_shape : tuple of ints (size along each dimension)
        The maximum shape of the data pieces. For fixed shape arraysets, all
        input tensors must have the same dimension size and rank as this
        specification. For variable-shape arraysets, tensor rank must match, but
        the size of each dimension may be less than or equal to the
        corresponding dimension here.
    schema_dtype : int
        The datatype numeric code (`np.dtype.num`) of the arrayset. All input
        tensors must exactly match this datatype.
    schema_default_backend : str
        backend specification for the schema default backend.
    schema_default_backend_opts : dict
        filter options for the default schema backend writer.
    schema_contains_subsamples: bool
        **TODO**

    Returns
    -------
    bytestring
        Bytes encoded representation of the schema.
    """
    schema_val = {
        'schema_hash': schema_hash,
        'schema_dtype': schema_dtype,
        'schema_is_var': schema_is_var,
        'schema_max_shape': schema_max_shape,
        'schema_default_backend': schema_default_backend,
        'schema_default_backend_opts': schema_default_backend_opts,
        'schema_contains_subsamples': schema_contains_subsamples,
    }

    db_schema_val = json.dumps(schema_val, separators=(',', ':')).encode()
    return db_schema_val


# -------------- db schema -> raw schema -------------------------------

def arrayset_record_schema_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_SCHEMA)) -> str:
    return db_key.decode()[_SPLT:]


def arrayset_record_schema_raw_val_from_db_val(db_val: bytes) -> RawArraysetSchemaVal:
    schema_spec = json.loads(db_val)
    schema_spec['schema_max_shape'] = tuple(schema_spec['schema_max_shape'])
    return RawArraysetSchemaVal(**schema_spec)


"""
Functions to convert total aset count records to/from python objects
---------------------------------------------------------------------
"""


def arrayset_record_count_range_key(aset_name: str) -> bytes:
    return f'{K_STGARR}{aset_name}{SEP_KEY}'.encode()


"""
Functions to convert metadata count records to/from python objects.
-------------------------------------------------------------------
"""

# ------------------ raw count -> db aset record count  --------------------


def metadata_range_key() -> bytes:
    """return the metadata db range counter key

    this is fixed at the current implementation, no arguments needed

    Parameters
    ----------

    Returns
    -------
    bytes
        db key to access the metadata count at
    """
    return K_STGMETA.encode()


"""
Functions to convert metadata records to/from python objects
-------------------------------------------------------------
"""


class MetadataRecordKey(NamedTuple):
    """Represents a Metadata Sample Record Key"""
    meta_name: Union[str, int]


class MetadataRecordVal(NamedTuple):
    """Represents a Metadata Sample Record Hash Value"""
    meta_hash: str


# -------------------- db -> raw (python) -----------------------------


def metadata_record_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_STGMETA)) -> MetadataRecordKey:
    """Convert and split a lmdb record key & value into python objects

    Parameters
    ----------
    db_key : bytes
        full lmdb record key

    Returns
    -------
    MetadataRecordKey
        the metadata name
    """
    meta_name = db_key.decode()[_SPLT:]
    if meta_name[0] == K_INT:
        meta_name = int(meta_name[1:])
    return MetadataRecordKey(meta_name)


def metadata_record_raw_val_from_db_val(db_val: bytes) -> MetadataRecordVal:
    """Convert and split a lmdb record value into python objects

    Parameters
    ----------
    db_val : bytes
        full lmdb record value

    Returns
    -------
    MetadataRecordVal
        containing the metadata hash
    """
    return MetadataRecordVal(db_val.decode())


# -------------------- raw (python) -> db -----------------------------


def metadata_record_db_key_from_raw_key(meta_name: Union[str, int]) -> bytes:
    """converts a python metadata name into the appropriate lmdb key

    Parameters
    ----------
    meta_name : Union[str, int]
        key / name assigned to the metadata piece

    Returns
    -------
    bytes
        Byte encoded db record key
    """
    if isinstance(meta_name, int):
        record_key = f'{K_STGMETA}{K_INT}{meta_name}'.encode()
    else:
        record_key = f'{K_STGMETA}{meta_name}'.encode()
    return record_key


def metadata_record_db_val_from_raw_val(meta_hash: str) -> bytes:
    """convert a python metadata hash into the appropriate lmdb value

    Parameters
    ----------
    meta_hash : string
        uuid of the data sample

    Returns
    -------
    bytes
        Byte encoded db record val.
    """
    return meta_hash.encode()


"""
Hash Parsing Methods
--------------------

The parsers defined in this section handle hash records for both data samples
and metadata, though currently each is written to seperate lmdb databases.

The hash db always exists in an unpacked structure, and contained records for
every hash stored in the repository. At the moment, the hash records assumes
the following:

    * HDF5 backed
    * All records available on the local disk
    * A single hash type is used (at the moment, xxh64_hexdigest)

These assumptions will change soon, and it is recomended to start here when
begining development on any of the above features.
"""

"""
Data Hash parsing functions used to convert db key/val to raw pyhon obj
-----------------------------------------------------------------------
"""

# -------------------- raw (python) -> db ----------------------------------------


def hash_schema_db_key_from_raw_key(schema_hash: str) -> bytes:
    return f'{K_SCHEMA}{schema_hash}'.encode()


def hash_data_db_key_from_raw_key(data_hash: str) -> bytes:
    return f'{K_HASH}{data_hash}'.encode()


# ----------------------------- db -> raw (python) ----------------------------


def hash_schema_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_SCHEMA)) -> str:
    return db_key.decode()[_SPLT:]


def hash_data_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_HASH)) -> str:
    return db_key.decode()[_SPLT:]


"""
Metadata/Label Hash parsing functions used to convert db key/val to raw pyhon obj
---------------------------------------------------------------------------------
"""

# -------------------- raw (python) -> db ----------------------------------------


def hash_meta_db_key_from_raw_key(meta_hash: str) -> bytes:
    return f'{K_HASH}{meta_hash}'.encode()


def hash_meta_db_val_from_raw_val(meta_val: str) -> bytes:
    return f'{meta_val}'.encode()


# ----------------------------- db -> raw (python) ----------------------------


def hash_meta_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_HASH)) -> str:
    return db_key.decode()[_SPLT:]


def hash_meta_raw_val_from_db_val(db_val: bytes) -> str:
    return db_val.decode()


# -------------------- Remote Work --------------------------------------------


def remote_db_key_from_raw_key(remote_name: str) -> bytes:
    """Get the remote db key val for a remote name

    Parameters
    ----------
    remote_name : str
        name of the remote location

    Returns
    -------
    bytes
        db key allowing access to address value at the name of the remote
    """
    return f'{K_REMOTES}{remote_name}'.encode()


def remote_raw_key_from_db_key(db_key: bytes, *, _SPLT=len(K_REMOTES)) -> str:
    """Get the remote name from a remote db key

    Parameters
    ----------
    db_key : bytes
        db key of the remote

    Returns
    -------
    str
        name of the remote
    """
    return db_key.decode()[_SPLT:]


def remote_db_val_from_raw_val(grpc_address: str) -> bytes:
    """Format a remote db value from it's grpc address string

    Parameters
    ----------
    grpc_address : str
        IP:PORT where the grpc server can be accessed

    Returns
    -------
    bytes
        formated representation of the grpc address suitable for storage in lmdb.
    """
    return grpc_address.encode()


def remote_raw_val_from_db_val(db_val: bytes) -> str:
    """Retrieve the address where a grpc server is running from a remote db value


    Parameters
    ----------
    db_val : bytes
        db value assigned to the desired remote name

    Returns
    -------
    str
        IP:PORT where the grpc server can be accessed.
    """
    return db_val.decode()


"""
Commit Parsing Methods
-----------------------

The parsers defined in this section handle commit (ref) records
"""


class CommitAncestorSpec(NamedTuple):
    is_merge_commit: bool
    master_ancestor: str
    dev_ancestor: str


class CommitUserSpec(NamedTuple):
    commit_time: float
    commit_message: str
    commit_user: str
    commit_email: str


class DigestAndUserSpec(NamedTuple):
    digest: str
    user_spec: CommitUserSpec


class DigestAndAncestorSpec(NamedTuple):
    digest: str
    ancestor_spec: CommitAncestorSpec


class DigestAndBytes(NamedTuple):
    digest: str
    raw: bytes


class DigestAndDbRefs(NamedTuple):
    digest: str
    db_kvs: Union[Tuple, Tuple[Tuple[bytes, bytes]]]


def _hash_func(recs: bytes) -> str:
    """hash a tuple of db formatted k, v pairs.

    Parameters
    ----------
    recs : bytes
        tuple to calculate the joined digest of

    Returns
    -------
    str
        hexdigest of the joined tuple data
    """
    return blake2b(recs, digest_size=20).hexdigest()


def cmt_final_digest(parent_digest: str, spec_digest: str, refs_digest: str,
                     *, tcode: str = 'a') -> str:
    """Determine digest of commit based on digests of its parent, specs, and refs.

    Parameters
    ----------
    parent_digest : str
        digest of the parent value
    spec_digest : str
        digest of the user spec value
    refs_digest : str
        digest of the data record values
    tcode : str, optional, kwarg-only
        hash calculation type code. Included to allow future updates to change
        hashing algorithm, kwarg-only, by default '0'

    Returns
    -------
    str
        digest of the commit with typecode prepended by '{tcode}='.
    """
    if tcode == 'a':
        sorted_digests = sorted([parent_digest, spec_digest, refs_digest])
        joined_bytes = CMT_DIGEST_JOIN_KEY.join(sorted_digests).encode()
        rawDigest = _hash_func(joined_bytes)
        digest = f'a={rawDigest}'
    else:
        raise ValueError(
            f'Invalid commit reference type code {tcode}. If encountered during '
            f'normal operation, please report to hangar development team.')
    return digest


"""
Commit Parent (ancestor) Lookup methods
---------------------------------------
"""

# ------------------------- raw -> db -----------------------------------------


def commit_parent_db_key_from_raw_key(commit_hash: str) -> bytes:
    return f'{commit_hash}'.encode()


def commit_parent_db_val_from_raw_val(master_ancestor: str,
                                      dev_ancestor: str = '',
                                      is_merge_commit: bool = False) -> DigestAndBytes:
    if is_merge_commit:
        str_val = f'{master_ancestor}{SEP_CMT}{dev_ancestor}'
    else:
        str_val = f'{master_ancestor}'
    db_val = str_val.encode()
    digest = _hash_func(db_val)
    return DigestAndBytes(digest=digest, raw=db_val)


# ------------------------------- db -> raw -----------------------------------


def commit_parent_raw_key_from_db_key(db_key: bytes) -> str:
    return db_key.decode()


def commit_parent_raw_val_from_db_val(db_val: bytes) -> DigestAndAncestorSpec:
    """Parse the value of a commit's parent value to find it's ancestors

    Parameters
    ----------
    db_val : bytes
        Lmdb value of the commit parent field.

    Returns
    -------
    DigestAndAncestorSpec
        `digest` of data writen to disk and `ancestor_spec`, Namedtuple
        containing fields for `is_merge_commit`, `master_ancestor`, and
        `dev_ancestor`
    """
    parentValDigest = _hash_func(db_val)

    commit_str = db_val.decode()
    commit_ancestors = commit_str.split(SEP_CMT)
    if len(commit_ancestors) == 1:
        is_merge_commit = False
        master_ancestor = commit_ancestors[0]
        dev_ancestor = ''
    else:
        is_merge_commit = True
        master_ancestor = commit_ancestors[0]
        dev_ancestor = commit_ancestors[1]

    ancestorSpec = CommitAncestorSpec(is_merge_commit, master_ancestor, dev_ancestor)
    return DigestAndAncestorSpec(digest=parentValDigest, ancestor_spec=ancestorSpec)


"""
Commit reference key and values.
--------------------------------
"""


def commit_ref_db_key_from_raw_key(commit_hash: str) -> bytes:
    return f'{commit_hash}{SEP_KEY}ref'.encode()


def _commit_ref_joined_kv_digest(joined_db_kvs: Iterable[bytes]) -> str:
    """reproducibly calculate digest from iterable of joined record k/v pairs.

    First calculate the digest of each element in the input iterable. As these
    elements contain the record type (meta key, arrayset name, sample key) as
    well as the data hash digest, any modification of any reference record will
    result in a different digest for that element. Then join all elements into
    single serialized bytestring.

    The output of this method is the hash digest of the serialized bytestring.

    Parameters
    ----------
    joined_db_kvs : Iterable[bytes]
        list or tuple of bytes where each element is the joining of kv pairs
        from the full commit references

    Returns
    -------
    str
        calculated digest of the commit ref record component
    """
    kv_digests = map(_hash_func, joined_db_kvs)
    joined_digests = CMT_DIGEST_JOIN_KEY.join(kv_digests).encode()
    res = _hash_func(joined_digests)
    return res


def commit_ref_db_val_from_raw_val(db_kvs: Iterable[Tuple[bytes, bytes]]) -> DigestAndBytes:
    """serialize and compress a list of db_key/db_value pairs for commit storage

    Parameters
    ----------
    db_kvs : Iterable[Tuple[bytes, bytes]]
        Iterable collection binary encoded db_key/db_val pairs.

    Returns
    -------
    DigestAndBytes
        `raw` serialized and compressed representation of the object. `digest`
        digest of the joined db kvs.
    """
    joined = tuple(map(CMT_KV_JOIN_KEY.join, db_kvs))
    refDigest = _commit_ref_joined_kv_digest(joined)
    pck = CMT_REC_JOIN_KEY.join(joined)
    raw = blosc.compress(pck, typesize=1, clevel=8, shuffle=blosc.NOSHUFFLE, cname='zstd')
    return DigestAndBytes(digest=refDigest, raw=raw)


def commit_ref_raw_val_from_db_val(commit_db_val: bytes) -> DigestAndDbRefs:
    """Load and decompress a commit ref db_val into python object memory.

    Parameters
    ----------
    commit_db_val : bytes
        Serialized and compressed representation of commit refs.

    Returns
    -------
    DigestAndDbRefs
        `digest` of the unpacked commit refs if desired for verification. `db_kvs`
        Iterable of binary encoded key/value pairs making up the repo state at the
        time of that commit. key/value pairs are already in sorted order.
    """
    uncomp_db_raw = blosc.decompress(commit_db_val)
    # if a commit has nothing in it (completely empty), the return from query == ()
    # the stored data is b'' from which the hash is calculated. We manually set these
    # values as the expected unpacking routine will not work correctly.
    if uncomp_db_raw == b'':
        refsDigest = _hash_func(b'')
        raw_db_kv_list = ()
    else:
        raw_joined_kvs_list = uncomp_db_raw.split(CMT_REC_JOIN_KEY)
        refsDigest = _commit_ref_joined_kv_digest(raw_joined_kvs_list)
        raw_db_kv_list = tuple(map(tuple, map(bytes.split, raw_joined_kvs_list)))

    return DigestAndDbRefs(digest=refsDigest, db_kvs=raw_db_kv_list)


"""
Commit spec reference keys and values
-------------------------------------
"""


def commit_spec_db_key_from_raw_key(commit_hash: str) -> bytes:
    return f'{commit_hash}{SEP_KEY}spec'.encode()


def commit_spec_db_val_from_raw_val(commit_time: float, commit_message: str,
                                    commit_user: str,
                                    commit_email: str) -> DigestAndBytes:
    """Serialize a commit specification from user values to a db store value

    Parameters
    ----------
    commit_time : float
        time since unix epoch that the commit was made
    commit_message : str
        user specified commit message to attach to the record
    commit_user : str
        globally configured user name of the repository committer
    commit_email : str
        globally configured user email of the repository committer

    Returns
    -------
    DigestAndBytes
        Two tuple containing ``digest`` and ``raw`` compressed binary encoded
        serialization of commit spec
    """
    spec_dict = {
        'commit_time': commit_time,
        'commit_message': commit_message,
        'commit_user': commit_user,
        'commit_email': commit_email,
    }

    db_spec_val = json.dumps(spec_dict, separators=(',', ':')).encode()
    digest = _hash_func(db_spec_val)
    comp_raw = blosc.compress(
        db_spec_val, typesize=8, clevel=9, shuffle=blosc.SHUFFLE, cname='zlib')
    return DigestAndBytes(digest=digest, raw=comp_raw)


def commit_spec_raw_val_from_db_val(db_val: bytes) -> DigestAndUserSpec:
    uncompressed_db_val = blosc.decompress(db_val)
    digest = _hash_func(uncompressed_db_val)
    commit_spec = json.loads(uncompressed_db_val)
    user_spec = CommitUserSpec(**commit_spec)
    return DigestAndUserSpec(digest=digest, user_spec=user_spec)
