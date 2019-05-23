import json
import re
from collections import namedtuple
from itertools import cycle
from time import sleep
from time import perf_counter
from random import randint

import blosc
import msgpack

from .. import config

HEAD = config.get('hangar.keys.head')
BRCH = config.get('hangar.keys.brch')
STGARR = config.get('hangar.keys.stgarr')
SCHEMA = config.get('hangar.keys.schema')
HASH = config.get('hangar.keys.hash')
STGMETA = config.get('hangar.keys.stgmeta')

LISTSEP = config.get('hangar.seps.list')
SEP = config.get('hangar.seps.key')
SLICESEP = config.get('hangar.seps.slice')
COMMITSEP = config.get('hangar.seps.commit')
HASHSEP = config.get('hangar.seps.hash')
REMOTES = config.get('hangar.keys.remotes')

cycle_list = [str(c).rjust(4, '0') for c in range(9_999)]
NAME_CYCLER = cycle(cycle_list)
RANDOM_NAME_SEED = str(randint(0, 999_999_999)).rjust(0, '0')
perf_counter()  # call to init monotonic start point


def generate_sample_name():
    ncycle = next(NAME_CYCLER)
    if ncycle == '0000':
        sleep(0.001)

    sec, subsec = str(perf_counter()).split('.')
    name = f'{RANDOM_NAME_SEED}{sec.rjust(6, "0")}{subsec.ljust(9, "0")}{ncycle}'
    return name


'''
Parsing functions used to deal with repository state The parsers defined in this
section handle repo/branch records.


Methods working with writer HEAD branch name
--------------------------------------------
'''

# --------------------- db HEAD key is fixed ----------------------


def repo_head_db_key():
    '''db_key of the head staging branch name.

    Returns
    -------
    bytestring
        lmdb key to query while looking up the head staging branch name
    '''
    db_key = HEAD.encode()
    return db_key


# --------------------- raw -> db --------------------------------


def repo_head_db_val_from_raw_val(branch_name):
    db_val = f'{BRCH}{branch_name}'.encode()
    return db_val


def repo_head_raw_val_from_db_val(db_val):
    raw_val = db_val.decode().replace(BRCH, '', 1)
    return raw_val


'''
Methods working with branch names / head commit values
------------------------------------------------------
'''

# ---------------------- raw -> db --------------------------------


def repo_branch_head_db_key_from_raw_key(branch_name):
    db_key = f'{BRCH}{branch_name}'.encode()
    return db_key


def repo_branch_head_db_val_from_raw_val(commit_hash):
    db_val = f'{commit_hash}'.encode()
    return db_val


# ---------------- db -> raw -----------------------------------


def repo_branch_head_raw_key_from_db_key(db_key):
    key_str = db_key.decode()
    branch_name = key_str.replace(BRCH, '', 1)
    return branch_name


def repo_branch_head_raw_val_from_db_val(db_val):
    try:
        commit_hash = db_val.decode()
    except AttributeError:
        commit_hash = ''
    return commit_hash


'''
Methods working with writer lock key/values
-------------------------------------------
'''

# ------------------- db key for lock is fixed -------------------


def repo_writer_lock_db_key():
    WRITER_LOCK = config.get('hangar.keys.writer_lock')
    db_key = f'{WRITER_LOCK}'.encode()
    return db_key


def repo_writer_lock_sentinal_db_val():
    WRITER_LOCK_SENTINEL = config.get('hangar.keys.writer_lock_sentinel')
    db_val = f'{WRITER_LOCK_SENTINEL}'.encode()
    return db_val


def repo_writer_lock_force_release_sentinal():
    return 'FORCE_RELEASE'


# ------------------------- raw -> db ------------------------------


def repo_writer_lock_db_val_from_raw_val(lock_uuid):
    db_val = f'{lock_uuid}'.encode()
    return db_val


# -------------------------- db -> raw ------------------------------


def repo_writer_lock_raw_val_from_db_val(db_val):
    lock_uuid = db_val.decode()
    return lock_uuid


'''
Parsing functions used to deal with unpacked records.
----------------------------------------------------

The parsers defined in this section handle unpacked data records.
Note that the unpacked structure for the staging area, and for a
commit checked out for reading is identical, with the only difference
being the checkout objects ability to perform write operations.

The following records can be parsed:
    * data records
    * dataset count records
    * dataset schema records
    * metadata records
    * metadata count records
'''

# ------------------- named tuple classes used ----------------------


RawDataRecordKey = namedtuple(
    typename='RawDataRecordKey', field_names=['dset_name', 'data_name'])


RawDataRecordVal = namedtuple(
    typename='RawDataRecordVal', field_names=['data_hash'])


RawDatasetSchemaVal = namedtuple(
    typename='RawDatasetSchemaVal',
    field_names=[
        'schema_uuid',
        'schema_hash',
        'schema_dtype',
        'schema_is_var',
        'schema_max_shape',
        'schema_is_named'])

'''
Parsing functions to convert lmdb data record keys/vals to/from python vars
----------------------------------------------------------------------------
'''

# -------------------- db -> raw (python) -----------------------------


def data_record_raw_key_from_db_key(db_key):
    '''Convert and split a lmdb record key & value into python objects

    Parameters
    ----------
    db_key : bytestring
        full lmdb record key

    Returns
    -------
    namedtuple
        Tuple containing the record dset_name, data_name
    '''
    key = db_key.decode()
    dset_name, data_name = key.replace(STGARR, '', 1).split(SEP)
    return RawDataRecordKey(dset_name, data_name)


def data_record_raw_val_from_db_val(db_val):
    '''Convert and split a lmdb record value into python objects

    Parameters
    ----------
    db_val : bytestring
        full lmdb record value

    Returns
    -------
    namedtuple
        Tuple containing the record data_hash
    '''
    data_hash = db_val.decode()
    return RawDataRecordVal(data_hash)


# -------------------- raw (python) -> db -----------------------------


def data_record_db_key_from_raw_key(dset_name, data_name):
    '''converts a python record spec into the appropriate lmdb key

    Parameters
    ----------
    dset_name : string
        name of the dataset for the record
    data_name : string
        name of the data sample for the record

    Returns
    -------
    bytesstring
        Byte encoded db record key
    '''
    record_key = f'{STGARR}{dset_name}{SEP}{data_name}'.encode()
    return record_key


def data_record_db_val_from_raw_val(data_hash: str) -> bytes:
    '''convert a python record spec into the appropriate lmdb value

    Parameters
    ----------
    # data_uuid : string
    #     uuid of the data sample
    data_hash : string
        hash of the data sample

    Returns
    -------
    bytestring
        Byte encoded db record val.
    '''
    record_val = f'{data_hash}'.encode()
    return record_val


'''
Functions to convert dataset count records to/from python objects.
------------------------------------------------------------------
'''

# ------------------ raw count -> db dset record count  --------------------


def dataset_record_count_db_key_from_raw_key(dset_name):
    db_record_count_key = f'{STGARR}{dset_name}'.encode()
    return db_record_count_key


def dataset_record_count_db_val_from_raw_val(dset_record_count):
    db_record_count_val = f'{dset_record_count}'.encode()
    return db_record_count_val


# ------------------ db dset record count -> raw count --------------------


def dataset_record_count_raw_key_from_db_key(db_key):
    dset_name = db_key.decode().replace(STGARR, '', 1)
    return dset_name


def dataset_record_count_raw_val_from_db_val(db_val):
    record_count = int(db_val.decode())
    return record_count


'''
Functions to convert dataset schema records to/from python objects.
--------------------------------------------------------------------
'''

# ----------------- raw schema -> db schema -----------------------------


def dataset_record_schema_db_key_from_raw_key(dset_name):
    '''Get the db schema key for a named dataset

    Parameters
    ----------
    dset_name : string
        the name of the dataset whose schema is found.

    Returns
    -------
    bytestring
        the db_key which can be used to query the schema
    '''
    db_schema_key = f'{SCHEMA}{dset_name}'.encode()
    return db_schema_key


def dataset_record_schema_db_val_from_raw_val(schema_uuid, schema_hash,
                                              schema_is_var, schema_max_shape,
                                              schema_dtype, schema_is_named):
    '''Format the db_value which includes all details of the dataset schema.

    Parameters
    ----------
    schema_uuid : string
        The uuid of the dataset at the time it was first initialized.
    schema_hash : string
        The hash of the schema calculated at initialization.
    schema_is_var : bool
        Are samples in the dataset variable shape or not?
    schema_max_shape : tuple of ints (size along each dimension)
        The maximum shape of the data pieces. For fixed shape datasets, all
        input tensors must have the same dimension size and rank as this
        specification. For variable-shape datasets, tensor rank must match, but
        the size of each dimension may be less than or equal to the
        corresponding dimension here.
    schema_dtype : int
        The datatype numeric code (`np.dtype.num`) of the dataset. All input
        tensors must exactally match this datatype.
    schema_is_named : bool
        Are samples in the datasets identifiable with names, or not.

    Returns
    -------
    bytestring
        Bytes encoded representation of the schema.
    '''
    schema_val = {
        'schema_uuid': schema_uuid,
        'schema_hash': schema_hash,
        'schema_dtype': schema_dtype,
        'schema_is_var': schema_is_var,
        'schema_max_shape': schema_max_shape,
        'schema_is_named': schema_is_named,
    }
    db_schema_val = json.dumps(schema_val, ensure_ascii=True).encode()
    return db_schema_val


# -------------- db schema -> raw schema -------------------------------

def dataset_record_schema_raw_key_from_db_key(db_key):
    dset_name = db_key.decode().replace(SCHEMA, '', 1)
    return dset_name


def dataset_record_schema_raw_val_from_db_val(db_val):
    schema_spec = json.loads(db_val)
    schema_spec['schema_max_shape'] = tuple(schema_spec['schema_max_shape'])
    raw_val = RawDatasetSchemaVal(**schema_spec)
    return raw_val


'''
Functions to convert total daset count records to/from python objects
---------------------------------------------------------------------
'''

# ------------------------ raw -> db ------------------------------------------


def dataset_total_count_db_key():
    db_key = STGARR.encode()
    return db_key


def dataset_total_count_db_val_from_raw_val(number_of_dsets):
    db_val = f'{number_of_dsets}'.encode()
    return db_val


# --------------------------- db -> raw ---------------------------------------


def dataset_total_count_raw_val_from_db_val(db_val):
    raw_val = int(db_val.decode())
    return raw_val


'''
Functions to convert metadata count records to/from python objects.
-------------------------------------------------------------------
'''

# ------------------ raw count -> db dset record count  --------------------


def metadata_count_db_key():
    '''return the metadata db count key

    this is fixed at the current implementation, no arguments needed

    Parameters
    ----------

    Returns
    -------
    bytestring
        db key to access the metadata count at
    '''
    db_key = STGMETA.encode()
    return db_key


def metadata_count_db_val_from_raw_val(metadata_record_count):
    '''return the metadata db count value from a raw value

    Parameters
    ----------
    metadata_record_count : int
        value to set in the db

    Returns
    -------
    bytestring
        db val of the new metadata count
    '''
    db_val = str(metadata_record_count).encode()
    return db_val


# ------------------ db dset record count -> raw count ------------------------
#
# Note: there is no need for a `metadata_count_raw_key_from_db_key()` function as
# the key is fixed and cannot be modified by callers.
#

def metadata_count_raw_val_from_db_val(db_val):
    record_count = int(db_val.decode())
    return record_count


'''
Functions to convert metadata records to/from python objects
-------------------------------------------------------------
'''


# -------------------- db -> raw (python) -----------------------------


def metadata_record_raw_key_from_db_key(db_key):
    '''Convert and split a lmdb record key & value into python objects

    Parameters
    ----------
    db_key : bytestring
        full lmdb record key

    Returns
    -------
    str
        string containing the metadata name
    '''
    meta_name = db_key.decode().replace(STGMETA, '', 1)
    return meta_name


def metadata_record_raw_val_from_db_val(db_val):
    '''Convert and split a lmdb record value into python objects

    Parameters
    ----------
    db_val : bytestring
        full lmdb record value

    Returns
    -------
    str
        string containing the metadata hash
    '''
    meta_hash = db_val.decode()
    return meta_hash


# -------------------- raw (python) -> db -----------------------------


def metadata_record_db_key_from_raw_key(meta_name):
    '''converts a python metadata name into the appropriate lmdb key

    Parameters
    ----------
    meta_name : string
        key / name assigned to the metadata piece

    Returns
    -------
    bytesstring
        Byte encoded db record key
    '''
    record_key = f'{STGMETA}{meta_name}'.encode()
    return record_key


def metadata_record_db_val_from_raw_val(meta_hash):
    '''convert a python metadata hash into the appropriate lmdb value

    Parameters
    ----------
    meta_hash : string
        uuid of the data sample

    Returns
    -------
    bytestring
        Byte encoded db record val.
    '''
    record_val = meta_hash.encode()
    return record_val


'''
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
'''

# ------------------- Named Tuple Specifications Used -------------------------

DataHashVal = namedtuple(
    typename='DataHashVal',
    field_names=['hdf5_file_schema', 'hdf5_schema_instance',
                 'hdf5_dataset', 'hdf5_dataset_idx', 'data_shape'])

# match and remove the following characters: '['   ']'   '('   ')'   ','
DataShapeReplacementRegEx = re.compile('[,\(\)\[\]]')

'''
Data Hash parsing functions used to convert db key/val to raw pyhon obj
-----------------------------------------------------------------------
'''

# -------------------- raw (python) -> db ----------------------------------------


def hash_schema_db_key_from_raw_key(schema_hash):
    key = f'{SCHEMA}{schema_hash}'
    db_key = key.encode()
    return db_key


def hash_data_db_key_from_raw_key(data_hash):
    key = f'{HASH}{data_hash}'
    db_key = key.encode()
    return db_key


def hash_data_db_val_from_raw_val(hdf5_file_schema, hdf5_schema_instance, hdf5_dataset,
                                  hdf5_dataset_idx, data_shape):
    '''converts the hdf5 data has spec to an appropriate db value

    .. Note:: This is the inverse of `hash_data_raw_val_from_db_val()`. Any changes in
              db value format must be reflected in both functions.

    Parameters
    ----------
    hdf5_file_schema : str
        directory name of the hdf5 schema to find this data piece in.
    hdf5_schema_instance : str
        file name (schema instance) of the hdf5 file to find this data piece in.
    hdf5_dataset : str
        collection (ie. hdf5 dataset) name to find find this data piece.
    hdf5_dataset_idx : int or str
        collection first axis index in which this data piece resides.
    data_shape : tuple
        shape of the data sample written to the collection idx. ie:
        what subslices of the hdf5 dataset should be read to retrieve
        the sample as recorded.

    Returns
    -------
    bytestring
        hash data db value recording all input specifications.
    '''
    # match and remove the following characters: '['   ']'   '('   ')'   ','
    # shape_str = re.sub('[,\(\)\[\]]', '', str(data_shape))
    shape_str = DataShapeReplacementRegEx.sub('', str(data_shape))

    hdf5_file_str = f'{LISTSEP}'.join([hdf5_file_schema, hdf5_schema_instance])
    hdf5_dset_str = f'{LISTSEP}'.join([hdf5_dataset, hdf5_dataset_idx])
    hdf5_str = f'{HASHSEP}'.join([hdf5_file_str, hdf5_dset_str])
    full_str = f'{SLICESEP}'.join([hdf5_str, shape_str])
    db_val = full_str.encode()
    return db_val


# ----------------------------- db -> raw (python) ----------------------------


def hash_schema_raw_key_from_db_key(db_key: bytes) -> str:
    raw_key = db_key.decode()
    data_hash = raw_key.replace(SCHEMA, '', 1)
    return data_hash


def hash_data_raw_key_from_db_key(db_key: bytes) -> str:
    # may be uncesessary
    raw_key = db_key.decode()
    data_hash = raw_key.replace(HASH, '', 1)
    return data_hash


def hash_data_raw_val_from_db_val(db_val: bytes) -> DataHashVal:
    '''converts an hdf5 data hash db val into an hdf5 data python spec.

    .. Note:: This is the inverse of `hash_data_db_val_from_raw_val()`. Any changes in
              db value format must be reflected in both functions.

    Parameters
    ----------
    db_val : bytestring
        data hash db value

    Returns
    -------
    namedtuple
        hdf5 data hash specification in DataHashVal named tuple format
    '''
    db_str = db_val.decode()
    hdf5_schema_vals, _, hdf5_dset_vals = db_str.partition(HASHSEP)
    hdf5_file_schema, hdf5_schema_instance = hdf5_schema_vals.split(LISTSEP)

    hdf5_vals, _, shape_vals = hdf5_dset_vals.rpartition(SLICESEP)
    hdf5_dataset, hdf5_dataset_idx = hdf5_vals.split(LISTSEP)
    if shape_vals == '':
        # if the data is of empty shape -> ()
        data_shape = ()
    else:
        data_shape_val = [int(dim) for dim in shape_vals.split(' ')]
        data_shape = tuple(data_shape_val)

    raw_val = DataHashVal(
        hdf5_file_schema,
        hdf5_schema_instance,
        hdf5_dataset,
        hdf5_dataset_idx,
        data_shape)

    return raw_val


'''
Metadata/Label Hash parsing functions used to convert db key/val to raw pyhon obj
---------------------------------------------------------------------------------
'''

# -------------------- raw (python) -> db ----------------------------------------


def hash_meta_db_key_from_raw_key(meta_hash):
    db_key = f'{HASH}{meta_hash}'.encode()
    return db_key


def hash_meta_db_val_from_raw_val(meta_val):
    db_val = f'{meta_val}'.encode()
    return db_val


# ----------------------------- db -> raw (python) ----------------------------


def hash_meta_raw_key_from_db_key(db_key):
    data_hash = db_key.decode().replace(HASH, '', 1)
    return data_hash


def hash_meta_raw_val_from_db_val(db_val):
    meta_val = db_val.decode()
    return meta_val


'''
Commit Parsing Methods
-----------------------

The parsers defined in this section handle commit (ref) records
'''

CommitAncestorSpec = namedtuple(
    typename='CommitAncestorSpec',
    field_names=['is_merge_commit', 'master_ancestor', 'dev_ancestor'])

CommitSpec = namedtuple(
    typename='CommitSpec',
    field_names=['commit_time', 'commit_message', 'commit_user', 'commit_email'])


'''
Commit Parent (ancestor) Lookup methods
---------------------------------------
'''

# ------------------------- raw -> db -----------------------------------------


def commit_parent_db_key_from_raw_key(commit_hash):
    db_key = f'{commit_hash}'.encode()
    return db_key


def commit_parent_db_val_from_raw_val(master_ancestor, dev_ancestor='', is_merge_commit=False):
    if is_merge_commit:
        str_val = f'{master_ancestor}{COMMITSEP}{dev_ancestor}'
    else:
        str_val = f'{master_ancestor}'
    db_val = str_val.encode()
    return db_val


# ------------------------------- db -> raw -----------------------------------


def commit_parent_raw_key_from_db_key(db_key):
    commit_hash = db_key.decode()
    return commit_hash


def commit_parent_raw_val_from_db_val(db_val):
    '''Parse the value of a commit's parent value to find it's ancestors

    Parameters
    ----------
    db_val : bytes
        Lmdb value of the commit parent field.

    Returns
    -------
    namedtuple
        Namedtuple containing fields for `is_merge_commit`, `master_ancestor`, and
        `dev_ancestor`
    '''
    commit_str = db_val.decode()
    commit_ancestors = commit_str.split(COMMITSEP)
    if len(commit_ancestors) == 1:
        is_merge_commit = False
        master_ancestor = commit_ancestors[0]
        dev_ancestor = ''
    else:
        is_merge_commit = True
        master_ancestor = commit_ancestors[0]
        dev_ancestor = commit_ancestors[1]

    return CommitAncestorSpec(is_merge_commit, master_ancestor, dev_ancestor)


'''
Commit reference key and values.
--------------------------------
'''


def commit_ref_db_key_from_raw_key(commit_hash):
    commit_ref_key = f'{commit_hash}{SEP}ref'.encode()
    return commit_ref_key


def commit_ref_db_val_from_raw_val(commit_db_key_val_list):
    '''serialize and compress a list of db_key/db_value pairs for commit storage

    Parameters
    ----------
    commit_db_key_val_list : iterable of 2-tuples
        Iterable collection binary encoded db_key/db_val pairs.

    Returns
    -------
    bytes
        Serialized and compressed representation of the object.
    '''
    serialized_db_list = msgpack.packb(
        tuple(commit_db_key_val_list), use_bin_type=True)

    zlibpacked = blosc.compress(
        serialized_db_list,
        cname='zlib',
        clevel=9,
        shuffle=blosc.SHUFFLE,
        typesize=1)

    print(blosc.get_cbuffer_sizes(zlibpacked))

    return zlibpacked


def commit_ref_raw_val_from_db_val(commit_db_val):
    '''Load and decompress a commit ref db_val into python object memory.

    Parameters
    ----------
    commit_db_val : bytes
        Serialized and compressed representation of commit refs.

    Returns
    -------
    tuple of two-tuple binary encoded key/values.
        Iterable of binary encoded key/value pairs making up the repo state at the time of
        that commit. key/value pairs are already in sorted order.
    '''
    uncompressed_db_list = blosc.decompress(commit_db_val)
    bytes_db_key_val_list = msgpack.unpackb(uncompressed_db_list, use_list=False)
    return bytes_db_key_val_list


'''
Commit spec reference keys and values
-------------------------------------
'''


def commit_spec_db_key_from_raw_key(commit_hash):
    commit_spec_key = f'{commit_hash}{SEP}spec'.encode()
    return commit_spec_key


def commit_spec_db_val_from_raw_val(commit_time, commit_message, commit_user, commit_email):
    '''Serialize a commit specification from user values to a db store value

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
    bytes
        binary encoded serialization of the commit spec.
    '''
    spec_dict = {
        'commit_time': commit_time,
        'commit_message': commit_message,
        'commit_user': commit_user,
        'commit_email': commit_email,
    }
    db_spec_val = json.dumps(spec_dict, ensure_ascii=True).encode()
    compressed_db_val = blosc.compress(
        db_spec_val,
        cname='zlib',
        clevel=9,
        shuffle=blosc.SHUFFLE)
    return compressed_db_val


def commit_spec_raw_val_from_db_val(db_val):
    uncompressed_db_val = blosc.decompress(db_val)
    commit_spec = json.loads(uncompressed_db_val, encoding='ASCII')
    raw_val = CommitSpec(**commit_spec)
    return raw_val


# -------------------- Remote Work --------------------------------------------

def remote_db_key_from_raw_key(remote_name: str) -> bytes:
    '''Get the remote db key val for a remote name

    Parameters
    ----------
    remote_name : str
        name of the remote location

    Returns
    -------
    bytes
        db key allowing access to address value at the name of the remote
    '''
    raw_key = f'{REMOTES}{remote_name}'
    db_key = raw_key.encode()
    return db_key


def remote_raw_key_from_db_key(db_key: bytes) -> str:
    '''Get the remote name from a remote db key

    Parameters
    ----------
    db_key : bytes
        db key of the remote

    Returns
    -------
    str
        name of the remote
    '''
    raw_key = db_key.decode()
    remote_name = raw_key[len(REMOTES):]
    return remote_name


def remote_db_val_from_raw_val(grpc_address: str) -> bytes:
    '''Format a remote db value from it's grpc address string

    Parameters
    ----------
    grpc_address : str
        IP:PORT where the grpc server can be accessed

    Returns
    -------
    bytes
        formated representation of the grpc address suitable for storage in lmdb.
    '''
    db_val = grpc_address.encode()
    return db_val


def remote_raw_val_from_db_val(db_val: bytes) -> str:
    '''Retrieve the address where a grpc server is running from a remote db value


    Parameters
    ----------
    db_val : bytes
        db value assigned to the desired remote name

    Returns
    -------
    str
        IP:PORT where the grpc server can be accessed.
    '''
    raw_val = db_val.decode()
    return raw_val
