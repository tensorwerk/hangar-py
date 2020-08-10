from typing import NamedTuple, Union, List, Optional

import numpy as np

from ..columns.constructors import open_file_handles, column_type_object_from_schema
from ..context import Environments
from ..records import (
    parsing,
    schema_spec_from_db_val,
    hash_schema_db_key_from_raw_key,
    hash_data_db_key_from_raw_key
)
from ..txnctx import TxnRegister


class ContentWriter(object):
    """Common methods to client & server which write content received.

    These are special methods configured especially for remote operations.
    They do not honor the public facing API or data write/read conventions
    established for users or the rest of Hangar internals.

    Parameters
    ----------
    envs : context.Environments
        main hangar environment context object.
    """

    def __init__(self, envs):

        self.env: Environments = envs
        self.txnctx: TxnRegister = TxnRegister()

    def commit(self, commit: str, parentVal: bytes, specVal: bytes,
               refVal: bytes) -> Union[str, bool]:
        """Write a commit record to the ref db

        Parameters
        ----------
        commit : str
            commit hash to write
        parentVal : bytes
            db formatted representation of commit parents
        specVal : bytes
            db formatted representation of the commit specs
        refVal : bytes
            db formated representation of commit record contents

        Returns
        -------
        str or False
            Commit hash if operation was successful.

            False if the commit hash existed in the db previously and
            no records were written.
        """
        commitSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)
        commitParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        commitRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        refTxn = self.txnctx.begin_writer_txn(self.env.refenv)
        try:
            cmtParExists = refTxn.put(commitParentKey, parentVal, overwrite=False)
            cmtRefExists = refTxn.put(commitRefKey, refVal, overwrite=False)
            cmtSpcExists = refTxn.put(commitSpecKey, specVal, overwrite=False)
        finally:
            self.txnctx.commit_writer_txn(self.env.refenv)

        ret = False if not all([cmtParExists, cmtRefExists, cmtSpcExists]) else commit
        return ret

    def schema(self, schema_hash: str, schemaVal: bytes) -> Union[str, bool]:
        """Write a column schema hash specification record to the db

        Parameters
        ----------
        schema_hash : str
            schema hash being written
        schemaVal : bytes
            db formatted representation of schema specification

        Returns
        -------
        str or False
            schema_hash written if operation was successful.

            False if the schema_hash existed in db and no records written.
        """
        schemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = self.txnctx.begin_writer_txn(self.env.hashenv)
        try:
            schemaExists = hashTxn.put(schemaKey, schemaVal, overwrite=False)
        finally:
            self.txnctx.commit_writer_txn(self.env.hashenv)

        ret = False if not schemaExists else schema_hash
        return ret


class DataWriter:

    def __init__(self, envs):

        self.env: Environments = envs
        self.txnctx: TxnRegister = TxnRegister()

        self._schema_hash_be_accessors = {}
        self._schema_hash_objects = {}
        self._is_cm = False

    def __enter__(self):
        self._is_cm = True
        self.hashTxn = self.txnctx.begin_writer_txn(self.env.hashenv)
        return self

    def __exit__(self, *exc):
        for be in self._schema_hash_be_accessors.values():
            be.close()
        self.txnctx.commit_writer_txn(self.env.hashenv)
        self._schema_hash_be_accessors.clear()
        self._schema_hash_objects.clear()
        self._is_cm = False
        self.hashTxn = None

    @property
    def is_cm(self):
        return self._is_cm

    def _open_new_backend(self, schema):
        be_accessor = open_file_handles(backends=[schema.backend],
                                        path=self.env.repo_path,
                                        mode='a',
                                        schema=schema,
                                        remote_operation=True)[schema.backend]
        self._schema_hash_be_accessors[schema.schema_hash_digest()] = be_accessor

    def _get_schema_object(self, schema_hash):
        schemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        schemaVal = self.hashTxn.get(schemaKey)

        schema_val = schema_spec_from_db_val(schemaVal)
        schema = column_type_object_from_schema(schema_val)

        if schema_hash != schema.schema_hash_digest():
            raise RuntimeError(schema.__dict__)

        self._schema_hash_objects[schema_hash] = schema
        return schema

    def _get_changed_schema_object(self, schema_hash, backend, backend_options):
        import copy
        if schema_hash in self._schema_hash_objects:
            base_schema = copy.deepcopy(self._schema_hash_objects[schema_hash])
        else:
            base_schema = copy.deepcopy(self._get_schema_object(schema_hash))

        base_schema.change_backend(backend, backend_options=backend_options)
        changed_schema = self._schema_hash_objects.setdefault(base_schema.schema_hash_digest(), base_schema)
        return changed_schema

    def data(self,
             schema_hash: str,
             data_digest: str,
             data: Union[str, int, np.ndarray],
             backend: Optional[str] = None,
             backend_options: Optional[dict] = None) -> str:
        """Write data content to the hash records database

        Parameters
        ----------
        schema_hash : str
            schema_hash currently being written
        backend : str, optional
            Manually specified backend code which will be used to record the
            data records. If not specified (``None``), the default backend
            recorded in the schema spec will be used, by default None

        Returns
        -------
        List[str]
            list of str of all data digests written by this method.
        """
        if schema_hash not in self._schema_hash_objects:
            self._get_schema_object(schema_hash)
        schema = self._schema_hash_objects[schema_hash]
        if (backend is not None) and ((backend != schema.backend) or (backend_options is not None)):
            schema = self._get_changed_schema_object(schema_hash, backend, backend_options)

        # Need because after changing, the schema_hash of the backend changes
        final_schema_hash = schema.schema_hash_digest()
        if final_schema_hash not in self._schema_hash_be_accessors:
            self._open_new_backend(schema)

        be_accessor = self._schema_hash_be_accessors[final_schema_hash]
        hashVal = be_accessor.write_data(data, remote_operation=True)
        hashKey = hash_data_db_key_from_raw_key(data_digest)
        self.hashTxn.put(hashKey, hashVal)
        return data_digest


RawCommitContent = NamedTuple('RawCommitContent', [('commit', str),
                                                   ('cmtParentVal', bytes),
                                                   ('cmtSpecVal', bytes),
                                                   ('cmtRefVal', bytes)])


class ContentReader(object):
    """Common methods to client & server which read content.

    These are special methods configured especially for remote operations.
    They do not honor the public facing API or data write/read conventions
    established for users or the rest of Hangar internals.

    Parameters
    ----------
    envs : context.Environments
        main hangar environment context object.
    """
    def __init__(self, envs):

        self.env: Environments = envs
        self.txnctx: TxnRegister = TxnRegister()

    def commit(self, commit: str) -> Union[RawCommitContent, bool]:
        """Read a commit with a given hash and get db formatted content

        Parameters
        ----------
        commit : str
            commit hash to read from the ref db

        Returns
        -------
        namedtuple or False
            nametuple with typename = RawCommitContent field_names = ('commit',
            'cmtParentVal', 'cmtSpecVal', 'cmtRefVal') if operation successful.

            False if commit does not exist with provided digest.
        """
        cmtRefKey = parsing.commit_ref_db_key_from_raw_key(commit)
        cmtParentKey = parsing.commit_parent_db_key_from_raw_key(commit)
        cmtSpecKey = parsing.commit_spec_db_key_from_raw_key(commit)

        reftxn = self.txnctx.begin_reader_txn(self.env.refenv)
        try:
            cmtRefVal = reftxn.get(cmtRefKey, default=False)
            cmtParentVal = reftxn.get(cmtParentKey, default=False)
            cmtSpecVal = reftxn.get(cmtSpecKey, default=False)
        finally:
            self.txnctx.abort_reader_txn(self.env.refenv)

        ret = RawCommitContent(commit, cmtParentVal, cmtSpecVal, cmtRefVal)

        if not all(ret) and not isinstance(ret.cmtParentVal, bytes):
            return False
        else:
            return ret

    def schema(self, schema_hash: str) -> Union[bytes, bool]:
        """Read db formatted schema val for a schema hash

        Parameters
        ----------
        schema_hash : str
            schema hash to look up

        Returns
        -------
        bytes or False
            db formatted representation of schema bytes if schema_hash exists

            False if the schema_hash does not exist in the db.
        """
        schemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = self.txnctx.begin_reader_txn(self.env.hashenv)
        try:
            schemaVal = hashTxn.get(schemaKey, default=False)
        finally:
            self.txnctx.abort_reader_txn(self.env.hashenv)

        ret = False if not schemaVal else schemaVal
        return ret
