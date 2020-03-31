from typing import NamedTuple, Union, Sequence, Tuple, List, Optional

import numpy as np

from ..columns.constructors import open_file_handles, column_type_object_from_schema
from ..context import Environments
from ..records import parsing
from ..records import (
    schema_spec_from_db_val,
    hash_schema_db_key_from_raw_key,
    hash_data_db_key_from_raw_key,
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

    def data(self,
             schema_hash: str,
             received_data: Sequence[Tuple[str, np.ndarray]],
             backend: Optional[str] = None,
             backend_options: Optional[dict] = None) -> List[str]:
        """Write data content to the hash records database

        Parameters
        ----------
        schema_hash : str
            schema_hash currently being written
        received_data : Sequence[Tuple[str, np.ndarray]]
            list of tuples, each specifying (digest, tensor) for data retrieved
            from the server. However, if a backend is manually specified which
            requires different input to the ``write_data`` method than a tensor,
            the second element can be replaced with what is appropriate for that
            situation.
        backend : str, optional
            Manually specified backend code which will be used to record the
            data records. If not specified (``None``), the default backend
            recorded in the schema spec will be used, by default None

        Returns
        -------
        List[str]
            list of str of all data digests written by this method.
        """
        schemaKey = hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = self.txnctx.begin_reader_txn(self.env.hashenv)
        try:
            schemaVal = hashTxn.get(schemaKey)
        finally:
            self.txnctx.abort_reader_txn(self.env.hashenv)
        schema_val = schema_spec_from_db_val(schemaVal)
        schema = column_type_object_from_schema(schema_val)

        if (backend is not None) and ((backend != schema.backend) or (backend_options is not None)):
            schema.change_backend(backend, backend_options=backend_options)

        be_accessor = open_file_handles(backends=[schema.backend],
                                        path=self.env.repo_path,
                                        mode='a',
                                        schema=schema,
                                        remote_operation=True)[schema.backend]
        saved_digests = []
        hashTxn = self.txnctx.begin_writer_txn(self.env.hashenv)
        try:
            for hdigest, tensor in received_data:
                hashVal = be_accessor.write_data(tensor, remote_operation=True)
                hashKey = hash_data_db_key_from_raw_key(hdigest)
                hashTxn.put(hashKey, hashVal)
                saved_digests.append(hdigest)
        finally:
            self.txnctx.commit_writer_txn(self.env.hashenv)
            be_accessor.close()
        return saved_digests


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
