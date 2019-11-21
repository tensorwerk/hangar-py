from typing import NamedTuple, Union, Sequence, Tuple, List, Optional

import numpy as np

from ..context import Environments, TxnRegister
from ..backends import BACKEND_ACCESSOR_MAP, backend_opts_from_heuristics
from ..records import parsing


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
        refTxn = TxnRegister().begin_writer_txn(self.env.refenv)
        try:
            cmtParExists = refTxn.put(commitParentKey, parentVal, overwrite=False)
            cmtRefExists = refTxn.put(commitRefKey, refVal, overwrite=False)
            cmtSpcExists = refTxn.put(commitSpecKey, specVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.refenv)

        ret = False if not all([cmtParExists, cmtRefExists, cmtSpcExists]) else commit
        return ret

    def schema(self, schema_hash: str, schemaVal: bytes) -> Union[str, bool]:
        """Write a arrayset schema hash specification record to the db

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
        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = TxnRegister().begin_writer_txn(self.env.hashenv)
        try:
            schemaExists = hashTxn.put(schemaKey, schemaVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.hashenv)

        ret = False if not schemaExists else schema_hash
        return ret

    def data(self,
             schema_hash: str,
             received_data: Sequence[Tuple[str, np.ndarray]],
             backend: Optional[str] = None,
             backend_opts: Optional[dict] = None) -> List[str]:
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
        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
        try:
            schemaVal = hashTxn.get(schemaKey)
        finally:
            TxnRegister().abort_reader_txn(self.env.hashenv)
        schema_val = parsing.arrayset_record_schema_raw_val_from_db_val(schemaVal)

        if backend is not None:
            if backend not in BACKEND_ACCESSOR_MAP:
                raise ValueError(f'Backend specifier: {backend} not known')
            if backend_opts is None:
                if backend == schema_val.schema_default_backend:
                    backend_opts = schema_val.schema_default_backend_opts
                else:
                    proto = np.zeros(
                        schema_val.schema_max_shape,
                        dtype=np.typeDict[schema_val.schema_dtype])
                    backend_opts = backend_opts_from_heuristics(
                        backend=backend,
                        array=proto,
                        named_samples=schema_val.schema_is_named,
                        variable_shape=schema_val.schema_is_var)
        else:
            backend = schema_val.schema_default_backend
            backend_opts = schema_val.schema_default_backend_opts

        accessor = BACKEND_ACCESSOR_MAP[backend]
        be_accessor = accessor(
            repo_path=self.env.repo_path,
            schema_shape=schema_val.schema_max_shape,
            schema_dtype=np.typeDict[int(schema_val.schema_dtype)])
        be_accessor.open(mode='a', remote_operation=True)
        be_accessor.backend_opts = backend_opts

        saved_digests = []
        hashTxn = TxnRegister().begin_writer_txn(self.env.hashenv)
        try:
            for hdigest, tensor in received_data:
                hashVal = be_accessor.write_data(tensor, remote_operation=True)
                hashKey = parsing.hash_data_db_key_from_raw_key(hdigest)
                hashTxn.put(hashKey, hashVal)
                saved_digests.append(hdigest)
        finally:
            be_accessor.close()
            TxnRegister().commit_writer_txn(self.env.hashenv)
        return saved_digests

    def label(self, digest: str, labelVal: bytes) -> Union[str, bool]:
        """write a metadata / label hash record & content to the db.

        Parameters
        ----------
        digest : str
            hash digest of the label value being written
        labelVal : bytes
            db formatted representation of the label content

        Returns
        -------
        Union[str, bool]
            digest if the operation was successful.

            False if some content already exists with the same digest in the
            db and no operation was performed.
        """
        labelHashKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = TxnRegister().begin_writer_txn(self.env.labelenv)
        try:
            labelExists = labelTxn.put(labelHashKey, labelVal, overwrite=False)
        finally:
            TxnRegister().commit_writer_txn(self.env.labelenv)

        ret = False if not labelExists else digest
        return ret


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

        reftxn = TxnRegister().begin_reader_txn(self.env.refenv)
        try:
            cmtRefVal = reftxn.get(cmtRefKey, default=False)
            cmtParentVal = reftxn.get(cmtParentKey, default=False)
            cmtSpecVal = reftxn.get(cmtSpecKey, default=False)
        finally:
            TxnRegister().abort_reader_txn(self.env.refenv)

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
        schemaKey = parsing.hash_schema_db_key_from_raw_key(schema_hash)
        hashTxn = TxnRegister().begin_reader_txn(self.env.hashenv)
        try:
            schemaVal = hashTxn.get(schemaKey, default=False)
        finally:
            TxnRegister().abort_reader_txn(self.env.hashenv)

        ret = False if not schemaVal else schemaVal
        return ret

    def label(self, digest: str) -> Union[bytes, bool]:
        """Read db formatted label / metadata val for a label / metadata digest

        Parameters
        ----------
        digest : str
            label digest to look up

        Returns
        -------
        bytes or False
            bytes db formatted representation of the label if digest exists

            False if the digest does not exist in the db.
        """
        labelKey = parsing.hash_meta_db_key_from_raw_key(digest)
        labelTxn = TxnRegister().begin_reader_txn(self.env.labelenv)
        try:
            labelVal = labelTxn.get(labelKey, default=False)
        finally:
            TxnRegister().abort_reader_txn(self.env.labelenv)

        ret = False if not labelVal else labelVal
        return ret
