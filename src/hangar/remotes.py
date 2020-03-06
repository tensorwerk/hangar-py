import logging
import tempfile
import time
import warnings
from collections import defaultdict
from contextlib import closing
from pathlib import Path
from typing import List, NamedTuple, Optional, Sequence

import grpc
import lmdb
from tqdm import tqdm

from .backends import backend_decoder
from .constants import LMDB_SETTINGS
from .context import Environments
from .records import hash_data_db_key_from_raw_key
from .records import heads, queries, summarize
from .records.commiting import (
    check_commit_hash_in_history,
    move_process_data_to_store,
    unpack_commit_ref,
)
from .remote.client import HangarClient
from .remote.content import ContentWriter, ContentReader
from .txnctx import TxnRegister
from .utils import is_suitable_user_key

logger = logging.getLogger(__name__)

RemoteInfo = NamedTuple('RemoteInfo', [('name', str), ('address', str)])


class Remotes(object):
    """Class which governs access to remote interactor objects.

    .. note::

       The remote-server implementation is under heavy development, and is
       likely to undergo changes in the Future. While we intend to ensure
       compatability between software versions of Hangar repositories written
       to disk, the API is likely to change. Please follow our process at:
       https://www.github.com/tensorwerk/hangar-py

    """

    def __init__(self, env: Environments):

        self._env: Environments = env
        self._repo_path: Path = self._env.repo_path
        self._client: Optional[HangarClient] = None

    def __verify_repo_initialized(self):
        """Internal method to verify repo initialized before operations occur

        Raises
        ------
        RuntimeError
            If the repository db environments have not been initialized at the
            specified repo path.
        """
        if not self._env.repo_is_initialized:
            raise RuntimeError(
                f'Path {self._repo_path} not Hangar Repo. Use `init_repo()` method')

    def add(self, name: str, address: str) -> RemoteInfo:
        """Add a remote to the repository accessible by `name` at `address`.

        Parameters
        ----------
        name : str
            the name which should be used to refer to the remote server (ie:
            'origin')
        address : str
            the IP:PORT where the hangar server is running

        Returns
        -------
        RemoteInfo
            Two-tuple containing (``name``, ``address``) of the remote added to
            the client's server list.

        Raises
        ------
        ValueError
            If provided name contains any non ascii letter characters
            characters, or if the string is longer than 64 characters long.
        ValueError
            If a remote with the provided name is already listed on this client,
            No-Op. In order to update a remote server address, it must be
            removed and then re-added with the desired address.
        """
        self.__verify_repo_initialized()
        if (not isinstance(name, str)) or (not is_suitable_user_key(name)):
            raise ValueError(
                f'Remote name {name} of type: {type(name)} invalid. Must be '
                f'string with only alpha-numeric (or "." "_" "-") ascii characters. '
                f'Must be <= 64 characters long.')

        succ = heads.add_remote(self._env.branchenv, name=name, address=address)
        if succ is False:
            raise ValueError(f'No-Op: Remote named: {name} already exists.')
        return RemoteInfo(name=name, address=address)

    def remove(self, name: str) -> RemoteInfo:
        """Remove a remote repository from the branch records

        Parameters
        ----------
        name : str
            name of the remote to remove the reference to

        Raises
        ------
        KeyError
            If a remote with the provided name does not exist

        Returns
        -------
        str
            The channel address which was removed at the given remote name
        """
        self.__verify_repo_initialized()
        try:
            address = heads.remove_remote(branchenv=self._env.branchenv, name=name)
        except KeyError as e:
            raise e
        return RemoteInfo(name=name, address=address)

    def list_all(self) -> List[RemoteInfo]:
        """List all remote names and addresses recorded in the client's repository.

        Returns
        -------
        List[RemoteInfo]
            list of namedtuple specifying (``name``, ``address``) for each
            remote server recorded in the client repo.
        """
        self.__verify_repo_initialized()
        res = []
        names = heads.get_remote_names(self._env.branchenv)
        for name in names:
            address = heads.get_remote_address(self._env.branchenv, name)
            res.append(RemoteInfo(name=name, address=address))
        return res

    def ping(self, name: str) -> float:
        """Ping remote server and check the round trip time.

        Parameters
        ----------
        name : str
            name of the remote server to ping

        Returns
        -------
        float
            round trip time it took to ping the server after the connection was
            established and requested client configuration was retrieved

        Raises
        ------
        KeyError
            If no remote with the provided name is recorded.
        ConnectionError
            If the remote server could not be reached.
        """
        self.__verify_repo_initialized()
        address = heads.get_remote_address(branchenv=self._env.branchenv, name=name)
        self._client = HangarClient(envs=self._env, address=address)
        with closing(self._client) as client:
            client: HangarClient
            start = time.time()
            client.ping_pong()
            elapsed = time.time() - start
        return elapsed

    def fetch(self, remote: str, branch: str) -> str:
        """Retrieve new commits made on a remote repository branch.

        This is semantically identical to a `git fetch` command. Any new commits
        along the branch will be retrieved, but placed on an isolated branch to
        the local copy (ie. ``remote_name/branch_name``). In order to unify
        histories, simply merge the remote branch into the local branch.

        Parameters
        ----------
        remote : str
            name of the remote repository to fetch from (ie. ``origin``)
        branch : str
            name of the branch to fetch the commit references for.

        Returns
        -------
        str
            Name of the branch which stores the retrieved commits.
        """
        self.__verify_repo_initialized()
        address = heads.get_remote_address(self._env.branchenv, name=remote)
        self._client = HangarClient(envs=self._env, address=address)
        CW = ContentWriter(self._env)

        with closing(self._client) as client:
            client: HangarClient

            # ----------------- setup / validate operations -------------------

            try:
                cHEAD = heads.get_branch_head_commit(self._env.branchenv, branch)
            except ValueError:
                # branch does not exist on local client
                try:
                    s_branch = client.fetch_branch_record(branch)
                    sHEAD = s_branch.rec.commit
                except grpc.RpcError as rpc_error:
                    if rpc_error.code() == grpc.StatusCode.NOT_FOUND:
                        # branch does not exist on remote
                        logger.error(rpc_error.details())
                    raise rpc_error
            else:
                c_bhistory = summarize.list_history(
                    self._env.refenv, self._env.branchenv, branch_name=branch)
                try:
                    s_branch = client.fetch_branch_record(branch)
                    sHEAD = s_branch.rec.commit
                except grpc.RpcError as rpc_error:
                    if rpc_error.code() == grpc.StatusCode.NOT_FOUND:
                        # branch does not exist on remote
                        logger.error(rpc_error.details())
                    raise rpc_error

                # verify histories are intact and should be synced
                if sHEAD == cHEAD:
                    warnings.warn(f'NoOp:  {sHEAD} == client HEAD {cHEAD}', UserWarning)
                    return branch
                elif sHEAD in c_bhistory['order']:
                    warnings.warn(
                        f'REJECTED: remote HEAD: {sHEAD} behind local: {cHEAD}', UserWarning)
                    return branch

            # ------------------- get data ------------------------------------

            mCmtResponse = client.fetch_find_missing_commits(branch)
            m_cmts = mCmtResponse.commits
            for commit in tqdm(m_cmts, desc='fetching commit data refs'):
                mSchemaResponse = client.fetch_find_missing_schemas(commit)
                for schema in mSchemaResponse.schema_digests:
                    schema_hash, schemaVal = client.fetch_schema(schema)
                    CW.schema(schema_hash, schemaVal)
                # Record missing data hash digests (does not get data itself)
                m_hashes = client.fetch_find_missing_hash_records(commit)
                m_schema_hash_map = defaultdict(list)
                for digest, schema_hash in m_hashes:
                    m_schema_hash_map[schema_hash].append((digest, schema_hash))
                for schema_hash, received_data in m_schema_hash_map.items():
                    CW.data(schema_hash, received_data, backend='50')

            # Get missing commit reference specification
            for commit in tqdm(m_cmts, desc='fetching commit spec'):
                cmt, parentVal, specVal, refVal = client.fetch_commit_record(commit)
                CW.commit(cmt, parentVal, specVal, refVal)

            # --------------------------- At completion -----------------------

            # Update (or create) remote branch pointer with new HEAD commit
            fetchBranchName = f'{remote}/{branch}'
            try:
                heads.create_branch(
                    self._env.branchenv, name=fetchBranchName, base_commit=sHEAD)
            except ValueError:
                heads.set_branch_head_commit(
                    self._env.branchenv, branch_name=fetchBranchName, commit_hash=sHEAD)

            return fetchBranchName

    def fetch_data(self,
                   remote: str,
                   branch: str = None,
                   commit: str = None,
                   *,
                   column_names: Optional[Sequence[str]] = None,
                   max_num_bytes: int = None,
                   retrieve_all_history: bool = False) -> List[str]:
        """Retrieve the data for some commit which exists in a `partial` state.

        Parameters
        ----------
        remote : str
            name of the remote to pull the data from
        branch : str, optional
            The name of a branch whose HEAD will be used as the data fetch
            point. If None, ``commit`` argument expected, by default None
        commit : str, optional
            Commit hash to retrieve data for, If None, ``branch`` argument
            expected, by default None
        column_names : Optional[Sequence[str]]
            Names of the columns which should be retrieved for the particular
            commits, any columns not named will not have their data fetched
            from the server. Default behavior is to retrieve all columns
        max_num_bytes : Optional[int]
            If you wish to limit the amount of data sent to the local machine,
            set a `max_num_bytes` parameter. This will retrieve only this
            amount of data from the server to be placed on the local disk.
            Default is to retrieve all data regardless of how large.
        retrieve_all_history : Optional[bool]
            if data should be retrieved for all history accessible by the parents
            of this commit HEAD. by default False

        Returns
        -------
        List[str]
            commit hashes of the data which was returned.

        Raises
        ------
            ValueError
                if branch and commit args are set simultaneously.
            ValueError
                if specified commit does not exist in the repository.
            ValueError
                if branch name does not exist in the repository.
        """
        self.__verify_repo_initialized()
        address = heads.get_remote_address(branchenv=self._env.branchenv, name=remote)
        self._client = HangarClient(envs=self._env, address=address)
        CW = ContentWriter(self._env)

        # ----------------- setup / validate operations -----------------------

        if all([branch, commit]):
            raise ValueError(f'``branch`` and ``commit`` args cannot be set simultaneously')
        if branch is not None:
            cmt = heads.get_branch_head_commit(self._env.branchenv, branch_name=branch)
        else:
            cmt = commit
            cmtExist = check_commit_hash_in_history(self._env.refenv, commit)
            if not cmtExist:
                raise ValueError(f'specified commit: {commit} does not exist in the repo.')

        # --------------- negotiate missing data to get -----------------------

        if retrieve_all_history is True:
            if isinstance(max_num_bytes, int):
                raise ValueError(
                    f'setting the maximum number of bytes transferred and requesting '
                    f'all history are incompatible arguments.')
            else:
                hist = summarize.list_history(self._env.refenv, self._env.branchenv, commit_hash=cmt)
                commits = hist['order']
        else:
            commits = [cmt]

        with tempfile.TemporaryDirectory() as tempD:
            # share unpacked ref db between dependent methods
            tmpDF = Path(tempD, 'test.lmdb')
            tmpDB = lmdb.open(path=str(tmpDF), **LMDB_SETTINGS)
            try:
                allHashs = set()
                # all history argument
                for commit in tqdm(commits, desc='counting objects'):
                    with tmpDB.begin(write=True) as txn:
                        with txn.cursor() as curs:
                            notEmpty = curs.first()
                            while notEmpty:
                                notEmpty = curs.delete()
                    unpack_commit_ref(self._env.refenv, tmpDB, commit)
                    # handle column_names option
                    if column_names is None:
                        column_names = queries.RecordQuery(tmpDB).column_names()
                    for asetn in column_names:
                        cmtData_hashs = queries.RecordQuery(tmpDB).column_data_hashes(asetn)
                        allHashs.update(cmtData_hashs)
            finally:
                tmpDB.close()
        hashTxn = TxnRegister().begin_reader_txn(self._env.hashenv)
        try:
            m_schema_hash_map = defaultdict(list)
            for hashVal in allHashs:
                hashKey = hash_data_db_key_from_raw_key(hashVal.digest)
                hashRef = hashTxn.get(hashKey)
                be_loc = backend_decoder(hashRef)
                if be_loc.backend == '50':
                    m_schema_hash_map[be_loc.schema_hash].append(hashVal.digest)
        finally:
            TxnRegister().abort_reader_txn(self._env.hashenv)

        # -------------------- download missing data --------------------------

        total_nbytes_seen = 0
        total_data = sum(len(v) for v in m_schema_hash_map.values())
        with closing(self._client) as client, tqdm(total=total_data, desc='fetching data') as pbar:
            client: HangarClient  # type hint
            stop = False
            for schema in m_schema_hash_map.keys():
                hashes = set(m_schema_hash_map[schema])
                while (len(hashes) > 0) and (not stop):
                    ret = client.fetch_data(schema, hashes)
                    # max_num_bytes option
                    if isinstance(max_num_bytes, int):
                        for idx, r_kv in enumerate(ret):
                            total_nbytes_seen += r_kv[1].nbytes
                            if total_nbytes_seen >= max_num_bytes:
                                ret = ret[0:idx]
                                stop = True
                                break
                    saved_digests = CW.data(schema, ret)
                    pbar.update(len(saved_digests))
                    hashes = hashes.difference(set(saved_digests))

        move_process_data_to_store(self._repo_path, remote_operation=True)
        return commits

    def push(self, remote: str, branch: str,
             *, username: str = '', password: str = '') -> str:
        """push changes made on a local repository to a remote repository.

        This method is semantically identical to a ``git push`` operation.
        Any local updates will be sent to the remote repository.

        .. note::

            The current implementation is not capable of performing a
            ``force push`` operation. As such, remote branches with diverged
            histories to the local repo must be retrieved, locally merged,
            then re-pushed. This feature will be added in the near future.

        Parameters
        ----------
        remote : str
            name of the remote repository to make the push on.
        branch : str
            Name of the branch to push to the remote. If the branch name does
            not exist on the remote, the it will be created
        username : str, optional, kwarg-only
            credentials to use for authentication if repository push restrictions
            are enabled, by default ''.
        password : str, optional, kwarg-only
            credentials to use for authentication if repository push restrictions
            are enabled, by default ''.

        Returns
        -------
        str
            Name of the branch which was pushed
        """
        self.__verify_repo_initialized()
        try:
            address = heads.get_remote_address(self._env.branchenv, name=remote)
            cHEAD = heads.get_branch_head_commit(self._env.branchenv, branch)
        except (KeyError, ValueError) as e:
            raise e from None

        CR = ContentReader(self._env)
        self._client = HangarClient(envs=self._env,
                                    address=address,
                                    auth_username=username,
                                    auth_password=password)

        # ----------------- setup / validate operations -------------------

        with closing(self._client) as client:
            client: HangarClient  # type hinting for development
            CR: ContentReader
            c_bhistory = summarize.list_history(refenv=self._env.refenv,
                                                branchenv=self._env.branchenv,
                                                branch_name=branch)
            try:
                s_branch = client.fetch_branch_record(branch)
            except grpc.RpcError as rpc_error:
                # Do not raise if error due to branch not existing on server
                if rpc_error.code() != grpc.StatusCode.NOT_FOUND:
                    raise rpc_error
            else:
                sHEAD = s_branch.rec.commit
                if sHEAD == cHEAD:
                    warnings.warn(
                        f'NoOp: server HEAD: {sHEAD} == client HEAD: {cHEAD}', UserWarning)
                    return branch
                elif (sHEAD not in c_bhistory['order']) and (sHEAD != ''):
                    warnings.warn(
                        f'REJECTED: server branch has commits not on client', UserWarning)
                    return branch

            # --------------- negotiate missing data to send -------------------

            try:
                # First push op verifies user permissions if push restricted (NOT SECURE)
                res = client.push_find_missing_commits(branch)
                m_commits = res.commits
            except grpc.RpcError as rpc_error:
                if rpc_error.code() == grpc.StatusCode.PERMISSION_DENIED:
                    raise PermissionError(f'{rpc_error.code()}: {rpc_error.details()}')
                else:
                    raise rpc_error

            m_schemas = set()
            m_schema_hashs = defaultdict(set)
            with tempfile.TemporaryDirectory() as tempD:
                tmpDF = Path(tempD, 'test.lmdb')
                tmpDB = lmdb.open(path=str(tmpDF), **LMDB_SETTINGS)
                for commit in tqdm(m_commits, desc='counting objects'):
                    # share unpacked ref db between dependent methods
                    with tmpDB.begin(write=True) as txn:
                        with txn.cursor() as curs:
                            notEmpty = curs.first()
                            while notEmpty:
                                notEmpty = curs.delete()
                    unpack_commit_ref(self._env.refenv, tmpDB, commit)
                    # schemas
                    schema_res = client.push_find_missing_schemas(commit, tmpDB=tmpDB)
                    m_schemas.update(schema_res.schema_digests)
                    # data hashs
                    m_cmt_schema_hashs = defaultdict(list)
                    mis_hashes_sch = client.push_find_missing_hash_records(commit, tmpDB=tmpDB)
                    for hsh, schema in mis_hashes_sch:
                        m_cmt_schema_hashs[schema].append(hsh)
                    for schema, hashes in m_cmt_schema_hashs.items():
                        m_schema_hashs[schema].update(hashes)
                tmpDB.close()

            # ------------------------- send data -----------------------------

            # schemas
            for m_schema in tqdm(m_schemas, desc='pushing schemas'):
                schemaVal = CR.schema(m_schema)
                if not schemaVal:
                    raise KeyError(f'no schema with hash: {m_schema} exists')
                client.push_schema(m_schema, schemaVal)
            # data
            total_data = sum([len(v) for v in m_schema_hashs.values()])
            with tqdm(total=total_data, desc='pushing data') as p:
                for dataSchema, dataHashes in m_schema_hashs.items():
                    client.push_data(dataSchema, dataHashes, pbar=p)
                    p.update(1)
            # commit refs
            for commit in tqdm(m_commits, desc='pushing commit refs'):
                cmtContent = CR.commit(commit)
                if not cmtContent:
                    raise KeyError(f'no commit with hash: {commit} exists')
                client.push_commit_record(commit=cmtContent.commit,
                                          parentVal=cmtContent.cmtParentVal,
                                          specVal=cmtContent.cmtSpecVal,
                                          refVal=cmtContent.cmtRefVal)

            # --------------------------- At completion -----------------------

            # update local remote HEAD pointer
            branchHead = heads.get_branch_head_commit(self._env.branchenv, branch)
            try:
                client.push_branch_record(branch, branchHead)
            except grpc.RpcError as rpc_error:
                # Do not raise if error due to branch not existing on server
                if rpc_error.code() != grpc.StatusCode.ALREADY_EXISTS:
                    logger.warning(f'CODE: {rpc_error.code()} DETAILS:{rpc_error.details()}')
                else:
                    raise rpc_error
            else:
                cRemoteBranch = f'{remote}/{branch}'
                if cRemoteBranch not in heads.get_branch_names(self._env.branchenv):
                    heads.create_branch(branchenv=self._env.branchenv,
                                        name=cRemoteBranch,
                                        base_commit=branchHead)
                else:
                    heads.set_branch_head_commit(branchenv=self._env.branchenv,
                                                 branch_name=cRemoteBranch,
                                                 commit_hash=branchHead)
            return branch
