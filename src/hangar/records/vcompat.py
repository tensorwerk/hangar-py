from pathlib import Path

import lmdb

from .parsing import (
    repo_version_db_key,
    repo_version_db_val_from_raw_val,
    repo_version_raw_spec_from_raw_string,
    repo_version_raw_val_from_db_val,
)
from .._version import Version
from ..constants import LMDB_SETTINGS, LMDB_BRANCH_NAME
from ..txnctx import TxnRegister
from ..utils import pairwise


def set_repository_software_version(branchenv: lmdb.Environment,
                                    ver_str: str,
                                    *,
                                    overwrite: bool = False) -> bool:
    """Write the repository software version to a particular value

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the head, branch, and version specs are stored
    ver_str : str
        semantic version style string representing version (ie. "0.1.0",
        "1.2.1", etc)
    overwrite : bool, optional
        If True, replace current value with new value; If False, do not
        overwrite if this key exists, by default False

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    versionKey = repo_version_db_key()
    ver_spec = repo_version_raw_spec_from_raw_string(v_str=ver_str)
    versionVal = repo_version_db_val_from_raw_val(v_spec=ver_spec)
    branchTxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        success = branchTxn.put(versionKey, versionVal, overwrite=overwrite)
    finally:
        TxnRegister().commit_writer_txn(branchenv)
    return success


def get_repository_software_version_spec(branchenv: lmdb.Environment) -> Version:
    """Get the repository version specification tuple.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the head, branch, and version specs are stored

    Returns
    -------
    Version
        This class abstracts handling of a project’s versions. A Version
        instance is comparison aware and can be compared and sorted using the
        standard Python interfaces.

    Raises
    ------
    KeyError
        If no version key is set for the repository
    """
    versionKey = repo_version_db_key()
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        versionVal = branchTxn.get(versionKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    if versionVal is False:
        raise KeyError('No version string is set for the repository')
    else:
        version_val = repo_version_raw_val_from_db_val(versionVal)
        return version_val


"""
Initial checking of repository versions
---------------------------------------
"""


def startup_check_repo_version(repo_path: Path) -> Version:
    """Determine repo version without having to have Environments ctx opened.

    Parameters
    ----------
    repo_path : Path
        path to the repository directory on disk

    Returns
    -------
    Version
        This class abstracts handling of a project’s versions. A Version
        instance is comparison aware and can be compared and sorted using the
        standard Python interfaces.

    Raises
    ------
    RuntimeError
        If for whatever reason, the branch file does not exist on disk.
        Execution should not reach this point.
    """
    brch_fp = repo_path.joinpath(LMDB_BRANCH_NAME)
    if not brch_fp.is_file():
        msg = f'Hangar Internal Error, startup_check_repo_version did not find '\
              f'brch db at: {brch_fp}. Execution should never reach this point. '\
              f'Please report this error to Hangar developers.'
        raise RuntimeError(msg)

    branchenv = lmdb.open(path=str(brch_fp), readonly=True, create=False, **LMDB_SETTINGS)
    spec = get_repository_software_version_spec(branchenv=branchenv)
    branchenv.close()
    return spec


incompatible_changes_after = [
    Version('0.2.0'),
    Version('0.3.0'),
    Version('0.4.0'),
    Version('0.5.0.dev0'),
    Version('0.5.0.dev1'),
]


def is_repo_software_version_compatible(repo_v: Version, curr_v: Version) -> bool:
    """Determine if the repo on disk and the current Hangar versions iscompatible.

    Parameters
    ----------
    repo_v : Version
        repository software writtern version.
    curr_v : Version
        currently active software version specification

    Returns
    -------
    bool
        True if compatible, False if not.
    """
    for start, end in pairwise(incompatible_changes_after):
        if (repo_v >= start) and (repo_v < end):
            if (curr_v < start) or (curr_v >= end):
                return False
            elif (curr_v >= start) and (curr_v < end):
                return True
    if (repo_v >= end) and (curr_v < end):
        return False
    return True
