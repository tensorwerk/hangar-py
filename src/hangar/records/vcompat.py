import os
from os.path import join as pjoin

import lmdb

from . import parsing
from .parsing import VersionSpec
from .. import constants as c
from ..context import TxnRegister
from ..utils import pairwise


"""
Repository version finding methods
----------------------------------

methods to set and get repostory sotware versions
"""


def set_repository_software_version(branchenv: lmdb.Environment, ver_str: str,
                                    *, overwrite: bool = False) -> bool:
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
    versionKey = parsing.repo_version_db_key()
    ver_spec = parsing.repo_version_raw_spec_from_raw_string(v_str=ver_str)
    versionVal = parsing.repo_version_db_val_from_raw_val(v_spec=ver_spec)
    branchTxn = TxnRegister().begin_writer_txn(branchenv)
    try:
        success = branchTxn.put(versionKey, versionVal, overwrite=overwrite)
    finally:
        TxnRegister().commit_writer_txn(branchenv)
    return success


def get_repository_software_version_spec(branchenv: lmdb.Environment) -> VersionSpec:
    """Get the repository version specification tuple.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the head, branch, and version specs are stored

    Returns
    -------
    VersionSpec
        NamedTuple containing major, minor, and micro versions.

    Raises
    ------
    KeyError
        If no version key is set for the repository
    """
    versionKey = parsing.repo_version_db_key()
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        versionVal = branchTxn.get(versionKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    if versionVal is False:
        raise KeyError('No version string is set for the repository')
    else:
        version_val = parsing.repo_version_raw_val_from_db_val(versionVal)
        return version_val


def get_repository_software_version_str(branchenv: lmdb.Environment) -> str:
    """Get the repository version string representation.

    Parameters
    ----------
    branchenv : lmdb.Environment
        db where the head, branch, and version specs are stored

    Returns
    -------
    str
        semantic version style string representation.

    Raises
    ------
    KeyError
        If no version key is set for the repository
    """
    versionKey = parsing.repo_version_db_key()
    branchTxn = TxnRegister().begin_reader_txn(branchenv)
    try:
        versionVal = branchTxn.get(versionKey, default=False)
    finally:
        TxnRegister().abort_reader_txn(branchenv)

    if versionVal is False:
        raise KeyError('No version string is set for the repository')
    else:
        ver_spec = parsing.repo_version_raw_val_from_db_val(versionVal)
        ver_Str = parsing.repo_version_raw_string_from_raw_spec(ver_spec)
        return ver_Str


"""
Initial checking of repository versions
---------------------------------------
"""


def startup_check_repo_version(repo_path: os.PathLike) -> VersionSpec:
    """Determine repo version without having to have Environments ctx opened.

    Parameters
    ----------
    repo_path : os.PathLike
        path to the repository directory on disk

    Returns
    -------
    VersionSpec
        NamedTuple containing ints for `major`, `minor`, `micro`, semantic
        software version

    Raises
    ------
    RuntimeError
        If for whatever reason, the branch file does not exist on disk.
        Execution should not reach this point.
    """
    brch_fp = pjoin(repo_path, c.LMDB_BRANCH_NAME)
    if not os.path.isfile(brch_fp):
        msg = f'Hangar Internal Error, startup_check_repo_version did not find '\
              f'brch db at: {brch_fp}. Execution should never reach this point. '\
              f'Please report this error to Hangar developers.'
        raise RuntimeError(msg)

    branchenv = lmdb.open(path=brch_fp, readonly=True, create=False, **c.LMDB_SETTINGS)
    spec = get_repository_software_version_spec(branchenv=branchenv)
    branchenv.close()
    return spec


"""
Version compatibility checking
------------------------------

Right now this is a dummy method, which just returns true, but it is important to
have as we move to compatible changes in the future.
"""


incompatible_changes_after = [
    VersionSpec(major=0, minor=2, micro=0),
    VersionSpec(major=0, minor=3, micro=0),
    VersionSpec(major=0, minor=4, micro=0)]


def is_repo_software_version_compatible(repo_v: VersionSpec, curr_v: VersionSpec) -> bool:
    """Determine if the repo on disk and the current Hangar versions iscompatible.

    Parameters
    ----------
    repo_v : VersionSpec
        repository software writtern version.
    curr_v : VersionSpec
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