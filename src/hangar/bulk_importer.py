__all__ = ('UDF_Return', 'run_bulk_import')

import concurrent.futures
import multiprocessing as mp
import os
import queue
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from inspect import getcallargs
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import NamedTuple, Union, Tuple, List, Iterator, Callable, Dict, TYPE_CHECKING
from operator import attrgetter, methodcaller

import numpy as np
from tqdm import tqdm

from .columns.common import open_file_handles
from .constants import DIR_DATA, DIR_DATA_REMOTE, DIR_DATA_STAGE, DIR_DATA_STORE
from .records import hashs
from .records.column_parsers import (
    hash_data_raw_key_from_db_key,
    hash_data_db_key_from_raw_key,
    flat_data_db_key_from_names,
    nested_data_db_key_from_names,
    data_record_db_val_from_digest,
)
from .txnctx import TxnRegister
from .utils import (
    ilen,
    grouper,
    is_valid_directory_path,
    isiterable,
)

if TYPE_CHECKING:
    import lmdb
    from . import Repository

UDF_T = Callable[..., Iterator['UDF_Return']]
KeyType = Union[str, int]


class UDF_Return(NamedTuple):
    """User-Defined Function return container for bulk importer read functions

    Attributes
    ----------
    column: str
        column name to place data into
    key: Union[KeyType, Tuple[KeyType, KeyType]]
        key (single value) to place flat sample into, or 2-tuple of keys for nested samples
    data: Union[np.ndarray, str, bytes]
        piece of data to place in the column with the provided key.
    """
    column: str
    key: Union[KeyType, Tuple[KeyType, KeyType]]
    data: Union[np.ndarray, str, bytes]

    def __eq__(self, other):
        if not self.__class__.__name__ == other.__class__.__name__:
            raise NotImplementedError

        if self.column != other.column:
            return False
        if self.key != other.key:
            return False

        if isinstance(self.data, np.ndarray):
            if not np.array_equal(self.data, other.data):
                return False
        elif self.data != other.data:
            return False
        return True


class _ContentDescriptionPrep(NamedTuple):
    column: str
    layout: str
    key: Union[Tuple[KeyType, KeyType], KeyType]
    digest: str
    udf_iter_idx:int

    def db_record_key(self):
        if self.layout == 'nested':
            db_key = nested_data_db_key_from_names(self.column, self.key[0], self.key[1])
        elif self.layout == 'flat':
            db_key = flat_data_db_key_from_names(self.column, self.key)
        else:
            raise ValueError(f'unknown column layout value {self.layout} encountered while formating db record key')
        return db_key

    def db_record_val(self):
        return data_record_db_val_from_digest(self.digest)


class _Task(NamedTuple):
    udf_kwargs: dict
    udf_iter_indices: Tuple[int]
    expected_digests: Tuple[str]

    def num_steps(self):
        return len(self.udf_iter_indices)


class _WrittenContentDescription(NamedTuple):
    """Description of data content piece saved in the multprocess content writter

    Attributes
    ----------
    digest: str
        digest of the data piece written.
    bespec: bytes
        backend location spec in db formated bytes representation.
    """
    digest: str
    bespec: bytes


def _num_steps_in_task_list(task_list: List[_Task]) -> int:
    num_steps_method = methodcaller('num_steps')
    return sum(map(num_steps_method, task_list))


def _check_user_input_func(
        columns,
        udf: UDF_T,
        udf_kwargs: List[dict],
        *,
        prerun_check_percentage: float = 0.02
):
    """Perform a few sanity tests to ensure kwargs and udf produces valid data.

    Parameters
    ----------
    columns
        initialized columns object dict.
    udf : UDF_T
        user provided function which takes some kwargs and generates one data sample.
    udf_kwargs : List[dict]
        kwarg dicts to unpack into UDF via `udf(**kwargs)`
    prerun_check_percentage : float, kwargonly, default=0.02
        value between (0.0, 1.0) representing what percentage of items in the full
        work list should be selected (at random) to be processed by udf &
        verified against the column schema.

        This is meant to serve as a quick sanity check (to test if is success is even
        possible) before launching the full pipeline with multiple worker processes.
    """

    for kwargs in udf_kwargs:
        try:
            getcallargs(udf, **kwargs)
        except Exception as e:
            print(f'Invalid call args passed to {udf} with kwargs: {kwargs}')
            raise e from None

        if not isiterable(udf(**kwargs)):
            raise TypeError(f'Input udf {udf} is not Iterable')

    num_choices_by_percent = ceil(len(udf_kwargs) * prerun_check_percentage)
    num_choices = max(max(2, num_choices_by_percent), 100)  # place upper/lower bounds.
    work_samples = random.choices(udf_kwargs, k=num_choices)

    for kwargs in tqdm(work_samples,
                       desc=f'Performing pre-run sanity check on {num_choices} samples'):
        first_results = []
        for first_res in udf(**kwargs):
            if not first_res.__class__.__name__ == UDF_Return.__name__:
                raise TypeError(
                    f'UDF must yield only values of type {UDF_Return}, recieved '
                    f'{type(first_res)} from input kwargs: {kwargs}')
            if first_res.column not in columns:
                raise ValueError(
                    f'UDF_Return column value {first_res.column} was not specified in bulk '
                    f'loader input. kwargs triggering this UDF_Return failure: {kwargs}')
            _col = columns[first_res.column]
            if _col.column_layout == 'flat':
                _col._set_arg_validate(first_res.key, first_res.data)
            else:
                _col._set_arg_validate(first_res.key[0], {first_res.key[1]: first_res.data})
            first_results.append(first_res)

        _DeterministicError = ValueError(
            f'contents returned in subbsequent calls to UDF with identical kwargs'
            f'yielded different results. UDFs MUST generate deterministic results '
            f'for the given inputs. Input kwargs generating this result: {kwargs}')
        second_len = 0
        for second_idx, second_res in enumerate(udf(**kwargs)):
            if not second_res == first_results[second_idx]:
                raise _DeterministicError
            second_len += 1
        if second_len != len(first_results):
            raise _DeterministicError
    return True


class _BatchProcessPrepare(mp.Process):
    """Image Thread"""

    def __init__(self, udf, schemas, column_layouts, in_queue, out_queue, *args, **kwargs):
        """
        Parameters
        ----------
        udf:
            user provided function which takes some set of kwargs to generate one data sample
        schemas:
            initialized schema object for the column. This is required in order to properly
            calculate the data hash digests.
        column_name:
            name of the column we are ading data for.
        in_queue:
            multiprocessing.Queue object which passes in kwargs to read data for one sample via `udf`
            as well as sample/subsample names to assign to the resulting data.
            tuple in form of `(kwargs, (samplen, [subsamplen,]))`
        out_queue:
            multiprocessing.Queue object which passes back sample keys formated for storage in ref db,
            serialized location spec, and hash digest of read / saved data.
        """
        super().__init__(*args, **kwargs)
        self.column_layouts = column_layouts
        self.udf = udf
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schemas = schemas

    def _input_tasks(self):
        while True:
            try:
                udf_kwargs = self.in_queue.get(True, 2)
            except queue.Empty:
                break
            yield udf_kwargs

    def run(self):

        for udf_kwargs in self._input_tasks():
            udf_kwargs_res = (
                (kwargs, self.udf(**kwargs)) for kwargs in udf_kwargs if isinstance(kwargs, dict)
            )

            content_digests = []
            for kwargs, udf_data_generator in udf_kwargs_res:
                # udf_data_generator: Iterable[UDF_Return]
                if kwargs is None:
                    continue

                udf_kwarg_content_digests = []
                for udf_iter_idx, udf_return in enumerate(udf_data_generator):
                    _column = udf_return.column
                    _schema = self.schemas[_column]
                    _layout = self.column_layouts[_column]
                    _key = udf_return.key
                    _data = udf_return.data

                    iscompat = _schema.verify_data_compatible(_data)
                    if not iscompat.compatible:
                        raise ValueError(f'data for key {_key} incompatible due to {iscompat.reason}')
                    digest = _schema.data_hash_digest(_data)
                    res = _ContentDescriptionPrep(_column, _layout, _key, digest, udf_iter_idx)
                    udf_kwarg_content_digests.append(res)

                content_digests.append((kwargs, udf_kwarg_content_digests))
            self.out_queue.put(content_digests)


class _BatchProcessWriter(mp.Process):
    """Image Thread"""

    def __init__(self, udf, backends, schemas, tmp_pth, in_queue, out_queue, *args, **kwargs):
        """
        Parameters
        ----------
        udf:
            user provided function which takes some set of kwargs to generate one data sample
        backend_instances:
            initialized hangar backend class instance which will write data to disk for all
            samples read in via this thread.
        schemas:
            initialized schema object for the column. This is required in order to properly
            calculate the data hash digests.
        in_queue:
            multiprocessing.Queue object which passes in kwargs to read data for one sample via `udf`
            as well as sample/subsample names to assign to the resulting data.
            tuple in form of `(kwargs, (samplen, [subsamplen,]))`
        out_queue:
            multiprocessing.Queue object which passes back sample keys formated for storage in ref db,
            serialized location spec, and hash digest of read / saved data.
        """
        super().__init__(*args, **kwargs)
        self.udf = udf
        self.backends = backends
        self.backend_instances = {}
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schemas = schemas
        self.tmp_pth = tmp_pth

    def _setup_isolated_process_backend_instances(self):
        """
        Because backend FileHandle classes have a reader checkout only condition
        check set on __getstate__, we open individual classes (and file) in the actual
        processes they will be used in (rather than trying to pickle)
        """
        for column_name, column_backend in self.backends.items():
            be_instance_map = open_file_handles(
                backends=[column_backend], path=self.tmp_pth, mode='a', schema=self.schemas[column_name])
            be_instance = be_instance_map[column_backend]
            self.backend_instances[column_name] = be_instance

    def _input_tasks(self):
        while True:
            try:
                tasks_list = self.in_queue.get(True, 2)
            except queue.Empty:
                break
            yield tasks_list

    def run(self):
        self._setup_isolated_process_backend_instances()
        for tasks_list in self._input_tasks():
            tasks = (
                (task, self.udf(**task.udf_kwargs)) for task in tasks_list if isinstance(task, _Task)
            )
            written_digests_locations = []
            for task, applied_udf in tasks:
                relevant_udf_indices = iter(task.udf_iter_indices)
                desired_udf_idx = next(relevant_udf_indices)
                for gen_idx, res in enumerate(applied_udf):
                    if gen_idx < desired_udf_idx:
                        continue

                    column = res.column
                    data = res.data
                    iscompat = self.schemas[column].verify_data_compatible(data)
                    if not iscompat.compatible:
                        raise ValueError(
                            f'data for task {task} yielding data for column: {res.column},'
                            f'key: {res.key} is is incompatible with column schema. Reason'
                            f'provided is: {iscompat.reason}'
                        )
                    digest = self.schemas[column].data_hash_digest(data)
                    location_spec = self.backend_instances[column].write_data(data)
                    res = _WrittenContentDescription(digest, location_spec)
                    written_digests_locations.append(res)
                    try:
                        desired_udf_idx = next(relevant_udf_indices)
                    except StopIteration:
                        break

            self.out_queue.put(written_digests_locations)


def _run_prepare_recipe(column_layouts, schemas, udf, udf_kwargs, *, ncpu=0, batch_size=10):
    if ncpu <= 0:
        ncpu = os.cpu_count() // 2
    q_size = ncpu * 2

    # setup queues
    in_queue = mp.Queue(maxsize=q_size)
    out_queue = mp.Queue(maxsize=q_size)

    dummy_groups = grouper(udf_kwargs, batch_size)
    n_queue_tasks = ilen(dummy_groups)
    grouped_keys_kwargs = grouper(udf_kwargs, batch_size)

    # start worker processes
    out, jobs = [], []
    for i in range(ncpu):
        t = _BatchProcessPrepare(
            udf=udf, schemas=schemas, column_layouts=column_layouts, in_queue=in_queue, out_queue=out_queue)
        jobs.append(t)
        t.start()

    # Populate queue with batched arguments
    for i in range(q_size):
        initial_keys_kwargs_group = next(grouped_keys_kwargs)
        in_queue.put(initial_keys_kwargs_group)

    # collect outputs and fill queue with more work if low
    # terminate if no more work should be done.
    with tqdm(total=len(udf_kwargs), desc='Constructing task recipe') as pbar:
        ngroups_processed = 0
        remaining = True
        while remaining is True:
            data_key_location_hash_digests = out_queue.get(True, 10)
            ngroups_processed += 1
            for saved in data_key_location_hash_digests:
                pbar.update(1)
                out.append(saved)
            try:
                while not in_queue.full():
                    keys_kwargs = next(grouped_keys_kwargs)
                    in_queue.put(keys_kwargs)
            except StopIteration:
                if ngroups_processed == n_queue_tasks:
                    remaining = False
    for j in jobs:
        j.join()
    return out


def _run_write_recipe_data(
        tmp_dir: Path, columns, schemas, udf: UDF_T, recipe_tasks: List[_Task],
        *,
        ncpu=0, batch_size=10
):
    if ncpu <= 0:
        ncpu = os.cpu_count() // 2
    q_size = ncpu * 2

    # setup queues
    in_queue = mp.Queue(maxsize=q_size)
    out_queue = mp.Queue(maxsize=q_size)

    dummy_groups = grouper(recipe_tasks, batch_size)
    n_queue_tasks = ilen(dummy_groups)
    grouped_keys_kwargs = grouper(recipe_tasks, batch_size)

    # start worker processes
    out, jobs = [], []
    for i in range(ncpu):
        backends = {}
        for col_name, column in columns.items():
            backends[col_name] = column.backend
        t = _BatchProcessWriter(
            udf=udf, backends=backends, schemas=schemas, tmp_pth=tmp_dir,
            in_queue=in_queue, out_queue=out_queue)
        jobs.append(t)
        t.start()

    # Populate queue with batched arguments
    for i in range(q_size):
        initial_keys_kwargs_group = next(grouped_keys_kwargs)
        in_queue.put(initial_keys_kwargs_group)

    # collect outputs and fill queue with more work if low
    # terminate if no more work should be done.
    nsteps = _num_steps_in_task_list(recipe_tasks)
    with tqdm(total=nsteps, desc='Executing Data Import Recipe') as pbar:
        ngroups_processed = 0
        remaining = True
        while remaining is True:
            data_key_location_hash_digests = out_queue.get()
            ngroups_processed += 1
            for saved in data_key_location_hash_digests:
                pbar.update(1)
                out.append(saved)
            try:
                while not in_queue.full():
                    keys_kwargs = next(grouped_keys_kwargs)
                    in_queue.put(keys_kwargs)
            except StopIteration:
                if ngroups_processed == n_queue_tasks:
                    remaining = False
    for j in jobs:
        j.join()
    return out


def _unify_recipe_contents(recipe: List[Tuple[dict, List[_ContentDescriptionPrep]]]) -> List[_ContentDescriptionPrep]:
    """Flatten and isolate all ContentDescriptionPrep in flat recipe list.

    Parameters
    ----------
    recipe: List[Tuple[dict, List[_ContentDescriptionPrep]]]

    Returns
    -------
    List[_ContentDescriptionPrep]
        Flat list where each element records a sample's column name, layout, keys, & digest.
    """
    unified_content = []
    for udf_kwargs, udf_contents in recipe:
        for content in udf_contents:
            unified_content.append(content)
    return unified_content


def _reduce_recipe_on_required_digests(recipe: List[Tuple[dict, List[_ContentDescriptionPrep]]], hashenv):
    """Before writing, eliminate duplicate steps which would write identical
    data and steps which would write data already recorded in the repository.

    Parameters
    ----------
    recipe: List[Tuple[dict, List[_ContentDescriptionPrep]]]

    Returns
    -------
    List[_Task]:
        reduced recipe tasks to serve as input for the mp writer.

    Notes
    -----
    - Any number of samples may be added which have unique keys/kwargs,
      but whose udf returns identical data. To avoid writing
      identical data to disk multiple times, we select just one sample
      (at random) for each unique digest in the recipe. We write the
      data to disk alongside the digest -> backend spec mapping. Once
      all unique data sample steps are written, we use the full sample
      step recipe to record the sample name -> digest mapping without
      needing to actually process the data that a full step execution
      would have produced produces.

    - A similar exclusion is made for steps which produce data which is
      already recorded in the repository. The only difference is that
      we do not process writing of these steps at all (for any sample).
      Since the digest -> backend spec map already exists, we just need
      to process to key -> digest mapping.
    """
    recipe_contents = _unify_recipe_contents(recipe)
    digest_getter = attrgetter('digest')
    recipe_digests = set(map(digest_getter, recipe_contents))

    hq = hashs.HashQuery(hashenv)
    recipe_digests_db = set(map(hash_data_db_key_from_raw_key, recipe_digests))
    existing_digests_db = hq.intersect_keys_db(recipe_digests_db)
    missing_digests_db = recipe_digests_db.difference(existing_digests_db)
    missing_digests = set(map(hash_data_raw_key_from_db_key, missing_digests_db))

    remaining_digests = set(missing_digests)
    task_list = []
    for udf_kwargs, content_prep_recipes in recipe:
        task_udf_kwargs = None  # Set to value if kwargs should be included
        udf_indices = []
        expected_digests = []
        for content_prep in content_prep_recipes:
            _digest = content_prep.digest
            if _digest in remaining_digests:
                udf_indices.append(content_prep.udf_iter_idx)
                expected_digests.append(_digest)
                task_udf_kwargs = udf_kwargs
                remaining_digests.remove(_digest)
        if task_udf_kwargs:
            _task = _Task(udf_kwargs, tuple(udf_indices), tuple(expected_digests))
            task_list.append(_task)

    return task_list


def _write_digest_to_bespec_mapping(
        executed_steps: List[_WrittenContentDescription],
        hashenv: 'lmdb.Environment',
        stagehashenv: 'lmdb.Environment'
):
    """Write written content digests and bespec to hash and stagehash db.
    """
    digests_bespecs = []
    for spec in executed_steps:
        dbSpec = spec.bespec
        dbDigest = hash_data_db_key_from_raw_key(spec.digest)
        digests_bespecs.append((dbDigest, dbSpec))

    hashtxn = TxnRegister().begin_writer_txn(hashenv)
    stagehashtxn = TxnRegister().begin_writer_txn(stagehashenv)
    try:
        for dbDigest, dbSpec in digests_bespecs:
            stagehashtxn.put(dbDigest, dbSpec)
            hashtxn.put(dbDigest, dbSpec)
    finally:
        TxnRegister().commit_writer_txn(hashenv)
        TxnRegister().commit_writer_txn(stagehashenv)


def _write_full_recipe_sample_key_to_digest_mapping(
        sample_steps: List[_ContentDescriptionPrep],
        dataenv: 'lmdb.Environment'
):
    """Write sample name -> digest key/value pairs in checkout data (stage) db.
    """
    db_kvs = []
    for step in sample_steps:
        staging_key = step.db_record_key()
        staging_val = step.db_record_val()
        db_kvs.append((staging_key, staging_val))

    datatxn = TxnRegister().begin_writer_txn(dataenv)
    try:
        for dbk, dbv in db_kvs:
            datatxn.put(dbk, dbv)
    finally:
        TxnRegister().commit_writer_txn(dataenv)


def _mock_hangar_directory_structure(dir_name: str) -> Path:
    """Setup folder structure of hangar repo within a temporary directory path.

    Parameters
    ----------
    dir_name
        directory path to create the hangar dir structure in.

    Returns
    -------
    mocked hangar directory path.
    """
    dirpth = Path(dir_name)
    is_valid_directory_path(dirpth)

    dirpth.joinpath(DIR_DATA_STORE).mkdir()
    dirpth.joinpath(DIR_DATA_STAGE).mkdir()
    dirpth.joinpath(DIR_DATA_REMOTE).mkdir()
    dirpth.joinpath(DIR_DATA).mkdir()
    return dirpth


def _move_tmpdir_data_files_to_repodir(repodir: Path, tmpdir: Path):
    tmp_stage_dir = tmpdir.joinpath(DIR_DATA_STAGE)
    tmp_data_dir = tmpdir.joinpath(DIR_DATA)
    hangar_stage_dir = repodir.joinpath(DIR_DATA_STAGE)
    hangar_data_dir = repodir.joinpath(DIR_DATA)

    task_list = []
    for be_pth in tmp_stage_dir.iterdir():
        if be_pth.is_dir():
            for fpth in be_pth.iterdir():
                if fpth.is_file() and not fpth.stem.startswith('.'):
                    tmp_stage_fp = tmp_stage_dir.joinpath(be_pth.name, fpth.name)
                    hangar_stage_fp = hangar_stage_dir.joinpath(be_pth.name, fpth.name)
                    task_list.append((tmp_stage_fp, hangar_stage_fp))

                    if hangar_stage_fp.suffix.endswith('dir'):
                        # data directories (ie. lmdb) have a stage_file suffix ending in
                        # 'dir' (for lmdb this is a suffix of `.lmdbdir`). The stage_file
                        # stem is the directory name which needs to be moved.
                        tmp_data_fp = tmp_data_dir.joinpath(be_pth.name, fpth.stem)
                        hangar_data_fp = hangar_data_dir.joinpath(be_pth.name, fpth.stem)
                    else:
                        # files are 1:1 copy of stage_file:data_file
                        tmp_data_fp = tmp_data_dir.joinpath(be_pth.name, fpth.name)
                        hangar_data_fp = hangar_data_dir.joinpath(be_pth.name, fpth.name)
                    task_list.append((tmp_data_fp, hangar_data_fp))

    _MoveException = False
    with ThreadPoolExecutor(max_workers=10) as e:
        future_result = [e.submit(shutil.move, str(src), str(dst)) for src, dst in task_list]
        for future in concurrent.futures.as_completed(future_result):
            try:
                future.result()
            except Exception:
                _MoveException = future.exception()
    if _MoveException is not False:
        print(f'Error encountered while persisting imported data in hangar repo directory.')
        print(f'Begining change set roll back.')
        for _, dest_fp in task_list:
            if dest_fp.is_file():
                os.remove(str(dest_fp))
                print(f'- {dest_fp}')
            elif dest_fp.is_dir():
                shutil.rmtree(str(dest_fp))
                print(f'- {dest_fp}')
        print(f'Roll back completed successfully')
        raise _MoveException
    return True


def run_bulk_import(
        repo: 'Repository',
        branch_name: str,
        column_names: List[str],
        udf: UDF_T,
        udf_kwargs: List[dict],
        *,
        ncpus: int = 0,
        autocommit: bool = True
):
    """Perform a bulk import operation.

    Parameters
    ----------
    repo : Repository
        Initialized repository object to import data into.
    branch_name : str
        Name of the branch to checkout and import data into.
    column_names : List[str]
        Names of all columns which data should be saved to.
    udf : UDF_T
        User-Defined Function (generator style; yielding an arbitrary number
        of values when iterated on) which is passed an unpacked kwarg dict as input
        and yields a single :class:`~.UDF_Return` instance at a time when iterated over.
        Cannot contain
    udf_kwargs : List[dict]
    ncpus : int, optional, default=0
    autocommit : bool, optional, default=True

    Returns
    -------
    """
    with closing(repo.checkout(write=True, branch=branch_name)) as co:
        columns, column_layouts, schemas = {}, {}, {}
        for name in column_names:
            _col = co.columns[name]
            _schema = _col._schema
            columns[name] = _col
            column_layouts[name] = _col.column_layout
            schemas[name] = _schema

        print(f'Validating Reader Function and Argument Input')
        _check_user_input_func(columns, udf, udf_kwargs)

        recipe = _run_prepare_recipe(column_layouts, schemas, udf, udf_kwargs, ncpu=ncpus)
        print('Unifying naieve recipe task set.')
        unified_recipe = _unify_recipe_contents(recipe)
        print('Pruning redundant ingest steps & eliminating tasks for data previously stored in hangar.')
        reduced_recipe = _reduce_recipe_on_required_digests(recipe, co._hashenv)

        nsteps_reduced_recipe = _num_steps_in_task_list(reduced_recipe)
        optim_percent = ((len(unified_recipe) - nsteps_reduced_recipe) / len(unified_recipe)) * 100
        print(f'Reduced recipe workload tasks by: {optim_percent:.2f}%')
        print(f' - Num tasks for naieve ingest  : {len(unified_recipe)}')
        print(f' - Num tasks after optimization : {nsteps_reduced_recipe}')

        hangardirpth = repo._repo_path
        if len(reduced_recipe) >= 1:
            print('Starting multiprocessed data importer.')
            with TemporaryDirectory(dir=str(hangardirpth)) as tmpdirname:
                tmpdirpth = _mock_hangar_directory_structure(tmpdirname)
                written_data_steps = _run_write_recipe_data(tmpdirpth, columns, schemas, udf, reduced_recipe, ncpu=4)
                print(f'Finalizing written data pieces in hangar repo directory...')
                _move_tmpdir_data_files_to_repodir(hangardirpth, tmpdirpth)
            _write_digest_to_bespec_mapping(written_data_steps, co._hashenv, co._stagehashenv)
        else:
            print('No actions requiring the data importer remain after optimizations. Skipping this step...')

        print(f'Mapping full recipe requested via UDF to optimized task set actually processed.')
        _write_full_recipe_sample_key_to_digest_mapping(unified_recipe, co._stageenv)

        if autocommit:
            print(f'autocommiting changes.')
            co.commit(f'Auto commit after bulk import of {len(unified_recipe)} samples to '
                      f'column {column_names} on branch {branch_name}')
        else:
            print(f'skipping autocommit')

        print('Buld data importer operation completed successfuly')
        return True


