import time
import os
import random
from contextlib import closing
from typing import NamedTuple, Union, Tuple, Optional, List, Dict, Iterable
import multiprocessing as mp
import queue
from inspect import getcallargs
from pathlib import Path
from tempfile import TemporaryDirectory

from tqdm import tqdm
import numpy as np

from .records import hashs
from .records.column_parsers import (
    hash_data_raw_key_from_db_key,
    hash_data_db_key_from_raw_key,
    flat_data_db_key_from_names,
    nested_data_db_key_from_names,
    data_record_db_val_from_digest
)
from .utils import (
    ilen,
    grouper,
    is_valid_directory_path,
    isiterable,
)
from .columns.common import open_file_handles
from .txnctx import TxnRegister


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


class ContentDescriptionPrep(NamedTuple):
    column: str
    layout: str
    key: Union[Tuple[KeyType, KeyType], KeyType]
    digest: Optional[str] = None
    udf_iter_idx: Optional[int] = None
    skip: bool = False

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


class Task(NamedTuple):
    udf_kwargs: dict
    udf_iter_indices: Tuple[int]
    expected_digests: Tuple[str]


class ContentDescription(NamedTuple):
    column: str
    layout: str
    key: Union[Tuple[KeyType, KeyType], KeyType]
    digest: Optional[str] = None
    bespec: Optional[bytes] = None
    udf_iter_idx: Optional[int] = None
    skip: bool = False


def _check_user_input_func(columns, udf, udf_kwargs, *, prerun_check_percentage: float = 0.02):
    """Perform a few sanity tests to ensure kwargs and udf produces valid data.

    Parameters
    ----------
    columns:
        initialized columns object dict.
    udf:
        user provided function which takes some kwargs and generates one data sample.
    udf_kwargs: list/tuple
        two element list/tuple where

        element 0 --> sample key (flat layout) or size == 2 list/tuple of
        samples/subsample key (nested layout).

        element 1 --> dict of kwargs which are passed to user specified
        data reader func to retrieve a single samples data from disk.
    prerun_check_percentage: float, kwargonly, default=0.02
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

    num_choices_by_percent = round(len(udf_kwargs) * prerun_check_percentage)
    num_choices = max(max(2, num_choices_by_percent), 100)  # place upper/lower bounds.
    work_samples = random.choices(udf_kwargs, k=num_choices)
    num_failed = 0

    for kwargs in tqdm(work_samples, desc='Performing pre-run sanity check'):
        results = udf(**kwargs)
        for res in results:
            if res.data is None:
                num_failed += 1
                if num_failed >= num_choices * 0.2:
                    raise ValueError(f'num failed exceeds max {num_failed}')
                continue

            assert res.column in columns
            _col = columns[res.column]
            if _col.column_layout == 'flat':
                _col._set_arg_validate(res.key, res.data)
            else:
                _col._set_arg_validate(res.key[0], {res.key[1]: res.data})
    return True


class BatchProcessPrepare(mp.Process):
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

    def run(self):

        while True:
            try:
                udf_kwargs = self.in_queue.get(True, 2)
            except queue.Empty:
                break

            udf_kwargs_res = (
                (kwargs, self.udf(**kwargs)) for kwargs in udf_kwargs if isinstance(kwargs, dict)
            )
            content_digests = []

            for kwargs, udf_data_generator in udf_kwargs_res:
                # udf_data_generator: Iterable[UDF_Return]
                udf_kwarg_content_digests = []
                for udf_iter_idx, udf_return in enumerate(udf_data_generator):
                    _column = udf_return.column
                    _layout = self.column_layouts[_column]
                    _key = udf_return.key
                    _data = udf_return.data

                    if _data is None:
                        res = ContentDescriptionPrep(column=_column,
                                                     layout=_layout,
                                                     key=_key,
                                                     udf_iter_idx=udf_iter_idx,
                                                     skip=True)
                        udf_kwarg_content_digests.append(res)
                        continue
                    elif _key is None or kwargs is None:
                        continue
                    else:
                        _schema = self.schemas[_column]
                        iscompat = _schema.verify_data_compatible(_data)
                        if not iscompat.compatible:
                            raise ValueError(f'data for key {_key} incompatible due to {iscompat.reason}')
                        digest = _schema.data_hash_digest(_data)
                        res = ContentDescriptionPrep(column=_column,
                                                     layout=_layout,
                                                     key=_key,
                                                     udf_iter_idx=udf_iter_idx,
                                                     digest=digest)
                        udf_kwarg_content_digests.append(res)
                    content_digests.append((kwargs, udf_kwarg_content_digests))
            self.out_queue.put(content_digests)


class BatchProcessWriter(mp.Process):
    """Image Thread"""

    def __init__(self, udf, backends, schemas, column_layouts, tmp_pth, in_queue, out_queue, *args, **kwargs):
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
        self.layouts = column_layouts
        self.tmp_pth = tmp_pth

    def run(self):

        for column_name, column_backend in self.backends.items():
            be_instance_map = open_file_handles(
                backends=[column_backend], path=self.tmp_pth, mode='a', schema=self.schemas[column_name])
            be_instance = be_instance_map[column_backend]
            self.backend_instances[column_name] = be_instance

        while True:
            # Grabs image path from queue
            try:
                tasks_list = self.in_queue.get(True, 2)
            except queue.Empty:
                break

            tasks = (
                (task, self.udf(**task.udf_kwargs)) for task in tasks_list if isinstance(task, Task)
            )
            saved_key_location_digests = []
            for task, applied_udf in tasks:
                relevant_udf_indices = iter(task.udf_iter_indices)
                desired_udf_idx = next(relevant_udf_indices)
                for gen_idx, res in enumerate(applied_udf):
                    if gen_idx < desired_udf_idx:
                        continue

                    column = res.column
                    layout = self.layouts[column]
                    keys = res.key
                    data = res.data
                    iscompat = self.schemas[column].verify_data_compatible(data)
                    if not iscompat.compatible:
                        raise ValueError(f'data for key {keys} incompatible due to {iscompat.reason}')
                    digest = self.schemas[column].data_hash_digest(data)
                    location_spec = self.backend_instances[column].write_data(data)
                    res = ContentDescription(column, layout, keys, digest, location_spec)
                    saved_key_location_digests.append(res)
                    try:
                        desired_udf_idx = next(relevant_udf_indices)
                    except StopIteration:
                        break

            self.out_queue.put(saved_key_location_digests)


def run_prepare_recipe(column_layouts, schemas, udf, udf_kwargs, *, ncpu=0, batch_size=10):
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
        t = BatchProcessPrepare(
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


def run_write_recipe_data(tmp_dir: Path, columns, column_layouts, schemas, udf: object, recipe_tasks: List[Task], *, ncpu=0, batch_size=10):

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
        t = BatchProcessWriter(
            udf=udf, backends=backends, schemas=schemas, column_layouts=column_layouts,
            tmp_pth=tmp_dir, in_queue=in_queue, out_queue=out_queue)
        jobs.append(t)
        t.start()

    # Populate queue with batched arguments
    for i in range(q_size):
        initial_keys_kwargs_group = next(grouped_keys_kwargs)
        in_queue.put(initial_keys_kwargs_group)

    # collect outputs and fill queue with more work if low
    # terminate if no more work should be done.
    with tqdm(total=len(recipe_tasks), desc='Executing Data Import Recipe') as pbar:
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


def reduce_recipe_on_required_digests(recipe: List[Tuple[dict, List[ContentDescriptionPrep]]], hashenv):
    """Before writing, eliminate duplicate steps which would write identical
    data and steps which would write data already recorded in the repository.

    Parameters
    ----------
    recipe: List[Tuple[dict, List[ContentDescriptionPrep]]]

    Returns
    -------
    List[Task]:
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

    - step.skip will only ever == True if user's udf decides not
      to use some piece of data and to not even include the sample at
      all (returning None when valid arguments were passed in to the
      udf)
    """
    all_digests = []
    for udf_kwargs, content_prep_recipes in recipe:
        for content_prep in content_prep_recipes:
            if content_prep.skip:
                continue
            all_digests.append(content_prep.digest)
    recipe_digests = set(all_digests)


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
            _task = Task(udf_kwargs, tuple(udf_indices), tuple(expected_digests))
            task_list.append(_task)

    return task_list


def unify_recipe_contents(recipe: List[Tuple[dict, List[ContentDescriptionPrep]]]) -> List[ContentDescriptionPrep]:
    """

    Parameters
    ----------
    recipe: List[Tuple[dict, List[ContentDescriptionPrep]]]

    Returns
    -------
    List[ContentDescriptionPrep]
        Flat list where each element records a sample's column name, layout, keys, & digest.
    """
    unified_content = []
    for udf_kwargs, udf_contents in recipe:
        for content in udf_contents:
            if not content.skip:
                unified_content.append(content)
    return unified_content


def write_digest_to_bespec_mapping(executed_steps, hashenv, stagehashenv):

    digests_bespecs = []
    for spec in executed_steps:
        res = (spec.digest, spec.bespec)
        digests_bespecs.append(res)

    hashtxn = TxnRegister().begin_writer_txn(hashenv)
    stagehashtxn = TxnRegister().begin_writer_txn(stagehashenv)
    try:
        for rawdigest, bespec in digests_bespecs:
            dbDigest = hash_data_db_key_from_raw_key(rawdigest)
            stagehashtxn.put(dbDigest, bespec)
            hashtxn.put(dbDigest, bespec)
    finally:
        TxnRegister().commit_writer_txn(hashenv)
        TxnRegister().commit_writer_txn(stagehashenv)


def write_full_recipe_sample_key_to_digest_mapping(sample_steps, dataenv):

    db_kvs = []
    for step in sample_steps:
        step: ContentDescriptionPrep
        staging_key = step.db_record_key()
        staging_val = step.db_record_val()
        db_kvs.append((staging_key, staging_val))

    datatxn = TxnRegister().begin_writer_txn(dataenv)
    try:
        for dbk, dbv in db_kvs:
            datatxn.put(dbk, dbv)
    finally:
        TxnRegister().commit_writer_txn(dataenv)


def mock_hangar_directory_structure(dir_name: str) -> Path:
    from .constants import DIR_DATA, DIR_DATA_REMOTE, DIR_DATA_STAGE, DIR_DATA_STORE

    dirpth = Path(dir_name)
    is_valid_directory_path(dirpth)

    dirpth.joinpath(DIR_DATA_STORE).mkdir()
    dirpth.joinpath(DIR_DATA_STAGE).mkdir()
    dirpth.joinpath(DIR_DATA_REMOTE).mkdir()
    dirpth.joinpath(DIR_DATA).mkdir()
    return dirpth


def move_tmpdir_data_files_to_stagedir(repodir: Path, tmpdir: Path):
    import shutil
    import concurrent.futures
    from concurrent.futures import ThreadPoolExecutor

    from .constants import DIR_DATA, DIR_DATA_STAGE

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
                    stage_src_dest = (tmp_stage_fp, hangar_stage_fp)
                    task_list.append(stage_src_dest)

                    tmp_data_fp = tmp_data_dir.joinpath(be_pth.name, fpth.name)
                    hangar_data_fp = hangar_data_dir.joinpath(be_pth.name, fpth.name)
                    data_src_dest = (tmp_data_fp, hangar_data_fp)
                    task_list.append(data_src_dest)

    with ThreadPoolExecutor(max_workers=10) as e:
        future_result = [e.submit(shutil.move, str(src), str(dst)) for src, dst in task_list]
        for future in concurrent.futures.as_completed(future_result):
            res = future.result()

    return True


def run_bulk_import(repo, branch_name, column_names, udf, udf_kwargs,
                    *, ncpus=0, autocommit=True):
    """Perform a bulk import operation.

    Parameters
    ----------
    repo : Repository
    branch_name : str
    column_names : str
    udf : object
    udf_kwargs : List[Tuple[Union[Tuple[KeyType, KeyType], KeyType], dict]]
    ncpus : int, optional, default=0
    autocommit : bool, optional, default=True
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

        recipe = run_prepare_recipe(column_layouts, schemas, udf, udf_kwargs, ncpu=ncpus)
        print('Optimizing base recipe against repostiory contents & duplicated sample data.')
        reduced_recipe_tasks = reduce_recipe_on_required_digests(recipe, co._hashenv)

        hangardirpth = repo._repo_path
        if len(reduced_recipe_tasks) >= 1:
            print('Starting multiprocessed data importer.')
            with TemporaryDirectory(dir=str(hangardirpth)) as tmpdirname:
                tmpdirpth = mock_hangar_directory_structure(tmpdirname)
                written_data_steps = run_write_recipe_data(tmpdirpth, columns, column_layouts, schemas, udf, reduced_recipe_tasks, ncpu=4)
                move_tmpdir_data_files_to_stagedir(hangardirpth, tmpdirpth)
            write_digest_to_bespec_mapping(written_data_steps, co._hashenv, co._stagehashenv)
        else:
            print('No actions requiring the data importer remain after optimizations. Skipping this step...')

        unified_recipe = unify_recipe_contents(recipe)
        write_full_recipe_sample_key_to_digest_mapping(unified_recipe, co._stageenv)

        if autocommit:
            print(f'autocommiting changes.')
            co.commit(f'Auto commit after bulk import of {len(unified_recipe)} samples to '
                      f'column {column_names} on branch {branch_name}')
        else:
            print(f'skipping autocommit')

        print('Buld data importer operation completed successfuly')
        return True


