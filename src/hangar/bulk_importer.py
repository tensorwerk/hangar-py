import time
import os
import random
from contextlib import closing
from typing import NamedTuple, Union, Tuple, Optional, List
import multiprocessing as mp
import queue
from collections import defaultdict
from inspect import getcallargs

from tqdm.auto import tqdm

from .records import hashs
from .records.column_parsers import (
    hash_data_db_key_from_raw_key,
    flat_data_db_key_from_names,
    nested_data_db_key_from_names,
    data_record_db_val_from_digest
)
from .utils import is_suitable_user_key, ilen, grouper
from .columns.common import open_file_handles
from .txnctx import TxnRegister


KeyType = Union[str, int]


class ContentDescription(NamedTuple):
    column: str
    layout: str
    key: Union[Tuple[KeyType, KeyType], KeyType]
    kwargs: dict
    digest: Optional[str] = None
    bespec: Optional[bytes] = None
    skip: bool = False


def _check_user_input_args(schema, keys_kwargs):
    """Validate user input specifying sample keys and reader func args/kwargs

    Parameters
    ----------
    schema:
        column schema
    keys_kwargs: list/tuple
        two element list/tuple where

        element 0 --> sample key (flat layout) or size == 2 list/tuple of
        samples/subsample key (nested layout).

        element 1 --> dict of kwargs which are passed to user specified
        data reader func to retrieve a single samples data from disk.
    """
    if not isinstance(keys_kwargs, (list, tuple)):
        raise TypeError(f'expected arg of type list/tuple, recieved {type(keys_kwargs)}')
    elif len(keys_kwargs) <= 1:
        raise ValueError(f'batch input must specify more than 1 data sample')

    sample_keys = []
    for sample_key_kwargs in keys_kwargs:
        if len(sample_key_kwargs) != 2:
            raise ValueError(
                f'all containers defining samples of batch input must be length 2 '
                f'where idx 0 == sample key and idx 1 == reader function kwargs dict.'
                f'Recieved {sample_key_kwargs} of length {len(sample_key_kwargs)}')

        key, kwargs = sample_key_kwargs
        if not isinstance(kwargs, dict):
            raise TypeError(f'reader input must be passed kwargs dict, not {type(kwargs)}')
        if schema.column_layout == 'nested':
            if not isinstance(key, (list, tuple)) or len(key) != 2:
                raise TypeError(f'Key {key} not list/tuple of length 2 for nested column.')
            if not all([is_suitable_user_key(k) for k in key]):
                raise ValueError(f'one of key values in {key} is not valid')
        elif schema.column_layout == 'flat':
            if not is_suitable_user_key(key):
                raise ValueError(f'key {key} is not valid')
        else:
            raise RuntimeError(f'schema layout {schema.column_layout} not supported.')
        sample_keys.append(tuple(key))

    # check that each key is unique
    unique_sample_keys = set(sample_keys)
    if len(sample_keys) != len(unique_sample_keys):
        raise ValueError(f'sample keys cannot be duplicated')
    return True


def _check_user_input_func(schema, read_func, keys_kwargs):
    """Perform a few sanity tests to ensure kwargs and reader_func produces valid data.

    Parameters
    ----------
    schema:
        initialized schema object for the column.
    read_func:
        user provided function which takes some kwargs and generates one data sample.
    keys_kwargs: list/tuple
        two element list/tuple where

        element 0 --> sample key (flat layout) or size == 2 list/tuple of
        samples/subsample key (nested layout).

        element 1 --> dict of kwargs which are passed to user specified
        data reader func to retrieve a single samples data from disk.
    """
    for sample, reader_args in keys_kwargs:
        try:
            getcallargs(read_func, **reader_args)
        except Exception as e:
            print(sample, reader_args)
            raise e from None

    failcount = 0
    numtries = min(10, len(keys_kwargs))
    maxfail = round(0.75 * numtries)
    for i in range(numtries):
        sample, func_kwargs = random.choice(keys_kwargs)
        res = read_func(**func_kwargs)
        if res is None:
            failcount += 1
            continue
        iscompat = schema.verify_data_compatible(res)
        if not iscompat.compatible:
            raise ValueError(f'Key {sample} kwargs {func_kwargs} data invalid {iscompat.reason}')
    if failcount >= maxfail:
        raise ValueError(f'reader func returned None on {failcount} / {numtries} samples.')

    return True


class BatchProcessPrepare(mp.Process):
    """Image Thread"""

    def __init__(self, read_func, schema, column_name, in_queue, out_queue, *args, **kwargs):
        """
        Parameters
        ----------
        read_func:
            user provided function which takes some set of kwargs to generate one data sample
        schema:
            initialized schema object for the column. This is required in order to properly
            calculate the data hash digests.
        column_name:
            name of the column we are ading data for.
        in_queue:
            multiprocessing.Queue object which passes in kwargs to read data for one sample via `read_func`
            as well as sample/subsample names to assign to the resulting data.
            tuple in form of `(kwargs, (samplen, [subsamplen,]))`
        out_queue:
            multiprocessing.Queue object which passes back sample keys formated for storage in ref db,
            serialized location spec, and hash digest of read / saved data.
        """
        super().__init__(*args, **kwargs)
        self.column_name = column_name
        self.layout = schema.column_layout
        self.read_func = read_func
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schema = schema

    def run(self):
        while True:
            try:
                sample_keys_kwargs = self.in_queue.get(True, 2)
            except queue.Empty:
                break

            sample_keys_and_data = (
                (keys, kwargs, self.read_func(**kwargs)) for keys, kwargs in sample_keys_kwargs
                if ((keys is not None) and (kwargs is not None))
            )
            content_digests = []
            for keys, kwargs, data in sample_keys_and_data:
                if data is None:
                    res = ContentDescription(self.column_name, self.layout, keys, kwargs, skip=True)
                    content_digests.append(res)
                    continue
                elif keys is None or kwargs is None:
                    continue

                iscompat = self.schema.verify_data_compatible(data)
                if not iscompat.compatible:
                    raise ValueError(f'data for key {keys} incompatible due to {iscompat.reason}')
                digest = self.schema.data_hash_digest(data)
                res = ContentDescription(self.column_name, self.layout, keys, kwargs, digest)
                content_digests.append(res)
            self.out_queue.put(content_digests)


class BatchProcessWriter(mp.Process):
    """Image Thread"""

    def __init__(self, read_func, backend_instance, schema, column_name, in_queue, out_queue, *args, **kwargs):
        """
        Parameters
        ----------
        read_func:
            user provided function which takes some set of kwargs to generate one data sample
        backend_instance:
            initialized hangar backend class instance which will write data to disk for all
            samples read in via this thread.
        schema:
            initialized schema object for the column. This is required in order to properly
            calculate the data hash digests.
        column_name:
            name of the column we are ading data for.
        in_queue:
            multiprocessing.Queue object which passes in kwargs to read data for one sample via `read_func`
            as well as sample/subsample names to assign to the resulting data.
            tuple in form of `(kwargs, (samplen, [subsamplen,]))`
        out_queue:
            multiprocessing.Queue object which passes back sample keys formated for storage in ref db,
            serialized location spec, and hash digest of read / saved data.
        """
        super().__init__(*args, **kwargs)
        self.column_name = column_name
        self.read_func = read_func
        self.backend_instance = backend_instance
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schema = schema
        self.layout = schema.column_layout

    def run(self):
        while True:
            # Grabs image path from queue
            try:
                sample_keys_kwargs = self.in_queue.get(True, 2)
            except queue.Empty:
                break

            sample_keys_and_data = (
                (keys, kwargs, self.read_func(**kwargs)) for keys, kwargs in sample_keys_kwargs
                if ((keys is not None) and (kwargs is not None))
            )

            saved_key_location_digests = []
            for keys, kwargs, data in sample_keys_and_data:
                if keys is None or kwargs is None or data is None:
                    continue

                iscompat = self.schema.verify_data_compatible(data)
                if not iscompat.compatible:
                    raise ValueError(f'data for key {keys} incompatible due to {iscompat.reason}')
                digest = self.schema.data_hash_digest(data)
                location_spec = self.backend_instance.write_data(data)
                res = ContentDescription(self.column_name, self.layout, keys, kwargs, digest, location_spec)
                saved_key_location_digests.append(res)
            self.out_queue.put(saved_key_location_digests)


def run_prepare_recipe(column, reader_func, keys_reader_kwargs, *, ncpu=0, batch_size=10):
    if ncpu <= 0:
        ncpu = os.cpu_count() // 2
    q_size = ncpu * 2

    # setup queues
    in_queue = mp.Queue(maxsize=q_size)
    out_queue = mp.Queue(maxsize=q_size)

    dummy_groups = grouper(keys_reader_kwargs, batch_size)
    n_queue_tasks = ilen(dummy_groups)
    grouped_keys_kwargs = grouper(keys_reader_kwargs, batch_size)

    # start worker processes
    out, jobs = [], []
    for i in range(ncpu):
        column_name = column.column
        schema = column._schema
        t = BatchProcessPrepare(
            read_func=reader_func, schema=schema, column_name=column_name, in_queue=in_queue, out_queue=out_queue)
        jobs.append(t)
        t.start()

    # Populate queue with batched arguments
    for i in range(q_size):
        initial_keys_kwargs_group = next(grouped_keys_kwargs)
        in_queue.put(initial_keys_kwargs_group)

    # collect outputs and fill queue with more work if low
    # terminate if no more work should be done.
    with tqdm(total=len(keys_reader_kwargs), desc='Constructing Import Recipe') as pbar:
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
                    time.sleep(0.0001)
            except StopIteration:
                if ngroups_processed == n_queue_tasks:
                    remaining = False
    for j in jobs:
        j.join()
    return out


def run_write_recipe_data(column, reader_func, keys_reader_kwargs, *, ncpu=0, batch_size=10):

    if ncpu <= 0:
        ncpu = os.cpu_count() // 2
    q_size = ncpu * 2

    # setup queues
    in_queue = mp.Queue(maxsize=q_size)
    out_queue = mp.Queue(maxsize=q_size)

    dummy_groups = grouper(keys_reader_kwargs, batch_size)
    n_queue_tasks = ilen(dummy_groups)
    grouped_keys_kwargs = grouper(keys_reader_kwargs, batch_size)

    # start worker processes
    out, jobs = [], []
    for i in range(ncpu):
        column_name = column.column
        backend = column.backend
        schema = column._schema
        pth = column._path
        be_instance_map = open_file_handles(
            backends=[backend], path=pth, mode='a', schema=column._schema)
        be_instance = be_instance_map[backend]
        t = BatchProcessWriter(
            read_func=reader_func, backend_instance=be_instance, schema=schema,
            column_name=column_name, in_queue=in_queue, out_queue=out_queue)
        jobs.append(t)
        t.start()

    # Populate queue with batched arguments
    for i in range(q_size):
        initial_keys_kwargs_group = next(grouped_keys_kwargs)
        in_queue.put(initial_keys_kwargs_group)

    # collect outputs and fill queue with more work if low
    # terminate if no more work should be done.
    with tqdm(total=len(keys_reader_kwargs), desc='Executing Data Import Recipe') as pbar:
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
                    time.sleep(0.0001)
            except StopIteration:
                if ngroups_processed == n_queue_tasks:
                    remaining = False
    for j in jobs:
        j.join()
    return out


def reduce_recipe_on_required_digests(recipe: List[ContentDescription], hashenv):
    """Before writing, eliminate duplicate steps which would write identical
    data and steps which would write data already recorded in the repository.

    Parameters
    ----------
    recipe: List[ContentDescription]

    Returns
    -------
    sample_steps:
        List[ContentDescription] containing all sample steps to write.
    reduced_recipe_keys_kwargs:
        List[Tuple[KeyType, dict]] input for the mp writer

    Notes
    -----
    - Any number of samples may be added which have unique keys/kwargs,
      but whose reader_func returns identical data. To avoid writing
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

    - step.skip will only ever == True if user's read_func decides not
      to use some piece of data and to not even include the sample at
      all (returning None when valid arguments were passed in to the
      reader_func)
    """
    sample_steps = [step for step in recipe if not step.skip]
    recipe_digests = set((el.digest for el in sample_steps))

    hq = hashs.HashQuery(hashenv)
    repository_digests = set(hq.list_all_hash_keys_raw())

    missing_digests = recipe_digests.difference(repository_digests)
    missing_data_sample_steps = [step for step in sample_steps if step.digest in missing_digests]

    digest_sample_steps_map = defaultdict(list)
    for spec in missing_data_sample_steps:
        digest_sample_steps_map[spec.digest].append(spec)

    single_steps_producing_unique_digests = []
    for spec, *_ in digest_sample_steps_map.values():
        single_steps_producing_unique_digests.append(spec)

    reduced_recipe_keys_kwargs = []
    for spec in single_steps_producing_unique_digests:
        item = (spec.key, spec.kwargs)
        reduced_recipe_keys_kwargs.append(item)

    return sample_steps, reduced_recipe_keys_kwargs


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
        if step.layout == 'nested':
            sample, subsample = step.key
            staging_key = nested_data_db_key_from_names(step.column, sample, subsample)
        elif step.layout == 'flat':
            staging_key = flat_data_db_key_from_names(step.column, step.key)
        else:
            raise ValueError(f'layout {step.layout} not supported')
        staging_val = data_record_db_val_from_digest(step.digest)
        db_kvs.append((staging_key, staging_val))

    datatxn = TxnRegister().begin_writer_txn(dataenv)
    try:
        for dbk, dbv in db_kvs:
            datatxn.put(dbk, dbv)
    finally:
        TxnRegister().commit_writer_txn(dataenv)


def run_bulk_import(repo, branch_name, column_name, reader_func, keys_kwargs,
                    *, ncpus=0, autocommit=True):
    """Perform a bulk import operation.

    Parameters
    ----------
    repo : Repository
    branch_name : str
    column_name : str
    reader_func : object
    keys_kwargs : List[Tuple[Union[Tuple[KeyType, KeyType], KeyType], dict]]
    ncpus : int, optional, default=0
    autocommit : bool, optional, default=True
    """
    with closing(repo.checkout(write=True, branch=branch_name)) as co:
        if co.diff.status() != 'CLEAN':
            raise RuntimeError(f'Cannot perform operation with uncommited changes in staging area.')

        column = co.columns[column_name]
        schema = column._schema

        print(f'Validating Reader Function and Argument Input')
        _check_user_input_args(schema, keys_kwargs)
        _check_user_input_func(schema, reader_func, keys_kwargs)

        recipe = run_prepare_recipe(column, reader_func, keys_kwargs, ncpu=ncpus)
        print('Optimizing base recipe against repostiory contents & duplicated sample data.')
        recipe_steps, reduced_keys_kwargs = reduce_recipe_on_required_digests(recipe, co._hashenv)

        if len(reduced_keys_kwargs) >= 1:
            print('Starting multiprocessed data importer.')
            try:
                written_data_steps = run_write_recipe_data(column, reader_func, reduced_keys_kwargs, ncpu=ncpus)
                write_digest_to_bespec_mapping(written_data_steps, co._hashenv, co._stagehashenv)
            except Exception:
                co.reset_staging_area(force=True)
                raise
        else:
            print('No actions requiring the data importer remain after optimizations. Skipping this step...')


        try:
            print(f'Adding changes to staging area.')
            write_full_recipe_sample_key_to_digest_mapping(recipe_steps, co._stageenv)
        except Exception:
            co.reset_staging_area(force=True)
            raise

        if autocommit:
            print(f'autocommiting changes.')
            co.commit(f'Auto commit after bulk import of {len(recipe_steps)} samples to '
                      f'column {column_name} on branch {branch_name}')
        else:
            print(f'skipping autocommit')

        print('Buld data importer operation completed successfuly')
        return True


