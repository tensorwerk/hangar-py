"""Bulk importer methods to ingest large quantities of data into Hangar.

The following module is designed to address challenges inherent to writing
massive amounts of data to a hangar repository via the standard API. Since
write-enabled checkouts are limited to processing in a single thread, the
time required to import hundreds of Gigabytes (or Terabytes) of data into
Hangar (from external sources) can become prohibitivly long. This module
implements a multi-processed importer which reduces import time nearly
linearly with the number of CPU cores allocated on a machine.

There are a number of challenges to overcome:

1. How to validate data against a column schema?

    - Does the column exist?

    - Are the key(s) valid?

    - Is the data a valid type/shape/precision valid for the the selected
      column schema?

2. How to handle duplicated data?

    -  If an identical piece of data is recorded in the repository already,
       only record the sample reference (do not write the data to disk again).

    - If the bulk import method would write identical pieces of data to the
      repository multiple times, and the data does not already exist, then that
      piece of content should only be written to disk once. Only sample
      references should be saved after that.

3. How to handle transactionality?

    - What happens if some column, sample keys, or data piece is invalid and
      cannot be written as desired?

    - How to rollback partial changes if the process is inturupted in
      the middle of a bulk import operation?

4. How to limit memory usage if many processes are trying to load and
   write large tensors?


Rough outline of steps:

    1. Validate UDF & Argument Signature

    2. Read, Validate, and Hash UDF results --> Task Recipe

    3. Prune Recipe

    4. Read, Validate, Write Data to Isolated Backend Storage

    5. Record Sample References in Isolated Environment

    6. If all successful, make isolated data known to repository core,
       otherwise abort to starting state.
"""
__all__ = ('UDF_Return', 'run_bulk_import')

import concurrent.futures
import multiprocessing as mp
import multiprocessing.queues as mpq
import os
import pickle
import queue
import random
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from inspect import signature, isgeneratorfunction
from math import ceil
from operator import attrgetter, methodcaller
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    NamedTuple, Union, Tuple, List, Iterator,
    Callable, Dict, Optional, TYPE_CHECKING
)

import cloudpickle
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
from .utils import grouper, is_valid_directory_path, bound

if TYPE_CHECKING:
    import lmdb
    from . import Repository
    from .typesystem.base import ColumnBase
    from .columns import ModifierTypes


UDF_T = Callable[..., Iterator['UDF_Return']]
KeyType = Union[str, int]


# ----------------- User Facing Potions of Bulk Data Loader -------------------


# noinspection PyUnresolvedReferences
class UDF_Return(NamedTuple):
    """User-Defined Function return container for bulk importer read functions

    Attributes
    ----------
    column
        column name to place data into
    key
        key to place flat sample into, or 2-tuple of keys for nested samples
    data
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
    """Perform a bulk import operation from a given user-defined function.

    In order to provide for arbitrary input data sources along with ensuring
    the core promises of hangar hold we require the following from users:

    Define some arbitrary function (ie "user-defined function" / "UDF") which
    accepts some arguments and yields data. The UDF must be a generator function,
    yielding only values which are of :class:`UDF_Return` type. The results
    yielded by the UDF must be deterministic for a given set of  inputs. This
    includes all values of the :class:`UDF_Return` (``columns`` and ``keys``,
    as well as ``data``).

    A list of input arguments to the UDF must be provided, this is formatted as a
    sequence  (list / tuple) of keyword-arg dictionaries, each of which must be
    valid when unpacked and bound to the UDF signature. Additionally, all columns
    must be  specified up front. If any columns are named a `UDF_Return`
    which were not pre-specified, the entire operation will fail.

    !!! note

        - This is an all-or-nothing operation, either all data is successfully
          read, validated, and written to the storage backends, or none of it
          is. A single maleformed key or data type/shape will cause the entire
          import operation to abort.

        - The input kwargs should be fairly small (of no consequence to load
          into memory), data out should be large. The results of the UDF
          will only be stored in memory for a very short period (just the time
          it takes to be validated against the column schema and compressed /
          flushed to disk).

        - Every step of the process is executed as a generator, lazily loading
          data the entire way. If possible, we recomend writing the UDF such that
          data is not allocated in memory before it is ready to be yielded.

        - If it is possible, the task recipe will be pruned and optimized in such
          a way that iteration over the UDF will be short circuted during the
          second pass (writing data to the backend). As this can greatly reduce
          processing time, we recomend trying to yield data pieces which are likely
          to be unique first from the UDF.

    !!! warning

        Please be aware that these methods should not be executed within a
        Jupyter Notebook / Jupyter Lab when running the bulk importer at scale.
        The internal implemenation makes significant use of multiprocess Queues
        for work distribution and recording. The heavy loads placed on the system
        have been observed to place strain on Jupyters ZeroMQ implementation,
        resulting in random failures which may or may not even display a traceback
        to indicate failure mode.

        A small sample set of data can be used within jupyter to test an
        implementation without problems, but for full scale operations it is best
        run in a script with the operations protected by a ``__main__`` block.

    Examples
    --------
        >>> import os
        >>> import numpy as np
        >>> from PIL import Image
        >>> from hangar.bulk_importer import UDF_Return, run_bulk_import
        >>> def image_loader(file_path):
        ...     im = Image.open(file_name)
        ...     arr = np.array(im.resize(512, 512))
        ...     im_record = UDF_Return(column='image', key=(category, sample), data=arr)
        ...     yield im_record
        ...
        ...     root, sample_file = os.path.split(file_path)
        ...     category = os.path.dirname(root)
        ...     sample_name, _ = os.path.splitext(sample_file)
        ...     path_record = UDF_Return(column='file_str', key=(category, sample_name), data=file_path)
        ...     yield path_record
        ...
        >>> udf_kwargs = [
        ...     {'file_path': '/foo/cat/image_001.jpeg'},
        ...     {'file_path': '/foo/cat/image_002.jpeg'},
        ...     {'file_path': '/foo/dog/image_001.jpeg'},
        ...     {'file_path': '/foo/bird/image_011.jpeg'},
        ...     {'file_path': '/foo/bird/image_003.jpeg'}
        ... ]
        >>> repo = Repository('foo/path/to/repo')
        >>> run_bulk_import(
        ...     repo, branch_name='master', column_names=['file_str', 'image'],
        ...     udf=image_loader, udf_kwargs=udf_kwargs)

    However, the following will not work, since the output is non-deterministic.

        >>> from hangar.bulk_importer import UDF_Return, run_bulk_import
        >>> def nondeterminstic(x, y):
        ...     first = str(x * y)
        ...     yield UDF_Return(column='valstr', key=f'{x}_{y}', data=first)
        ...
        ...     second = str(x * y * random())
        ...     yield UDF_Return(column='valstr', key=f'{x}_{y}', data=second)
        ...
        >>> udf_kwargs = [
        ...     {'x': 1, 'y': 2},
        ...     {'x': 1, 'y': 3},
        ...     {'x': 2, 'y': 4},
        ... ]
        >>> run_bulk_import(
        ...     repo, branch_name='master', column_names=['valstr'],
        ...     udf=image_loader, udf_kwargs=udf_kwargs)
        Traceback (most recent call last):
          `File "<stdin>", line 1, in <module>`
        TypeError: contents returned in subbsequent calls to UDF with identical
          kwargs yielded different results. UDFs MUST generate deterministic
          results for the given inputs. Input kwargs generating this result:
          {'x': 1, 'y': 2}.

    Not all columns must be returned from every input to the UDF, the number of
    data pieces yielded can also vary arbitrarily (so long as the results are
    deterministic for a particular set of inputs)

        >>> import numpy as np
        >>> from hangar.bulk_importer import UDF_Return, run_bulk_import
        >>> def maybe_load(x_arr, y_arr, sample_name, columns=['default']):
        ...     for column in columns:
        ...         arr = np.multiply(x_arr, y_arr)
        ...         yield UDF_Return(column=column, key=sample_name, data=arr)
        ...     #
        ...     # do some strange processing which only outputs another column sometimes
        ...     if len(columns) == 1:
        ...         other = np.array(x_arr.shape) * np.array(y_arr.shape)
        ...         yield UDF_Return(column='strange_column', key=sample_name, data=other)
        ...
        >>> udf_kwargs = [
        ...     {'x_arr': np.arange(10), 'y_arr': np.arange(10) + 1, 'sample_name': 'sample_1'},
        ...     {'x_arr': np.arange(10), 'y_arr': np.arange(10) + 1, 'sample_name': 'sample_2', 'columns': ['foo', 'bar', 'default']},
        ...     {'x_arr': np.arange(10) * 2, 'y_arr': np.arange(10), 'sample_name': 'sample_3'},
        ... ]
        >>> run_bulk_import(
        ...     repo, branch_name='master',
        ...     column_names=['default', 'foo', 'bar', 'strange_column'],
        ...     udf=maybe_load, udf_kwargs=udf_kwargs)

    Parameters
    ----------
    repo
        Initialized repository object to import data into.
    branch_name
        Name of the branch to checkout and import data into.
    column_names
        Names of all columns which data should be saved to.
    udf
        User-Defined Function (generator style; yielding an arbitrary number
        of values when iterated on) which is passed an unpacked kwarg dict as input
        and yields a single :class:`~.UDF_Return` instance at a time when iterated over.
        Cannot contain
    udf_kwargs
        A sequence of keyword argument dictionaries which are individually unpacked
        as inputs into the user-defined function (UDF). the keyword argument dictionaries
    ncpus
        Number of Parallel processes to read data files & write to hangar backend stores
        in. If <= 0, then the default is set to ``num_cpus / 2``. The value of this
        parameter should never exceed the total CPU count of the system. Import time
        scales mostly linearly with ncpus. Optimal performance is achieved by balancing
        memory usage of the ``UDF`` function and backend storage writer processes against
        the total system memory.
        generally increase linearly up to
    autocommit
        Control whether a commit should be made after successfully importing the
        specified data to the staging area of the branch.
    """
    _BATCH_SIZE = 10  # TODO: Is this necessary?

    columns: Dict[str, 'ModifierTypes'] = {}
    column_layouts: Dict[str, str] = {}
    schemas: Dict[str, 'ColumnBase'] = {}

    with closing(repo.checkout(write=True, branch=branch_name)) as co:
        for name in column_names:
            _col = co.columns[name]
            _schema = _col._schema
            columns[name] = _col
            column_layouts[name] = _col.column_layout
            schemas[name] = _schema

        print(f'Validating Reader Function and Argument Input')
        _check_user_input_func(columns=columns, udf=udf, udf_kwargs=udf_kwargs)
        serialized_udf = _serialize_udf(udf)

        ncpu = _process_num_cpus(ncpus)
        print(f'Using {ncpu} worker processes')

        recipe = _run_prepare_recipe(
            column_layouts=column_layouts,
            schemas=schemas,
            udf=serialized_udf,
            udf_kwargs=udf_kwargs,
            ncpu=ncpu,
            batch_size=_BATCH_SIZE)
        print('Unifying naieve recipe task set.')
        unified_recipe = _unify_recipe_contents(recipe)
        print('Pruning redundant steps & eliminating tasks on data stored in hangar.')
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
                written_data_steps = _run_write_recipe_data(
                    tmp_dir=tmpdirpth,
                    columns=columns,
                    schemas=schemas,
                    udf=serialized_udf,
                    recipe_tasks=reduced_recipe,
                    ncpu=ncpu,
                    batch_size=_BATCH_SIZE)
                print(f'Finalizing written data pieces in hangar repo directory...')
                _move_tmpdir_data_files_to_repodir(repodir=hangardirpth, tmpdir=tmpdirpth)
            _write_digest_to_bespec_mapping(
                executed_steps=written_data_steps,
                hashenv=co._hashenv,
                stagehashenv=co._stagehashenv)
        else:
            print('No actions requiring the data import remain after optimizations.')

        print(f'Mapping full recipe requested via UDF to optimized task set actually processed.')
        _write_full_recipe_sample_key_to_digest_mapping(sample_steps=unified_recipe, dataenv=co._stageenv)

        if autocommit:
            print(f'autocommiting changes.')
            co.commit(f'Auto commit after bulk import of {len(unified_recipe)} samples to '
                      f'column {column_names} on branch {branch_name}')
        else:
            print(f'skipping autocommit')

        print('Buld data importer operation completed successfuly')
        return


# ---------------- Internal Implementation of Bulk Data Loader ----------------


class _ContentDescriptionPrep(NamedTuple):
    column: str
    layout: str
    key: Union[Tuple[KeyType, KeyType], KeyType]
    digest: str
    udf_iter_idx: int

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
    udf_iter_indices: Tuple[int, ...]
    expected_digests: Tuple[str, ...]

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


def _serialize_udf(udf: UDF_T) -> bytes:
    raw = cloudpickle.dumps(udf, protocol=pickle.HIGHEST_PROTOCOL)
    return raw


def _deserialize_udf(raw: bytes) -> UDF_T:
    udf = cloudpickle.loads(raw)
    return udf


def _process_num_cpus(ncpus: int) -> int:
    """Determine how many workerprocesses to spin up in bulk importer

    Parameters
    ----------
    ncpus: int
        User specified number of worker processes. If <= 0 set to num CPU cores / 2.

    Returns
    -------
    int
    """
    node_cpus = os.cpu_count()
    if ncpus <= 0:
        cpu_try = ceil(node_cpus / 2)
        ncpus = bound(1, node_cpus, cpu_try)
    elif ncpus > node_cpus:
        warnings.warn(
            f'Input number of CPUs exceeds maximum on node. {ncpus} > {node_cpus}',
            category=UserWarning
        )
    return ncpus


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
    if not isgeneratorfunction(udf):
        raise TypeError(f'UDF {udf} is not a user defined generator function.')

    try:
        _raw_udf = _serialize_udf(udf)
        _deserialized = _deserialize_udf(_raw_udf)
    except (pickle.PicklingError, pickle.UnpicklingError) as e:
        my_err = RuntimeError(f'Could not pickle/unpickle UDF {udf} using cloudpickle.')
        raise my_err from e

    sig = signature(udf)
    for idx, kwargs in enumerate(tqdm(udf_kwargs, desc='Validating argument signature')):
        try:
            sig.bind(**kwargs)
        except TypeError as e:
            my_err = TypeError(f'Value {kwargs} at index {idx} of `udf_kwargs` is invalid.')
            raise my_err from e

    num_choices_by_percent = ceil(len(udf_kwargs) * prerun_check_percentage)
    num_choices = bound(2, 100, num_choices_by_percent)
    work_samples = random.choices(udf_kwargs, k=num_choices)
    for kwargs in tqdm(work_samples, desc=f'Performing pre-run sanity check'):
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


class _MPQueue(mpq.Queue):
    """Interuptable Multiprocess Queue class which does not throw errors.
    """

    def __init__(self, *args, **kwargs):
        ctx = mp.get_context()
        super().__init__(*args, **kwargs, ctx=ctx)

    def safe_get(self, timeout=0.5):
        try:
            if timeout is None:
                return self.get(False)
            else:
                return self.get(True, timeout)
        except queue.Empty:
            return None

    def safe_put(self, item, timeout=0.5) -> bool:
        try:
            self.put(item, False, timeout)
            return True
        except queue.Full:
            return False

    def drain(self):
        item = self.safe_get()
        while item:
            yield item
            item = self.safe_get()

    def safe_close(self) -> int:
        num_left = sum(1 for __ in self.drain())
        self.close()
        self.join_thread()
        return num_left


class _BatchProcessPrepare(mp.Process):

    def __init__(
            self,
            udf: bytes,
            schemas: Dict[str, 'ColumnBase'],
            column_layouts: Dict[str, str],
            in_queue: _MPQueue,
            out_queue: _MPQueue,
            *args, **kwargs
    ):
        """Read all data generated by all UDF(**udf_kwargs) input.

        Validates reader function works, yields correct UDF_Return type, keys/columns
        are compatible names, data schema is suitable for column, and calculates digest
        of data and index location into UDF iteration.

        Parameters
        ----------
        udf
            user provided function yielding UDF_Return instances when iterated over
        schemas
            dict mapping column names -> initialized schema objects. This is required in
            order to properly calculate the data hash digests.
        column_layouts
            dict mapping column names -> column layout string
        in_queue
            queue contianing work pieces (kwargs) to process via UDF `mp.Queue[List[dict]]`
        out_queue
            queue containing mp.Queue[List[Tuple[dict, List[_ContentDescriptionPrep]]]]
            mapping kwargs -> content description read in.
        """
        super().__init__(*args, **kwargs)
        self.column_layouts = column_layouts
        self._udf_raw: bytes = udf
        self.udf: Optional[UDF_T] = None
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schemas = schemas

    def _setup(self):
        self.udf = _deserialize_udf(self._udf_raw)

    def _input_tasks(self) -> Iterator[List[dict]]:
        udf_kwargs = self.in_queue.safe_get(timeout=2.0)
        while udf_kwargs is not None:
            yield udf_kwargs
            udf_kwargs = self.in_queue.safe_get()

    def run(self):
        self._setup()
        for udf_kwargs in self._input_tasks():
            udf_kwargs_res = (
                (kwargs, self.udf(**kwargs)) for kwargs in udf_kwargs if isinstance(kwargs, dict)
            )
            content_digests = []
            for kwargs, udf_data_generator in udf_kwargs_res:
                if kwargs is None:
                    continue

                udf_kwarg_content_digests = []
                for udf_iter_idx, udf_return in enumerate(udf_data_generator):
                    _column = udf_return.column
                    _key = udf_return.key
                    _data = udf_return.data
                    _schema = self.schemas[_column]
                    _layout = self.column_layouts[_column]

                    iscompat = _schema.verify_data_compatible(_data)
                    if not iscompat.compatible:
                        raise ValueError(f'data for key {_key} incompatible due to {iscompat.reason}')
                    digest = _schema.data_hash_digest(_data)
                    res = _ContentDescriptionPrep(_column, _layout, _key, digest, udf_iter_idx)
                    udf_kwarg_content_digests.append(res)
                content_digests.append((kwargs, udf_kwarg_content_digests))
            self.out_queue.safe_put(content_digests)


def _run_prepare_recipe(
        column_layouts: Dict[str, str],
        schemas: Dict[str, 'ColumnBase'],
        udf: bytes,
        udf_kwargs: List[dict],
        *,
        ncpu: int = 0,
        batch_size: int = 10
) -> List[Tuple[dict, List[_ContentDescriptionPrep]]]:

    # Setup & populate queue with batched arguments
    in_queue = _MPQueue()
    out_queue = _MPQueue()
    n_queue_tasks = ceil(len(udf_kwargs) / batch_size)
    for keys_kwargs in grouper(udf_kwargs, batch_size):
        in_queue.safe_put(keys_kwargs)

    out, jobs = [], []
    try:
        # start worker processes
        for _ in range(ncpu):
            t = _BatchProcessPrepare(
                udf=udf,
                schemas=schemas,
                column_layouts=column_layouts,
                in_queue=in_queue,
                out_queue=out_queue)
            jobs.append(t)
            t.start()

        # collect outputs and fill queue with more work if low
        # terminate if no more work should be done.
        with tqdm(total=len(udf_kwargs), desc='Constructing task recipe') as pbar:
            ngroups_processed = 0
            while ngroups_processed < n_queue_tasks:
                data_key_location_hash_digests = out_queue.safe_get(timeout=30)
                if data_key_location_hash_digests is None:
                    continue
                ngroups_processed += 1
                for saved in data_key_location_hash_digests:
                    pbar.update(1)
                    out.append(saved)

        in_queue.safe_close()
        out_queue.safe_close()
        for j in jobs:
            try:
                j.join(timeout=0.2)
            except mp.TimeoutError:
                j.terminate()
    except (KeyboardInterrupt, InterruptedError):
        in_queue.safe_close()
        out_queue.safe_close()
        while jobs:
            j = jobs.pop()
            if j.is_alive():
                print(f'terminating PID {j.pid}')
                j.terminate()
            else:
                exitcode = j.exitcode
                if exitcode:
                    print(f'PID {j.pid} exitcode: {exitcode}')
        raise
    return out


class _BatchProcessWriter(mp.Process):

    def __init__(
            self,
            udf: bytes,
            backends: Dict[str, str],
            schemas: Dict[str, 'ColumnBase'],
            tmp_pth: Path,
            in_queue: _MPQueue,
            out_queue: _MPQueue,
            *args, **kwargs
    ):
        """

        Parameters
        ----------
        udf
            user provided function yielding UDF_Return instances when iterated over.
        backends
            dict mapping column name -> backend code.
        schemas
            dict mapping column names -> initialized schema objects. This is required in
            order to properly calculate the data hash digests.
        tmp_pth
            tempdir path to write data to
        in_queue
            grouped task lists `mp.Queue[List[_Task]]`
        out_queue
            written content description `mp.Queue[List[_WrittenContentDescription]]`
        args
        kwargs
        """
        super().__init__(*args, **kwargs)
        self._udf_raw: bytes = udf
        self.udf: Optional[UDF_T] = None
        self.backends = backends
        self.backend_instances = {}
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.schemas = schemas
        self.tmp_pth = tmp_pth

    def _setup(self):
        """
        Because backend FileHandle classes have a reader checkout only condition
        check set on __getstate__, we open individual classes (and file) in the actual
        processes they will be used in (rather than trying to pickle)
        """
        self.udf = _deserialize_udf(self._udf_raw)

        for column_name, column_backend in self.backends.items():
            be_instance_map = open_file_handles(
                backends=[column_backend],
                path=self.tmp_pth,
                mode='a',
                schema=self.schemas[column_name])
            be_instance = be_instance_map[column_backend]
            self.backend_instances[column_name] = be_instance

    def _input_tasks(self) -> Iterator[List[_Task]]:
        tasks_list = self.in_queue.safe_get(timeout=2)
        while tasks_list is not None:
            yield tasks_list
            tasks_list = self.in_queue.safe_get()

    @contextmanager
    def _enter_backends(self):
        try:
            for be in self.backend_instances.keys():
                self.backend_instances[be].__enter__()
            yield
        finally:
            for be in self.backend_instances.keys():
                self.backend_instances[be].__exit__()

    def run(self):
        self._setup()
        with self._enter_backends():
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
                        digest = self.schemas[column].data_hash_digest(data)
                        location_spec = self.backend_instances[column].write_data(data)
                        res = _WrittenContentDescription(digest, location_spec)
                        written_digests_locations.append(res)
                        try:
                            desired_udf_idx = next(relevant_udf_indices)
                        except StopIteration:
                            break
                self.out_queue.safe_put(written_digests_locations)


def _run_write_recipe_data(
        tmp_dir: Path,
        columns: Dict[str, 'ModifierTypes'],
        schemas: Dict[str, 'ColumnBase'],
        udf: bytes,
        recipe_tasks: List[_Task],
        *,
        ncpu=0,
        batch_size=10
) -> List[_WrittenContentDescription]:

    # Setup & populate queue with batched arguments
    in_queue = _MPQueue()
    out_queue = _MPQueue()
    n_queue_tasks = ceil(len(recipe_tasks) / batch_size)
    for keys_kwargs in grouper(recipe_tasks, batch_size):
        in_queue.put_nowait(keys_kwargs)

    out, jobs = [], []
    try:
        # start worker processes
        backends = {}
        for col_name, column in columns.items():
            backends[col_name] = column.backend
        for _ in range(ncpu):
            t = _BatchProcessWriter(
                udf=udf,
                backends=backends,
                schemas=schemas,
                tmp_pth=tmp_dir,
                in_queue=in_queue,
                out_queue=out_queue)
            jobs.append(t)
            t.start()

        # collect outputs and fill queue with more work if low
        # terminate if no more work should be done.
        nsteps = _num_steps_in_task_list(recipe_tasks)
        with tqdm(total=nsteps, desc='Executing Data Import Recipe') as pbar:
            ngroups_processed = 0
            while ngroups_processed < n_queue_tasks:
                data_key_location_hash_digests = out_queue.safe_get(timeout=30)
                if data_key_location_hash_digests is None:
                    continue
                ngroups_processed += 1
                for saved in data_key_location_hash_digests:
                    pbar.update(1)
                    out.append(saved)
        in_queue.safe_close()
        out_queue.safe_close()
        for j in jobs:
            try:
                j.join(timeout=0.2)
            except mp.TimeoutError:
                j.terminate()
    except (KeyboardInterrupt, InterruptedError):
        in_queue.safe_close()
        out_queue.safe_close()
        while jobs:
            j = jobs.pop()
            if j.is_alive():
                print(f'terminating PID {j.pid}')
                j.terminate()
            else:
                exitcode = j.exitcode
                if exitcode:
                    print(f'PID {j.pid} exitcode: {exitcode}')
        raise
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

    _MoveException = None
    num_workers = bound(5, 32, os.cpu_count() + 4)
    with ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix='hangar_import_shutil') as e:
        future_result = [e.submit(shutil.move, str(src), str(dst)) for src, dst in task_list]
        for future in concurrent.futures.as_completed(future_result):
            if future.exception() is not None:
                _MoveException = future.exception()

    if _MoveException is not None:
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
