import atexit
import os
import weakref
from contextlib import suppress
from functools import partial
from collections import namedtuple
from uuid import uuid4

import lmdb
import warnings

from .arrayset import Arraysets
from .diff import ReaderUserDiff, WriterUserDiff
from .merger import select_merge_algorithm
from .metadata import MetadataReader, MetadataWriter
from .records import commiting, hashs, heads
from .utils import cm_weakref_obj_proxy


class ReaderCheckout(object):
    """Checkout the repository as it exists at a particular branch.

    This class is instantiated automatically from a repository checkout
    operation. This object will govern all access to data and interaction methods
    the user requests.

        >>> co = repo.checkout()
        >>> isinstance(co, ReaderCheckout)
        True

    If a commit hash is provided, it will take precedent over the branch name
    parameter. If neither a branch not commit is specified, the staging
    environment's base branch ``HEAD`` commit hash will be read.

        >>> co = repo.checkout(commit='foocommit')
        >>> co.commit_hash
        'foocommit'
        >>> co.close()
        >>> co = repo.checkout(branch='testbranch')
        >>> co.commit_hash
        'someothercommithashhere'
        >>> co.close()

    Unlike :class:`WriterCheckout`, any number of :class:`ReaderCheckout`
    objects can exist on the repository independently. Like the
    ``write-enabled`` variant, the :meth:`close` method should be called after
    performing the necessary operations on the repo. However, as there is no
    concept of a ``lock`` for ``read-only`` checkouts, this is just to free up
    memory resources, rather than changing recorded access state.

    In order to reduce the chance that the python interpreter is shut down
    without calling :meth:`close`,  - a common mistake during ipython / jupyter
    sessions - an `atexit <https://docs.python.org/3/library/atexit.html>`_
    hook is registered to :meth:`close`. If properly closed by the user, the
    hook is unregistered after completion with no ill effects. So long as a the
    process is NOT terminated via non-python ``SIGKILL``, fatal internal python
    error, or or special ``os exit`` methods, cleanup will occur on interpreter
    shutdown and resources will be freed. If a non-handled termination method
    does occur, the implications of holding resources varies on a per-OS basis.
    While no risk to data integrity is observed, repeated misuse may require a
    system reboot in order to achieve expected performance characteristics.
    """

    def __init__(self,
                 base_path: os.PathLike, labelenv: lmdb.Environment,
                 dataenv: lmdb.Environment, hashenv: lmdb.Environment,
                 branchenv: lmdb.Environment, refenv: lmdb.Environment,
                 commit: str):
        """Developer documentation of init method.

        Parameters
        ----------
        base_path : str
            directory path to the Hangar repository on disk
        labelenv : lmdb.Environment
            db where the label dat is stored
        dataenv : lmdb.Environment
            db where the checkout record data is unpacked and stored.
        hashenv : lmdb.Environment
            db where the hash records are stored.
        branchenv : lmdb.Environment
            db where the branch records are stored.
        refenv : lmdb.Environment
            db where the commit references are stored.
        commit : str
            specific commit hash to checkout
        """
        self._commit_hash = commit
        self._repo_path = base_path
        self._labelenv = labelenv
        self._dataenv = dataenv
        self._hashenv = hashenv
        self._branchenv = branchenv
        self._refenv = refenv
        self._is_conman = False

        self._metadata = MetadataReader(
            mode='r',
            repo_pth=self._repo_path,
            dataenv=self._dataenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_commit(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            cmtrefenv=self._dataenv)
        self._differ = ReaderUserDiff(
            commit_hash=self._commit_hash,
            branchenv=self._branchenv,
            refenv=self._refenv)
        atexit.register(self.close)

    def _repr_pretty_(self, p, cycle):
        """pretty repr for printing in jupyter notebooks
        """
        self.__verify_checkout_alive()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : False\
                \n    Commit Hash  : {self._commit_hash}\
                \n    Num Arraysets : {len(self._arraysets)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self.__verify_checkout_alive()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'labelenv={self._labelenv} '\
              f'dataenv={self._dataenv} '\
              f'hashenv={self._hashenv} '\
              f'commit={self._commit_hash})'
        return res

    def __enter__(self):
        self._is_conman = True
        return self

    def __exit__(self, *exc):
        self._is_conman = False

    def __verify_checkout_alive(self):
        """Validates that the checkout object has not been closed

        Raises
        ------
        PermissionError
            if the checkout was previously close
        """
        p_hasattr = partial(hasattr, self)
        if not all(map(p_hasattr, ['_metadata', '_arraysets', '_differ'])):
            e = PermissionError(
                f'Unable to operate on past checkout objects which have been '
                f'closed. No operation occurred. Please use a new checkout.')
            raise e from None

    def __getitem__(self, index):
        """Dictionary style access to arraysets and samples

        Checkout object can be thought of as a "dataset" ("dset") mapping a
        view of samples across arraysets.

            >>> dset = repo.checkout(branch='master')

        Get an arrayset contained in the checkout.

            >>> dset['foo']
            ArraysetDataReader

        Get a specific sample from ``'foo'`` (returns a single array)

            >>> dset['foo', '1']
            np.array([1])

        Get multiple samples from ``'foo'`` (retuns a list of arrays, in order
        of input keys)

            >>> dset['foo', ['1', '2', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]

        Get sample from multiple arraysets (returns namedtuple of arrays, field
        names = arrayset names)

            >>> dset[('foo', 'bar', 'baz'), '1']
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        Get multiple samples from multiple arraysets(returns list of namedtuple
        of array sorted in input key order, field names = arrayset names)

            >>> dset[('foo', 'bar'), ('1', '2')]
            [ArraysetData(foo=array([1]), bar=array([11])),
             ArraysetData(foo=array([2]), bar=array([22]))]

        Get samples from all arraysets (shortcut syntax)

            >>> out = dset[:, ('1', '2')]
            >>> out = dset[..., ('1', '2')]
            >>> out
            [ArraysetData(foo=array([1]), bar=array([11]), baz=array([111])),
             ArraysetData(foo=array([2]), bar=array([22]), baz=array([222]))]

            >>> out = dset[:, '1']
            >>> out = dset[..., '1']
            >>> out
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        .. warning::

            It is possible for an :class:`~.arrayset.Arraysets` name to be an
            invalid field name for a ``namedtuple`` object. The python docs state:

                Any valid Python identifier may be used for a fieldname except for
                names starting with an underscore. Valid identifiers consist of
                letters, digits, and underscores but do not start with a digit or
                underscore and cannot be a keyword such as class, for, return,
                global, pass, or raise.

            In addition, names must be unique, and cannot containing a period
            (``.``) or dash (``-``) character. If a namedtuple would normally be
            returned during some operation, and the field name is invalid, a
            :class:`UserWarning` will be emitted indicating that any suspect fields
            names will be replaced with the positional index as is customary in the
            python standard library. The ``namedtuple`` docs explain this by
            saying:

                If rename is true, invalid fieldnames are automatically replaced with
                positional names. For example, ['abc', 'def', 'ghi', 'abc'] is
                converted to ['abc', '_1', 'ghi', '_3'], eliminating the keyword def
                and the duplicate fieldname abc.

            The next section demonstrates the implications and workarounds for this
            issue

        As an example, if we attempted to retrieve samples from arraysets with
        the names: ``['raw', 'data.jpeg', '_garba-ge', 'try_2']``, two of the
        four would be renamed:

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> print(out)
            ArraysetData(raw=array([0]), _1=array([1]), _2=array([2]), try_2==array([3]))
            >>> print(out._fields)
            ('raw', '_1', '_2', 'try_2')
            >>> out.raw
            array([0])
            >>> out._2
            array([4])

        In cases where the input arraysets are explicitly specified, then, then
        it is guarrenteed that the order of fields in the resulting tuple is
        exactally what was requested in the input

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> out._fields
            ('raw', '_1', '_2', 'try_2')
            >>> reorder = dset[('data.jpeg', 'try_2', 'raw', '_garba-ge'), '1']
            >>> reorder._fields
            ('_0', 'try_2', 'raw', '_3')

        However, if an ``Ellipsis`` (``...``) or ``slice`` (``:``) syntax is
        used to select arraysets, *the order in which arraysets are placed into
        the namedtuple IS NOT predictable.* Should any arrayset have an invalid
        field name, it will be renamed according to it's positional index, but
        will not contain any identifying mark. At the moment, there is no
        direct way to infer the arraayset name from this strcture alone. This
        limitation will be addressed in a future release.

        Do NOT rely on any observed patterns. For this corner-case, **the ONLY
        guarrentee we provide is that structures returned from multi-sample
        queries have the same order in every ``ArraysetData`` tuple returned in
        that queries result list!** Should another query be made with
        unspecified ordering, you should assume that the indices of the
        arraysets in the namedtuple would have changed from the previous
        result!!

            >>> print(dset.arraysets.keys())
            ('raw', 'data.jpeg', '_garba-ge', 'try_2']
            >>> out = dset[:, '1']
            >>> out._fields
            ('_0', 'raw', '_2', 'try_2')
            >>>
            >>> # ordering in a single query is preserved
            ...
            >>> multi_sample = dset[..., ('1', '2')]
            >>> multi_sample[0]._fields
            ('try_2', '_1', 'raw', '_3')
            >>> multi_sample[1]._fields
            ('try_2', '_1', 'raw', '_3')
            >>>
            >>> # but it may change upon a later query
            >>> multi_sample2 = dset[..., ('1', '2')]
            >>> multi_sample2[0]._fields
            ('_0', '_1', 'raw', 'try_2')
            >>> multi_sample2[1]._fields
            ('_0', '_1', 'raw', 'try_2')

        Parameters
        ----------
        index
            Please see detailed explanation above for full options. Hard coded
            options are the order to specification.

            The first element (or collection) specified must be ``str`` type and
            correspond to an arrayset name(s). Alternativly the Ellipsis operator
            (``...``) or unbounded slice operator (``:`` <==> ``slice(None)``) can
            be used to indicate "select all" behavior.

            If a second element (or collection) is present, the keys correspond to
            sample names present within (all) the specified arraysets. If a key is
            not present in even on arrayset, the entire ``get`` operation will
            abort with ``KeyError``. If desired, the same selection syntax can be
            used with the :meth:`~hangar.checkout.ReaderCheckout.get` method, which
            will not Error in these situations, but simply return ``None`` values
            in the appropriate position for keys which do not exist.

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            single arrayset parameter, no samples specified

        :class:`numpy.ndarray`
            Single arrayset specified, single sample key specified

        List[:class:`numpy.ndarray`]
            Single arrayset, multiple samples array data for each sample is
            returned in same order sample keys are recieved.

        List[NamedTuple[``*``:class:`numpy.ndarray`]]
            Multiple arraysets, multiple samples. Each arrayset's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each arrayset via a common sample key. Each sample
            key is returned values as an individual element in the
            List. The sample order is returned in the same order it wasw recieved.

        Warns
        -----
        UserWarning
            Arrayset names contains characters which are invalid as namedtuple fields.

        Notes
        -----

        *  All specified arraysets must exist

        *  All specified sample `keys` must exist in all specified arraysets,
           otherwise standard exception thrown

        *  Slice syntax cannot be used in sample `keys` field

        *  Slice syntax for arrayset field cannot specify `start`, `stop`, or
           `step` fields, it is soley a shortcut syntax for 'get all arraysets' in
           the ``:`` or ``slice(None)`` form

        .. seealso:

            :meth:`~hangar.checkout.ReaderCheckout.get`

        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__verify_checkout_alive()
                self.__enter__()

            if isinstance(index, str):
                return self.arraysets[index]
            elif not isinstance(index, (tuple, list)):
                raise TypeError(f'Unknown index: {index} type: {type(index)}')
            if len(index) > 2:
                raise ValueError(f'index of len > 2 not allowed: {index}')

            arraysets, samples = index
            return self.get(arraysets, samples, except_missing=True)

        finally:
            if tmpconman:
                self.__exit__()

    def get(self, arraysets, samples, *, except_missing=False):
        """View of sample data across arraysets gracefully handeling missing sample keys.

        Please see :meth:`__getitem__` for full description. This method is
        identical with a single exception: if a sample key is not present in an
        arrayset, this method will plane a null ``None`` value in it's return
        slot rather than throwing a ``KeyError`` like the dict style access
        does.

        Parameters
        ----------
        arraysets: Union[str, Iterable[str], Ellipses, slice(None)]

            Name(s) of the arraysets to query. The Ellipsis operator (``...``)
            or unbounded slice operator (``:`` <==> ``slice(None)``) can be
            used to indicate "select all" behavior.

        samples: Union[str, int, Iterable[Union[str, int]]]

            Names(s) of the samples to query

        except_missing: bool, **KWARG ONLY**

            If False, will not throw exceptions on missing sample key value.
            Will raise KeyError if True and missing key found.

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            single arrayset parameter, no samples specified

        :class:`numpy.ndarray`
            Single arrayset specified, single sample key specified

        List[:class:`numpy.ndarray`]
            Single arrayset, multiple samples array data for each sample is
            returned in same order sample keys are recieved.

        List[NamedTuple[``*``:class:`numpy.ndarray`]]
            Multiple arraysets, multiple samples. Each arrayset's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each arrayset via a common sample key. Each sample
            key is returned values as an individual element in the
            List. The sample order is returned in the same order it wasw recieved.

        Warns
        -----
        UserWarning
            Arrayset names contains characters which are invalid as namedtuple fields.
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__verify_checkout_alive()
                self.__enter__()

            # Arrayset Parsing
            if (arraysets is Ellipsis) or isinstance(arraysets, slice):
                arraysets = list(self._arraysets._arraysets.values())
            elif isinstance(arraysets, str):
                arraysets = [self._arraysets._arraysets[arraysets]]
            elif isinstance(arraysets, (tuple, list)):
                arraysets = [self._arraysets._arraysets[aname] for aname in arraysets]
            else:
                raise TypeError(f'Arraysets: {arraysets} type: {type(arraysets)}')
            nAsets = len(arraysets)
            try:
                aset_names = [aset.name for aset in arraysets]
                ArraysetData = namedtuple('ArraysetData', aset_names)
            except ValueError:
                warnings.warn(
                    'Arrayset names contains characters which are invalid as namedtuple fields. '
                    'All suspect field names will be replaced by their positional names '
                    '(ie "_0" for element 0, "_4" for element 4)', UserWarning)
                ArraysetData = namedtuple('ArraysetData', aset_names, rename=True)

            # Sample Parsing
            if isinstance(samples, (str, int)):
                samples = [samples]
            elif not isinstance(samples, (tuple, list)):
                raise TypeError(f'Samples idx: {samples} type: {type(samples)}')
            nSamples = len(samples)

            # Data Retrieval
            asetsSamplesData = []
            for aset in arraysets:
                aset_samples = []
                for sample in samples:
                    try:
                        arr = aset.get(sample)
                    except KeyError as e:
                        if except_missing:
                            raise e
                        arr = None
                    aset_samples.append(arr)
                if nAsets == 1:
                    asetsSamplesData = aset_samples
                    if nSamples == 1:
                        asetsSamplesData = asetsSamplesData[0]
                    break
                asetsSamplesData.append(aset_samples)
            else:  # N.B. for-else conditional (ie. 'no break')
                tmp = map(ArraysetData._make, zip(*asetsSamplesData))
                asetsSamplesData = list(tmp)
                if len(asetsSamplesData) == 1:
                    asetsSamplesData = asetsSamplesData[0]

            return asetsSamplesData

        finally:
            if tmpconman:
                self.__exit__()

    @property
    def arraysets(self) -> Arraysets:
        """Provides access to arrayset interaction object.

        Can be used to either return the arraysets accessor for all elements or
        a single arrayset instance by using dictionary style indexing.

            >>> co = repo.checkout(write=False)
            >>> len(co.arraysets)
            1
            >>> print(co.arraysets.keys())
            ['foo']

            >>> fooAset = co.arraysets['foo']
            >>> fooAset.dtype
            np.fooDtype

            >>> asets = co.arraysets
            >>> fooAset = asets['foo']
            >>> fooAset = asets.get('foo')
            >>> fooAset.dtype
            np.fooDtype

        .. seealso::

            The class :class:`~.arrayset.Arraysets` contains all methods
            accessible by this property accessor

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            weakref proxy to the arraysets object which behaves exactly like a
            arraysets accessor class but which can be invalidated when the writer
            lock is released.
        """
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._arraysets)
        return wr

    @property
    def metadata(self) -> MetadataReader:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`~hangar.metadata.MetadataReader` contains all methods
            accessible by this property accessor

        Returns
        -------
        MetadataReader
            weakref proxy to the metadata object which behaves exactly like a
            metadata class but which can be invalidated when the writer lock is
            released.
        """
        self.__verify_checkout_alive()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

    @property
    def diff(self) -> ReaderUserDiff:
        """Access the differ methods for a read-only checkout.

        .. seealso::

            The class :class:`ReaderUserDiff` contains all methods accessible
            by this property accessor

        Returns
        -------
        ReaderUserDiff
            weakref proxy to the differ object (and contained methods) which behaves
            exactly like the differ class but which can be invalidated when the
            writer lock is released.
        """
        self.__verify_checkout_alive()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def commit_hash(self) -> str:
        """Commit hash this read-only checkout's data is read from.

            >>> co.commit_hash
            foohashdigesthere

        Returns
        -------
        str
            commit hash of the checkout
        """
        self.__verify_checkout_alive()
        return self._commit_hash

    def close(self) -> None:
        """Gracefully close the reader checkout object.

        Though not strictly required for reader checkouts (as opposed to
        writers), closing the checkout after reading will free file handles and
        system resources, which may improve performance for repositories with
        multiple simultaneous read checkouts.
        """
        self.__verify_checkout_alive()
        with suppress(AttributeError):
            self._arraysets._close()

        for asetn in (self._arraysets._arraysets.keys()):
            for attr in list(self._arraysets._arraysets[asetn].__dir__()):
                with suppress(AttributeError, TypeError):
                    delattr(self._arraysets._arraysets[asetn], attr)

        for attr in list(self._arraysets.__dir__()):
            with suppress(AttributeError, TypeError):
                # adding `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                delattr(self._arraysets, f'_self_{attr}')

        for attr in list(self._metadata.__dir__()):
            with suppress(AttributeError, TypeError):
                # adding `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                delattr(self._metadata, f'_self_{attr}')

        with suppress(AttributeError):
            del self._arraysets
        with suppress(AttributeError):
            del self._metadata
        with suppress(AttributeError):
            del self._differ

        del self._commit_hash
        del self._repo_path
        del self._labelenv
        del self._dataenv
        del self._hashenv
        del self._branchenv
        del self._refenv
        del self._is_conman
        atexit.unregister(self.close)
        return


# --------------- Write enabled checkout ---------------------------------------


class WriterCheckout(object):
    """Checkout the repository at the head of a given branch for writing.

    This is the entry point for all writing operations to the repository, the
    writer class records all interactions in a special ``"staging"`` area,
    which is based off the state of the repository as it existed at the
    ``HEAD`` commit of a branch.

        >>> co = repo.checkout(write=True)
        >>> co.branch_name
        'master'
        >>> co.commit_hash
        'masterheadcommithash'
        >>> co.close()

    At the moment, only one instance of this class can write data to the
    staging area at a time. After the desired operations have been completed,
    it is crucial to call :meth:`close` to release the writer lock. In
    addition, after any changes have been made to the staging area, the branch
    ``HEAD`` cannot be changed. In order to checkout another branch ``HEAD``
    for writing, you must either :meth:`commit` the changes, or perform a
    hard-reset of the staging area to the last commit via
    :meth:`reset_staging_area`.

    In order to reduce the chance that the python interpreter is shut down
    without calling :meth:`close`, which releases the writer lock - a common
    mistake during ipython / jupyter sessions - an `atexit
    <https://docs.python.org/3/library/atexit.html>`_ hook is registered to
    :meth:`close`. If properly closed by the user, the hook is unregistered
    after completion with no ill effects. So long as a the process is NOT
    terminated via non-python SIGKILL, fatal internal python error, or or
    special os exit methods, cleanup will occur on interpreter shutdown and the
    writer lock will be released. If a non-handled termination method does
    occur, the :py:meth:`~.Repository.force_release_writer_lock` method must be
    called manually when a new python process wishes to open the writer
    checkout.
    """

    def __init__(self,
                 repo_pth: os.PathLike,
                 branch_name: str,
                 labelenv: lmdb.Environment,
                 hashenv: lmdb.Environment,
                 refenv: lmdb.Environment,
                 stageenv: lmdb.Environment,
                 branchenv: lmdb.Environment,
                 stagehashenv: lmdb.Environment,
                 mode: str = 'a'):
        """Developer documentation of init method.

        Parameters
        ----------
        repo_pth : str
            local file path of the repository.
        branch_name : str
            name of the branch whose ``HEAD`` commit will for the starting state
            of the staging area.
        labelenv : lmdb.Environment
            db where the label dat is stored
        hashenv lmdb.Environment
            db where the hash records are stored.
        refenv : lmdb.Environment
            db where the commit record data is unpacked and stored.
        stagenv : lmdb.Environment
            db where the stage record data is unpacked and stored.
        branchenv : lmdb.Environment
            db where the head record data is unpacked and stored.
        stagehashenv: lmdb.Environment
            db where the staged hash record data is stored.
        mode : str, optional
            open in write or read only mode, default is 'a' which is write-enabled.
        """
        self._is_conman = False
        self._repo_path = repo_pth
        self._branch_name = branch_name
        self._writer_lock = str(uuid4())

        self._refenv = refenv
        self._hashenv = hashenv
        self._labelenv = labelenv
        self._stageenv = stageenv
        self._branchenv = branchenv
        self._stagehashenv = stagehashenv

        self._arraysets: Arraysets = None
        self._differ: WriterUserDiff = None
        self._metadata: MetadataWriter = None
        self.__setup()
        atexit.register(self.close)

    def _repr_pretty_(self, p, cycle):
        """pretty repr for printing in jupyter notebooks
        """
        self.__acquire_writer_lock()
        res = f'Hangar {self.__class__.__name__}\
                \n    Writer       : True\
                \n    Base Branch  : {self._branch_name}\
                \n    Num Arraysets : {len(self._arraysets)}\
                \n    Num Metadata : {len(self._metadata)}\n'
        p.text(res)

    def __repr__(self):
        self.__acquire_writer_lock()
        res = f'{self.__class__}('\
              f'base_path={self._repo_path} '\
              f'branch_name={self._branch_name} ' \
              f'labelenv={self._labelenv} '\
              f'hashenv={self._hashenv} '\
              f'refenv={self._refenv} '\
              f'stageenv={self._stageenv} '\
              f'branchenv={self._branchenv})\n'
        return res

    def __enter__(self):
        self.__acquire_writer_lock()
        self._is_conman = True
        self.arraysets.__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self.arraysets.__exit__(*exc)

    def __acquire_writer_lock(self):
        """Ensures that this class instance holds the writer lock in the database.

        Raises
        ------
        PermissionError
            If the checkout was previously closed (no :attr:``_writer_lock``) or if
            the writer lock value does not match that recorded in the branch db
        """
        try:
            self._writer_lock
        except AttributeError:
            with suppress(AttributeError):
                del self._arraysets
            with suppress(AttributeError):
                del self._metadata
            with suppress(AttributeError):
                del self._differ
            err = f'Unable to operate on past checkout objects which have been '\
                  f'closed. No operation occurred. Please use a new checkout.'
            raise PermissionError(err) from None

        try:
            heads.acquire_writer_lock(self._branchenv, self._writer_lock)
        except PermissionError as e:
            with suppress(AttributeError):
                del self._arraysets
            with suppress(AttributeError):
                del self._metadata
            with suppress(AttributeError):
                del self._differ
            raise e from None

    def __setup(self):
        """setup the staging area appropriately for a write enabled checkout.

        On setup, we cannot be sure what branch the staging area was previously
        checked out on, and we cannot be sure if there are any 'uncommitted
        changes' in the staging area (ie. the staging area is ``DIRTY``). The
        setup methods here ensure that we can safety make any changes to the
        staging area without overwriting uncommitted changes, and then perform
        the setup steps to checkout staging area state at that point in time.

        Raises
        ------
        ValueError
            if there are changes previously made in the staging area which were
            based on one branch's ``HEAD``, but a different branch was specified to
            be used for the base of this checkout.
        """
        self.__acquire_writer_lock()
        current_head = heads.get_staging_branch_head(self._branchenv)
        currentDiff = WriterUserDiff(stageenv=self._stageenv,
                                     refenv=self._refenv,
                                     branchenv=self._branchenv,
                                     branch_name=current_head)
        if currentDiff.status() == 'DIRTY':
            if current_head != self._branch_name:
                e = ValueError(
                    f'Unable to check out branch: {self._branch_name} for writing '
                    f'as the staging area has uncommitted changes on branch: '
                    f'{current_head}. Please commit or stash uncommitted changes '
                    f'before checking out a different branch for writing.')
                self.close()
                raise e
        else:
            if current_head != self._branch_name:
                try:
                    cmt = heads.get_branch_head_commit(
                        branchenv=self._branchenv, branch_name=self._branch_name)
                except ValueError as e:
                    self.close()
                    raise e
                commiting.replace_staging_area_with_commit(
                    refenv=self._refenv, stageenv=self._stageenv, commit_hash=cmt)
                heads.set_staging_branch_head(
                    branchenv=self._branchenv, branch_name=self._branch_name)

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

    def __getitem__(self, index):
        """Dictionary style access to arraysets and samples

        Checkout object can be thought of as a "dataset" ("dset") mapping a
        view of samples across arraysets.

            >>> dset = repo.checkout(branch='master')

        Get an arrayset contained in the checkout.

            >>> dset['foo']
            ArraysetDataReader

        Get a specific sample from ``'foo'`` (returns a single array)

            >>> dset['foo', '1']
            np.array([1])

        Get multiple samples from ``'foo'`` (retuns a list of arrays, in order
        of input keys)

            >>> dset['foo', ['1', '2', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]

        Get sample from multiple arraysets (returns namedtuple of arrays, field
        names = arrayset names)

            >>> dset[('foo', 'bar', 'baz'), '1']
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        Get multiple samples from multiple arraysets(returns list of namedtuple
        of array sorted in input key order, field names = arrayset names)

            >>> dset[('foo', 'bar'), ('1', '2')]
            [ArraysetData(foo=array([1]), bar=array([11])),
             ArraysetData(foo=array([2]), bar=array([22]))]

        Get samples from all arraysets (shortcut syntax)

            >>> out = dset[:, ('1', '2')]
            >>> out = dset[..., ('1', '2')]
            >>> out
            [ArraysetData(foo=array([1]), bar=array([11]), baz=array([111])),
             ArraysetData(foo=array([2]), bar=array([22]), baz=array([222]))]

            >>> out = dset[:, '1']
            >>> out = dset[..., '1']
            >>> out
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        .. warning::

            It is possible for an :class:`~.arrayset.Arraysets` name to be an
            invalid field name for a ``namedtuple`` object. The python docs state:

                Any valid Python identifier may be used for a fieldname except for
                names starting with an underscore. Valid identifiers consist of
                letters, digits, and underscores but do not start with a digit or
                underscore and cannot be a keyword such as class, for, return,
                global, pass, or raise.

            In addition, names must be unique, and cannot containing a period
            (``.``) or dash (``-``) character. If a namedtuple would normally be
            returned during some operation, and the field name is invalid, a
            :class:`UserWarning` will be emitted indicating that any suspect fields
            names will be replaced with the positional index as is customary in the
            python standard library. The ``namedtuple`` docs explain this by
            saying:

                If rename is true, invalid fieldnames are automatically replaced with
                positional names. For example, ['abc', 'def', 'ghi', 'abc'] is
                converted to ['abc', '_1', 'ghi', '_3'], eliminating the keyword def
                and the duplicate fieldname abc.

            The next section demonstrates the implications and workarounds for this
            issue

        As an example, if we attempted to retrieve samples from arraysets with
        the names: ``['raw', 'data.jpeg', '_garba-ge', 'try_2']``, two of the
        four would be renamed:

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> print(out)
            ArraysetData(raw=array([0]), _1=array([1]), _2=array([2]), try_2==array([3]))
            >>> print(out._fields)
            ('raw', '_1', '_2', 'try_2')
            >>> out.raw
            array([0])
            >>> out._2
            array([4])

        In cases where the input arraysets are explicitly specified, then, then
        it is guarrenteed that the order of fields in the resulting tuple is
        exactally what was requested in the input

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> out._fields
            ('raw', '_1', '_2', 'try_2')
            >>> reorder = dset[('data.jpeg', 'try_2', 'raw', '_garba-ge'), '1']
            >>> reorder._fields
            ('_0', 'try_2', 'raw', '_3')

        However, if an ``Ellipsis`` (``...``) or ``slice`` (``:``) syntax is
        used to select arraysets, *the order in which arraysets are placed into
        the namedtuple IS NOT predictable.* Should any arrayset have an invalid
        field name, it will be renamed according to it's positional index, but
        will not contain any identifying mark. At the moment, there is no
        direct way to infer the arraayset name from this strcture alone. This
        limitation will be addressed in a future release.

        Do NOT rely on any observed patterns. For this corner-case, **the ONLY
        guarrentee we provide is that structures returned from multi-sample
        queries have the same order in every ``ArraysetData`` tuple returned in
        that queries result list!** Should another query be made with
        unspecified ordering, you should assume that the indices of the
        arraysets in the namedtuple would have changed from the previous
        result!!

            >>> print(dset.arraysets.keys())
            ('raw', 'data.jpeg', '_garba-ge', 'try_2']
            >>> out = dset[:, '1']
            >>> out._fields
            ('_0', 'raw', '_2', 'try_2')
            >>>
            >>> # ordering in a single query is preserved
            ...
            >>> multi_sample = dset[..., ('1', '2')]
            >>> multi_sample[0]._fields
            ('try_2', '_1', 'raw', '_3')
            >>> multi_sample[1]._fields
            ('try_2', '_1', 'raw', '_3')
            >>>
            >>> # but it may change upon a later query
            >>> multi_sample2 = dset[..., ('1', '2')]
            >>> multi_sample2[0]._fields
            ('_0', '_1', 'raw', 'try_2')
            >>> multi_sample2[1]._fields
            ('_0', '_1', 'raw', 'try_2')

        Parameters
        ----------
        index
            Please see detailed explanation above for full options.

            The first element (or collection) specified must be ``str`` type and
            correspond to an arrayset name(s). Alternativly the Ellipsis operator
            (``...``) or unbounded slice operator (``:`` <==> ``slice(None)``) can
            be used to indicate "select all" behavior.

            If a second element (or collection) is present, the keys correspond to
            sample names present within (all) the specified arraysets. If a key is
            not present in even on arrayset, the entire ``get`` operation will
            abort with ``KeyError``. If desired, the same selection syntax can be
            used with the :meth:`~hangar.checkout.WriterCheckout.get` method, which
            will not Error in these situations, but simply return ``None`` values
            in the appropriate position for keys which do not exist.

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            single arrayset parameter, no samples specified

        :class:`numpy.ndarray`
            Single arrayset specified, single sample key specified

        List[:class:`numpy.ndarray`]
            Single arrayset, multiple samples array data for each sample is
            returned in same order sample keys are recieved.

        List[NamedTuple[``*``:class:`numpy.ndarray`]]
            Multiple arraysets, multiple samples. Each arrayset's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each arrayset via a common sample key. Each sample
            key is returned values as an individual element in the
            List. The sample order is returned in the same order it wasw recieved.

        Warns
        -----
        UserWarning
            Arrayset names contains characters which are invalid as namedtuple fields.

        Notes
        -----

        *  All specified arraysets must exist

        *  All specified sample `keys` must exist in all specified arraysets,
           otherwise standard exception thrown

        *  Slice syntax cannot be used in sample `keys` field

        *  Slice syntax for arrayset field cannot specify `start`, `stop`, or
           `step` fields, it is soley a shortcut syntax for 'get all arraysets' in
           the ``:`` or ``slice(None)`` form

        .. seealso:

            :meth:`~hangar.checkout.WriterCheckout.get`

        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__acquire_writer_lock()
                self.__enter__()

            if isinstance(index, str):
                return self.arraysets[index]
            elif not isinstance(index, (tuple, list)):
                raise TypeError(f'Unknown index: {index} type: {type(index)}')
            if len(index) > 2:
                raise ValueError(f'index of len > 2 not allowed: {index}')

            arraysets, samples = index
            return self.get(arraysets, samples, except_missing=True)

        finally:
            if tmpconman:
                self.__exit__()

    def get(self, arraysets, samples, *, except_missing=False):
        """View of samples across arraysets which handles missing sample keys.

        Please see :meth:`__getitem__` for full description. This method is
        identical with a single exception: if a sample key is not present in an
        arrayset, this method will plane a null ``None`` value in it's return
        slot rather than throwing a ``KeyError`` like the dict style access
        does.

        Parameters
        ----------
        arraysets: Union[str, Iterable[str], Ellipses, slice(None)]

            Name(s) of the arraysets to query. The Ellipsis operator (``...``)
            or unbounded slice operator (``:`` <==> ``slice(None)``) can be
            used to indicate "select all" behavior.

        samples: Union[str, int, Iterable[Union[str, int]]]

            Names(s) of the samples to query

        except_missing: bool, *kwarg-only*

            If False, will not throw exceptions on missing sample key value.
            Will raise KeyError if True and missing key found.

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            single arrayset parameter, no samples specified

        :class:`numpy.ndarray`
            Single arrayset specified, single sample key specified

        List[:class:`numpy.ndarray`]
            Single arrayset, multiple samples array data for each sample is
            returned in same order sample keys are recieved.

        List[NamedTuple[``*``:class:`numpy.ndarray`]]
            Multiple arraysets, multiple samples. Each arrayset's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each arrayset via a common sample key. Each sample
            key is returned values as an individual element in the List. The
            sample order is returned in the same order it wasw recieved.

        Warns
        -----
        UserWarning
            Arrayset names contains characters which are invalid as namedtuple fields.
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__acquire_writer_lock()
                self.__enter__()

            # Arrayset Parsing
            if (arraysets is Ellipsis) or isinstance(arraysets, slice):
                arraysets = list(self._arraysets._arraysets.values())
            elif isinstance(arraysets, str):
                arraysets = [self._arraysets._arraysets[arraysets]]
            elif isinstance(arraysets, (tuple, list)):
                arraysets = [self._arraysets._arraysets[aname] for aname in arraysets]
            else:
                raise TypeError(f'Arraysets: {arraysets} type: {type(arraysets)}')
            nAsets = len(arraysets)
            try:
                aset_names = [aset.name for aset in arraysets]
                ArraysetData = namedtuple('ArraysetData', aset_names)
            except ValueError:
                warnings.warn(
                    'Arrayset names contains characters which are invalid as namedtuple fields. '
                    'All suspect field names will be replaced by their positional names '
                    '(ie "_0" for element 0, "_4" for element 4)', UserWarning)
                ArraysetData = namedtuple('ArraysetData', aset_names, rename=True)

            # Sample Parsing
            if isinstance(samples, (str, int)):
                samples = [samples]
            elif not isinstance(samples, (tuple, list)):
                raise TypeError(f'Samples idx: {samples} type: {type(samples)}')
            nSamples = len(samples)

            # Data Retrieval
            asetsSamplesData = []
            for aset in arraysets:
                aset_samples = []
                for sample in samples:
                    try:
                        arr = aset.get(sample)
                    except KeyError as e:
                        if except_missing:
                            raise e
                        arr = None
                    aset_samples.append(arr)
                if nAsets == 1:
                    asetsSamplesData = aset_samples
                    if nSamples == 1:
                        asetsSamplesData = asetsSamplesData[0]
                    break
                asetsSamplesData.append(aset_samples)
            else:  # N.B. for-else conditional (ie. 'no break')
                tmp = map(ArraysetData._make, zip(*asetsSamplesData))
                asetsSamplesData = list(tmp)
                if len(asetsSamplesData) == 1:
                    asetsSamplesData = asetsSamplesData[0]

            return asetsSamplesData

        finally:
            if tmpconman:
                self.__exit__()

    def __setitem__(self, index, value):
        """Syntax for setting items.

        Checkout object can be thought of as a "dataset" ("dset") mapping a view
        of samples across arraysets:

            >>> dset = repo.checkout(branch='master', write=True)

        Add single sample to single arrayset

            >>> dset['foo', 1] = np.array([1])
            >>> dset['foo', 1]
            array([1])

        Add multiple samples to single arrayset

            >>> dset['foo', [1, 2, 3]] = [np.array([1]), np.array([2]), np.array([3])]
            >>> dset['foo', [1, 2, 3]]
            [array([1]), array([2]), array([3])]

        Add single sample to multiple arraysets

            >>> dset[['foo', 'bar'], 1] = [np.array([1]), np.array([11])]
            >>> dset[:, 1]
            ArraysetData(foo=array([1]), bar=array([11]))

        Parameters
        ----------
        index: Union[Iterable[str], Iterable[str, int]]
            Please see detailed explanation above for full options.The first
            element (or collection) specified must be ``str`` type and correspond
            to an arrayset name(s). The second element (or collection) are keys
            corresponding to sample names which the data should be written to.

            Unlike the :meth:`__getitem__` method, only ONE of the ``arrayset``
            name(s) or ``sample`` key(s) can specify multiple elements at the same
            time. Ie. If multiple ``arraysets`` are specified, only one sample key
            can be set, likewise if multiple ``samples`` are specified, only one
            ``arrayset`` can be specified. When specifying multiple ``arraysets``
            or ``samples``, each data piece to be stored must reside as individual
            elements (``np.ndarray``) in a List or Tuple. The number of keys and
            the number of values must match exactally.

        values: Union[:class:`numpy.ndarray`, Iterable[:class:`numpy.ndarray`]]
            Data to store in the specified arraysets/sample keys. When
            specifying multiple ``arraysets`` or ``samples``, each data piece
            to be stored must reside as individual elements (``np.ndarray``) in
            a List or Tuple. The number of keys and the number of values must
            match exactally.

        Notes
        -----

        *  No slicing syntax is supported for either arraysets or samples. This
           is in order to ensure explicit setting of values in the desired
           fields/keys

        *  Add multiple samples to multiple arraysets not yet supported.

        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__acquire_writer_lock()
                self.__enter__()

            if not isinstance(index, (tuple, list)):
                raise ValueError(f'Idx: {index} does not specify arrayset(s) AND sample(s)')
            elif len(index) > 2:
                raise ValueError(f'Index of len > 2 invalid. To multi-set, pass in lists')
            asetsIdx, sampleNames = index

            # Parse Arraysets
            if isinstance(asetsIdx, str):
                asets = [self._arraysets._arraysets[asetsIdx]]
            elif isinstance(asetsIdx, (tuple, list)):
                asets = [self._arraysets._arraysets[aidx] for aidx in asetsIdx]
            else:
                raise TypeError(f'Arrayset idx: {asetsIdx} of type: {type(asetsIdx)}')
            nAsets = len(asets)

            # Parse sample names
            if isinstance(sampleNames, (str, int)):
                sampleNames = [sampleNames]
            elif not isinstance(sampleNames, (list, tuple)):
                raise TypeError(f'Sample names: {sampleNames} type: {type(sampleNames)}')
            nSamples = len(sampleNames)

            # Verify asets
            if (nAsets > 1) and (nSamples > 1):
                raise SyntaxError(
                    'Not allowed to specify BOTH multiple samples AND multiple'
                    'arraysets in `set` operation in current Hangar implementation')

            elif (nAsets == 1) and (nSamples == 1):
                aset = asets[0]
                sampleName = sampleNames[0]
                aset[sampleName] = value

            elif nAsets >= 2:
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f'Value: {value} not list/tuple of np.ndarray')
                elif not (len(value) == nAsets):
                    raise ValueError(f'Num values: {len(value)} != num arraysets {nAsets}')
                for aset, val in zip(asets, value):
                    isCompat = aset._verify_array_compatible(val)
                    if not isCompat.compatible:
                        raise ValueError(isCompat.reason)
                for sampleName in sampleNames:
                    for aset, val in zip(asets, value):
                        aset[sampleName] = val

            else:  # nSamples >= 2
                if not isinstance(value, (list, tuple)):
                    raise TypeError(f'Value: {value} not list/tuple of np.ndarray')
                elif not (len(value) == nSamples):
                    raise ValueError(f'Num values: {len(value)} != num samples {nSamples}')
                for aset in asets:
                    for val in value:
                        isCompat = aset._verify_array_compatible(val)
                        if not isCompat.compatible:
                            raise ValueError(isCompat.reason)
                for aset in asets:
                    for sampleName, val in zip(sampleNames, value):
                        aset[sampleName] = val
                return None
        finally:
            if tmpconman:
                self.__exit__()

    @property
    def arraysets(self) -> Arraysets:
        """Provides access to arrayset interaction object.

        Can be used to either return the arraysets accessor for all elements or
        a single arrayset instance by using dictionary style indexing.

            >>> co = repo.checkout(write=True)
            >>> asets = co.arraysets
            >>> len(asets)
            0
            >>> fooAset = asets.init_arrayset('foo', shape=(10, 10), dtype=np.uint8)
            >>> len(co.arraysets)
            1
            >>> print(co.arraysets.keys())
            ['foo']
            >>> fooAset = co.arraysets['foo']
            >>> fooAset.dtype
            np.fooDtype
            >>> fooAset = asets.get('foo')
            >>> fooAset.dtype
            np.fooDtype

        .. seealso::

            The class :class:`~.arrayset.Arraysets` contains all methods accessible
            by this property accessor

        Returns
        -------
        :class:`~.arrayset.Arraysets`
            weakref proxy to the arraysets object which behaves exactly like a
            arraysets accessor class but which can be invalidated when the writer
            lock is released.
        """
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._arraysets)
        return wr

    @property
    def metadata(self) -> MetadataWriter:
        """Provides access to metadata interaction object.

        .. seealso::

            The class :class:`hangar.metadata.MetadataWriter` contains all methods
            accessible by this property accessor

        Returns
        -------
        MetadataWriter
            weakref proxy to the metadata object which behaves exactly like a
            metadata class but which can be invalidated when the writer lock is
            released.
        """
        self.__acquire_writer_lock()
        wr = cm_weakref_obj_proxy(self._metadata)
        return wr

    @property
    def diff(self) -> WriterUserDiff:
        """Access the differ methods which are aware of any staged changes.

        .. seealso::

            The class :class:`hangar.diff.WriterUserDiff` contains all methods
            accessible by this property accessor

        Returns
        -------
        WriterUserDiff
            weakref proxy to the differ object (and contained methods) which behaves
            exactly like the differ class but which can be invalidated when the
            writer lock is released.
        """
        self.__acquire_writer_lock()
        wr = weakref.proxy(self._differ)
        return wr

    @property
    def branch_name(self) -> str:
        """Branch this write enabled checkout's staging area was based on.

        Returns
        -------
        str
            name of the branch whose commit ``HEAD`` changes are staged from.
        """
        self.__acquire_writer_lock()
        return self._branch_name

    @property
    def commit_hash(self) -> str:
        """Commit hash which the staging area of `branch_name` is based on.

        Returns
        -------
        str
            commit hash
        """
        self.__acquire_writer_lock()
        cmt = heads.get_branch_head_commit(branchenv=self._branchenv,
                                           branch_name=self._branch_name)
        return cmt

    def merge(self, message: str, dev_branch: str) -> str:
        """Merge the currently checked out commit with the provided branch name.

        If a fast-forward merge is possible, it will be performed, and the
        commit message argument to this function will be ignored.

        Parameters
        ----------
        message : str
            commit message to attach to a three-way merge
        dev_branch : str
            name of the branch which should be merge into this branch (`master`)

        Returns
        -------
        str
            commit hash of the new commit for the `master` branch this checkout
            was started from.
        """
        self.__acquire_writer_lock()
        commit_hash = select_merge_algorithm(
            message=message,
            branchenv=self._branchenv,
            stageenv=self._stageenv,
            refenv=self._refenv,
            stagehashenv=self._stagehashenv,
            master_branch=self._branch_name,
            dev_branch=dev_branch,
            repo_path=self._repo_path,
            writer_uuid=self._writer_lock)

        for asetHandle in self._arraysets.values():
            with suppress(KeyError):
                asetHandle._close()

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)

        return commit_hash

    def commit(self, commit_message: str) -> str:
        """Commit the changes made in the staging area on the checkout branch.

        Parameters
        ----------
        commit_message : str, optional
            user proved message for a log of what was changed in this commit.
            Should a fast forward commit be possible, this will NOT be added to
            fast-forward ``HEAD``.

        Returns
        -------
        str
            The commit hash of the new commit.

        Raises
        ------
        RuntimeError
            If no changes have been made in the staging area, no commit occurs.
        """
        self.__acquire_writer_lock()

        open_asets = []
        for arrayset in self._arraysets.values():
            if arrayset._is_conman:
                open_asets.append(arrayset.name)
        open_meta = self._metadata._is_conman

        try:
            if open_meta:
                self._metadata.__exit__()
            for asetn in open_asets:
                self._arraysets[asetn].__exit__()

            if self._differ.status() == 'CLEAN':
                e = RuntimeError('No changes made in staging area. Cannot commit.')
                raise e from None

            self._arraysets._close()
            commit_hash = commiting.commit_records(message=commit_message,
                                                   branchenv=self._branchenv,
                                                   stageenv=self._stageenv,
                                                   refenv=self._refenv,
                                                   repo_path=self._repo_path)
            # purge recs then reopen file handles so that we don't have to invalidate
            # previous weakproxy references like if we just called :meth:``__setup```
            hashs.clear_stage_hash_records(self._stagehashenv)
            self._arraysets._open()

        finally:
            for asetn in open_asets:
                self._arraysets[asetn].__enter__()
            if open_meta:
                self._metadata.__enter__()

        return commit_hash

    def reset_staging_area(self) -> str:
        """Perform a hard reset of the staging area to the last commit head.

        After this operation completes, the writer checkout will automatically
        close in the typical fashion (any held references to :attr:``arrayset``
        or :attr:``metadata`` objects will finalize and destruct as normal), In
        order to perform any further operation, a new checkout needs to be
        opened.

        .. warning::

            This operation is IRREVERSIBLE. all records and data which are note
            stored in a previous commit will be permanently deleted.

        Returns
        -------
        str
            Commit hash of the head which the staging area is reset to.

        Raises
        ------
        RuntimeError
            If no changes have been made to the staging area, No-Op.
        """
        self.__acquire_writer_lock()
        print(f'Hard reset requested with writer_lock: {self._writer_lock}')

        if self._differ.status() == 'CLEAN':
            e = RuntimeError(f'No changes made in staging area. No reset necessary.')
            raise e from None

        self._arraysets._close()
        hashs.remove_stage_hash_records_from_hashenv(self._hashenv, self._stagehashenv)
        hashs.clear_stage_hash_records(self._stagehashenv)
        hashs.delete_in_process_data(self._repo_path)

        branch_head = heads.get_staging_branch_head(self._branchenv)
        head_commit = heads.get_branch_head_commit(self._branchenv, branch_head)
        commiting.replace_staging_area_with_commit(refenv=self._refenv,
                                                   stageenv=self._stageenv,
                                                   commit_hash=head_commit)

        self._metadata = MetadataWriter(
            mode='a',
            repo_pth=self._repo_path,
            dataenv=self._stageenv,
            labelenv=self._labelenv)
        self._arraysets = Arraysets._from_staging_area(
            repo_pth=self._repo_path,
            hashenv=self._hashenv,
            stageenv=self._stageenv,
            stagehashenv=self._stagehashenv)
        self._differ = WriterUserDiff(
            stageenv=self._stageenv,
            refenv=self._refenv,
            branchenv=self._branchenv,
            branch_name=self._branch_name)
        return head_commit

    def close(self) -> None:
        """Close all handles to the writer checkout and release the writer lock.

        Failure to call this method after the writer checkout has been used will
        result in a lock being placed on the repository which will not allow any
        writes until it has been manually cleared.
        """
        self.__acquire_writer_lock()

        if hasattr(self, '_arraysets') and (getattr(self, '_arraysets') is not None):
            self._arraysets._close()

            for asetn in (self._arraysets._arraysets.keys()):
                for attr in list(self._arraysets._arraysets[asetn].__dir__()):
                    with suppress(AttributeError, TypeError):
                        delattr(self._arraysets._arraysets[asetn], attr)

            for attr in list(self._arraysets.__dir__()):
                with suppress(AttributeError, TypeError):
                    # prepending `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                    delattr(self._arraysets, f'_self_{attr}')

        if hasattr(self, '_metadata') and (getattr(self, '_arraysets') is not None):
            for attr in list(self._metadata.__dir__()):
                with suppress(AttributeError, TypeError):
                    # prepending `_self_` addresses `WeakrefProxy` in `wrapt.ObjectProxy`
                    delattr(self._metadata, f'_self_{attr}')

        with suppress(AttributeError):
            del self._arraysets
        with suppress(AttributeError):
            del self._metadata
        with suppress(AttributeError):
            del self._differ

        heads.release_writer_lock(self._branchenv, self._writer_lock)

        del self._refenv
        del self._hashenv
        del self._labelenv
        del self._stageenv
        del self._branchenv
        del self._stagehashenv
        del self._repo_path
        del self._writer_lock
        del self._branch_name
        del self._is_conman
        atexit.unregister(self.close)
        return
