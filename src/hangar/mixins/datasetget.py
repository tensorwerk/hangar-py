from functools import reduce
from operator import getitem as op_getitem
from contextlib import ExitStack

# noinspection PyUnresolvedReferences
class GetMixin:
    """Mixin methods for the checkout object classes.

    Used since the read and write enabled checkouts have the same :meth:`__get__`
    and :meth:`get` methods
    """

    def __getitem__(self, index):
        """Dictionary style access to columns and samples

        Checkout object can be thought of as a "dataset" ("dset") mapping a
        view of samples across columns.

            >>> dset = repo.checkout(branch='master')
            >>>
            # Get an column contained in the checkout.
            >>> dset['foo']
            ColumnDataReader
            >>>
            # Get a specific sample from ``'foo'`` (returns a single array)
            >>> dset['foo', '1']
            np.array([1])
            >>>
            # Get multiple samples from ``'foo'`` (returns a list of arrays, in order
            # of input keys)
            >>> dset[['foo', '1'], ['foo', '2'],  ['foo', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]
            >>>
            # Get sample from multiple columns, column/data returned is ordered
            # in same manner as input of func.
            >>> dset[['foo', '1'], ['bar', '1'],  ['baz', '1']]
            [np.array([1]), np.ndarray([1, 1]), np.ndarray([1, 1, 1])]
            >>>
            # Get multiple samples from multiple columns\
            >>> keys = [(col, str(samp)) for samp in range(2) for col in ['foo', 'bar']]
            >>> keys
            [('foo', '0'), ('bar', '0'), ('foo', '1'), ('bar', '1')]
            >>> dset[keys]
            [np.array([1]), np.array([1, 1]), np.array([2]), np.array([2, 2])]

        Arbitrary column layouts are supported by simply adding additional members
        to the keys for each piece of data. For example, getting data from a column
        with a nested layout:

            >>> dset['nested_col', 'sample_1', 'subsample_0']
            np.array([1, 0])
            >>>
            # a sample accessor object can be retrieved at will...
            >>> dset['nested_col', 'sample_1']
            <class 'FlatSubsampleReader'>(column_name='nested_col', sample_name='sample_1')
            >>>
            # to get all subsamples in a nested sample use the Ellipsis operator
            >>> dset['nested_col', 'sample_1', ...]
            {'subsample_0': np.array([1, 0]),
             'subsample_1': np.array([1, 1]),
             ...
             'subsample_n': np.array([1, 255])}

        Retrieval of data from different column types can be mixed and combined
        as desired. For example, retrieving data from both flat and nested columns
        simultaneously:

            >>> dset[('nested_col', 'sample_1', '0'), ('foo', '0')]
            [np.array([1, 0]), np.array([0])]
            >>> dset[('nested_col', 'sample_1', ...), ('foo', '0')]
            [{'subsample_0': np.array([1, 0]), 'subsample_1': np.array([1, 1])},
             np.array([0])]
            >>> dset[('foo', '0'), ('nested_col', 'sample_1')]
            [np.array([0]),
             <class 'FlatSubsampleReader'>(column_name='nested_col', sample_name='sample_1')]

        If a column or data key does not exist, then this method will raise a KeyError.
        As an alternative, missing keys can be gracefully handeled by calling :meth:`get()`
        instead. This method does not (by default) raise an error if a key is missing.
        Instead, a (configurable) default value is simply inserted in it's place.

            >>> dset['foo', 'DOES_NOT_EXIST']
            -------------------------------------------------------------------
            KeyError                           Traceback (most recent call last)
            <ipython-input-40-731e6ea62fb8> in <module>
            ----> 1 res = co['foo', 'DOES_NOT_EXIST']
            KeyError: 'DOES_NOT_EXIST'

        Parameters
        ----------
        index
            column name, sample key(s) or sequence of list/tuple of column name,
            sample keys(s) which should be retrieved in the operation.

            Please see detailed explanation above for full explanation of accepted
            argument format / result types.

        Returns
        -------
        :class:`~.columns.column.Columns`
            single column parameter, no samples specified
        Any
            Single column specified, single sample key specified
        List[Any]
            arbitrary columns, multiple samples array data for each sample is
            returned in same order sample keys are received.
        """
        # not using kwargs since this could be in a tight loop.
        # kwargs: default-None, except_missing=True
        return self._get_in(index, None, True)

    def get(self, keys, default=None, except_missing=False):
        """View of sample data across columns gracefully handling missing sample keys.

        Please see :meth:`__getitem__()` for full description. This method is
        identical with a single exception: if a sample key is not present in an
        column, this method will plane a null ``None`` value in it's return
        slot rather than throwing a ``KeyError`` like the dict style access
        does.

        Parameters
        ----------
        keys
            sequence of column name (and optionally) sample key(s) or sequence of
            list/tuple of column name, sample keys(s) which should be retrieved in
            the operation.

            Please see detailed explanation in :meth:`__getitem__()` for full
            explanation of accepted argument format / result types.

        default: Any, optional
            default value to insert in results for the case where some column
            name / sample key is not found, and the `except_missing` parameter
            is set to False.

        except_missing: bool, optional
            If False, will not throw exceptions on missing sample key value.
            Will raise KeyError if True and missing key found.

        Returns
        -------
        :class:`~.columns.column.Columns`
            single column parameter, no samples specified
        Any
            Single column specified, single sample key specified
        List[Any]
            arbitrary columns, multiple samples array data for each sample is
            returned in same order sample keys are received.
        """
        return self._get_in(keys, default, except_missing)

    def _get_in(self, keys, default=None, except_missing=False,
                *, _EXCEPTION_CLASSES = (KeyError, IndexError, TypeError)):
        """Internal method to get data from columns within a nested set of dicts.

        Parameters
        ----------
        keys
            sequence of column name (and optionally) sample key(s) or sequence of
            list/tuple of column name, sample keys(s) which should be retrieved in
            the operation.

            Please see detailed explanation in :meth:`__getitem__()` for full
            explanation of accepted argument format / result types.

        default: Any, optional
            default value to insert in results for the case where some column
            name / sample key is not found, and the `except_missing` parameter
            is set to False.

        except_missing: bool, optional
            If False, will not throw exceptions on missing sample key value.
            Will raise KeyError if True and missing key found.

        Returns
        -------
        Any
            Single column specified, single sample key specified
        List[Any]
            arbitrary columns, multiple samples array data for each sample is
            returned in same order sample keys are received.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if isinstance(keys, str):
                return self.columns[keys]

            _COLUMNS = self._columns
            if len(keys) >= 2 and any([isinstance(k, (list, tuple)) for k in keys]):
                res = []
                for key in keys:
                    try:
                        tmp = reduce(op_getitem, key, _COLUMNS)
                        res.append(tmp)
                    except _EXCEPTION_CLASSES:
                        if except_missing:
                            raise
                        res.append(default)
                return res
            else:
                try:
                    return reduce(op_getitem, keys, _COLUMNS)
                except _EXCEPTION_CLASSES:
                    if except_missing:
                        raise
                    return default
