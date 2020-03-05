from functools import reduce
from operator import getitem as op_getitem
from contextlib import ExitStack

from collections import namedtuple
import warnings


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

        Get an column contained in the checkout.

            >>> dset['foo']
            ColumnDataReader

        Get a specific sample from ``'foo'`` (returns a single array)

            >>> dset['foo', '1']
            np.array([1])

        Get multiple samples from ``'foo'`` (returns a list of arrays, in order
        of input keys)

            >>> dset[['foo', '1'], ['foo', '2'],  ['foo', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]

        Get sample from multiple columns, column/data returned is ordered
        in same manner as input of func.

            >>> dset[['foo', '1'], ['bar', '1'],  ['baz', '1']]
            [np.array([1]), np.ndarray([1, 1]), np.ndarray([1, 1, 1])]

        Get multiple samples from multiple columns(returns list of namedtuple
        of array sorted in input key order, field names = column names)

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
            >>> # a sample accessor object can be retrieved at will...
            >>> dset['nested_col', 'sample_1']
            <class 'FlatSubsampleReader'>(column_name='nested_col', sample_name='sample_1')
            >>>
            >>> # to get all subsamples in a nested sample use the Ellipsis operator
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


        TODO: REMOVE/Rework following documentation
        #
        # Get samples from all columns (shortcut syntax)
        #
        #     >>> out = dset[:, ('1', '2')]
        #     >>> out
        #     [ColumnData(foo=array([1]), bar=array([11]), baz=array([111])),
        #      ColumnData(foo=array([2]), bar=array([22]), baz=array([222]))]
        #     >>> out = dset[..., ('1', '2')]
        #     >>> out
        #     [ColumnData(foo=array([1]), bar=array([11]), baz=array([111])),
        #      ColumnData(foo=array([2]), bar=array([22]), baz=array([222]))]
        #     >>>
        #     >>> out = dset[:, '1']
        #     >>> out
        #     ColumnData(foo=array([1]), bar=array([11]), baz=array([111]))
        #     >>> out = dset[..., '1']
        #     >>> out
        #     ColumnData(foo=array([1]), bar=array([11]), baz=array([111]))
        #
        #
        # .. warning::
        #
        #     It is possible for an :class:`~.columns.column.Columns` name to be an
        #     invalid field name for a ``namedtuple`` object. The python docs state:
        #
        #         Any valid Python identifier may be used for a fieldname except for
        #         names starting with an underscore. Valid identifiers consist of
        #         letters, digits, and underscores but do not start with a digit or
        #         underscore and cannot be a keyword such as class, for, return,
        #         global, pass, or raise.
        #
        #     In addition, names must be unique, and cannot containing a period
        #     (``.``) or dash (``-``) character. If a namedtuple would normally be
        #     returned during some operation, and the field name is invalid, a
        #     :class:`UserWarning` will be emitted indicating that any suspect fields
        #     names will be replaced with the positional index as is customary in the
        #     python standard library. The ``namedtuple`` docs explain this by
        #     saying:
        #
        #         If rename is true, invalid fieldnames are automatically replaced with
        #         positional names. For example, ['abc', 'def', 'ghi', 'abc'] is
        #         converted to ['abc', '_1', 'ghi', '_3'], eliminating the keyword def
        #         and the duplicate fieldname abc.
        #
        #     The next section demonstrates the implications and workarounds for this
        #     issue
        #
        # As an example, if we attempted to retrieve samples from columns with
        # the names: ``['raw', 'data.jpeg', '_garba-ge', 'try_2']``, two of the
        # four would be renamed:
        #
        #     >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
        #     >>> print(out)
        #     ColumnData(raw=array([0]), _1=array([1]), _2=array([2]), try_2==array([3]))
        #     >>> print(out._fields)
        #     ('raw', '_1', '_2', 'try_2')
        #     >>> out.raw
        #     array([0])
        #     >>> out._2
        #     array([4])
        #
        # In cases where the input columns are explicitly specified, then, then
        # it is guaranteed that the order of fields in the resulting tuple is
        # exactly what was requested in the input
        #
        #     >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
        #     >>> out._fields
        #     ('raw', '_1', '_2', 'try_2')
        #     >>> reorder = dset[('data.jpeg', 'try_2', 'raw', '_garba-ge'), '1']
        #     >>> reorder._fields
        #     ('_0', 'try_2', 'raw', '_3')
        #
        # However, if an ``Ellipsis`` (``...``) or ``slice`` (``:``) syntax is
        # used to select columns, *the order in which columns are placed into
        # the namedtuple IS NOT predictable.* Should any column have an invalid
        # field name, it will be renamed according to it's positional index, but
        # will not contain any identifying mark. At the moment, there is no
        # direct way to infer the column name from this structure alone. This
        # limitation will be addressed in a future release.
        #
        # Do NOT rely on any observed patterns. For this corner-case, **the ONLY
        # guarantee we provide is that structures returned from multi-sample
        # queries have the same order in every ``ColumnData`` tuple returned in
        # that queries result list!** Should another query be made with
        # unspecified ordering, you should assume that the indices of the
        # columns in the namedtuple would have changed from the previous
        # result!!
        #
        #     >>> print(dset.columns.keys())
        #     ('raw', 'data.jpeg', '_garba-ge', 'try_2']
        #     >>> out = dset[:, '1']
        #     >>> out._fields
        #     ('_0', 'raw', '_2', 'try_2')
        #     >>>
        #     >>> # ordering in a single query is preserved
        #     >>> multi_sample = dset[..., ('1', '2')]
        #     >>> multi_sample[0]._fields
        #     ('try_2', '_1', 'raw', '_3')
        #     >>> multi_sample[1]._fields
        #     ('try_2', '_1', 'raw', '_3')
        #     >>>
        #     >>> # but it may change upon a later query
        #     >>> multi_sample2 = dset[..., ('1', '2')]
        #     >>> multi_sample2[0]._fields
        #     ('_0', '_1', 'raw', 'try_2')
        #     >>> multi_sample2[1]._fields
        #     ('_0', '_1', 'raw', 'try_2')

        Parameters
        ----------
        index
            Please see detailed explanation above for full options. Hard coded
            options are the order to specification.

            # TODO: Rewrite documentation
            #
            # The first element (or collection) specified must be ``str`` type and
            # correspond to an column name(s). Alternatively the Ellipsis operator
            # (``...``) or unbounded slice operator (``:`` <==> ``slice(None)``) can
            # be used to indicate "select all" behavior.
            #
            # If a second element (or collection) is present, the keys correspond to
            # sample names present within (all) the specified columns. If a key is
            # not present in even on column, the entire ``get`` operation will
            # abort with ``KeyError``. If desired, the same selection syntax can be
            # used with the :meth:`~hangar.checkout.ReaderCheckout.get` method, which
            # will not Error in these situations, but simply return ``None`` values
            # in the appropriate position for keys which do not exist.

        Returns
        -------
        :class:`~.columns.column.Columns`
            single column parameter, no samples specified

        Any
            Single column specified, single sample key specified

        List[Any]
            arbitrary columns, multiple samples array data for each sample is
            returned in same order sample keys are received.


        .. seealso:

            :meth:`~hangar.checkout.ReaderCheckout.get`
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if isinstance(index, str):
                return self.columns[index]
            else:
                return self.get_in(index, except_missing=True)

    def get(self, keys, default=None, except_missing=False):
        """View of sample data across columns gracefully handling missing sample keys.

        Please see :meth:`__getitem__` for full description. This method is
        identical with a single exception: if a sample key is not present in an
        column, this method will plane a null ``None`` value in it's return
        slot rather than throwing a ``KeyError`` like the dict style access
        does.

        # TODO: Rewrite docstring
        #
        # Parameters
        # ----------
        # columns: Union[str, Iterable[str], Ellipses, slice(None)]
        #
        #     Name(s) of the columns to query. The Ellipsis operator (``...``)
        #     or unbounded slice operator (``:`` <==> ``slice(None)``) can be
        #     used to indicate "select all" behavior.
        #
        # samples: Union[str, int, Iterable[Union[str, int]]]
        #
        #     Names(s) of the samples to query
        #
        # except_missing: bool, **KWARG ONLY**
        #
        #     If False, will not throw exceptions on missing sample key value.
        #     Will raise KeyError if True and missing key found.

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
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if isinstance(keys, str):
                return self.columns[keys]

            return self.get_in(keys, default, except_missing)

    def get_in(self, keys, default=None, except_missing=False):
        """Returns coll[i0][i1]...[iX] where [i0, i1, ..., iX]==keys.

        If column[i0][i1]...[iX] cannot be found, returns ``default``, unless
        ``except_missing`` is specified, then it raises KeyError or IndexError.

        ``get_in`` is a generalization of ``operator.getitem`` for nested data
        structures such as dictionaries and lists.

        Parameters
        ----------
        keys
        default
        except_missing

        Returns
        -------

        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if len(keys) >= 2 and any([isinstance(k, (list, tuple)) for k in keys]):
                res = []
                for key in keys:
                    try:
                        res.append(reduce(op_getitem, key, self.columns))
                    except (KeyError, IndexError, TypeError):
                        if except_missing:
                            raise
                        res.append(default)
                return res
            else:
                try:
                    return reduce(op_getitem, keys, self.columns)
                except (KeyError, IndexError, TypeError):
                    if except_missing:
                        raise
                    return default
