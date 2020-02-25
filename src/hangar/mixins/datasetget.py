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

            >>> dset['foo', ['1', '2', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]

        Get sample from multiple columns (returns namedtuple of arrays, field
        names = column names)

            >>> dset[('foo', 'bar', 'baz'), '1']
            ColumnData(foo=array([1]), bar=array([11]), baz=array([111]))

        Get multiple samples from multiple columns(returns list of namedtuple
        of array sorted in input key order, field names = column names)

            >>> dset[('foo', 'bar'), ('1', '2')]
            [ColumnData(foo=array([1]), bar=array([11])),
             ColumnData(foo=array([2]), bar=array([22]))]

        Get samples from all columns (shortcut syntax)

            >>> out = dset[:, ('1', '2')]
            >>> out
            [ColumnData(foo=array([1]), bar=array([11]), baz=array([111])),
             ColumnData(foo=array([2]), bar=array([22]), baz=array([222]))]
            >>> out = dset[..., ('1', '2')]
            >>> out
            [ColumnData(foo=array([1]), bar=array([11]), baz=array([111])),
             ColumnData(foo=array([2]), bar=array([22]), baz=array([222]))]
            >>>
            >>> out = dset[:, '1']
            >>> out
            ColumnData(foo=array([1]), bar=array([11]), baz=array([111]))
            >>> out = dset[..., '1']
            >>> out
            ColumnData(foo=array([1]), bar=array([11]), baz=array([111]))

        .. warning::

            It is possible for an :class:`~.columns.column.Columns` name to be an
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

        As an example, if we attempted to retrieve samples from columns with
        the names: ``['raw', 'data.jpeg', '_garba-ge', 'try_2']``, two of the
        four would be renamed:

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> print(out)
            ColumnData(raw=array([0]), _1=array([1]), _2=array([2]), try_2==array([3]))
            >>> print(out._fields)
            ('raw', '_1', '_2', 'try_2')
            >>> out.raw
            array([0])
            >>> out._2
            array([4])

        In cases where the input columns are explicitly specified, then, then
        it is guaranteed that the order of fields in the resulting tuple is
        exactly what was requested in the input

            >>> out = dset[('raw', 'data.jpeg', '_garba-ge', 'try_2'), '1']
            >>> out._fields
            ('raw', '_1', '_2', 'try_2')
            >>> reorder = dset[('data.jpeg', 'try_2', 'raw', '_garba-ge'), '1']
            >>> reorder._fields
            ('_0', 'try_2', 'raw', '_3')

        However, if an ``Ellipsis`` (``...``) or ``slice`` (``:``) syntax is
        used to select columns, *the order in which columns are placed into
        the namedtuple IS NOT predictable.* Should any column have an invalid
        field name, it will be renamed according to it's positional index, but
        will not contain any identifying mark. At the moment, there is no
        direct way to infer the column name from this structure alone. This
        limitation will be addressed in a future release.

        Do NOT rely on any observed patterns. For this corner-case, **the ONLY
        guarantee we provide is that structures returned from multi-sample
        queries have the same order in every ``ColumnData`` tuple returned in
        that queries result list!** Should another query be made with
        unspecified ordering, you should assume that the indices of the
        columns in the namedtuple would have changed from the previous
        result!!

            >>> print(dset.columns.keys())
            ('raw', 'data.jpeg', '_garba-ge', 'try_2']
            >>> out = dset[:, '1']
            >>> out._fields
            ('_0', 'raw', '_2', 'try_2')
            >>>
            >>> # ordering in a single query is preserved
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
            correspond to an column name(s). Alternatively the Ellipsis operator
            (``...``) or unbounded slice operator (``:`` <==> ``slice(None)``) can
            be used to indicate "select all" behavior.

            If a second element (or collection) is present, the keys correspond to
            sample names present within (all) the specified columns. If a key is
            not present in even on column, the entire ``get`` operation will
            abort with ``KeyError``. If desired, the same selection syntax can be
            used with the :meth:`~hangar.checkout.ReaderCheckout.get` method, which
            will not Error in these situations, but simply return ``None`` values
            in the appropriate position for keys which do not exist.

        Returns
        -------
        :class:`~.columns.column.Columns`
            single column parameter, no samples specified

        Any
            Single column specified, single sample key specified

        List[Any]
            Single column, multiple samples array data for each sample is
            returned in same order sample keys are received.

        List[NamedTuple[``*``Any]]
            Multiple columns, multiple samples. Each column's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each column via a common sample key. Each sample
            key is returned values as an individual element in the
            List. The sample order is returned in the same order it was received.

        Warns
        -----
        UserWarning
            Column names contains characters which are invalid as namedtuple fields.

        Notes
        -----

        *  All specified columns must exist

        *  All specified sample `keys` must exist in all specified columns,
           otherwise standard exception thrown

        *  Slice syntax cannot be used in sample `keys` field

        *  Slice syntax for column field cannot specify `start`, `stop`, or
           `step` fields, it is solely a shortcut syntax for 'get all columns' in
           the ``:`` or ``slice(None)`` form

        .. seealso:

            :meth:`~hangar.checkout.ReaderCheckout.get`

        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            if isinstance(index, str):
                return self.columns[index]
            elif not isinstance(index, (tuple, list)):
                raise TypeError(f'Unknown index: {index} type: {type(index)}')
            if len(index) > 2:
                raise ValueError(f'index of len > 2 not allowed: {index}')
            columns, samples = index
            return self.get(columns, samples, except_missing=True)

    def get(self, columns, samples, *, except_missing=False):
        """View of sample data across columns gracefully handling missing sample keys.

        Please see :meth:`__getitem__` for full description. This method is
        identical with a single exception: if a sample key is not present in an
        column, this method will plane a null ``None`` value in it's return
        slot rather than throwing a ``KeyError`` like the dict style access
        does.

        Parameters
        ----------
        columns: Union[str, Iterable[str], Ellipses, slice(None)]

            Name(s) of the columns to query. The Ellipsis operator (``...``)
            or unbounded slice operator (``:`` <==> ``slice(None)``) can be
            used to indicate "select all" behavior.

        samples: Union[str, int, Iterable[Union[str, int]]]

            Names(s) of the samples to query

        except_missing: bool, **KWARG ONLY**

            If False, will not throw exceptions on missing sample key value.
            Will raise KeyError if True and missing key found.

        Returns
        -------
        :class:`~.columns.column.Columns`
            single column parameter, no samples specified

        Any
            Single column specified, single sample key specified

        List[Any]
            Single column, multiple samples array data for each sample is
            returned in same order sample keys are received.

        List[NamedTuple[``*``Any]]
            Multiple columns, multiple samples. Each column's name is used
            as a field in the NamedTuple elements, each NamedTuple contains
            arrays stored in each column via a common sample key. Each sample
            key is returned values as an individual element in the
            List. The sample order is returned in the same order it was received.

        Warns
        -----
        UserWarning
            Column names contains characters which are invalid as namedtuple fields.
        """
        with ExitStack() as stack:
            if not self._is_conman:
                stack.enter_context(self)

            # Column Parsing
            if (columns is Ellipsis) or isinstance(columns, slice):
                columns = list(self._columns._columns.values())
            elif isinstance(columns, str):
                columns = [self._columns._columns[columns]]
            elif isinstance(columns, (tuple, list)):
                columns = [self._columns._columns[aname] for aname in columns]
            else:
                raise TypeError(f'Columns index {columns} of type {type(columns)} invalid.')
            nAsets = len(columns)
            aset_names = [aset.column for aset in columns]
            try:
                ColumnData = namedtuple('ColumnData', aset_names)
            except ValueError:
                warnings.warn(
                    'Column names contains characters which are invalid as namedtuple fields. '
                    'All suspect field names will be replaced by their positional names '
                    '(ie "_0" for element 0, "_4" for element 4)', UserWarning)
                ColumnData = namedtuple('ColumnData', aset_names, rename=True)

            # Sample Parsing
            if isinstance(samples, (str, int)):
                samples = [samples]
            elif not isinstance(samples, (tuple, list)):
                raise TypeError(f'Samples index `{samples}` of type `{type(samples)}` invalid.')
            nSamples = len(samples)

            # Data Retrieval
            asetsSamplesData = []
            for aset in columns:
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
                # noinspection PyUnresolvedReferences
                tmp = map(ColumnData._make, zip(*asetsSamplesData))
                asetsSamplesData = list(tmp)
                if len(asetsSamplesData) == 1:
                    asetsSamplesData = asetsSamplesData[0]

            return asetsSamplesData
