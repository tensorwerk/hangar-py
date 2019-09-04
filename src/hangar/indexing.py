"""Indexing base class for checkout objects.

These methods allow a checkout object to be viewed and accessed in manners
similar to a "dataset".
"""
from collections import namedtuple


class CheckoutIndexer(object):

    def __init__(self):
        self._is_conman = False

    def __getitem__(self, index):
        """Dictionary style access to arraysets and samples

        Checkout object can be thought of as a "dataset" ("dset") mapping a view
        of samples across arraysets:

            >>> dset = repo.checkout(branch='master')

        Get an arrayset of the dataset (i.e. a "column" of the dataset?)

            >>> dset['foo']
            ArraysetDataReader

        Get a specific sample from 'foo' (returns a single array)

            >>> dset['foo', '1']
            np.array([1])

        Get multiple samples from 'foo' (retuns a list of arrays, in order of input keys)

            >>> dset['foo', ['1', '2', '324']]
            [np.array([1]), np.ndarray([2]), np.ndarray([324])]

        Get sample from multiple arraysets (returns namedtuple of arrays, field
        names = arrayset names)

            >>> dset[('foo', 'bar', 'baz'), '1']
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        Get multiple samples from multiple arraysets(returns list of namedtuple of
        array sorted in input key order, field names = arrayset names)

            >>> dset[('foo', 'bar'), ('1', '2')]
            [ArraysetData(foo=array([1]), bar=array([11])),
             ArraysetData(foo=array([2]), bar=array([22]))]

        Get samples from all arraysets (shortcut syntax)

            Example 1:
            >>> out = dset[:, ('1', '2')]
            >>> out = dset[..., ('1', '2')]
            >>> out
            [ArraysetData(foo=array([1]), bar=array([11]), baz=array([111])),
             ArraysetData(foo=array([2]), bar=array([22]), baz=array([222]))]

            Example 2:
            >>> out = dset[:, '1']
            >>> out = dset[..., '1']
            >>> out
            ArraysetData(foo=array([1]), bar=array([11]), baz=array([111]))

        Notes
        -----

        1. All specified arraysets must exist
        2. All specified sample `keys` must exist in all specified arraysets,
           otherwise standard exception thrown
        3. Slice syntax cannot be used in sample `keys` field
        4. Slice syntax for arrayset field cannot specify `start`, `stop`, or
           `step` fields, it is soley a shortcut syntax for 'get all arraysets' in
           the ``:`` or ``slice(None)`` form
        """
        if isinstance(index, str):
            return self.arraysets[index]
        elif not isinstance(index, (tuple, list)):
            raise TypeError(f'Unknown index: {index} type: {type(index)}')
        if len(index) > 2:
            raise ValueError(f'index of len > 2 not allowed: {index}')

        arraysets, samples = index
        return self.get(arraysets, samples, _except_missing_key=True)

    def get(self, arraysets, samples, *, _except_missing_key=False):
        """

        One difference is that if a sample key is not present in an arrayset,
        the corresponding value returned will be ``None`` instead of raising a
        ``KeyError``.
        """
        try:
            tmpconman = not self._is_conman
            if tmpconman:
                self.__enter__()

            # Arrayset Parsing
            if (arraysets is (Ellipsis, slice)):
                arraysets = self._arraysets.keys()
            elif isinstance(arraysets, str):
                arraysets = [arraysets]
            elif not isinstance(arraysets, (tuple, list)):
                raise TypeError(f'Arraysets: {arraysets} type: {type(arraysets)}')
            nAsets = len(arraysets)
            ArraysetData = namedtuple('ArraysetData', arraysets)

            # Sample Parsing
            if isinstance(samples, (str, int)):
                samples = [samples]
            elif not isinstance(samples, (tuple, list)):
                raise TypeError(f'Samples idx: {samples} type: {type(samples)}')
            nSamples = len(samples)

            # Data Retrieval
            asetsSamplesData = []
            for sample in samples:
                if nAsets == 1:
                    for aset in arraysets:
                        try:
                            asetsData = self._arraysets._arraysets[aset].get(sample)
                        except KeyError as e:
                            if _except_missing_key:
                                raise e
                            else:
                                asetsData = None
                else:
                    arrays = []
                    for aset in arraysets:
                        try:
                            arr = self._arraysets._arraysets[aset].get(sample)
                            arrays.append(arr)
                        except KeyError as e:
                            if _except_missing_key:
                                raise e
                            else:
                                arrays.append(None)
                    asetsData = ArraysetData(*arrays)

                if nSamples == 1:
                    asetsSamplesData = asetsData
                else:
                    asetsSamplesData.append(asetsData)

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

        Notes
        -----

        1. No slicing syntax is supported for either arraysets or samples. This
           is in order to ensure explicit setting of values in the desired
           fields/keys
        2. Add multiple samples to multiple arraysets not yet supported.
        """

        try:
            tmpconman = not self._is_conman
            if tmpconman:
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
                asets = []
                for asetIdx in asetsIdx:
                    asets.append(self._arraysets._arraysets[asetIdx])
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
                for aset in asets:
                    for sampleName in sampleNames:
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

    def __enter__(self):
        self._is_conman = True
        self._arraysets.__enter__()
        return self

    def __exit__(self, *exc):
        self._is_conman = False
        self._arraysets.__exit__(*exc)
