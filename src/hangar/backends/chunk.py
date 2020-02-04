"""
Portions of this code have been taken and modified from the "PyTables" project.

URL:      https://github.com/PyTables/PyTables
File:     tables/leaf.py
Commit:   1e7b14e87507c2392265321fe18b2f1f5920ea7f
Accessed: 23 JAN 2020

PyTables License
-------------------------------------------------------------------------------
License: BSD
URL:     https://github.com/PyTables/PyTables/blob/1e7b14e875/LICENSE.txt
"""
import numpy as np
import math


SizeType = np.int64


def _csformula(expected_mb):
    """Return the fitted chunksize for expected_mb."""

    # For a basesize of 8 KB, this will return:
    # 8 KB for datasets <= 1 MB
    # 1 MB for datasets >= 10 TB
    basesize = 8 * 1024  # 8 KB is a good minimum
    return basesize * int(2 ** math.log10(expected_mb))


def _limit_es(expected_mb):
    """Protection against creating too small or too large chunks."""

    if expected_mb < 1:  # < 1 MB
        expected_mb = 1
    elif expected_mb > 10 ** 7:  # > 10 TB
        expected_mb = 10 ** 7
    return expected_mb


def _calc_chunksize(expected_mb):
    """Compute the optimum HDF5 chunksize for I/O purposes.

    Rational: HDF5 takes the data in bunches of chunksize length to write the
    on disk. A BTree in memory is used to map structures on disk. The more
    chunks that are allocated for a dataset the larger the B-tree. Large
    B-trees take memory and causes file storage overhead as well as more disk
    I/O and higher contention for the meta data cache.  You have to balance
    between memory and I/O overhead (small B-trees) and time to access to data
    (big B-trees). The tuning of the chunksize parameter affects the
    performance and the memory consumed. This is based on my own experiments
    and, as always, your mileage may vary.
    """

    expected_mb = _limit_es(expected_mb)
    zone = int(math.log10(expected_mb))
    expected_mb = 10 ** zone
    chunksize = _csformula(expected_mb)
    # XXX: Multiply by 8 seems optimal for sequential access
    return chunksize * 24


def _rowsize(shape, maindim, itemsize):
    """"The size of the rows in bytes in dimensions orthogonal to *maindim*."

    shape:
        Shape of the sample to fit in the row

    maindim:
        The dimension along which iterators work. Its value is 0 (i.e. the first
        dimension) when the dataset is not extendable, and self.extdim (where
        available) for extendable ones.

    itemsize:
        nbytes of each element

    The meaning of *atomic* is that individual elements of a cell can not be
    extracted directly by indexing (i.e.  __getitem__()) the dataset; e.g. if a
    dataset has shape (2, 2) and its atoms have shape (3,), to get the third
    element of the cell at (1, 0) one should use dataset[1,0][2] instead of
    dataset[1,0,2].
    """
    rowsize = itemsize
    for i, dim in enumerate(shape):
        if i != maindim:
            rowsize *= dim
    return rowsize


def calc_chunkshape(shape, expectedrows, itemsize, maindim):
    """Calculate the shape for the HDF5 chunk.

    shape:
        Shape of the sample to fit in the row

    expectedrows:
        how many samples will fit into the file container

    itemsize:
        nbytes of each element

    maindim:
        The dimension along which iterators work. Its value is 0 (i.e. the first
        dimension) when the dataset is not extendable, and self.extdim (where
        available) for extendable ones.

        may want to set to shape.index(max(shape))
    """

    # In case of a scalar shape, return the unit chunksize
    if shape == ():
        return (SizeType(1),)

    MB = 1024 * 1024
    # if shape is sufficiently small, no need to further chunk
    # At time of writing, set to be less than 1MB since that is
    # the limit to hdf5 chunk cache.
    if ((np.prod(shape) * itemsize) < MB) and (shape != ()):
        return shape

    # Compute the chunksize
    rsize = _rowsize(shape, maindim, itemsize)
    expected_mb = (expectedrows * rsize) // MB
    chunksize = _calc_chunksize(expected_mb)

    # Compute the chunknitems
    chunknitems = chunksize // itemsize
    # Safeguard against itemsizes being extremely large
    if chunknitems == 0:
        chunknitems = 1
    chunkshape = list(shape)
    # Check whether trimming the main dimension is enough
    chunkshape[maindim] = 1
    newchunknitems = np.prod(chunkshape, dtype=SizeType)
    if newchunknitems <= chunknitems:
        chunkshape[maindim] = chunknitems // newchunknitems
    else:
        # No, so start trimming other dimensions as well
        for j in range(len(chunkshape)):
            # Check whether trimming this dimension is enough
            chunkshape[j] = 1
            newchunknitems = np.prod(chunkshape, dtype=SizeType)
            if newchunknitems <= chunknitems:
                chunkshape[j] = chunknitems // newchunknitems
                break
        else:
            # Ops, we ran out of the loop without a break
            # Set the last dimension to chunknitems
            chunkshape[-1] = chunknitems

    # safeguard outputing chunks which are larger than shape
    if chunkshape[maindim] > shape[maindim]:
        chunkshape[maindim] = shape[maindim]

    return tuple(SizeType(s) for s in chunkshape)
