"""
Portions of this code have been adapted from the "scikit-image" project.


URL:      https://github.com/scikit-image/scikit-image
Commit:   bde5a9bc3106d68ab9a4ca3dfed4f866fdd6a129
Accessed: 06 AUG 2019

Scikit-Image License
-------------------------------------------------------------------------------
Copyright (C) 2019, the scikit-image team
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

 1. Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the
    distribution.
 3. Neither the name of skimage nor the names of its contributors may be
    used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

__all__ = ['imread', 'imsave']

import numpy as np
from PIL import Image


def imread(fname):
    """Load an image from file.

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.

    Notes
    -----
    Files are read using the Python Imaging Library.
    See PIL docs [1]_ for a list of supported formats.

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    """
    if isinstance(fname, str):
        with open(fname, 'rb') as f:
            im = Image.open(f)
            return np.array(im)
    else:
        im = Image.open(fname)
        return np.array(im)


def imsave(fname, arr, format_str=None, **kwargs):
    """Save an image to disk.

    Parameters
    ----------
    fname : str or file-like object
        Name of destination file.
    arr : ndarray of uint8 or float
        Array (image) to save.  Arrays of data-type uint8 should have
        values in [0, 255], whereas floating-point arrays must be
        in [0, 1].
    format_str: str
        Format to save as, this is defaulted to PNG if using a file-like
        object; this will be derived from the extension if fname is a string
    kwargs: dict
        Keyword arguments to the Pillow save function (or tifffile save
        function, for Tiff files). These are format dependent. For example,
        Pillow's JPEG save function supports an integer ``quality`` argument
        with values in [1, 95], while TIFFFile supports a ``compress``
        integer argument with values in [0, 9].

    Notes
    -----
    Use the Python Imaging Library.
    See PIL docs [1]_ for a list of other supported formats.

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    """
    # default to PNG if file-like object
    if not isinstance(fname, str) and format_str is None:
        format_str = "PNG"
    # Check for png in filename
    if (isinstance(fname, str) and fname.lower().endswith(".png")):
        format_str = "PNG"

    if arr.dtype.kind == 'b':
        arr = arr.astype(np.uint8)

    if arr.ndim not in (2, 3):
        raise ValueError("Invalid shape for image array: %s" % (arr.shape, ))

    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError("Invalid number of channels in image array.")

    img = Image.fromarray(arr)
    img.save(fname, format=format_str, **kwargs)