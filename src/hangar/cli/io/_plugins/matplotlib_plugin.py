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
from collections import namedtuple
import numpy as np
try:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.image
except ImportError:
    raise ImportError(
        "Matplotlib could not be found. Please try pip install matplotlib "
        "or refer to http://matplotlib.org for further instructions.")

from warnings import warn


_default_colormap = 'gray'
_nonstandard_colormap = 'viridis'
_diverging_colormap = 'RdBu'


ImageProperties = namedtuple('ImageProperties',
                             ['signed', 'out_of_range_float', 'unsupported_dtype'])

# For integers Numpy uses `_integer_types` basis internally, and builds a leaky
# `np.XintYY` abstraction on top of it. This leads to situations when, for
# example, there are two np.Xint64 dtypes with the same attributes but
# different object references. In order to avoid any potential issues, we use
# the basis dtypes here. For more information, see:
# - https://github.com/scikit-image/scikit-image/issues/3043 For convenience,
#   for these dtypes we indicate also the possible bit depths (some of them are
#   platform specific). For the details, see:
#   http://www.unix.org/whitepapers/64bit.html
_integer_types = (np.byte, np.ubyte,          # 8 bits
                  np.short, np.ushort,        # 16 bits
                  np.intc, np.uintc,          # 16 or 32 or 64 bits
                  np.int_, np.uint,           # 32 or 64 bits
                  np.longlong, np.ulonglong)  # 64 bits
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)

_supported_types = list(dtype_range.keys())


def _get_image_properties(image):
    """Determine nonstandard properties of an input image.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    ip : ImageProperties named tuple
        The properties of the image:

        - signed: whether the image has negative values.
        - out_of_range_float: if the image has floating point data
          outside of [-1, 1].
        - unsupported_dtype: if the image data type is not a
          standard skimage type, e.g. ``numpy.uint64``.
    """
    immin, immax = np.min(image), np.max(image)
    imtype = image.dtype.type
    try:
        lo, hi = dtype_range[imtype]
    except KeyError:
        lo, hi = immin, immax

    signed = immin < 0
    out_of_range_float = (np.issubdtype(image.dtype, np.floating) and
                          (immin < lo or immax > hi))
    unsupported_dtype = image.dtype not in _supported_types

    return ImageProperties(signed, out_of_range_float, unsupported_dtype)


def _raise_warnings(image_properties):  # pragma: no cover
    """Raise the appropriate warning for each nonstandard image type.

    Parameters
    ----------
    image_properties : ImageProperties named tuple
        The properties of the considered image.
    """
    ip = image_properties
    if ip.unsupported_dtype:
        warn("Non-standard image type; displaying image with "
             "stretched contrast.")
    if ip.out_of_range_float:
        warn("Float image out of standard range; displaying "
             "image with stretched contrast.")


def _get_display_range(image):  # pragma: no cover
    """Return the display range for a given set of image properties.

    Parameters
    ----------
    image : array
        The input image.

    Returns
    -------
    lo, hi : same type as immin, immax
        The display range to be used for the input image.
    cmap : string
        The name of the colormap to use.
    """
    ip = _get_image_properties(image)
    immin, immax = np.min(image), np.max(image)
    if ip.signed:
        magnitude = max(abs(immin), abs(immax))
        lo, hi = -magnitude, magnitude
        cmap = _diverging_colormap
    elif any(ip):
        _raise_warnings(ip)
        lo, hi = immin, immax
        cmap = _nonstandard_colormap
    else:
        lo = 0
        imtype = image.dtype.type
        hi = dtype_range[imtype][1]
        cmap = _default_colormap
    return lo, hi, cmap


def imshow(image, ax=None, show_cbar=None, **kwargs):
    """Show the input image and return the current axes.

    By default, the image is displayed in greyscale, rather than
    the matplotlib default colormap.

    Images are assumed to have standard range for their type. For
    example, if a floating point image has values in [0, 0.5], the
    most intense color will be gray50, not white.

    If the image exceeds the standard range, or if the range is too
    small to display, we fall back on displaying exactly the range of
    the input image, along with a colorbar to clearly indicate that
    this range transformation has occurred.

    For signed images, we use a diverging colormap centered at 0.

    Parameters
    ----------
    image : array, shape (M, N[, 3])
        The image to display.
    ax: `matplotlib.axes.Axes`, optional
        The axis to use for the image, defaults to plt.gca().
    show_cbar: boolean, optional.
        Whether to show the colorbar (used to override default behavior).
    **kwargs : Keyword arguments
        These are passed directly to `matplotlib.pyplot.imshow`.

    Returns
    -------
    ax_im : `matplotlib.pyplot.AxesImage`
        The `AxesImage` object returned by `plt.imshow`.
    """
    import matplotlib.pyplot as plt

    lo, hi, cmap = _get_display_range(image)

    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', cmap)
    kwargs.setdefault('vmin', lo)
    kwargs.setdefault('vmax', hi)

    ax = ax or plt.gca()
    ax_im = ax.imshow(image, **kwargs)
    if (cmap != _default_colormap and show_cbar is not False) or show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(ax_im, cax=cax)
    ax.get_figure().tight_layout()

    return ax_im


imread = matplotlib.image.imread


def _app_show():  # pragma: no cover
    from matplotlib.pyplot import show
    show()