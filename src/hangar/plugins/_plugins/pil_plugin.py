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

from .img_dtype import img_as_ubyte, img_as_uint


def imread(fname, dtype=None, img_num=None, **kwargs):
    """Load an image from file.

    Parameters
    ----------
    fname : str or file
       File name or file-like-object.
    dtype : numpy dtype object or string specifier
       Specifies data type of array elements.
    img_num : int, optional
       Specifies which image to read in a file with multiple images
       (zero-indexed).
    kwargs : keyword pairs, optional
        Addition keyword arguments to pass through.

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
            return pil_to_ndarray(im, dtype=dtype, img_num=img_num)
    else:
        im = Image.open(fname)
        return pil_to_ndarray(im, dtype=dtype, img_num=img_num)


def pil_to_ndarray(image, dtype=None, img_num=None):
    """Import a PIL Image object to an ndarray, in memory.

    Parameters
    ----------
    Refer to ``imread``.

    """
    try:
        # this will raise an IOError if the file is not readable
        image.getdata()[0]
    except IOError as e:
        site = "http://pillow.readthedocs.org/en/latest/installation.html#external-libraries"
        pillow_error_message = str(e)
        error_message = ('Could not load "%s" \n'
                         'Reason: "%s"\n'
                         'Please see documentation at: %s'
                         % (image.filename, pillow_error_message, site))
        raise ValueError(error_message)
    frames = []
    grayscale = None
    i = 0
    while 1:
        try:
            image.seek(i)
        except EOFError:
            break

        frame = image

        if img_num is not None and img_num != i:
            image.getdata()[0]
            i += 1
            continue

        if image.format == 'PNG' and image.mode == 'I' and dtype is None:
            dtype = 'uint16'

        if image.mode == 'P':
            if grayscale is None:
                grayscale = _palette_is_grayscale(image)

            if grayscale:
                frame = image.convert('L')
            else:
                if image.format == 'PNG' and 'transparency' in image.info:
                    frame = image.convert('RGBA')
                else:
                    frame = image.convert('RGB')

        elif image.mode == '1':
            frame = image.convert('L')

        elif 'A' in image.mode:
            frame = image.convert('RGBA')

        elif image.mode == 'CMYK':
            frame = image.convert('RGB')

        if image.mode.startswith('I;16'):
            shape = image.size
            dtype = '>u2' if image.mode.endswith('B') else '<u2'
            if 'S' in image.mode:
                dtype = dtype.replace('u', 'i')
            frame = np.fromstring(frame.tobytes(), dtype)
            frame.shape = shape[::-1]

        else:
            frame = np.array(frame, dtype=dtype)

        frames.append(frame)
        i += 1

        if img_num is not None:
            break

    if hasattr(image, 'fp') and image.fp:
        image.fp.close()

    if img_num is None and len(frames) > 1:
        return np.array(frames)
    elif frames:
        return frames[0]
    elif img_num:
        raise IndexError('Could not find image  #%s' % img_num)


def _palette_is_grayscale(pil_image):
    """Return True if PIL image in palette mode is grayscale.

    Parameters
    ----------
    pil_image : PIL image
        PIL Image that is in Palette mode.

    Returns
    -------
    is_grayscale : bool
        True if all colors in image palette are gray.
    """
    if pil_image.mode != 'P':
        raise ValueError('pil_image.mode must be equal to "P".')
    # get palette as an array with R, G, B columns
    palette = np.asarray(pil_image.getpalette()).reshape((256, 3))
    # Not all palette colors are used; unused colors have junk values.
    start, stop = pil_image.getextrema()
    valid_palette = palette[start:stop + 1]
    # Image is grayscale if channel differences (R - G and G - B)
    # are all zero.
    return np.allclose(np.diff(valid_palette), 0)


def ndarray_to_pil(arr, format_str=None):
    """Export an ndarray to a PIL object.

    Parameters
    ----------
    Refer to ``imsave``.

    """
    if arr.ndim == 3:
        arr = img_as_ubyte(arr)
        mode = {3: 'RGB', 4: 'RGBA'}[arr.shape[2]]

    elif format_str in ['png', 'PNG']:
        mode = 'I;16'
        mode_base = 'I'

        if arr.dtype.kind == 'f':
            arr = img_as_uint(arr)

        elif arr.max() < 256 and arr.min() >= 0:
            arr = arr.astype(np.uint8)
            mode = mode_base = 'L'

        else:
            arr = img_as_uint(arr)

    else:
        arr = img_as_ubyte(arr)
        mode = 'L'
        mode_base = 'L'

    array_buffer = arr.tobytes()

    if arr.ndim == 2:
        im = Image.new(mode_base, arr.T.shape)
        im.frombytes(array_buffer, 'raw', mode)
    else:
        image_shape = (arr.shape[1], arr.shape[0])
        im = Image.frombytes(mode, image_shape, array_buffer)
    return im


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
    All images besides single channel PNGs are converted using `img_as_uint8`.
    Single Channel PNGs have the following behavior:
    - Integer values in [0, 255] and Boolean types -> img_as_uint8
    - Floating point and other integers -> img_as_uint16

    References
    ----------
    .. [1] http://pillow.readthedocs.org/en/latest/handbook/image-file-formats.html
    """
    # default to PNG if file-like object
    if not isinstance(fname, str) and format_str is None:
        format_str = "PNG"
    # Check for png in filename
    if (isinstance(fname, str)
            and fname.lower().endswith(".png")):
        format_str = "PNG"

    arr = np.asanyarray(arr)

    if arr.dtype.kind == 'b':
        arr = arr.astype(np.uint8)

    if arr.ndim not in (2, 3):
        raise ValueError("Invalid shape for image array: %s" % (arr.shape, ))

    if arr.ndim == 3:
        if arr.shape[2] not in (3, 4):
            raise ValueError("Invalid number of channels in image array.")

    img = ndarray_to_pil(arr, format_str=format_str)
    img.save(fname, format=format_str, **kwargs)