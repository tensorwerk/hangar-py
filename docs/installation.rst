.. _ref_installation:

============
Installation
============

For general usage it is recommended that you use a pre-built version of Hangar,
either from a Python Distribution, or a pre-built wheel from PyPi.


Pre-Built Installation
======================


Python Distributions
--------------------

If you do not already use a Python Distribution, we recommend the `Anaconda
<https://www.anaconda.com/distribution/>`_ (or `Miniconda
<https://docs.conda.io/en/latest/miniconda.html>`_) distribution, which supports
all major operating systems (Windows, MacOSX, & the typical Linux variations).
Detailed usage instructions are available `on the anaconda website
<https://docs.anaconda.com/anaconda/>`_.

To install Hangar via the Anaconda Distribution (from the `conda-forge conda
channel <https://anaconda.org/conda-forge/hangar>`_)::

    conda install -c conda-forge hangar


Wheels (PyPi)
-------------

If you have an existing python installation on your computer, pre-built Hangar Wheels
can be installed via pip from the Python Package Index (PyPi)::

    pip install hangar


Source Installation
===================


To install Hangar from source, clone the repository from `Github
<https://github.com/tensorwerk/hangar-py>`_::

    git clone https://github.com/tensorwerk/hangar-py.git
    cd hangar-py
    python setup.py install

Or use pip on the local package if you want to install all dependencies
automatically in a development environment::

    pip install -e .


Source installation in Google colab
-----------------------------------
Google colab comes with an older version of ``h5py`` pre-installed which is not
compatible with hangar. If you need to install hangar from the source in 
google colab, make sure to uninstall the existing ``h5py`` ::

    !pip uninstall h5py

Then follow the Source Installation steps given above.
