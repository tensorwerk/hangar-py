.. _ref-backends:

.. note::

   The following documentation contains highly technical descriptions of the
   data writing and loading backends of the Hangar core. It is intended for
   developer use only, with the functionality described herein being completely
   hidden from regular users.

   Any questions or comments can be directed to the `Hangar Github Issues Page
   <https://github.com/tensorwerk/hangar-py/issues>`_

=================
Backend selection
=================

.. automodule:: hangar.backends.__init__


Backend Specifications
======================

.. toctree::
   :maxdepth: 2
   :titlesonly:

   ./backends/hdf5_00
   ./backends/hdf5_01
   ./backends/numpy_10
   ./backends/lmdb_30
   ./backends/remote_50
