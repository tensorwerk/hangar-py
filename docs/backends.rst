.. _ref-backends:

.. note::

   The following documentation contains highly technical descriptions of the
   data writing and loading backends of the hangar core. It is intended for
   developer use only, with the functionality described herein being completely
   hidden from regular users.

   Any questions or comments can be directed to the `Hangar Github Issues Page
   <https://github.com/tensorwerk/hangar-py/issues>`_

=================
Backend selection
=================

.. automodule:: hangar.backends.selection


Backend Specifications
======================

.. toctree::
   :maxdepth: 2
   :titlesonly:

   ./backends/local_hdf5
   ./backends/local_np_mmap
   ./backends/remote_unknown