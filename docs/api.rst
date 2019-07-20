==========
Python API
==========

This is the python API for the Hangar project.


Repository
==========

.. automodule:: hangar.repository
   :members:

.. autoclass:: Remotes
   :members:

Write Enabled Checkout
======================

.. autoclass:: hangar.checkout.WriterCheckout
   :members:

Datasets
--------

.. autoclass:: hangar.dataset.Datasets
   :members: keys, values, items, get, iswriteable, multi_add, init_dataset, remove_dset
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Dataset Data
------------

.. autoclass:: hangar.dataset.DatasetDataWriter
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataWriter
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Differ
------

.. autoclass:: hangar.diff.WriterUserDiff
   :members:


Read Only Checkout
==================

.. autoclass:: hangar.checkout.ReaderCheckout
   :members:

Datasets
--------

.. autoclass:: hangar.dataset.Datasets
   :members: keys, values, items, get, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__

Dataset Data
------------

.. autoclass:: hangar.dataset.DatasetDataReader
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataReader
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__

Differ
------

.. autoclass:: hangar.diff.ReaderUserDiff
   :members: