==========
Python API
==========

This is the python API for the Hangar project.


Repository
==========

.. automodule:: hangar.repository
   :members:


Write Enabled Checkout
======================

.. autoclass:: hangar.checkout.WriterCheckout
   :members:

Datasets
--------

.. autoclass:: hangar.dataset.Datasets
   :members: keys, values, items, get, iswriteable, add, init_dataset, remove_dset
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Dataset Data
------------

.. autoclass:: hangar.dataset.DatasetDataWriter
   :members: keys, values, items, get, name, dtype, shape, variable_shape, named_samples, iswriteable, add, remove
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataWriter
   :members: keys, values, items, get, iswriteable, add, remove
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__

Differ
------

.. autoclass:: hangar.diff.WriterUserDiff
   :members: commit, commit_hash, branch_name, status, branch, staged


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
   :members: keys, values, items, get, name, dtype, shape, variable_shape, named_samples, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataReader
   :members: keys, values, items, get, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__

Differ
------

.. autoclass:: hangar.diff.ReaderUserDiff
   :members: commit, commit_hash, branch