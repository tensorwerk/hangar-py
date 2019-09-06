.. _ref-api:

==========
Python API
==========

This is the python API for the Hangar project.


Repository
==========

.. automodule:: hangar.repository
   :members:

.. autoclass:: Remotes()
   :members:
   :exclude-members: __init__

Write Enabled Checkout
======================

.. autoclass:: hangar.checkout.WriterCheckout()
   :members:
   :special-members: __getitem__, __setitem__
   :exclude-members: __init__

Arraysets
---------

.. autoclass:: hangar.arrayset.Arraysets()
   :members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Arrayset Data
-------------

.. autoclass:: hangar.arrayset.ArraysetDataWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Differ
------

.. autoclass:: hangar.diff.WriterUserDiff()
   :members:
   :exclude-members: __init__


Read Only Checkout
==================

.. autoclass:: hangar.checkout.ReaderCheckout()
   :members:
   :special-members: __getitem__
   :exclude-members: __init__

Arraysets
----------

.. autoclass:: hangar.arrayset.Arraysets()
   :members: keys, values, items, get, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Arrayset Data
-------------

.. autoclass:: hangar.arrayset.ArraysetDataReader()
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Metadata
--------

.. autoclass:: hangar.metadata.MetadataReader()
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Differ
------

.. autoclass:: hangar.diff.ReaderUserDiff()
   :members:
   :exclude-members: __init__


ML Framework Dataloaders
========================

Tensorflow
----------

.. autofunction:: hangar.make_tf_dataset

Pytorch
-------

.. autofunction:: hangar.make_torch_dataset
