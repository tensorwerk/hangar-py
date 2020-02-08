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

.. autoclass:: hangar.columns.column.Arraysets()
   :members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Sample Level Arrayset Data
--------------------------

.. autoclass:: hangar.columns.SampleWriterModifier()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Subsample Level Arrayset Data
-----------------------------

.. autoclass:: hangar.columns.SubsampleWriterModifier()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


.. autoclass:: hangar.columns.SubsampleWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Metadata
--------

.. autoclass:: hangar.columns.metadata.MetadataWriter()
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

.. autoclass:: hangar.columns.column.Arraysets()
   :members: keys, values, items, get, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Sample Level Arrayset Data
--------------------------

.. autoclass:: hangar.columns.SampleReaderModifier()
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Subsample Level Arrayset Data
-----------------------------

.. autoclass:: hangar.columns.SubsampleReaderModifier()
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


.. autoclass:: hangar.columns.SubsampleReader()
   :members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Metadata
--------

.. autoclass:: hangar.columns.metadata.MetadataReader()
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
