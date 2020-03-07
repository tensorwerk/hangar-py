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

Checkout
--------

.. autoclass:: hangar.checkout.WriterCheckout()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __len__, __contains__, __iter__
   :exclude-members: __init__

Columns
-------

.. autoclass:: hangar.columns.column.Columns()
   :members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Flat Column Layout Container
----------------------------

.. autoclass:: hangar.columns.layout_flat.FlatSampleWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Nested Column Layout Container
------------------------------

.. autoclass:: hangar.columns.layout_nested.NestedSampleWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

.. autoclass:: hangar.columns.layout_nested.FlatSubsampleWriter()
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

Checkout
--------

.. autoclass:: hangar.checkout.ReaderCheckout()
   :members:
   :inherited-members:
   :special-members: __getitem__, __len__, __contains__, __iter__
   :exclude-members: __init__


Flat Column Layout Container
----------------------------

.. autoclass:: hangar.columns.layout_flat.FlatSampleWriter()
   :members:
   :inherited-members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__


Nested Column Layout Container
------------------------------

.. autoclass:: hangar.columns.layout_nested.NestedSampleReader()
   :members:
   :inherited-members:
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

.. autoclass:: hangar.columns.layout_nested.FlatSubsampleReader()
   :members:
   :inherited-members:
   :special-members: __getitem__,, __contains__, __len__, __iter__
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
