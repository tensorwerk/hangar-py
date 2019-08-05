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
   :exclude-members: __init__

Cellstores
----------

.. autoclass:: hangar.cellstore.Cellstores()
   :members:
   :special-members: __getitem__, __setitem__, __delitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Cellstore Data
--------------

.. autoclass:: hangar.cellstore.CellstoreDataWriter()
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
   :exclude-members: __init__

Cellstores
----------

.. autoclass:: hangar.cellstore.Cellstores()
   :members: keys, values, items, get, iswriteable
   :special-members: __getitem__, __contains__, __len__, __iter__
   :exclude-members: __init__

Cellstore Data
--------------

.. autoclass:: hangar.cellstore.CellstoreDataReader()
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
