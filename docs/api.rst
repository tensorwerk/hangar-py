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


.. autoclass:: hangar.dataset.Datasets
   :members:

.. autoclass:: hangar.dataset.DatasetDataWriter
   :members:
   :inherited-members:


.. autoclass:: hangar.metadata.MetadataWriter
   :members:
   :inherited-members:
   :undoc-members:


Read Only Checkout
==================

.. autoclass:: hangar.checkout.ReaderCheckout
   :members:

.. autoclass:: hangar.dataset.Datasets
   :members: keys, values, items, get

.. autoclass:: hangar.dataset.DatasetDataReader
   :members: keys, values, items, get

.. autoclass:: hangar.metadata.MetadataReader
   :members:
   :undoc-members: