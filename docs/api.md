Python API 
==========

This is the python API for the Hangar project.

Repository
----------

::: hangar.repository

Remotes
-------

::: hangar.remotes.Remotes

Write Enabled Checkout
----------------------

### Checkout

::: hangar.checkout.WriterCheckout

### Columns

::: hangar.columns.column.Columns

### Flat Column Layout Container

::: hangar.columns.layout_flat.FlatSampleWriter

### Nested Column Layout Container

::: hangar.columns.layout_nested.NestedSampleWriter

::: hangar.columns.layout_nested.FlatSubsampleWriter

### Differ

::: hangar.diff.WriterUserDiff

### Bulk Importer

::: hangar.bulk_importer.run_bulk_import

Read Only Checkout
------------------

### Checkout

::: hangar.checkout.ReaderCheckout

### Flat Column Layout Container

::: hangar.columns.layout_flat.FlatSampleReader

### Nested Column Layout Container

::: hangar.columns.layout_nested.NestedSampleReader

:::hangar.columns.layout_nested.FlatSubsampleReader

### Differ

::: hangar.diff.ReaderUserDiff

ML Framework Dataloaders
------------------------

### Tensorflow

::: hangar.make_tf_dataset

### Pytorch

::: hangar.make_torch_dataset
