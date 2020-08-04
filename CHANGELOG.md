Change Log
==========

[v0.5.2](https://github.com/tensorwerk/hangar-py/compare/v0.5.1...v0.5.2) (2020-05-08)
--------------------------------------------------------------------------------------

### New Features

-   New column data type supporting arbitrary `bytes` data.
    ([\#198](https://github.com/tensorwerk/hangar-py/pull/198))
    [\@rlizzo](https://github.com/rlizzo)

### Improvements

-   `str` typed columns can now accept data containing any unicode
    code-point. In prior releases data containing any `non-ascii`
    character could not be written to this column type.
    ([\#198](https://github.com/tensorwerk/hangar-py/pull/198))
    [\@rlizzo](https://github.com/rlizzo)

### Bug Fixes

-   Fixed issue where `str` and (newly added) `bytes` column data could
    not be fetched / pushed between a local client repository and remote
    server. ([\#198](https://github.com/tensorwerk/hangar-py/pull/198))
    [\@rlizzo](https://github.com/rlizzo)

[v0.5.1](https://github.com/tensorwerk/hangar-py/compare/v0.5.0...v0.5.1) (2020-04-05)
--------------------------------------------------------------------------------------

### BugFixes

-   Fixed issue where importing `make_torch_dataloader` or
    `make_tf_dataloader` under python 3.6 Would raise a `NameError`
    irrigardless of if the package is installed.
    ([\#196](https://github.com/tensorwerk/hangar-py/pull/196))
    [\@rlizzo](https://github.com/rlizzo)

[v0.5.0](https://github.com/tensorwerk/hangar-py/compare/v0.4.0...v0.5.0) (2020-04-4)
-------------------------------------------------------------------------------------

### Improvements

-   Python 3.8 is now fully supported.
    ([\#193](https://github.com/tensorwerk/hangar-py/pull/193))
    [\@rlizzo](https://github.com/rlizzo)
-   Major backend overhaul which defines column layouts and data types
    in the same interchangable / extensable manner as storage backends.
    This will allow rapid development of new layouts and data type
    support as new use cases are discovered by the community.
    ([\#184](https://github.com/tensorwerk/hangar-py/pull/184))
    [\@rlizzo](https://github.com/rlizzo)
-   Column and backend classes are now fully serializable (pickleable)
    for `read-only` checkouts.
    ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   Modularized internal structure of API classes to easily allow new
    columnn layouts / data types to be added in the future.
    ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   Improved type / value checking of manual specification for column
    `backend` and `backend_options`.
    ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   Standardized column data access API to follow python standard
    library `dict` methods API.
    ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   Memory usage of arrayset checkouts has been reduced by \~70% by
    using C-structs for allocating sample record locating info.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)
-   Read times from the `HDF5_00` and `HDF5_01` backend have been
    reduced by 33-38% (or more for arraysets with many samples) by
    eliminating redundant computation of chunked storage B-Tree.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)
-   Commit times and checkout times have been reduced by 11-18% by
    optimizing record parsing and memory allocation.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)

### New Features

-   Added `str` type column with same behavior as `ndarray` column
    (supporting both single-level and nested layouts) added to replace
    functionality of removed `metadata` container.
    ([\#184](https://github.com/tensorwerk/hangar-py/pull/184))
    [\@rlizzo](https://github.com/rlizzo)
-   New backend based on `LMDB` has been added (specifier of `lmdb_30`).
    ([\#184](https://github.com/tensorwerk/hangar-py/pull/184))
    [\@rlizzo](https://github.com/rlizzo)
-   Added `.diff()` method to `Repository` class to enable diffing
    changes between any pair of commits / branches without needing to
    open the diff base in a checkout.
    ([\#183](https://github.com/tensorwerk/hangar-py/pull/183))
    [\@rlizzo](https://github.com/rlizzo)
-   New CLI command `hangar diff` which reports a summary view of
    changes made between any pair of commits / branches.
    ([\#183](https://github.com/tensorwerk/hangar-py/pull/183))
    [\@rlizzo](https://github.com/rlizzo)
-   Added `.log()` method to `Checkout` objects so graphical commit
    graph or machine readable commit details / DAG can be queried when
    operating on a particular commit.
    ([\#183](https://github.com/tensorwerk/hangar-py/pull/183))
    [\@rlizzo](https://github.com/rlizzo)
-   \"string\" type columns now supported alongside \"ndarray\" column
    type. ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   New \"column\" API, which replaces \"arrayset\" name.
    ([\#180](https://github.com/tensorwerk/hangar-py/pull/180))
    [\@rlizzo](https://github.com/rlizzo)
-   Arraysets can now contain \"nested subsamples\" under a common
    sample key.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)
-   New API to add and remove samples from and arrayset.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)
-   Added `repo.size_nbytes` and `repo.size_human` to report disk usage
    of a repository on disk.
    ([\#174](https://github.com/tensorwerk/hangar-py/pull/174))
    [\@rlizzo](https://github.com/rlizzo)
-   Added method to traverse the entire repository history and
    cryptographically verify integrity.
    ([\#173](https://github.com/tensorwerk/hangar-py/pull/173))
    [\@rlizzo](https://github.com/rlizzo)

### Changes

-   Argument syntax of `__getitem__()` and `get()` methods of
    `ReaderCheckout` and `WriterCheckout` classes. The new format
    supports handeling arbitrary arguments specific to retrieval of data
    from any column type.
    ([\#183](https://github.com/tensorwerk/hangar-py/pull/183))
    [\@rlizzo](https://github.com/rlizzo)

### Removed

-   `metadata` container for `str` typed data has been completly
    removed. It is replaced by a highly extensible and much more
    user-friendly `str` typed column.
    ([\#184](https://github.com/tensorwerk/hangar-py/pull/184))
    [\@rlizzo](https://github.com/rlizzo)
-   `__setitem__()` method in `WriterCheckout` objects. Writing data to
    columns via a checkout object is no longer supported.
    ([\#183](https://github.com/tensorwerk/hangar-py/pull/183))
    [\@rlizzo](https://github.com/rlizzo)

### Bug Fixes

-   Backend data stores no longer use file symlinks, improving
    compatibility with some types file systems.
    ([\#171](https://github.com/tensorwerk/hangar-py/pull/171))
    [\@rlizzo](https://github.com/rlizzo)
-   All arrayset types (\"flat\" and \"nested subsamples\") and backend
    readers can now be pickled \-- for parallel processing \-- in a
    read-only checkout.
    ([\#179](https://github.com/tensorwerk/hangar-py/pull/179))
    [\@rlizzo](https://github.com/rlizzo)

### Breaking changes

-   New backend record serialization format is incompatible with
    repositories written in version 0.4 or earlier.
-   New arrayset API is incompatible with Hangar API in version 0.4 or
    earlier.

[v0.4.0](https://github.com/tensorwerk/hangar-py/compare/v0.3.0...v0.4.0) (2019-11-21)
--------------------------------------------------------------------------------------

### New Features

-   Added ability to delete branch names/pointers from a local
    repository via both API and CLI.
    ([\#128](https://github.com/tensorwerk/hangar-py/pull/128))
    [\@rlizzo](https://github.com/rlizzo)
-   Added `local` keyword arg to arrayset key/value iterators to return
    only locally available samples
    ([\#131](https://github.com/tensorwerk/hangar-py/pull/131))
    [\@rlizzo](https://github.com/rlizzo)
-   Ability to change the backend storage format and options applied to
    an `arrayset` after initialization.
    ([\#133](https://github.com/tensorwerk/hangar-py/pull/133))
    [\@rlizzo](https://github.com/rlizzo)
-   Added blosc compression to HDF5 backend by default on PyPi
    installations.
    ([\#146](https://github.com/tensorwerk/hangar-py/pull/146))
    [\@rlizzo](https://github.com/rlizzo)
-   Added Benchmarking Suite to Test for Performance Regressions in PRs.
    ([\#155](https://github.com/tensorwerk/hangar-py/pull/155))
    [\@rlizzo](https://github.com/rlizzo)
-   Added new backend optimized to increase speeds for fixed size
    arrayset access.
    ([\#160](https://github.com/tensorwerk/hangar-py/pull/160))
    [\@rlizzo](https://github.com/rlizzo)

### Improvements

-   Removed `msgpack` and `pyyaml` dependencies. Cleaned up and improved
    remote client/server code.
    ([\#130](https://github.com/tensorwerk/hangar-py/pull/130))
    [\@rlizzo](https://github.com/rlizzo)
-   Multiprocess Torch DataLoaders allowed on Linux and MacOS.
    ([\#144](https://github.com/tensorwerk/hangar-py/pull/144))
    [\@rlizzo](https://github.com/rlizzo)
-   Added CLI options `commit`, `checkout`, `arrayset create`, &
    `arrayset remove`.
    ([\#150](https://github.com/tensorwerk/hangar-py/pull/150))
    [\@rlizzo](https://github.com/rlizzo)
-   Plugin system revamp.
    ([\#134](https://github.com/tensorwerk/hangar-py/pull/134))
    [\@hhsecond](https://github.com/hhsecond)
-   Documentation Improvements and Typo-Fixes.
    ([\#156](https://github.com/tensorwerk/hangar-py/pull/156))
    [\@alessiamarcolini](https://github.com/alessiamarcolini)
-   Removed implicit removal of arrayset schema from checkout if every
    sample was removed from arrayset. This could potentially result in
    dangling accessors which may or may not self-destruct (as expected)
    in certain edge-cases.
    ([\#159](https://github.com/tensorwerk/hangar-py/pull/159))
    [\@rlizzo](https://github.com/rlizzo)
-   Added type codes to hash digests so that calculation function can be
    updated in the future without breaking repos written in previous
    Hangar versions.
    ([\#165](https://github.com/tensorwerk/hangar-py/pull/165))
    [\@rlizzo](https://github.com/rlizzo)

### Bug Fixes

-   Programatic access to repository log contents now returns branch
    heads alongside other log info.
    ([\#125](https://github.com/tensorwerk/hangar-py/pull/125))
    [\@rlizzo](https://github.com/rlizzo)
-   Fixed minor bug in types of values allowed for `Arrayset` names vs
    `Sample` names.
    ([\#151](https://github.com/tensorwerk/hangar-py/pull/151))
    [\@rlizzo](https://github.com/rlizzo)
-   Fixed issue where using checkout object to access a sample in
    multiple arraysets would try to create a `namedtuple` instance with
    invalid field names. Now incompatible field names are automatically
    renamed with their positional index.
    ([\#161](https://github.com/tensorwerk/hangar-py/pull/161))
    [\@rlizzo](https://github.com/rlizzo)
-   Explicitly raise error if `commit` argument is set while checking
    out a repository with `write=True`.
    ([\#166](https://github.com/tensorwerk/hangar-py/pull/166))
    [\@rlizzo](https://github.com/rlizzo)

### Breaking changes

-   New commit reference serialization format is incompatible with
    repositories written in version 0.3.0 or earlier.

[v0.3.0](https://github.com/tensorwerk/hangar-py/compare/v0.2.0...v0.3.0) (2019-09-10)
--------------------------------------------------------------------------------------

### New Features

-   API addition allowing reading and writing arrayset data from a
    checkout object directly.
    ([\#115](https://github.com/tensorwerk/hangar-py/pull/115))
    [\@rlizzo](https://github.com/rlizzo)
-   Data importer, exporters, and viewers via CLI for common file
    formats. Includes plugin system for easy extensibility in the
    future. ([\#103](https://github.com/tensorwerk/hangar-py/pull/103))
    ([\@rlizzo](https://github.com/rlizzo),
    [\@hhsecond](https://github.com/hhsecond))

### Improvements

-   Added tutorial on working with remote data.
    ([\#113](https://github.com/tensorwerk/hangar-py/pull/113))
    [\@rlizzo](https://github.com/rlizzo)
-   Added Tutorial on Tensorflow and PyTorch Dataloaders.
    ([\#117](https://github.com/tensorwerk/hangar-py/pull/117))
    [\@hhsecond](https://github.com/hhsecond)
-   Large performance improvement to diff/merge algorithm (\~30x
    previous).
    ([\#112](https://github.com/tensorwerk/hangar-py/pull/112))
    [\@rlizzo](https://github.com/rlizzo)
-   New commit hash algorithm which is much more reproducible in the
    long term.
    ([\#120](https://github.com/tensorwerk/hangar-py/pull/120))
    [\@rlizzo](https://github.com/rlizzo)
-   HDF5 backend updated to increase speed of reading/writing variable
    sized dataset compressed chunks
    ([\#120](https://github.com/tensorwerk/hangar-py/pull/120))
    [\@rlizzo](https://github.com/rlizzo)

### Bug Fixes

-   Fixed ML Dataloaders errors for a number of edge cases surrounding
    partial-remote data and non-common keys.
    ([\#110](https://github.com/tensorwerk/hangar-py/pull/110)) (
    [\@hhsecond](https://github.com/hhsecond),
    [\@rlizzo](https://github.com/rlizzo))

### Breaking changes

-   New commit hash algorithm is incompatible with repositories written
    in version 0.2.0 or earlier

[v0.2.0](https://github.com/tensorwerk/hangar-py/compare/v0.1.1...v0.2.0) (2019-08-09)
--------------------------------------------------------------------------------------

### New Features

-   Numpy memory-mapped array file backend added.
    ([\#70](https://github.com/tensorwerk/hangar-py/pull/70))
    [\@rlizzo](https://github.com/rlizzo)
-   Remote server data backend added.
    ([\#70](https://github.com/tensorwerk/hangar-py/pull/70))
    [\@rlizzo](https://github.com/rlizzo)
-   Selection heuristics to determine appropriate backend from arrayset
    schema. ([\#70](https://github.com/tensorwerk/hangar-py/pull/70))
    [\@rlizzo](https://github.com/rlizzo)
-   Partial remote clones and fetch operations now fully supported.
    ([\#85](https://github.com/tensorwerk/hangar-py/pull/85))
    [\@rlizzo](https://github.com/rlizzo)
-   CLI has been placed under test coverage, added interface usage to
    docs. ([\#85](https://github.com/tensorwerk/hangar-py/pull/85))
    [\@rlizzo](https://github.com/rlizzo)
-   TensorFlow and PyTorch Machine Learning Dataloader Methods
    (*Experimental Release*).
    ([\#91](https://github.com/tensorwerk/hangar-py/pull/91)) lead:
    [\@hhsecond](https://github.com/hhsecond), co-author:
    [\@rlizzo](https://github.com/rlizzo), reviewed by:
    [\@elistevens](https://github.com/elistevens)

### Improvements

-   Record format versioning and standardization so to not break
    backwards compatibility in the future.
    ([\#70](https://github.com/tensorwerk/hangar-py/pull/70))
    [\@rlizzo](https://github.com/rlizzo)
-   Backend addition and update developer protocols and documentation.
    ([\#70](https://github.com/tensorwerk/hangar-py/pull/70))
    [\@rlizzo](https://github.com/rlizzo)
-   Read-only checkout arrayset sample `get` methods now are multithread
    and multiprocess safe.
    ([\#84](https://github.com/tensorwerk/hangar-py/pull/84))
    [\@rlizzo](https://github.com/rlizzo)
-   Read-only checkout metadata sample `get` methods are thread safe if
    used within a context manager.
    ([\#101](https://github.com/tensorwerk/hangar-py/pull/101))
    [\@rlizzo](https://github.com/rlizzo)
-   Samples can be assigned integer names in addition to `string` names.
    ([\#89](https://github.com/tensorwerk/hangar-py/pull/89))
    [\@rlizzo](https://github.com/rlizzo)
-   Forgetting to close a `write-enabled` checkout before terminating
    the python process will close the checkout automatically for many
    situations.
    ([\#101](https://github.com/tensorwerk/hangar-py/pull/101))
    [\@rlizzo](https://github.com/rlizzo)
-   Repository software version compatability methods added to ensure
    upgrade paths in the future.
    ([\#101](https://github.com/tensorwerk/hangar-py/pull/101))
    [\@rlizzo](https://github.com/rlizzo)
-   Many tests added (including support for Mac OSX on Travis-CI). lead:
    [\@rlizzo](https://github.com/rlizzo), co-author:
    [\@hhsecond](https://github.com/hhsecond)

### Bug Fixes

-   Diff results for fast forward merges now returns sensible results.
    ([\#77](https://github.com/tensorwerk/hangar-py/pull/77))
    [\@rlizzo](https://github.com/rlizzo)
-   Many type annotations added, and developer documentation improved.
    [\@hhsecond](https://github.com/hhsecond) &
    [\@rlizzo](https://github.com/rlizzo)

### Breaking changes

-   Renamed all references to `datasets` in the API / world-view to
    `arraysets`.
-   These are backwards incompatible changes. For all versions \> 0.2,
    repository upgrade utilities will be provided if breaking changes
    occur.

[v0.1.1](https://github.com/tensorwerk/hangar-py/compare/v0.1.0...v0.1.1) (2019-05-24)
--------------------------------------------------------------------------------------

### Bug Fixes

-   Fixed typo in README which was uploaded to PyPi

[v0.1.0](https://github.com/tensorwerk/hangar-py/compare/v0.0.0...v0.1.0) (2019-05-24)
--------------------------------------------------------------------------------------

### New Features

-   Remote client-server config negotiation and administrator
    permissions.
    ([\#10](https://github.com/tensorwerk/hangar-py/pull/10))
    [\@rlizzo](https://github.com/rlizzo)
-   Allow single python process to access multiple repositories
    simultaneously.
    ([\#20](https://github.com/tensorwerk/hangar-py/pull/20))
    [\@rlizzo](https://github.com/rlizzo)
-   Fast-Forward and 3-Way Merge and Diff methods now fully supported
    and behaving as expected.
    ([\#32](https://github.com/tensorwerk/hangar-py/pull/32))
    [\@rlizzo](https://github.com/rlizzo)

### Improvements

-   Initial test-case specification.
    ([\#14](https://github.com/tensorwerk/hangar-py/pull/14))
    [\@hhsecond](https://github.com/hhsecond)
-   Checkout test-case work.
    ([\#25](https://github.com/tensorwerk/hangar-py/pull/25))
    [\@hhsecond](https://github.com/hhsecond)
-   Metadata test-case work.
    ([\#27](https://github.com/tensorwerk/hangar-py/pull/27))
    [\@hhsecond](https://github.com/hhsecond)
-   Any potential failure cases raise exceptions instead of silently
    returning. ([\#16](https://github.com/tensorwerk/hangar-py/pull/16))
    [\@rlizzo](https://github.com/rlizzo)
-   Many usability improvements in a variety of commits.

### Bug Fixes

-   Ensure references to checkout arrayset or metadata objects cannot
    operate after the checkout is closed.
    ([\#41](https://github.com/tensorwerk/hangar-py/pull/41))
    [\@rlizzo](https://github.com/rlizzo)
-   Sensible exception classes and error messages raised on a variety of
    situations (Many commits). [\@hhsecond](https://github.com/hhsecond)
    & [\@rlizzo](https://github.com/rlizzo)
-   Many minor issues addressed.

### API Additions

-   Refer to API documentation
    ([\#23](https://github.com/tensorwerk/hangar-py/pull/23))

### Breaking changes

-   All repositories written with previous versions of Hangar are liable
    to break when using this version. Please upgrade versions
    immediately.

[v0.0.0](https://github.com/tensorwerk/hangar-py/commit/2aff3805c66083a7fbb2ebf701ceaf38ac5165c7) (2019-04-15)
--------------------------------------------------------------------------------------------------------------

-   First Public Release of Hangar!
