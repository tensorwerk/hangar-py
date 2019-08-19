==========
Change Log
==========

`In-Progress`_
==============

Improvements
------------

* Added tutorial on working with remote data.
  (`#113 <https://github.com/tensorwerk/hangar-py/pull/113>`__) `@rlizzo <https://github.com/rlizzo>`__

`v0.2.0`_ (2019-08-09)
======================

New Features
------------

* Numpy memory-mapped array file backend added.
  (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Remote server data backend added.
  (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Selection heuristics to determine appropriate backend from arrayset schema.
  (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Partial remote clones and fetch operations now fully supported.
  (`#85 <https://github.com/tensorwerk/hangar-py/pull/85>`__) `@rlizzo <https://github.com/rlizzo>`__
* CLI has been placed under test coverage, added interface usage to docs.
  (`#85 <https://github.com/tensorwerk/hangar-py/pull/85>`__) `@rlizzo <https://github.com/rlizzo>`__
* TensorFlow and PyTorch Machine Learning Dataloader Methods (*Experimental Release*).
  (`#91 <https://github.com/tensorwerk/hangar-py/pull/91>`__)
  lead: `@hhsecond <https://github.com/hhsecond>`__, co-author: `@rlizzo <https://github.com/rlizzo>`__,
  reviewed by: `@elistevens <https://github.com/elistevens>`__

Improvements
------------

* Record format versioning and standardization so to not break backwards compatibility in the future.
  (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Backend addition and update developer protocols and documentation.
  (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Read-only checkout arrayset sample ``get`` methods now are multithread and multiprocess safe.
  (`#84 <https://github.com/tensorwerk/hangar-py/pull/84>`__) `@rlizzo <https://github.com/rlizzo>`__
* Read-only checkout metadata sample ``get`` methods are thread safe if used within a context manager.
  (`#101 <https://github.com/tensorwerk/hangar-py/pull/101>`__) `@rlizzo <https://github.com/rlizzo>`__
* Samples can be assigned integer names in addition to ``string`` names.
  (`#89 <https://github.com/tensorwerk/hangar-py/pull/89>`__) `@rlizzo <https://github.com/rlizzo>`__
* Forgetting to close a ``write-enabled`` checkout before terminating the python process will close the
  checkout automatically for many situations.
  (`#101 <https://github.com/tensorwerk/hangar-py/pull/101>`__) `@rlizzo <https://github.com/rlizzo>`__
* Repository software version compatability methods added to ensure upgrade paths in the future.
  (`#101 <https://github.com/tensorwerk/hangar-py/pull/101>`__) `@rlizzo <https://github.com/rlizzo>`__
* Many tests added (including support for Mac OSX on Travis-CI).
  lead: `@rlizzo <https://github.com/rlizzo>`__, co-author: `@hhsecond <https://github.com/hhsecond>`__

Bug Fixes
---------

* Diff results for fast forward merges now returns sensible results.
  (`#77 <https://github.com/tensorwerk/hangar-py/pull/77>`__) `@rlizzo <https://github.com/rlizzo>`__
* Many type annotations added, and developer documentation improved.
  `@hhsecond <https://github.com/hhsecond>`__ & `@rlizzo <https://github.com/rlizzo>`__

Breaking changes
----------------

* Renamed all references to ``datasets`` in the API / world-view to ``arraysets``.
* These are backwards incompatible changes. For all versions > 0.2, repository upgrade utilities will
  be provided if breaking changes occur.


`v0.1.1`_ (2019-05-24)
===========================

Bug Fixes
---------

* Fixed typo in README which was uploaded to PyPi


`v0.1.0`_ (2019-05-24)
===========================

New Features
------------

* Remote client-server config negotiation and administrator permissions.
  (`#10 <https://github.com/tensorwerk/hangar-py/pull/10>`__) `@rlizzo <https://github.com/rlizzo>`__
* Allow single python process to access multiple repositories simultaneously.
  (`#20 <https://github.com/tensorwerk/hangar-py/pull/20>`__) `@rlizzo <https://github.com/rlizzo>`__
* Fast-Forward and 3-Way Merge and Diff methods now fully supported and behaving as expected.
  (`#32 <https://github.com/tensorwerk/hangar-py/pull/32>`__) `@rlizzo <https://github.com/rlizzo>`__

Improvements
------------

* Initial test-case specification.
  (`#14 <https://github.com/tensorwerk/hangar-py/pull/14>`__) `@hhsecond <https://github.com/hhsecond>`__
* Checkout test-case work.
  (`#25 <https://github.com/tensorwerk/hangar-py/pull/25>`__) `@hhsecond <https://github.com/hhsecond>`__
* Metadata test-case work.
  (`#27 <https://github.com/tensorwerk/hangar-py/pull/27>`__) `@hhsecond <https://github.com/hhsecond>`__
* Any potential failure cases raise exceptions instead of silently returning.
  (`#16 <https://github.com/tensorwerk/hangar-py/pull/16>`__) `@rlizzo <https://github.com/rlizzo>`__
* Many usability improvements in a variety of commits.


Bug Fixes
---------

* Ensure references to checkout arrayset or metadata objects cannot operate after the checkout is closed.
  (`#41 <https://github.com/tensorwerk/hangar-py/pull/41>`__) `@rlizzo <https://github.com/rlizzo>`__
* Sensible exception classes and error messages raised on a variety of situations (Many commits).
  `@hhsecond <https://github.com/hhsecond>`__ & `@rlizzo <https://github.com/rlizzo>`__
* Many minor issues addressed.

API Additions
-------------

* Refer to API documentation (`#23 <https://github.com/tensorwerk/hangar-py/pull/23>`__)

Breaking changes
----------------

* All repositories written with previous versions of Hangar are liable to break when using this version. Please upgrade versions immediately.


`v0.0.0`_ (2019-04-15)
======================

* First Public Release of Hangar!

.. _v0.0.0: https://github.com/tensorwerk/hangar-py/commit/2aff3805c66083a7fbb2ebf701ceaf38ac5165c7
.. _v0.1.0: https://github.com/tensorwerk/hangar-py/compare/v0.0.0...v0.1.0
.. _v0.1.1: https://github.com/tensorwerk/hangar-py/compare/v0.1.0...v0.1.1
.. _v0.2.0: https://github.com/tensorwerk/hangar-py/compare/v0.1.1...v0.2.0
.. _In-Progress: https://github.com/tensorwerk/hangar-py/compare/v0.2.0...master
