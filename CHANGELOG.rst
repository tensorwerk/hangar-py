==========
Change Log
==========


`In-Progress`_
==============

New Features
------------

* Numpy memory-mapped array file backend added. (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Remote server data backend added. (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Selection heuristics to determine appropriate backend from dataset schema. (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__


Improvements
------------

* Record format versioning and standardization so to not break backwards compatibility in the future. (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Backend addition and update developer protocols and documentation. (`#70 <https://github.com/tensorwerk/hangar-py/pull/70>`__) `@rlizzo <https://github.com/rlizzo>`__
* Read-only checkout dataset sample ``get`` methods now are multithread and multiprocess safe. (`#84 <https://github.com/tensorwerk/hangar-py/pull/84>`__) `@rlizzo <https://github.com/rlizzo>`__
* Many tests added (including support for Mac OSX on Travis-CI).

Bug Fixes
---------

* Diff results for fast forward merges now returns sensible results (`#77 <https://github.com/tensorwerk/hangar-py/pull/77>`__) `@rlizzo <https://github.com/rlizzo>`__

Breaking changes
----------------

* These are backwards incompatible changes. For all versions > 0.2, repository upgrade utilities will be provided if breaking changes occur.


`v0.1.1`_ (2019-05-24)
===========================

Bug Fixes
---------

* Fixed typo in README which was uploaded to PyPi


`v0.1.0`_ (2019-05-24)
===========================

New Features
------------

* Remote client-server config negotiation and administrator permissions (`#10 <https://github.com/tensorwerk/hangar-py/pull/10>`__) `@rlizzo <https://github.com/rlizzo>`__
* Allow single python process to access multiple repositories simultaneously (`#20 <https://github.com/tensorwerk/hangar-py/pull/20>`__) `@rlizzo <https://github.com/rlizzo>`__
* Fast-Forward and 3-Way Merge and Diff methods now fully supported and behaving as expected (`#32 <https://github.com/tensorwerk/hangar-py/pull/32>`__) `@rlizzo <https://github.com/rlizzo>`__

Improvements
------------

* Initial test-case specification (`#14 <https://github.com/tensorwerk/hangar-py/pull/14>`__) `@hhsecond <https://github.com/hhsecond>`__
* Checkout test-case work (`#25 <https://github.com/tensorwerk/hangar-py/pull/25>`__) `@hhsecond <https://github.com/hhsecond>`__
* Metadata test-case work (`#27 <https://github.com/tensorwerk/hangar-py/pull/27>`__) `@hhsecond <https://github.com/hhsecond>`__
* Any potential failure cases raise exceptions instead of silently returning (`#16 <https://github.com/tensorwerk/hangar-py/pull/16>`__) `@rlizzo <https://github.com/rlizzo>`__
* Many usability improvements in a variety of commits

Bug Fixes
---------

* Ensure references to checkout dataset or metadata objects cannot operate after the checkout is closed. (`#41 <https://github.com/tensorwerk/hangar-py/pull/41>`__) `@rlizzo <https://github.com/rlizzo>`__
* Sensible exception classes and error messages raised on a variety of situations (Many commits) `@hhsecond <https://github.com/hhsecond>`__ & `@rlizzo <https://github.com/rlizzo>`__
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
.. _In-Progress: https://github.com/tensorwerk/hangar-py/compare/v0.1.1...master