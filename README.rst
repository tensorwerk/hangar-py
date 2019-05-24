========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/hangar-py/badge/?style=flat
    :target: https://readthedocs.org/projects/hangar-py
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/tensorwerk/hangar-py.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/tensorwerk/hangar-py

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/tensorwerk/hangar-py?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/rlizzo/hangar-py

.. |version| image:: https://img.shields.io/pypi/v/hangar.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/hangar

.. |commits-since| image:: https://img.shields.io/github/commits-since/tensorwerk/hangar-py/v0.1.1.svg
    :alt: Commits since latest release
    :target: https://github.com/tensorwerk/hangar-py/compare/v0.1.1...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/hangar.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/hangar

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/hangar.svg
    :alt: Supported versions
    :target: https://pypi.org/project/hangar

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/hangar.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/hangar


.. end-badges

Hangar is version control for tensor data. Commit, branch, merge, revert, and
collaborate in the data-defined software era.

* Free software: Apache 2.0 license

What is Hangar?
===============

Hangar is based off the belief that too much time is spent collecting, managing,
and creating home-brewed version control systems for data. At it's core Hangar
is designed to solve many of the same problems faced by traditional code version
control system (ie. ``Git``), just adapted for numerical data:

* Time travel through the historical evolution of a dataset
* Zero-cost Branching to enable exploratory analysis and collaboration
* Cheap Merging to build datasets over time (with multiple collaborators)
* Completely abstracted organization and management of data files on disk
* Ability to only retrieve a small portion of the data (as needed) while still
  maintaining complete historical record
* Ability to push and pull changes directly to collaborators or a central server
  (ie a truly distributed version control system)

The ability of version control systems to perform these tasks for codebases is
largely taken for granted by almost every developer today; However, we are
in-fact standing on the shoulders of giants, with decades of engineering which
has resulted in these phenomenally useful tools. Now that a new era of
"Data-Defined software" is taking hold, we find there is a strong need for
analogous version control systems which are designed to handle numerical data at
large scale... Welcome to Hangar!


The Hangar Workflow:

::

       Checkout Branch
              |
              ▼
     Create/Access Datasets
              |
              ▼
    Add/Remove/Update Samples
              |
              ▼
           Commit

Log Style Output:

::

    *   5254ec (master) : merge commit combining training updates and new validation samples
    |\
    | * 650361 (add-validation-data) : Add validation labels and image data in isolated branch
    * | 5f15b4 : Add some metadata for later reference and add new training samples received after initial import
    |/
    *   baddba : Initial commit adding training images and labels


Learn more about what Hangar is all about at https://hangar-py.readthedocs.io/


Installation
============

Hangar is in early alpha development release!

::

    pip install hangar

Documentation
=============

https://hangar-py.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox