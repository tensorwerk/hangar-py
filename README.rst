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
    :target: https://ci.appveyor.com/project/tensorwerk/hangar-py

.. |version| image:: https://img.shields.io/pypi/v/hangar.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/hangar

.. |commits-since| image:: https://img.shields.io/github/commits-since/tensorwerk/hangar-py/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/tensorwerk/hangar-py/compare/v0.0.0...master

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

Hangar is based off of the founded from the belief that too much time is spent collecting,
managing, and creating home-brewed version control systems for data. Hangar is
a `Git` inspired tool to version and work with data!

Installation
============

Hangar is in early alpha development release!

Pip package to be released shortly!

::

    python setup.py

.. ::

..     pip install hangar

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