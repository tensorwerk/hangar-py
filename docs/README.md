Overview
========

+-----------------------------------+-----------------------------------+
| docs                              |                                   |
+-----------------------------------+-----------------------------------+
| tests                             | | [![Build Status](https://github |
|                                   | .com/tensorwerk/hangar-py/workflo |
|                                   | ws/Run%20Test%20Suite/badge.svg?b |
|                                   | ranch=master)](https://github.com |
|                                   | /tensorwerk/hangar-py/actions?que |
|                                   | ry=workflow%3A%22Run+Test+Suite%2 |
|                                   | 2+branch%3Amaster+event%3Apush+is |
|                                   | %3Acompleted)                     |
|                                   |   [![Code Coverage](https://codec |
|                                   | ov.io/gh/tensorwerk/hangar-py/bra |
|                                   | nch/master/graph/badge.svg)](http |
|                                   | s://codecov.io/gh/tensorwerk/hang |
|                                   | ar-py)                            |
|                                   | | [![Language grade: Python](http |
|                                   | s://img.shields.io/lgtm/grade/pyt |
|                                   | hon/g/tensorwerk/hangar-py.svg?lo |
|                                   | go=lgtm&logoWidth=18)](https://lg |
|                                   | tm.com/projects/g/tensorwerk/hang |
|                                   | ar-py/context:python)             |
+-----------------------------------+-----------------------------------+
| package                           | | [![PyPI Package latest release] |
|                                   | (https://img.shields.io/pypi/v/ha |
|                                   | ngar.svg)](https://pypi.org/proje |
|                                   | ct/hangar)                        |
|                                   |   [![PyPI Wheel](https://img.shie |
|                                   | lds.io/pypi/wheel/hangar.svg)](ht |
|                                   | tps://pypi.org/project/hangar)    |
|                                   |   [![Conda-Forge Latest Version]( |
|                                   | https://img.shields.io/conda/vn/c |
|                                   | onda-forge/hangar.svg)](https://a |
|                                   | naconda.org/conda-forge/hangar)   |
|                                   | | [![Supported versions](https:// |
|                                   | img.shields.io/pypi/pyversions/ha |
|                                   | ngar.svg)](https://pypi.org/proje |
|                                   | ct/hangar)                        |
|                                   |   [![Supported implementations](h |
|                                   | ttps://img.shields.io/pypi/implem |
|                                   | entation/hangar.svg)](https://pyp |
|                                   | i.org/project/hangar)             |
|                                   | | [![GitHub license](https://img. |
|                                   | shields.io/github/license/tensorw |
|                                   | erk/hangar-py)](https://github.co |
|                                   | m/tensorwerk/hangar-py/blob/maste |
|                                   | r/LICENSE)                        |
+-----------------------------------+-----------------------------------+

Hangar is version control for tensor data. Commit, branch, merge,
revert, and collaborate in the data-defined software era.

-   Free software: Apache 2.0 license

What is Hangar?
---------------

Hangar is based off the belief that too much time is spent collecting,
managing, and creating home-brewed version control systems for data. At
it\'s core Hangar is designed to solve many of the same problems faced
by traditional code version control system (ie. `Git`), just adapted for
numerical data:

-   Time travel through the historical evolution of a dataset.
-   Zero-cost Branching to enable exploratory analysis and collaboration
-   Cheap Merging to build datasets over time (with multiple
    collaborators)
-   Completely abstracted organization and management of data files on
    disk
-   Ability to only retrieve a small portion of the data (as needed)
    while still maintaining complete historical record
-   Ability to push and pull changes directly to collaborators or a
    central server (ie a truly distributed version control system)

The ability of version control systems to perform these tasks for
codebases is largely taken for granted by almost every developer today;
However, we are in-fact standing on the shoulders of giants, with
decades of engineering which has resulted in these phenomenally useful
tools. Now that a new era of \"Data-Defined software\" is taking hold,
we find there is a strong need for analogous version control systems
which are designed to handle numerical data at large scale\... Welcome
to Hangar!

The Hangar Workflow:

       Checkout Branch
            |
            ▼
     Create/Access Data
            | 
            ▼
    Add/Remove/Update Samples
            | 
            ▼
          Commit

Log Style Output:

```
*   5254ec (master) : merge commit combining training updates and new validation samples
|\
| * 650361 (add-validation-data) : Add validation labels and image data in isolated branch
* | 5f15b4 : Add some metadata for later reference and add new training samples received after initial import
|/
*   baddba : Initial commit adding training images and labels
```

Learn more about what Hangar is all about at
<https://hangar-py.readthedocs.io/>

Installation
------------

Hangar is in early alpha development release!

    pip install hangar

Documentation
-------------

<https://hangar-py.readthedocs.io/>

Development
-----------

To run the all tests run:

    tox

Note, to combine the coverage data from all the tox environments run:

+------+---------------------------------------------------------------+
| Wind |     set PYTEST_ADDOPTS=--cov-append                           |
| ows  |     tox                                                       |
+------+---------------------------------------------------------------+
| Othe |     PYTEST_ADDOPTS=--cov-append tox                           |
| r    |                                                               |
+------+---------------------------------------------------------------+
