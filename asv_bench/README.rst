Hangar Performance Benchmarking Suite
=====================================

This directory contains the hangar performance benchmarking suite. Benchmarks
are run via the `Airspeed Velocity (ASV)
<https://https://asv.readthedocs.io/>`_ project tooling.

As ASV sets up and manages it's own virtual environments and source
installations, benchmark execution is not run via ``tox``. Please refer to the
`ASV Docs <https://https://asv.readthedocs.io/>`_ for information on how to run
this suite locally. All pull requests to the main Hangar GitHub Repo are
automatically benchmarked by our CI servers, and all results are compared to
the ``master`` branch to identify any regressions early in the PR process.

* Benchmark Results Repo: https://github.com/tensorwerk/hangar-benchmarks

* Benchmark Web View: https://tensorwerk.com/hangar-benchmarks
