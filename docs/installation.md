Installation
============

For general usage it is recommended that you use a pre-built version of
Hangar, either from a Python Distribution, or a pre-built wheel from
PyPi.

Pre-Built Installation
----------------------

### Python Distributions

If you do not already use a Python Distribution, we recommend the
[Anaconda](https://www.anaconda.com/distribution/) (or 
[Miniconda](https://docs.conda.io/en/latest/miniconda.html)) distribution,
which supports all major operating systems (Windows, MacOSX, & the
typical Linux variations). Detailed usage instructions are available [on
the anaconda website](https://docs.anaconda.com/anaconda/).

To install Hangar via the Anaconda Distribution (from the [conda-forge
conda channel](https://anaconda.org/conda-forge/hangar)):

<div class="termy">

```console
$ conda install -c conda-forge hangar

Collecting package metadata (current_repodata.json): done
// Solving environment: done

<span data-ty></span>
## Package Plan ##

  environment location: /Users/rick/miniconda3/envs/new-install

  added / updated specs:
    - hangar
<span data-ty></span>
// 
The following NEW packages will be INSTALLED:

  blosc              pkgs/main/osx-64::blosc-1.19.0-hab81aa3_0
  c-ares             conda-forge/osx-64::c-ares-1.16.1-haf1e3a3_0
  click              conda-forge/noarch::click-7.1.2-pyh9f0ad1d_0
  ...
  zstd               conda-forge/osx-64::zstd-1.4.5-h0384e3a_1

// 
<span data-ty="input" data-ty-typeDelay="1000" data-ty-prompt="Proceed ([y]/n)? ">y</span>
 
Preparing transaction:
---> 100%
Preparing transaction: done
Verifying transaction:
---> 100%
Verifying transaction: done
Executing transaction:
---> 100%
Executing transaction: done

$ hangar --version 
hangar, version 0.5.2
```

</div>

### Wheels (PyPi)

If you have an existing python installation on your computer, pre-built
Hangar Wheels can be installed via pip from the Python Package Index
(PyPi):

<div class="termy">

```console
$ pip install hangar
---> 100%
$ hangar --version
hangar, version 0.5.2
```

</div>


Source Installation
-------------------

To install Hangar from source, clone the repository from 
[Github](https://github.com/tensorwerk/hangar-py):

    git clone https://github.com/tensorwerk/hangar-py.git
    cd hangar-py
    python setup.py install

Or use pip on the local package if you want to install all dependencies
automatically in a development environment:

    pip install -e .

### Source installation in Google colab

Google colab comes with an older version of `h5py` pre-installed which
is not compatible with hangar. If you need to install hangar from the
source in google colab, make sure to uninstall the existing `h5py` :

    !pip uninstall h5py

Then follow the Source Installation steps given above.
