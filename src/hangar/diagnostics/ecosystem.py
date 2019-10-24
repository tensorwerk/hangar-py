from typing import Dict, List, Tuple, Union


required_packages = [
    ('hangar', lambda p: p.__version__),
    ('click', lambda p: p.__version__),
    ('lmdb', lambda p: p.__version__),
    ('h5py', lambda p: p.__version__),
    ('hdf5plugin', lambda p: p.version),
    ('numpy', lambda p: p.__version__),
    ('blosc', lambda p: p.__version__),
    ('tqdm', lambda p: p.__version__),
    ('wrapt', lambda p: p.__version__),
    ('grpc', lambda p: p.__version__),
    ('xxhash', lambda p: p.VERSION),
]


def get_versions() -> dict:
    """Return information on software, machine, installed versions of packages.

    dict
        host, package, and `optional` package info.
    """
    d = {'host': get_system_info(),
         'packages': get_package_info(required_packages),
         'optional': get_optional_info()}
    return d


def get_system_info() -> List[Tuple[str, str]]:
    """Return local computer python, OS, and Machine info

    Returns
    -------
    List[Tuple[str, str]]
        field collected and value of the system parameter.
    """
    import locale
    import os
    import platform
    import struct
    import sys

    (sysname, nodename, release,
     version, machine, processor) = platform.uname()

    try:
        loc = locale.getlocale()
    except ValueError:  # pragma: no cover
        loc = None

    host = [
        ('python', f'{sys.version_info[:]}'),
        ('python-bits', f'{struct.calcsize("P") * 8}'),
        ('OS', f'{sysname}'),
        ('OS-release', f'{release}'),
        ('machine', f'{machine}'),
        ('processor', f'{processor}'),
        ('byteorder', f'{sys.byteorder}'),
        ('LC_ALL', f'{os.environ.get("LC_ALL", "None")}'),
        ('LANG', f'{os.environ.get("LANG", "None")}'),
        ('LOCALE', f'{loc}'),
        ('cpu-count', f'{os.cpu_count()}'),
    ]

    return host


def get_optional_info() -> Dict[str, Union[str, bool]]:
    """Get optional package info (tensorflow, pytorch, hdf5_bloscfilter, etc.)

    Returns
    -------
    Dict[str, Union[str, False]]
        package name, package version (if installed, otherwise False)
    """
    res = {}
    try:
        import h5py
        bloscFilterAvail = h5py.h5z.filter_avail(32001)
    except ImportError:  # pragma: no cover
        bloscFilterAvail = False
    res['blosc-hdf5-plugin'] = bloscFilterAvail

    try:
        import torch
        torchVersion = torch.__version__
    except ImportError:  # pragma: no cover
        torchVersion = False
    res['pytorch'] = torchVersion

    try:
        import tensorflow
        tensorflowVersion = tensorflow.__version__
    except ImportError:  # pragma: no cover
        tensorflowVersion = False
    res['tensorflow'] = tensorflowVersion

    return res


def get_package_info(pkgs):
    """ get package versions for the passed required & optional packages.

    Using local imports to avoid import overhead on interpreter startup.
    """
    import importlib

    pversions = []
    for modname, ver_f in pkgs:
        try:
            mod = importlib.import_module(modname)
            ver = ver_f(mod)
            pversions.append((modname, ver))
        except Exception:  # pragma: no cover
            pversions.append((modname, None))

    return pversions
