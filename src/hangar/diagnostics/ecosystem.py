import importlib
import locale
import os
import platform
import struct
import sys


required_packages = [
    ('hangar', lambda p: p.__version__),
    ('click', lambda p: p.__version__),
    ('msgpack', lambda p: '.'.join([str(v) for v in p.version])),
    ('lmdb', lambda p: p.__version__),
    ('h5py', lambda p: p.__version__),
    ('numpy', lambda p: p.__version__),
    ('blosc', lambda p: p.__version__),
    ('yaml', lambda p: p.__version__),
    ('tqdm', lambda p: p.__version__),
    ('wrapt', lambda p: p.__version__),
    ('grpc', lambda p: p.__version__),
]


def get_versions():
    """
    Return basic information on our software installation, and out installed versions of packages.
    """

    d = {'host': get_system_info(),
         'packages': get_package_info(required_packages),
         'optional': get_optional_info()}
    return d


def get_system_info():
    (sysname, nodename, release,
     version, machine, processor) = platform.uname()

    try:
        loc = locale.getlocale()
    except ValueError:
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


def get_optional_info() -> dict:
    res = {}
    try:
        import h5py
        bloscFilterAvail = h5py.h5z.filter_avail(32001)
    except ImportError:
        bloscFilterAvail = False
    res['blosc-hdf5-plugin'] = bloscFilterAvail

    try:
        import torch
        torchVersion = torch.__version__
    except ImportError:
        torchVersion = False
    res['pytorch'] = torchVersion
    return res


def get_package_info(pkgs):
    """ get package versions for the passed required & optional packages """

    pversions = []
    for modname, ver_f in pkgs:
        try:
            mod = importlib.import_module(modname)
            ver = ver_f(mod)
            pversions.append((modname, ver))
        except Exception:
            pversions.append((modname, None))

    return pversions
