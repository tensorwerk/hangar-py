import importlib
import locale
import os
import platform
import struct
import sys

from .. import config
from ..utils import ignoring

required_packages = [('hangar', lambda p: p.__version__),
                     ('click', lambda p: p.__version__),
                     ('msgpack', lambda p: '.'.join([str(v) for v in p.version])),
                     ('lmdb', lambda p: p.__version__),
                     ('h5py', lambda p: p.__version__),
                     ('numpy', lambda p: p.__version__),
                     ('blosc', lambda p: p.__version__),
                     ('yaml', lambda p: p.__version__),
                     ('xxhash', lambda p: p.VERSION)]


def get_versions():
    """
    Return basic information on our software installation, and out installed versions of packages.
    """

    d = {'host': get_system_info(),
         'packages': get_package_info(required_packages),
         'config': config.config}
    return d


def get_system_info():
    (sysname, nodename, release,
     version, machine, processor) = platform.uname()

    try:
        loc = locale.getlocale()
    except ValueError:
        loc = None

    host = [('python', f'{sys.version_info[:]}'),
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


def version_of_package(pkg):
    """ Try a variety of common ways to get the version of a package """
    with ignoring(AttributeError):
        return pkg.__version__
    with ignoring(AttributeError):
        return str(pkg.version)
    with ignoring(AttributeError):
        return '.'.join(map(str, pkg.version_info))
    with ignoring(AttributeError):
        return str(pkg.VERSION)
    return None


def get_package_info(pkgs):
    """ get package versions for the passed required & optional packages """

    pversions = []
    for pkg in pkgs:
        if isinstance(pkg, (tuple, list)):
            modname, ver_f = pkg
        else:
            modname = pkg
            ver_f = version_of_package

        if ver_f is None:
            ver_f = version_of_package

        try:
            mod = importlib.import_module(modname)
            ver = ver_f(mod)
            pversions.append((modname, ver))
        except Exception:
            pversions.append((modname, None))

    return pversions
