#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import platform
import sys
from os.path import join
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion
from setuptools import setup, Extension, find_packages


# Use `setup.py [] --debug` for a debug build of hangar
HANGAR_DEBUG_BUILD = False


# Set deployment target for mac
#
# Need to ensure that extensions are built for macos 10.9 when compiling on a
# 10.9 system or above, overriding distutils behavior which is to target
# the version used to build the current python binary.
#
# TO OVERRIDE:
#   set MACOSX_DEPLOYMENT_TARGET before calling setup.py
#
# From https://github.com/pandas-dev/pandas/pull/24274
# 3-Clause BSD License: https://github.com/pandas-dev/pandas/blob/master/LICENSE
if sys.platform == 'darwin':
    if 'MACOSX_DEPLOYMENT_TARGET' not in os.environ:
        current_system = LooseVersion(platform.mac_ver()[0])
        python_target = LooseVersion(get_config_var('MACOSX_DEPLOYMENT_TARGET'))
        if python_target < '10.9' and current_system >= '10.9':
            os.environ['MACOSX_DEPLOYMENT_TARGET'] = '10.9'


class LazyCommandClass(dict):
    """
    Lazy command class that defers operations requiring Cython and numpy until
    they've actually been downloaded and installed by setup_requires.
    """

    def __contains__(self, key):
        return key in ['build_ext', 'bdist_wheel', 'sdist'] or super().__contains__(key)

    def __setitem__(self, key, value):
        if key == 'build_ext':
            raise AssertionError("build_ext overridden!")
        super().__setitem__(key, value)

    def __getitem__(self, key):
        if key == 'build_ext':
            return self.make_build_ext_cmd()
        elif key == 'bdist_wheel':
            return self.make_bdist_wheel_cmd()
        elif key  == 'sdist':
            return self.make_sdist_cmd()
        else:
            return super().__getitem__(key)

    def make_build_ext_cmd(self):
        """Returns a command class implementing 'build_ext'.
        """
        from Cython.Distutils.build_ext import new_build_ext as cython_build_ext
        from Cython.Compiler.Main import default_options

        default_options['language_level'] = 3
        default_options['compiler_directives']['embedsignature'] = True
        default_options['compiler_directives']['emit_code_comments'] = True
        if HANGAR_DEBUG_BUILD is True:
            default_options['annotate'] = True
            default_options['emit_linenums'] = True
            default_options['gdb_debug'] = True

        class build_ext(cython_build_ext):
            def build_extensions(self):
                cython_build_ext.build_extensions(self)

        return build_ext

    def make_bdist_wheel_cmd(self):
        """Returns a command class implementing 'bdist_wheel'.
        """
        from wheel.bdist_wheel import bdist_wheel

        class bdist_wheel_cmd(bdist_wheel):
            def run(self):
                # This may modify package_data:
                bdist_wheel.run(self)

        return bdist_wheel_cmd

    def make_sdist_cmd(self):
        """A command class implementing 'sdist'.
        """
        from distutils.command.sdist import sdist as _sdist

        class sdist(_sdist):
            def run(self):
                # Make sure the compiled Cython files in the distribution are up-to-date
                # so we generate .c files correctly (.so will be removed)
                _sdist.run(self)

        return sdist

# Pass command line flags to setup.py script
# handle --lflags=[FLAGS] --cflags=[FLAGS]
args = sys.argv[:]
for arg in args:
    if arg.find('--debug') == 0:
        HANGAR_DEBUG_BUILD = True
        sys.argv.remove(arg)

# Source files for build
CYTHON_SOURCES = [
    join('src', 'hangar', 'optimized_utils.pyx'),
    join('src', 'hangar', 'backends', 'specs.pyx'),
    join('src', 'hangar', 'backends', 'specparse.pyx'),
    join('src', 'hangar', 'records', 'recordstructs.pyx'),
    join('src', 'hangar', 'records', 'column_parsers.pyx'),
    join('src', 'hangar', 'records', 'hashmachine.pyx'),
]
CYTHON_HEADERS = [
    join('src', 'hangar', 'optimized_utils.pxd'),
    join('src', 'hangar', 'backends', 'specs.pxd'),
    join('src', 'hangar', 'records', 'recordstructs.pxd'),
]

__extensions = []
for source in CYTHON_SOURCES:
    module_name = os.path.splitext(source)[0]
    if module_name + '.pxd' in CYTHON_HEADERS:
        deps = module_name + '.pxd'
    else:
        deps = None
    if module_name.startswith(f'src{os.sep}'):
        module_name = module_name.lstrip(f'src{os.sep}')
    module_name = module_name.replace(os.sep, '.')
    ext = Extension(module_name,
                    include_dirs=[],
                    define_macros=[],
                    sources=[source],
                    depends=[deps] if deps else [],
                    library_dirs=[],
                    libraries=[],
                    extra_link_args=[],
                    extra_compile_args=[],
                    language="c")
    __extensions.append(ext)

with open('README.rst') as f:
    README_RST = f.read()

SHORT_DESCRIPTION = (
    'Hangar is version control for tensor data. Commit, branch, merge, '
    'revert, and collaborate in the data-defined software era.'
)

SETUP_REQUIRES = [
    'cython>=0.27',
    'setuptools>=40.0',
    'wheel>=0.30',
]

INSTALL_REQUIRES = [
    'blosc>=1.8',
    'click',
    'grpcio',
    'protobuf',
    'h5py>=2.9',
    'hdf5plugin>=2.0',
    'lmdb>=0.94',
    'numpy',
    'tqdm',
    'wrapt',
    'xxhash',
]

setup(
    name='hangar',
    version='0.5.1',
    license='Apache 2.0',
    # Package Meta Info (for PyPi)
    description=SHORT_DESCRIPTION,
    long_description=README_RST,
    long_description_content_type='text/x-rst',
    author='Richard Izzo',
    author_email='rick@tensorwerk.com',
    maintainer='Richard Izzo',
    maintainer_email='rick@tensorwerk.com',
    url='https://github.com/tensorwerk/hangar-py',
    project_urls={
        'Documentation': 'https://hangar-py.readthedocs.io/',
        'Changelog': 'https://hangar-py.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/tensorwerk/hangar-py/issues',
    },
    platforms=['any'],
    # Module Source Files
    ext_modules=__extensions,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'': ['*.ini', '*.proto']},
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': ['hangar = hangar.cli:main']
    },
    # Requirements
    python_requires='>= 3.6.0',
    install_requires=INSTALL_REQUIRES,
    setup_requires=SETUP_REQUIRES,
    # hooks into `sdist`, `bdist_wheel`, `bdist_ext` commands.
    cmdclass=LazyCommandClass(),
    # PyPi classifiers
    # http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: Unix',
        'Programming Language :: Cython',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Database',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Version Control',
        'Topic :: Utilities',
    ],
)
