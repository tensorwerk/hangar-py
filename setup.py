#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import io

from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='hangar',
    version='0.4.0',
    license='Apache 2.0',
    description=
    'Hangar is version control for tensor data. Commit, branch, merge, revert, and collaborate in the data-defined software era.',
    long_description=read('README.rst'),
    author='Richard Izzo',
    author_email='rick@tensorwerk.com',
    url='https://github.com/tensorwerk/hangar-py',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Database',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
    project_urls={
        'Documentation': 'https://hangar-py.readthedocs.io/',
        'Changelog':
        'https://hangar-py.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/tensorwerk/hangar-py/issues',
    },
    keywords=[],
    python_requires='>= 3.6.0',
    install_requires=[
        'blosc',
        'click',
        'grpcio',
        'protobuf',
        'h5py==2.9.0',
        'hdf5plugin==2.0.0',
        'lmdb>=0.94,<=0.96',
        'numpy',
        'tqdm',
        'wrapt',
        'xxhash',
    ],
    entry_points={
        'console_scripts': ['hangar = hangar.cli:main']
    },
)
