#! /usr/bin/env python
#
# Copyright (C) 2017-2019 Jonathan Lansey


DESCRIPTION = "matviz: matrix data visualization"
LONG_DESCRIPTION = """\
MatViz is a library for making nice looking and useful graphics in Python. It is built on top of `matplotlib <https://matplotlib.org/>`_ and closely integrated with `numpy <https://numpy.org/>`_ and `pandas <https://pandas.pydata.org/>`_ data structures.
It also pulls from and is inspired by `seaborn <https://https://seaborn.pydata.org/>`_
Here is some of the functionality that matviz offers:
- A histogram function that compares multiple distributions with default settings to maximize utility
- A 2D histogram function with fun tricks like smoothing and plotting timeseries data
- Specialized support for using complex numbers in place of x and y (z = x + 1j * y)
- Taking any figure and making it look nicer than the matplotlib defaults
- Streamgraph implementation with lots of additional features, based on stackedplot
- Lots of plotting functions to import by default and make the coding environment similar to Matlab.
"""

DISTNAME = 'matviz'
MAINTAINER = 'Jonathan Lansey'
MAINTAINER_EMAIL = 'jonathan@lansey.net'
URL = 'https://github.com/JLansey/matviz'
LICENSE = 'BSD (3-clause)'
DOWNLOAD_URL = 'https://github.com/JLansey/matviz'

INSTALL_REQUIRES = [
    'numpy>=1.9.3',
    'scipy>=0.14.0',
    'pandas>=0.15.2',
    'matplotlib>=1.4.3',
    'seaborn>=0.5.1',
    'simplejson',
    'mpld3',
    'scikit-learn',
    'PyMuPDF'
]

PACKAGES = [
    'matviz'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.9',
    'License :: OSI Approved :: BSD License',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Multimedia :: Graphics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(
        name=DISTNAME,
        author=MAINTAINER,
        author_email=MAINTAINER_EMAIL,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        license=LICENSE,
        url=URL,
        use_scm_version={'write_to': 'matviz/_version.py'},
        setup_requires=['setuptools_scm'],
        download_url=DOWNLOAD_URL,
        install_requires=INSTALL_REQUIRES,
        packages=PACKAGES,
        classifiers=CLASSIFIERS
    )