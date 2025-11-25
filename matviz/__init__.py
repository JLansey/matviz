try:
    from ._version import __version__
except ImportError:
    # For development installs where setuptools_scm hasn't generated _version.py yet
    __version__ = "0.0.0+dev"