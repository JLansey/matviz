# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information
project = "matviz"
copyright = "2019, Jonathan C. Lansey"
author = "Jonathan C. Lansey"

# -- General configuration
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# Notebook execution
nb_execution_mode = "off"

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

# Numpydoc settings
numpydoc_show_class_members = False

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- Options for HTML output
html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/JLansey/matviz",
    "navbar_align": "left",
    "navigation_with_keys": False,
}

html_static_path = ["_static"]

# Suppress warnings about missing references for built-in types
nitpicky = False
