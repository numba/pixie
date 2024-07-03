# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
from pixie import __version__  # noqa: E402

project = 'pixie'
copyright = '2024, Numba Developers'
author = 'Numba Developers'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    'sphinx_rtd_theme',
]
nb_execution_mode = "off"

templates_path = ['_templates']
exclude_patterns = []

language = 'English'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']
