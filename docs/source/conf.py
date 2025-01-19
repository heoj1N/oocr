import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'OOCR Training'
copyright = '2024, Your Name'
author = 'Your Name'
release = '0.1.0'  # Add version number

# Extensions
extensions = [
    'sphinx.ext.autodoc',      # API documentation
    'sphinx.ext.napoleon',     # Support for Google/NumPy docstrings
    'sphinx.ext.viewcode',     # Add links to source code
    'sphinx.ext.githubpages',  # GitHub pages support
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
]

# Theme settings
html_theme = 'sphinx_rtd_theme'  # Use Read the Docs theme
html_static_path = ['_static']

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False

# Autodoc settings
autodoc_member_order = 'bysource'
autodoc_typehints = 'description' 