import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'TrOCR Training'
copyright = '2024, Your Name'
author = 'Your Name'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
]

# Theme settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static'] 