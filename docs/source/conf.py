# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import sys
#sys.path.insert(0, os.path.abspath('../privkit/'))
from pathlib import Path

# go up two levels from /docs/source to the package root
sys.path.insert(0, str(Path().resolve().parent.parent))

# -- Project information -----------------------------------------------------

project = 'Privkit'
copyright = '2024, Privkit'
author = 'Privkit'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    #"sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    # -- General configuration ------------------------------------------------
    "sidebar_includehidden": True,
    "use_edit_page_button": False,
    "logo": {
        "text": "Privkit",
        "image_dark": "_static/logo-dark.svg",
    },
    "logo": {
        "alt_text": "privkit homepage",
        "image_relative": "logos/logo_grey_word.svg",
        "image_light": "logos/logo_grey_word.svg",
        "image_dark": "logos/logo_white_word.svg",
    },
    "navbar_align": "left",
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "footer_start": ["copyright"],
    "footer_center": ["sphinx-version"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/privkit",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_favicon = "logos/favicon.svg"

html_static_path = ["logos"]

html_short_title = "privkit"

html_context = {}