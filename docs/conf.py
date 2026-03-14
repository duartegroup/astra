import os
import sys

# In case the project was not installed
sys.path.insert(0, os.path.abspath(".."))

import astra  # noqa: E402

# -- Project information -----------------------------------------------------

project = "ASTRA"
copyright = "2025, Wojtek Treyde"
author = "Wojtek Treyde"
version = astra.__version__
release = astra.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.extlinks",
    "sphinx_design",
    "sphinx_copybutton",
    "notfound.extension",
    "myst_nb",
]

nb_execution_mode = "off"

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "special-members": True,
    "inherited-members": True,
    "show-inheritance": True,
}

autosummary_generate = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_ivar = False
napoleon_numpy_docstring = True

source_suffix = ".rst"
master_doc = "index"
language = "English"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = "ASTRA"

html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#f59e0b",
        "color-brand-content": "#fbbf24",
        "color-background-primary": "#0b0f1a",
        "color-background-secondary": "#0f1729",
        "color-sidebar-background": "#0b0f1a",
        "color-foreground-primary": "#e2e8f0",
        "color-foreground-secondary": "#94a3b8",
        "color-highlighted-background": "#1e293b",
        "color-admonition-background": "#101f3a",
    },
    "light_css_variables": {
        "color-brand-primary": "#b45309",
        "color-brand-content": "#d97706",
    },
}

pygments_style = "tango"
pygments_dark_style = "monokai"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

# -- LaTeX output ------------------------------------------------------------

htmlhelp_basename = "astradoc"

latex_documents = [
    (master_doc, "astra.tex", "ASTRA Documentation", "astra", "manual"),
]

man_pages = [(master_doc, "astra", "ASTRA Documentation", [author], 1)]

texinfo_documents = [
    (
        master_doc,
        "astra",
        "ASTRA Documentation",
        author,
        "astra",
        "Automated model selection using statistical testing "
        "for robust algorithms",
        "Miscellaneous",
    ),
]

# -- Extension configuration -------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}
