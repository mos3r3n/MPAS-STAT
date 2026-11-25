"""Sphinx configuration for the MPAS-STAT documentation."""
import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

project = "MPAS-STAT"
author = "MPAS-STAT contributors"
copyright = f"{date.today().year}, {author}"
try:
    from importlib.metadata import version as _version
    release = _version("mpas-stat")
except Exception:
    release = "dev"
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "USER_GUIDE.md", "Thumbs.db", ".DS_Store"]

# MyST (Markdown) -------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "fieldlist", "tasklist"]
myst_heading_anchors = 3

# autodoc / autosummary ---------------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_mock_imports = ["cartopy", "numba"]
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# HTML ------------------------------------------------------------------------
html_theme = "furo"
html_title = "MPAS-STAT"
html_static_path = []
html_theme_options = {
    "source_repository": "https://github.com/mos3r3n/MPAS-STAT/",
    "source_branch": "main",
    "source_directory": "docs/",
}
