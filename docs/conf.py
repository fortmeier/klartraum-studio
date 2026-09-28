# Sphinx configuration for the Klartraum Studio documentation.

import os

project = "Klartraum Studio"
author = "Dirk Fortmeier"
copyright = "2025, Dirk Fortmeier"

extensions = [
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx_needs",
]

source_suffix = {".md": "markdown"}
exclude_patterns = ["_build", ".venv"]

myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

primary_domain = "cpp"
highlight_language = "cpp"

# Links into the Klartraum Engine API reference. KLARTRAUM_INVENTORY selects
# the inventory: unset uses the published engine docs, a path uses a local
# objects.inv (e.g. from a local engine docs build), and "off" builds without
# engine links.
KLARTRAUM_DOCS = "https://fortmeier.github.io/klartraum/docs/"
_inventory = os.environ.get("KLARTRAUM_INVENTORY", "")
if _inventory != "off":
    intersphinx_mapping = {"klartraum": (KLARTRAUM_DOCS, _inventory or None)}
intersphinx_disabled_reftypes = ["std:doc"]

# Requirements, specifications and tests are sphinx-needs objects. IDs are
# upper case with a type prefix, e.g. REQ_EDITOR_001.
needs_types = [
    {"directive": "req", "title": "Requirement", "prefix": "REQ_", "color": "#BFD8D2", "style": "node"},
    {"directive": "spec", "title": "Specification", "prefix": "SPEC_", "color": "#FEDCD2", "style": "node"},
    {"directive": "test", "title": "Test", "prefix": "TEST_", "color": "#DCB239", "style": "node"},
]
needs_id_required = True
needs_id_regex = r"^(REQ|SPEC|TEST)_[A-Z0-9_]+$"
needs_fields = {
    "status": {
        "description": "draft, open, implemented or verified",
        "schema": {"type": "string", "enum": ["draft", "open", "implemented", "verified"]},
    },
    "gtest": {
        "description": "GoogleTest case that implements a test need, e.g. Suite.name",
        "schema": {"type": "string"},
        "nullable": True,
    },
}
needs_links = {
    "implements": {"incoming": "is implemented by", "outgoing": "implements"},
    "verifies": {"incoming": "is verified by", "outgoing": "verifies"},
}
needs_build_json = True

html_theme = "furo"
html_title = "Klartraum Studio"
html_static_path = ["_static"]
