# coding=utf-8
"""Pytest configuration for the canonical ``dnawalker`` test suite.

Ensures the repository root is importable when tests are invoked by absolute
path from outside the checkout, without relying on removed compatibility
modules.
"""

import os
import sys

# Repo root is the parent of this ``tests/`` directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
