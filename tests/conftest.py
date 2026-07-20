# coding=utf-8
"""Pytest configuration for the validation-experiments test suite.

Ensures the repository root is importable so tests can ``import pysim`` and
``import validation_common`` regardless of the directory pytest is invoked from.
"""

import os
import sys

# Repo root is the parent of this ``tests/`` directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
