# coding=utf-8
"""Small, shared helpers for binding results to binary artifacts."""

import hashlib
import hmac
import os
import re
from numbers import Integral


def sha256_file(path, chunk_size=1024 * 1024):
    """Return the SHA-256 digest of a regular file."""
    path = os.fspath(path)
    if (isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size <= 0):
        raise ValueError(
            f"chunk_size must be a positive integer, got {chunk_size!r}"
        )

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_matching_sha256(path, expected, artifact_name):
    """Return the actual digest and reject a present, mismatched expectation.

    ``expected=None`` is reserved for explicitly supported legacy artifacts that
    predate hash provenance. A checkpoint that contains a hash must contain a
    valid 64-digit SHA-256 value and match the file exactly.
    """
    actual = sha256_file(path)
    if expected is None:
        return actual
    if (not isinstance(expected, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", expected) is None):
        raise ValueError(
            f"Checkpoint {artifact_name} SHA-256 is invalid: {expected!r}"
        )
    if not hmac.compare_digest(actual, expected.lower()):
        raise ValueError(
            f"Checkpoint {artifact_name} SHA-256 mismatch: "
            f"expected {expected.lower()}, got {actual}"
        )
    return actual


def optional_checkpoint_seed(value, field_name):
    """Return a checkpoint seed as ``int`` without lossy coercion.

    ``None`` is accepted for legacy checkpoints. Present values must be exact
    non-boolean integers in the uint32 range used by the training configuration.
    """
    if value is None:
        return None
    if (isinstance(value, bool)
            or not isinstance(value, Integral)
            or not 0 <= int(value) <= 2 ** 32 - 1):
        raise ValueError(
            f"Checkpoint {field_name} must be an integer in "
            f"[0, {2 ** 32 - 1}], got {value!r}"
        )
    return int(value)


def optional_positive_int(value, field_name):
    """Return an optional exact positive integer from checkpoint metadata."""
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) <= 0
    ):
        raise ValueError(
            f"Checkpoint {field_name} must be a positive integer, got {value!r}"
        )
    return int(value)
