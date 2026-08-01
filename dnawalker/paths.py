"""Repository-owned paths used by configuration, artifacts, and outputs.

Source checkouts resolve their root from this module's location. Installed
packages can run against an external checkout by starting inside that checkout
or by setting ``DNAWALKER_PROJECT_ROOT`` explicitly.
"""

import os
from pathlib import Path


_PROJECT_ROOT_ENV = "DNAWALKER_PROJECT_ROOT"


def _has_project_layout(candidate: Path) -> bool:
    """Return whether *candidate* provides the required repository configs."""
    return (
        (candidate / "configs" / "common.ini").is_file()
        and (candidate / "configs" / "cnn.ini").is_file()
        and (candidate / "configs" / "transformer.ini").is_file()
    )


def _discover_repo_root() -> Path:
    """Locate the external project root used for configs and large artifacts."""
    configured = os.environ.get(_PROJECT_ROOT_ENV)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if not _has_project_layout(candidate):
            raise RuntimeError(
                f"{_PROJECT_ROOT_ENV} does not point to a DNA Walker project "
                f"root with configs/common.ini: {candidate}"
            )
        return candidate

    source_candidate = Path(__file__).resolve().parents[1]
    if _has_project_layout(source_candidate):
        return source_candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _has_project_layout(candidate):
            return candidate

    raise RuntimeError(
        "DNA Walker project root not found. Run from a source checkout or set "
        f"{_PROJECT_ROOT_ENV}=/path/to/checkout."
    )


REPO_ROOT = _discover_repo_root()
CONFIG_DIR = REPO_ROOT / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "common.ini"
DEFAULT_CNN_CONFIG = CONFIG_DIR / "cnn.ini"
DEFAULT_TRANSFORMER_CONFIG = CONFIG_DIR / "transformer.ini"
DEFAULT_SMOKE_PROFILE = CONFIG_DIR / "profiles" / "smoke.ini"
DATA_DIR = REPO_ROOT / "data"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
