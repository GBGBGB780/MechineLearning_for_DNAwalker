# coding=utf-8
"""
dnawalker.studies.protocol — Shared helpers for the validation-experiments suite.

This canonical module hosts cross-cutting helpers used by the recovery,
warm-start, and multi-seed experiment implementations. Root
``validation_common.py`` remains a same-module-object compatibility facade; new
code imports this package module directly.

Scope of this file grows task-by-task. Currently implemented:
  - ``PARAM_NAMES``            : the seven trainable parameters (pysim order)
  - ``MissingSeedError``       : raised when a stochastic step lacks a seed
  - ``require_seed``           : seed guard enforcing Requirement 6.2
  - ``require_int``            : strict integer/range validation
  - ``require_finite_real``    : strict finite-real/range validation
  - ``derive_batch_seed``      : deterministic per-batch seed (Requirements 2.10, 3.12, 6.1)
  - ``load_configured_ranges`` : Configured_Ranges loader (Requirement 6.x)
  - ``lhs_unit``               : Latin Hypercube sample in [0,1)^dim (Requirement 2.1)
  - ``lhs_params``             : LHS Parameter_Set draw scaled into ranges (Requirement 2.1)
  - ``relative_error``         : signed relative error (Requirement 2.6, 2.12)
  - ``r_squared``              : coefficient of determination R² (Requirement 2.7)
  - ``distribution_stats``     : mean/median/std/min/max/p5/p95 (Requirement 2.9)
  - ``make_predictor``         : pluggable predictor factory (Requirement 5.3)
  - ``write_json``             : indented JSON writer + dir creation (Requirement 6.3)
"""

import json
import os
import tempfile
from numbers import Integral, Real

import numpy as np

from dnawalker.physics import simulator as pysim
from dnawalker.config import Config
from dnawalker.paths import DEFAULT_CONFIG
# Deterministic per-batch seed derivation is shared by generation and validation.
from dnawalker.shared.seeding import derive_batch_seed as _derive_batch_seed

derive_batch_seed = _derive_batch_seed

# The seven trainable physical parameters, in the canonical pysim order:
# ['E_b', 'E_b_azo_trans', 'E_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']
PARAM_NAMES = pysim.PARAM_NAMES

# Default source for configured parameter ranges.
_DEFAULT_CONFIG_PATH = os.fspath(DEFAULT_CONFIG)


class MissingSeedError(RuntimeError):
    """Raised when a stochastic step is requested without a configured seed.

    Enforces Requirement 6.2: an experiment started without a fixed seed for a
    stochastic step must stop *before* performing that step and report the
    missing-seed condition.
    """


def require_seed(seed, step_name):
    """Return ``seed`` when it is a valid integer seed, else raise.

    A valid seed is a non-negative Python ``int`` (``bool`` is rejected even
    though it is an ``int`` subclass). Negative values are rejected because
    NumPy's random generators do not accept them.

    Args:
        seed: The candidate seed value.
        step_name: Human-readable name of the stochastic step being guarded.

    Returns:
        int: The validated seed, unchanged.

    Raises:
        MissingSeedError: If ``seed`` is negative, ``None``, or not an ``int``.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise MissingSeedError(step_name)
    return seed


def require_int(value, name, *, minimum=None, maximum=None):
    """Return an exact integer after optional inclusive range validation.

    Unlike ``int(value)``, this helper never truncates floats, parses strings,
    or accepts booleans. NumPy integer scalars are accepted and normalized to a
    Python ``int`` for JSON serialization and downstream library calls.
    """
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {result}")
    return result


def require_finite_real(value, name, *, minimum=None, maximum=None,
                        strict_minimum=False, strict_maximum=False):
    """Return a finite real number after optional range validation."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if minimum is not None:
        invalid = result <= minimum if strict_minimum else result < minimum
        if invalid:
            op = ">" if strict_minimum else ">="
            raise ValueError(f"{name} must be {op} {minimum}, got {result}")
    if maximum is not None:
        invalid = result >= maximum if strict_maximum else result > maximum
        if invalid:
            op = "<" if strict_maximum else "<="
            raise ValueError(f"{name} must be {op} {maximum}, got {result}")
    return result


def load_configured_ranges(config_path=None):
    """Load the Configured_Ranges from ``[TRAINING_PARAMETER_RANGES]``.

    Reuses the existing ``config_loader`` API and maps the config's trainable
    parameter names onto ``pysim.PARAM_NAMES`` by case-insensitive name. The
    returned dict is keyed by ``pysim.PARAM_NAMES`` and ordered to match.

    Args:
        config_path: Optional path to a config file. Defaults to the canonical
            ``configs/common.ini`` resolved via ``dnawalker.paths``.

    Returns:
        dict: ``{param_name: (min, max)}`` for the seven trainable params,
        keyed and ordered by ``pysim.PARAM_NAMES``.
    """
    cfg_path = config_path if config_path is not None else _DEFAULT_CONFIG_PATH
    config = Config(cfg_path)

    # Map config trainable names onto pysim.PARAM_NAMES by NAME (case-insensitive)
    # via the shared param_io.build_name_map, then re-key the ranges so they are
    # addressable by the canonical parameter name. build_name_map raises if the
    # config's parameter set does not match pysim.PARAM_NAMES (guards against a
    # silently mislabeled reorder), unlike the old positional dict(zip(...)).
    from dnawalker.shared.parameters import build_name_map
    name_map = build_name_map(config.get_trainable_param_names())
    raw_ranges = config.get_param_ranges()  # {trainable_name: (min, max)}
    mapped = {name_map[k]: v for k, v in raw_ranges.items()}

    # Return in canonical pysim.PARAM_NAMES order.
    return {name: mapped[name] for name in PARAM_NAMES}


def lhs_unit(n, dim, seed):
    """Return a Latin Hypercube sample of shape ``(n, dim)`` in ``[0, 1)``.

    Reuses the ``dnawalker.data.generate`` LHS pattern verbatim: prefer
    ``scipy.stats.qmc.LatinHypercube(d=dim, seed=seed)`` and fall back to a
    stratified random sampler (one stratum per row, jittered uniformly) when
    SciPy is unavailable. Both paths are seeded so a repeated call with the same
    ``seed`` reproduces the same draw (Requirement 2.1 / 6.1).

    Args:
        n: Number of sample rows to draw (a positive integer).
        dim: Dimensionality of each sample (number of columns).
        seed: Integer seed driving the sampler.

    Returns:
        np.ndarray: Array of shape ``(n, dim)`` with every value in ``[0, 1)``.
    """
    n = require_int(n, "n", minimum=1)
    dim = require_int(dim, "dim", minimum=1)
    seed = require_seed(seed, "lhs_unit")

    try:
        from scipy.stats import qmc
    except ImportError:
        qmc = None

    if qmc is not None:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(n)

    # Fallback only when scipy.stats.qmc is genuinely unavailable. Runtime
    # failures such as MemoryError or invalid arguments must remain visible.
    rng = np.random.default_rng(seed)
    out = np.zeros((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def lhs_params(n, ranges, seed):
    """Draw ``n`` Parameter_Sets via LHS, scaled into the Configured_Ranges.

    Produces a unit-cube Latin Hypercube sample with :func:`lhs_unit` and scales
    each column ``i`` into ``ranges[PARAM_NAMES[i]] = (min, max)`` using
    ``min + (max - min) * unit`` — the same affine scaling ``dnawalker.data.generate`` applies
    via ``MIN_VALS + (MAX_VALS - MIN_VALS) * unit_sample``. The returned array is
    keyed positionally to :data:`PARAM_NAMES`, so column ``i`` corresponds to
    ``PARAM_NAMES[i]`` (Requirement 2.1).

    Args:
        n: Number of Parameter_Sets to draw (a positive integer).
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` keyed by
            ``pysim.PARAM_NAMES``.
        seed: Integer seed driving the sampler.

    Returns:
        np.ndarray: Array of shape ``(n, 7)`` whose column ``i`` lies within
        ``ranges[PARAM_NAMES[i]]``.
    """
    dim = len(PARAM_NAMES)
    unit = lhs_unit(n, dim, seed)

    mins = np.array([ranges[name][0] for name in PARAM_NAMES], dtype=np.float64)
    maxs = np.array([ranges[name][1] for name in PARAM_NAMES], dtype=np.float64)

    return mins + (maxs - mins) * unit


def relative_error(estimated, truth):
    """Return the signed relative error ``(estimated - truth) / abs(truth)``.

    Works elementwise for both Python scalars and numpy arrays, so it can be
    applied to a single (estimated, truth) pair or to whole per-parameter
    columns at once. The result has the same shape as the broadcast of its
    inputs (Requirement 2.6).

    Zero-truth handling: this function intentionally does **not** special-case
    ``truth == 0``. Where ``truth`` is exactly 0 the quotient evaluates to
    ``+/-inf`` (when ``estimated != 0``) or ``nan`` (when ``estimated == 0``).
    Per Requirement 2.12 the *caller* (the recovery-metrics step, task 4.8) is
    responsible for detecting truth-equals-zero samples, excluding them from the
    relative-error distribution, and counting them. Callers should therefore
    filter the non-finite entries this function returns for those samples.

    Args:
        estimated: Estimated value(s); scalar or numpy array.
        truth: Ground-truth value(s); scalar or numpy array, broadcastable
            against ``estimated``.

    Returns:
        The signed relative error, as a numpy array when either input is an
        array (truth==0 positions yield ``inf``/``nan``), or a Python/numpy
        float for scalar inputs.
    """
    estimated = np.asarray(estimated, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    # Suppress the expected divide-by-zero / 0-over-0 warnings: the resulting
    # inf/nan values are the documented contract and are filtered by callers.
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (estimated - truth) / np.abs(truth)
    # Preserve scalar-in/scalar-out ergonomics for single-value callers.
    if result.ndim == 0:
        return float(result)
    return result


def r_squared(y_true, y_pred):
    """Return the coefficient of determination R² between truth and prediction.

    Computed as ``1 - SS_res / SS_tot`` where
    ``SS_res = sum((y_true - y_pred) ** 2)`` and
    ``SS_tot = sum((y_true - mean(y_true)) ** 2)`` over the flattened inputs
    (Requirement 2.7).

    Degenerate ``SS_tot == 0`` convention: ``SS_tot`` is zero exactly when every
    ``y_true`` value is identical (a constant target), making R² undefined in
    the usual sense. In that case this function returns ``1.0`` if the
    predictions match the (constant) truth perfectly (``SS_res == 0``) and
    ``0.0`` otherwise. This keeps a perfect fit scoring 1.0 while a non-perfect
    fit against a constant target scores the baseline 0.0, avoiding ``inf``/
    ``nan`` blow-ups in the reported metrics.

    Args:
        y_true: Ground-truth values; scalar or array-like.
        y_pred: Predicted values; scalar or array-like, same shape as
            ``y_true``.

    Returns:
        float: The R² value (``<= 1.0`` for non-degenerate inputs).
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    if ss_tot == 0.0:
        # Constant target: perfect fit -> 1.0, otherwise the baseline 0.0.
        return 1.0 if ss_res == 0.0 else 0.0

    return 1.0 - ss_res / ss_tot


def distribution_stats(values):
    """Summarize a 1-D sample as a dict of descriptive statistics.

    Returns the mean, median, standard deviation, minimum, maximum, and the 5th
    and 95th percentiles of ``values`` (Requirement 2.9). Percentiles are
    computed with :func:`numpy.percentile` at 5 and 95. The standard deviation
    uses numpy's default population convention (``ddof=0``).

    Args:
        values: A 1-D array-like of finite numeric samples (non-empty).

    Returns:
        dict: ``{"mean", "median", "std", "min", "max", "p5", "p95"}`` with
        plain ``float`` values.
    """
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "values must be a non-empty finite numeric sample"
        ) from exc
    if arr.size == 0:
        raise ValueError("values must contain at least one sample")
    if not np.all(np.isfinite(arr)):
        raise ValueError("values must contain only finite samples")

    with np.errstate(over="ignore", invalid="ignore"):
        stats = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p95": float(np.percentile(arr, 95)),
        }
    if not all(np.isfinite(value) for value in stats.values()):
        raise ValueError("distribution statistics overflowed")
    return stats


def make_predictor(model, **overrides):
    """Build a pluggable Inverse_Model predictor for the validation suite.

    Factory returning a predictor object that exposes the common inverse-model
    interface used across the suite:

      - ``predict(X) -> np.ndarray`` of shape ``(N, 7)`` in physical-parameter
        space (accepts a single ``(3, T)`` curve set or a batch ``(N, 3, T)``);
      - ``get_param_names() -> list[str]`` naming the seven trainable params.

    Both concrete predictors already implement this contract identically (see
    ``dnawalker.cnn.inference`` and
    ``dnawalker.transformer.inference``), which is what lets the recovery
    and warm-start experiments treat the model as a pluggable ``Predictor``
    (Requirement 5.3).

    The heavy deep-learning modules (which import ``torch``) are imported
    **lazily inside this function** so that merely importing ``validation_common``
    stays light: the stats/LHS/seed helpers can be used without ``torch`` being
    installed or loaded. ``torch`` is only pulled in when a predictor is actually
    constructed.

    Args:
        model: Which Inverse_Model to build. ``'cnn'`` selects
            :class:`NanorobotPredictor`; ``'transformer'`` selects
            :class:`TransformerPredictor`. Case-insensitive.
        **overrides: Constructor keyword arguments forwarded to the selected
            predictor. The two predictors accept different override keywords:

            * ``model='cnn'`` → ``NanorobotPredictor(config_file=None,
              model_path=None, config_override_file=None)``;
            * ``model='transformer'`` → ``TransformerPredictor(model_path=None,
              parent_config_override_file=None,
              transformer_config_override_file=None)``.

            Only pass keywords supported by the chosen predictor.

    Returns:
        An initialized predictor instance exposing ``predict`` and
        ``get_param_names``.

    Raises:
        ValueError: If ``model`` is not ``'cnn'`` or ``'transformer'``.
    """
    key = str(model).strip().lower()

    if key == "cnn":
        # Lazy package imports keep torch out of ``import validation_common``.
        from dnawalker.cnn.inference import NanorobotPredictor
        return NanorobotPredictor(**overrides)

    if key == "transformer":
        from dnawalker.transformer.inference import TransformerPredictor
        return TransformerPredictor(**overrides)

    raise ValueError(
        f"Unknown model {model!r}; expected 'cnn' or 'transformer'."
    )


def build_predictor_overrides(
        model, *, model_path=None, config_path=None,
        transformer_config_path=None):
    """Translate release CLI paths into model-specific predictor arguments."""
    key = str(model).strip().lower()
    if key not in {"cnn", "transformer"}:
        raise ValueError(
            f"Unknown model {model!r}; expected 'cnn' or 'transformer'."
        )

    def absolute(path):
        return os.path.abspath(os.fspath(path)) if path else None

    overrides = {}
    if model_path:
        overrides["model_path"] = absolute(model_path)
    if key == "cnn":
        if transformer_config_path:
            raise ValueError(
                "transformer_config_path is invalid for the CNN predictor"
            )
        if config_path:
            overrides["config_override_file"] = absolute(config_path)
        return overrides

    if config_path:
        overrides["parent_config_override_file"] = absolute(config_path)
    transformer_path = transformer_config_path or config_path
    if transformer_path:
        overrides["transformer_config_override_file"] = absolute(
            transformer_path
        )
    return overrides


class _NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that renders common numpy scalar/array types natively.

    Trivially handles the numpy types the experiment metrics may carry so that
    writing a metrics dict does not fail on ``np.float64``/``np.int64`` values
    or small ``np.ndarray`` columns. Anything else falls through to the default
    encoder (and raises as usual for genuinely unserializable objects).
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _json_finite(obj):
    """Return a JSON-native tree, replacing non-finite numbers with ``None``."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return _json_finite(obj.tolist())
    if isinstance(obj, dict):
        return {key: _json_finite(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_finite(value) for value in obj]
    return obj


def write_json(path, obj):
    """Write ``obj`` to ``path`` as indented JSON, creating its directory.

    Ensures the parent Results_Directory exists (``os.makedirs(...,
    exist_ok=True)``) before writing, then serializes ``obj`` with
    ``indent=2`` following the existing project output convention (cf.
    model ``evaluate`` workflows). Common numpy scalar/array types are
    handled transparently. Non-finite numbers are represented as JSON ``null``;
    ``allow_nan=False`` guarantees the output is valid RFC 8259 JSON rather
    than Python's non-standard ``NaN``/``Infinity`` extensions.

    Args:
        path: Destination file path for the JSON output.
        obj: A JSON-serializable object (typically the experiment metrics dict).
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.",
        suffix=".tmp",
        dir=parent,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                _json_finite(obj),
                f,
                indent=2,
                cls=_NumpyJSONEncoder,
                allow_nan=False,
            )
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
    except BaseException:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise
