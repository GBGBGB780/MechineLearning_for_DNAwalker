# coding=utf-8
"""Property-based tests for ``validation_common`` shared helpers.

Currently covers Property 19 (the seed guard). Further property tests for
``validation_common`` (LHS sampling, statistics helpers, the predictor factory
and JSON writer) are added by later tasks (2.4, 2.6, 2.7, 2.8, 2.10).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dnawalker.studies import protocol as validation_common


# Feature: validation-experiments, Property 19: The seed guard refuses stochastic steps without a seed
#
# Validates: Requirements 6.2, 6.1
#
# Two complementary obligations:
#   (a) For any non-int / None argument (None, floats, strings, lists, and the
#       bool literals True/False which are rejected even though bool subclasses
#       int), ``require_seed(arg, "step")`` raises ``MissingSeedError`` BEFORE any
#       stochastic work is performed. The guard is a pure validation function:
#       it does no stochastic work itself, so a raised error necessarily precedes
#       any downstream stochastic step that would consume the seed.
#   (b) For any non-negative int seed, ``require_seed`` returns that seed
#       unchanged. Negative seeds are rejected before NumPy sees them.


# Non-int values that must be rejected. Booleans are listed explicitly because
# ``bool`` is an ``int`` subclass yet is not a valid seed.
_non_int_values = st.one_of(
    st.none(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.lists(st.integers()),
    st.just(True),
    st.just(False),
)


@settings(max_examples=100)
@given(arg=_non_int_values)
def test_require_seed_rejects_non_int_arguments(arg):
    """Non-int / None args raise MissingSeedError before any stochastic work."""
    try:
        validation_common.require_seed(arg, "step")
    except validation_common.MissingSeedError:
        # Expected: the guard refused the stochastic step.
        pass
    else:
        raise AssertionError(
            "require_seed should have raised MissingSeedError for "
            "non-int argument {!r} (type {})".format(arg, type(arg).__name__)
        )


@settings(max_examples=100)
@given(seed=st.integers(min_value=0))
def test_require_seed_returns_int_seed_unchanged(seed):
    """Any non-negative int seed is returned unchanged."""
    result = validation_common.require_seed(seed, "step")
    assert result == seed
    assert type(result) is int
    # A genuine bool would have been rejected by the branch above; guard here too.
    assert not isinstance(result, bool)


@pytest.mark.parametrize("seed", [-1, -(2 ** 63)])
def test_require_seed_rejects_negative_integers(seed):
    with pytest.raises(validation_common.MissingSeedError):
        validation_common.require_seed(seed, "step")


@pytest.mark.parametrize("value", [1.5, "2", True, None])
def test_require_int_rejects_coercible_non_integers(value):
    with pytest.raises(ValueError, match="integer"):
        validation_common.require_int(value, "count", minimum=1)


def test_require_int_accepts_numpy_integer_and_enforces_bounds():
    assert validation_common.require_int(np.int64(3), "count", minimum=1) == 3
    with pytest.raises(ValueError, match="<= 3"):
        validation_common.require_int(4, "count", maximum=3)


@pytest.mark.parametrize("value", [True, "0.1", np.nan, np.inf, 0.0, -1.0])
def test_require_finite_real_rejects_invalid_positive_values(value):
    with pytest.raises(ValueError):
        validation_common.require_finite_real(
            value, "value", minimum=0.0, strict_minimum=True
        )


@pytest.mark.parametrize(
    ("base_seed", "batch_index"),
    [(-1, 0), (0, -1), (1.5, 0), (0, True), (2 ** 32, 0)],
)
def test_derive_batch_seed_rejects_invalid_inputs(base_seed, batch_index):
    with pytest.raises(ValueError):
        validation_common.derive_batch_seed(base_seed, batch_index)


# Feature: validation-experiments, Property 5: Latin Hypercube draws produce the requested count within Configured_Ranges
#
# Validates: Requirements 2.1
#
# Two obligations checked for any valid requested count ``n`` and any seed:
#   (a) ``lhs_params(n, ranges, seed)`` returns an ``(n, 7)`` array in which every
#       value in column ``i`` lies within ``ranges[PARAM_NAMES[i]] = [min, max]``.
#       ``n`` is generated in [1, 50] for test speed; the property holds for all
#       n in [1, 1_000_000].
#   (b) Reproducibility: the same ``(n, seed)`` yields a byte-identical array.
#
# Seeds are generated as non-negative integers: the underlying sampler
# (``scipy.stats.qmc.LatinHypercube``) requires a non-negative integer seed.

# Configured_Ranges are fixed for a given config; load once at import time.
_RANGES = validation_common.load_configured_ranges()
_PARAM_NAMES = validation_common.PARAM_NAMES


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_lhs_params_count_and_within_configured_ranges(n, seed):
    """LHS draws yield (n, 7) with every column within its Configured_Range."""
    draw = validation_common.lhs_params(n, _RANGES, seed)

    # Shape: exactly the requested count of 7-dimensional Parameter_Sets.
    assert draw.shape == (n, len(_PARAM_NAMES))

    # Every value in column i must lie within ranges[PARAM_NAMES[i]] = [min, max].
    for i, name in enumerate(_PARAM_NAMES):
        lo, hi = _RANGES[name]
        col = draw[:, i]
        assert (col >= lo).all(), (
            "column {} ({}) has value below min {}".format(i, name, lo)
        )
        assert (col <= hi).all(), (
            "column {} ({}) has value above max {}".format(i, name, hi)
        )


@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=50),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_lhs_params_is_reproducible_for_same_n_and_seed(n, seed):
    """Same (n, seed) produces an identical array (Requirement 2.1 / 6.1)."""
    first = validation_common.lhs_params(n, _RANGES, seed)
    second = validation_common.lhs_params(n, _RANGES, seed)
    assert np.array_equal(first, second)


# Feature: validation-experiments, Property 7: Relative error is computed correctly and zero-truth values are excluded
#
# Validates: Requirements 2.6, 2.12
#
# Two obligations:
#   (a) Correctness for non-zero truth. For any finite ``estimated`` and any
#       non-zero finite ``truth``, ``relative_error(estimated, truth)`` equals the
#       signed relative error ``(estimated - truth) / abs(truth)`` (checked with
#       ``math.isclose``), and equals exactly 0 when ``estimated == truth``.
#   (b) Zero-truth exclusion (the Requirement 2.12 filtering done by the caller).
#       For an array of (estimated, truth) pairs mixing exact-zero and non-zero
#       truths, every relative-error value at a ``truth == 0`` position is
#       non-finite (inf or nan), and applying ``np.isfinite`` filtering (the
#       caller's exclusion rule) retains EXACTLY the non-zero-truth entries — all
#       of which are finite.
#
# Magnitudes are bounded so that non-zero-truth entries cannot overflow to inf:
# with |estimated| <= 1e6 and |truth| in [1e-3, 1e6] the quotient is at most
# ~2e9, comfortably finite. This makes the finite/non-finite partition coincide
# exactly with the non-zero/zero-truth partition.

import math

_finite_estimated = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_nonzero_truth = st.one_of(
    st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e6, max_value=-1e-3, allow_nan=False, allow_infinity=False),
)
# A truth value is either exactly 0.0 (to be excluded) or a bounded non-zero float.
_truth_or_zero = st.one_of(st.just(0.0), _nonzero_truth)


@settings(max_examples=100, deadline=None)
@given(estimated=_finite_estimated, truth=_nonzero_truth)
def test_relative_error_matches_definition_for_nonzero_truth(estimated, truth):
    """relative_error == (estimated - truth)/abs(truth) for non-zero truth."""
    got = validation_common.relative_error(estimated, truth)
    expected = (estimated - truth) / abs(truth)
    assert math.isclose(got, expected, rel_tol=1e-9, abs_tol=1e-12)

    # Equal estimate and truth yield exactly zero relative error.
    assert validation_common.relative_error(truth, truth) == 0.0


@settings(max_examples=100, deadline=None)
@given(pairs=st.lists(st.tuples(_finite_estimated, _truth_or_zero), min_size=1, max_size=50))
def test_relative_error_zero_truth_entries_are_excluded_by_finite_filter(pairs):
    """Zero-truth positions are non-finite; finite-filtering keeps exactly the
    non-zero-truth entries (the Requirement 2.12 exclusion rule)."""
    estimated = np.array([p[0] for p in pairs], dtype=np.float64)
    truth = np.array([p[1] for p in pairs], dtype=np.float64)

    rel = validation_common.relative_error(estimated, truth)

    zero_mask = truth == 0.0
    finite_mask = np.isfinite(rel)

    # (1) Every relative error at a truth==0 position is non-finite (inf or nan).
    assert not np.isfinite(rel[zero_mask]).any()

    # (2) The caller's exclusion rule (np.isfinite filtering) retains EXACTLY the
    #     non-zero-truth entries.
    assert np.array_equal(finite_mask, ~zero_mask)

    # (3) Every retained relative error is finite.
    assert np.isfinite(rel[finite_mask]).all()


# Feature: validation-experiments, Property 8: R-squared satisfies its defining invariants
#
# Validates: Requirements 2.7
#
# ``r_squared(y_true, y_pred) = 1 - SS_res / SS_tot`` with
# ``SS_res = sum((y_true - y_pred)**2)`` and ``SS_tot = sum((y_true - mean)**2)``.
# Three defining invariants are checked over non-degenerate inputs (``SS_tot > 0``,
# i.e. non-constant ``y_true``), plus an explicit check of the documented
# degenerate convention for a constant target:
#
#   (a) Upper bound: R² <= 1.0 (within fp tolerance) for ANY predictions, because
#       ``SS_res >= 0`` and ``SS_tot > 0`` force ``1 - SS_res/SS_tot <= 1``.
#   (b) Perfect fit: when ``y_pred == y_true`` exactly, ``SS_res == 0`` so R² == 1.0.
#   (c) Mean baseline: when ``y_pred`` is the constant ``mean(y_true)``,
#       ``SS_res == SS_tot`` so R² == 0.0 (within fp tolerance).
#
# Non-constant ``y_true`` is *constructed* (not filtered) to guarantee ``SS_tot > 0``:
# the last element is set to ``y_true[0] + bump`` with ``bump >= 1.0``, so
# ``max - min >= bump >= 1`` and ``SS_tot > 0`` always. Magnitudes are bounded to
# +/-1e6 (and bump <= 1e6) so the sums of squares stay comfortably finite.

_r2_finite = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
_r2_bump = st.floats(
    min_value=1.0, max_value=1e6, allow_nan=False, allow_infinity=False
)


@st.composite
def _nonconstant_truth(draw):
    """Build a length>=2 finite ``y_true`` array with guaranteed variance.

    The last element is forced to differ from the first by ``bump >= 1.0`` so
    ``max - min >= 1`` and therefore ``SS_tot > 0`` (non-degenerate).
    """
    n = draw(st.integers(min_value=2, max_value=30))
    values = draw(st.lists(_r2_finite, min_size=n, max_size=n))
    bump = draw(_r2_bump)
    y_true = np.array(values, dtype=np.float64)
    y_true[-1] = y_true[0] + bump  # guarantee spread -> SS_tot > 0
    return y_true


@st.composite
def _nonconstant_truth_and_pred(draw):
    """A non-constant ``y_true`` paired with an arbitrary finite ``y_pred``."""
    y_true = draw(_nonconstant_truth())
    n = y_true.shape[0]
    pred_values = draw(st.lists(_r2_finite, min_size=n, max_size=n))
    y_pred = np.array(pred_values, dtype=np.float64)
    return y_true, y_pred


@settings(max_examples=100, deadline=None)
@given(data=_nonconstant_truth_and_pred())
def test_r_squared_is_at_most_one(data):
    """R² <= 1.0 (within fp tolerance) for any predictions on non-constant truth."""
    y_true, y_pred = data
    r2 = validation_common.r_squared(y_true, y_pred)
    assert r2 <= 1.0 + 1e-9, "R² exceeded 1.0: {!r}".format(r2)


@settings(max_examples=100, deadline=None)
@given(y_true=_nonconstant_truth())
def test_r_squared_is_one_for_perfect_fit(y_true):
    """R² == 1.0 when predictions equal the (non-constant) truth exactly."""
    r2 = validation_common.r_squared(y_true, y_true.copy())
    assert math.isclose(r2, 1.0, rel_tol=1e-9, abs_tol=1e-9), (
        "perfect fit should give R²==1.0, got {!r}".format(r2)
    )


@settings(max_examples=100, deadline=None)
@given(y_true=_nonconstant_truth())
def test_r_squared_is_zero_for_constant_mean_prediction(y_true):
    """R² ≈ 0.0 when the prediction is the constant mean of the truth."""
    y_pred = np.full_like(y_true, np.mean(y_true))
    r2 = validation_common.r_squared(y_true, y_pred)
    assert math.isclose(r2, 0.0, rel_tol=1e-9, abs_tol=1e-9), (
        "mean-baseline prediction should give R²≈0.0, got {!r}".format(r2)
    )


def test_r_squared_degenerate_constant_target_convention():
    """Documented degenerate convention: constant target (SS_tot == 0).

    Perfect prediction -> 1.0; any imperfect prediction -> 0.0.
    """
    constant = np.array([3.0, 3.0, 3.0, 3.0], dtype=np.float64)
    # Perfect fit against the constant target scores 1.0.
    assert validation_common.r_squared(constant, constant.copy()) == 1.0
    # Any imperfect fit against the constant target scores the baseline 0.0.
    imperfect = np.array([3.0, 3.0, 3.0, 3.5], dtype=np.float64)
    assert validation_common.r_squared(constant, imperfect) == 0.0


# Feature: validation-experiments, Property 9: Distribution statistics are correctly ordered
#
# Validates: Requirements 2.9
#
# For any non-empty finite 1-D sample, ``distribution_stats(values)`` returns the
# seven descriptive statistics with these invariants:
#   (a) Ordering: min <= p5 <= median <= p95 <= max (within fp tolerance). This
#       follows because the 5th/50th/95th percentiles are order statistics of the
#       sample bracketed by its min and max.
#   (b) Mean lies within [min, max]: any arithmetic mean of finite values is
#       bounded by the sample's extremes (within fp tolerance).
#   (c) Non-negative std: numpy's standard deviation is always >= 0.
#   (d) The returned dict has EXACTLY the keys {mean, median, std, min, max, p5, p95}.
#
# Magnitudes are bounded to +/-1e9 so the mean and std computations stay
# comfortably finite and free of overflow. A small absolute+relative tolerance
# absorbs floating-point rounding in the percentile interpolation and the mean.

_dist_finite = st.floats(
    min_value=-1e9, max_value=1e9, allow_nan=False, allow_infinity=False
)


@settings(max_examples=100, deadline=None)
@given(values=st.lists(_dist_finite, min_size=1, max_size=200))
def test_distribution_stats_are_correctly_ordered(values):
    """min <= p5 <= median <= p95 <= max; mean in [min, max]; std >= 0; exact keys."""
    stats = validation_common.distribution_stats(values)

    # (d) Exactly the expected keys, nothing more, nothing less.
    assert set(stats.keys()) == {"mean", "median", "std", "min", "max", "p5", "p95"}

    # Scale-aware tolerance: percentile interpolation and mean accumulation can
    # round by a few ULPs at magnitudes up to 1e9.
    scale = max(abs(stats["min"]), abs(stats["max"]), 1.0)
    tol = 1e-6 * scale

    # (a) Ordering of the order statistics.
    assert stats["min"] <= stats["p5"] + tol
    assert stats["p5"] <= stats["median"] + tol
    assert stats["median"] <= stats["p95"] + tol
    assert stats["p95"] <= stats["max"] + tol

    # (b) The mean is bracketed by the sample extremes.
    assert stats["min"] - tol <= stats["mean"] <= stats["max"] + tol

    # (c) Standard deviation is non-negative.
    assert stats["std"] >= 0.0


@pytest.mark.parametrize(
    "values",
    [[], [np.nan], [np.inf], [1.0, -np.inf], ["not-a-number"]],
)
def test_distribution_stats_rejects_empty_or_nonfinite_samples(values):
    with pytest.raises(ValueError, match="sample|finite|numeric"):
        validation_common.distribution_stats(values)


def test_distribution_stats_rejects_overflowed_statistics():
    with pytest.raises(ValueError, match="overflowed"):
        validation_common.distribution_stats([1e308, -1e308])


# ---------------------------------------------------------------------------
# Task 2.10 — Example/unit tests for the predictor factory and JSON writer.
#
# Validates: Requirements 5.3, 6.3
#
# These are plain pytest example tests (NOT Hypothesis property tests). They
# avoid loading torch / the heavy DL models by faking the lazily-imported
# predictor modules. ``make_predictor`` imports its predictor classes lazily
# *inside* the function from the canonical package paths. Injecting fake modules
# into ``sys.modules`` before the call makes the factory construct the fakes
# instead of the real (torch-dependent) classes.
# ---------------------------------------------------------------------------

import json
import sys
import types

class _FakeCNNPredictor:
    """Stand-in for ``NanorobotPredictor`` exposing the predictor interface."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.kind = "cnn"

    def predict(self, X):
        # Shape contract is (N, 7); content is irrelevant for the interface test.
        return np.zeros((1, len(validation_common.PARAM_NAMES)), dtype=np.float64)

    def get_param_names(self):
        return list(validation_common.PARAM_NAMES)


class _FakeTransformerPredictor:
    """Stand-in for ``TransformerPredictor`` exposing the predictor interface."""

    def __init__(self, **overrides):
        self.overrides = overrides
        self.kind = "transformer"

    def predict(self, X):
        return np.zeros((1, len(validation_common.PARAM_NAMES)), dtype=np.float64)

    def get_param_names(self):
        return list(validation_common.PARAM_NAMES)


def _install_fake_predictor_modules(monkeypatch):
    """Inject fake legacy and canonical predictor modules without loading torch."""
    cnn_mod = types.ModuleType("dnawalker.cnn.inference")
    cnn_mod.NanorobotPredictor = _FakeCNNPredictor
    tf_mod = types.ModuleType("dnawalker.transformer.inference")
    tf_mod.TransformerPredictor = _FakeTransformerPredictor

    # Canonical package paths used by validation_common.make_predictor.
    monkeypatch.setitem(
        sys.modules, "dnawalker.cnn.inference", cnn_mod
    )
    monkeypatch.setitem(
        sys.modules, "dnawalker.transformer.inference", tf_mod
    )
    # Legacy paths remain injectable for downstream compatibility checks.
    monkeypatch.setitem(sys.modules, "train_cnn.inference_cnn", cnn_mod)
    monkeypatch.setitem(
        sys.modules, "train_transformer.inference_transformer", tf_mod
    )


def test_make_predictor_cnn_constructs_faked_class_with_interface(monkeypatch):
    """make_predictor('cnn') builds NanorobotPredictor and exposes the interface."""
    _install_fake_predictor_modules(monkeypatch)

    predictor = validation_common.make_predictor("cnn")

    # The factory constructed the (faked) CNN predictor class.
    assert isinstance(predictor, _FakeCNNPredictor)
    assert predictor.kind == "cnn"

    # The returned object exposes the common predictor interface.
    assert callable(getattr(predictor, "predict", None))
    assert callable(getattr(predictor, "get_param_names", None))
    assert predictor.get_param_names() == list(validation_common.PARAM_NAMES)
    assert predictor.predict(np.zeros((3, 10))).shape == (
        1,
        len(validation_common.PARAM_NAMES),
    )


def test_make_predictor_transformer_constructs_faked_class_with_interface(monkeypatch):
    """make_predictor('transformer') builds TransformerPredictor with interface."""
    _install_fake_predictor_modules(monkeypatch)

    predictor = validation_common.make_predictor("transformer")

    assert isinstance(predictor, _FakeTransformerPredictor)
    assert predictor.kind == "transformer"

    assert callable(getattr(predictor, "predict", None))
    assert callable(getattr(predictor, "get_param_names", None))
    assert predictor.get_param_names() == list(validation_common.PARAM_NAMES)
    assert predictor.predict(np.zeros((3, 10))).shape == (
        1,
        len(validation_common.PARAM_NAMES),
    )


def test_make_predictor_is_case_insensitive(monkeypatch):
    """Model selection is case-insensitive (e.g. 'CNN', 'Transformer')."""
    _install_fake_predictor_modules(monkeypatch)

    assert isinstance(validation_common.make_predictor("CNN"), _FakeCNNPredictor)
    assert isinstance(
        validation_common.make_predictor("Transformer"), _FakeTransformerPredictor
    )


def test_make_predictor_forwards_override_kwargs(monkeypatch):
    """Constructor keyword overrides are forwarded to the selected predictor."""
    _install_fake_predictor_modules(monkeypatch)

    predictor = validation_common.make_predictor("cnn", model_path="/tmp/fake.pth")
    assert predictor.overrides == {"model_path": "/tmp/fake.pth"}


def test_make_predictor_rejects_unknown_model():
    """make_predictor('bogus') raises ValueError (no heavy imports needed)."""
    with pytest.raises(ValueError):
        validation_common.make_predictor("bogus")


def test_write_json_creates_directory_and_round_trips_dict(tmp_path):
    """write_json creates the Results_Directory and round-trips a plain dict."""
    out_path = tmp_path / "validation" / "nested" / "metrics.json"
    # Neither the file nor its parent directory exist yet.
    assert not out_path.parent.exists()

    payload = {
        "experiment": "synthetic_parameter_recovery",
        "model": "cnn",
        "seed": 42,
        "retained_sample_count": 947,
        "parameters": {
            "E_b": {"r_squared": 0.981, "zero_truth_excluded_count": 0},
        },
        "flags": [True, False],
    }

    validation_common.write_json(str(out_path), payload)

    # The directory was created and the file written.
    assert out_path.parent.is_dir()
    assert out_path.is_file()

    # Round-trip equality: reading the JSON back yields the original dict.
    with open(str(out_path)) as f:
        loaded = json.load(f)
    assert loaded == payload


def test_write_json_handles_numpy_types(tmp_path):
    """write_json serializes numpy scalars/arrays as JSON-native values."""
    out_path = tmp_path / "numpy_metrics.json"

    payload = {
        "np_float": np.float64(0.0123),
        "np_int": np.int64(947),
        "np_bool": np.bool_(True),
        "np_array": np.array([1.0, 2.5, 3.0], dtype=np.float64),
        "nested": {"col": np.array([4, 5, 6], dtype=np.int64)},
    }

    # Must not raise even though the dict contains numpy types.
    validation_common.write_json(str(out_path), payload)

    with open(str(out_path)) as f:
        loaded = json.load(f)

    # Values come back as JSON-native Python types (float/int/bool/list).
    assert isinstance(loaded["np_float"], float)
    assert math.isclose(loaded["np_float"], 0.0123, rel_tol=1e-12)
    assert isinstance(loaded["np_int"], int)
    assert loaded["np_int"] == 947
    assert isinstance(loaded["np_bool"], bool)
    assert loaded["np_bool"] is True
    assert loaded["np_array"] == [1.0, 2.5, 3.0]
    assert loaded["nested"]["col"] == [4, 5, 6]


def test_write_json_produces_valid_indented_json(tmp_path):
    """The written file is valid JSON and is indented (multi-line) formatting."""
    out_path = tmp_path / "formatted.json"
    payload = {"a": 1, "b": {"c": 2, "d": [1, 2, 3]}}

    validation_common.write_json(str(out_path), payload)

    raw = out_path.read_text()
    # Valid JSON: parses back to the original object.
    assert json.loads(raw) == payload
    # Indented formatting (indent=2) yields multiple lines with leading spaces.
    lines = raw.splitlines()
    assert len(lines) > 1
    assert any(line.startswith("  ") for line in lines)


def test_write_json_replaces_non_finite_values_with_null(tmp_path):
    """Metrics files must be strict JSON, not Python's NaN/Infinity dialect."""
    out_path = tmp_path / "strict.json"
    payload = {
        "inf": float("inf"),
        "negative_inf": np.float64("-inf"),
        "nan": float("nan"),
        "nested": [1.0, np.array([2.0, np.nan])],
    }

    validation_common.write_json(str(out_path), payload)

    raw = out_path.read_text(encoding="utf-8")
    assert "Infinity" not in raw
    assert "NaN" not in raw
    assert json.loads(raw) == {
        "inf": None,
        "negative_inf": None,
        "nan": None,
        "nested": [1.0, [2.0, None]],
    }


def test_write_json_failure_leaves_no_partial_file_or_temp(tmp_path):
    out_path = tmp_path / "invalid.json"
    with pytest.raises(TypeError):
        validation_common.write_json(out_path, {"unsupported": object()})
    assert not out_path.exists()
    assert list(tmp_path.iterdir()) == []
