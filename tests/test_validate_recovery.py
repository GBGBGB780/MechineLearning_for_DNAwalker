# coding=utf-8
"""Property-based tests for ``validate_recovery`` (Recovery_Experiment).

Currently covers Property 10 (retained samples partition cleanly and exclude
invalid simulations). Further recovery property/example/smoke tests (4.3, 4.5,
4.7, 4.11, 4.12) are added by later tasks.

These tests exercise the REAL ``pysim`` physics simulator through
``validate_recovery.generate_ground_truth`` and are therefore SLOW. Sample
counts (``n``) and the draw budget (``max_total_draws``) are kept small so the
suite finishes in a reasonable time while still running >=100 Hypothesis
examples.
"""

import json
import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pysim
import validation_common
import validate_recovery


# Configured_Ranges are fixed for a given config; load once at import time.
_RANGES = validation_common.load_configured_ranges()
_NUM_PARAMS = len(validation_common.PARAM_NAMES)


# Feature: validation-experiments, Property 10: Retained samples partition cleanly and exclude invalid simulations
#
# Validates: Requirements 2.3
#
# ``generate_ground_truth(n, ranges, seed, max_total_draws)`` returns
# ``(Y_true, X_curves, n_excluded_invalid, exhausted)`` (it does NOT return the
# total number of draws consumed). The property checks, for any small ``n``, any
# modest draw budget, and any non-negative seed:
#
#   (a) Retained samples truly exclude invalid simulations. Re-simulating every
#       retained Parameter_Set (each row of ``Y_true``) through the REAL
#       ``pysim.run_simulation`` yields ``dt_used >= 0``. Because ``pysim`` is
#       deterministic, a retained sample that re-simulates valid confirms the
#       function kept only valid draws (Requirement 2.3).
#
#   (b) The partition is clean. Internally every processed draw is either
#       retained (valid) or excluded (invalid), so
#       ``retained + n_excluded_invalid == total_drawn`` with
#       ``total_drawn <= max_total_draws``. Since ``total_drawn`` is not returned,
#       the equivalent observable invariant is asserted:
#         * ``len(Y_true) + n_excluded_invalid <= max_total_draws``
#           (the retained/excluded counts partition only consumed draws), and
#         * ``len(Y_true) <= n`` and ``len(Y_true) == X_curves.shape[0]``
#           (Y_true and X_curves stay aligned and never exceed the request), and
#         * ``not exhausted`` implies ``len(Y_true) == n`` (the budget was
#           sufficient, so exactly ``n`` valid samples were retained).
#
# ``n`` is kept in [1, 3] and ``max_total_draws`` in [3, 12] because each draw
# invokes the expensive REAL simulator; the property holds for all valid values.
# Seeds are generated as non-negative ints (the LHS sampler requires a
# non-negative integer seed).
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=3),
    max_total_draws=st.integers(min_value=3, max_value=12),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_retained_samples_partition_cleanly_and_exclude_invalid(n, max_total_draws, seed):
    """Retained samples are all valid and counts partition the consumed draws."""
    Y_true, X_curves, n_excluded_invalid, exhausted = (
        validate_recovery.generate_ground_truth(n, _RANGES, seed, max_total_draws)
    )

    retained_count = len(Y_true)

    # --- Structural invariants -------------------------------------------------
    # Y_true has the expected (m, 7) shape and X_curves is row-aligned to it.
    assert Y_true.ndim == 2 and Y_true.shape[1] == _NUM_PARAMS
    assert retained_count <= n, (
        "retained {} exceeds requested {}".format(retained_count, n)
    )
    assert retained_count == X_curves.shape[0], (
        "Y_true rows ({}) and X_curves rows ({}) disagree".format(
            retained_count, X_curves.shape[0]
        )
    )

    # --- (b) Clean partition over the consumed draws ---------------------------
    # retained + excluded == total_drawn <= max_total_draws. total_drawn is not
    # returned, so assert the equivalent observable bound.
    assert n_excluded_invalid >= 0
    assert retained_count + n_excluded_invalid <= max_total_draws, (
        "retained ({}) + excluded ({}) exceeds draw budget ({})".format(
            retained_count, n_excluded_invalid, max_total_draws
        )
    )

    # A sufficient budget (not exhausted) must yield exactly n retained samples.
    if not exhausted:
        assert retained_count == n, (
            "not exhausted but retained {} != requested {}".format(
                retained_count, n
            )
        )

    # --- (a) Retained samples truly exclude invalid simulations ----------------
    # Re-simulate every retained Parameter_Set; each must be valid (dt_used >= 0).
    for i, params in enumerate(Y_true):
        _signals, dt_used = pysim.run_simulation(params)
        assert dt_used >= 0, (
            "retained sample {} re-simulates invalid (dt_used={})".format(i, dt_used)
        )


# Feature: validation-experiments, Property 11: Recovery sampling is reproducible under a fixed seed
#
# Validates: Requirements 2.10, 6.1
#
# The Recovery_Experiment SHALL use a fixed, recorded random seed for sampling
# so that a repeated run produces identical retained samples (Requirement 2.10),
# and every stochastic step in the Validation_Suite SHALL use fixed, recorded
# seeds (Requirement 6.1).
#
# This property checks determinism end-to-end through the REAL simulator: for
# any small ``n``, any modest draw budget, and any non-negative seed, two
# independent calls to ``generate_ground_truth(n, ranges, seed, max_total_draws)``
# with the SAME ``seed`` must produce byte-for-byte identical results:
#
#   * identical retained Parameter_Sets (``Y_true`` array-equal),
#   * identical simulated curves (``X_curves`` array-equal),
#   * identical excluded-invalid count (``n_excluded_invalid``), and
#   * identical exhaustion flag (``exhausted``).
#
# Because the per-batch seeds are derived deterministically from the base seed
# (and ``pysim`` is deterministic), the redraw-on-invalid loop must replay
# exactly the same draws on a repeated call. ``n`` is kept in [1, 3] and
# ``max_total_draws`` in [3, 12] because each draw invokes the expensive REAL
# simulator; the property holds for all valid values. Seeds are generated as
# non-negative ints (the LHS sampler requires a non-negative integer seed).
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=3),
    max_total_draws=st.integers(min_value=3, max_value=12),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_recovery_sampling_is_reproducible_under_a_fixed_seed(n, max_total_draws, seed):
    """Two same-seed runs of generate_ground_truth produce identical results."""
    Y_true_a, X_curves_a, n_excluded_a, exhausted_a = (
        validate_recovery.generate_ground_truth(n, _RANGES, seed, max_total_draws)
    )
    Y_true_b, X_curves_b, n_excluded_b, exhausted_b = (
        validate_recovery.generate_ground_truth(n, _RANGES, seed, max_total_draws)
    )

    # Retained Parameter_Sets are byte-for-byte identical (same shape + values).
    assert Y_true_a.shape == Y_true_b.shape, (
        "Y_true shapes differ across same-seed runs: {} vs {}".format(
            Y_true_a.shape, Y_true_b.shape
        )
    )
    assert np.array_equal(Y_true_a, Y_true_b), (
        "Y_true differs across same-seed runs (seed={})".format(seed)
    )

    # Corresponding simulated curves are identical too.
    assert X_curves_a.shape == X_curves_b.shape, (
        "X_curves shapes differ across same-seed runs: {} vs {}".format(
            X_curves_a.shape, X_curves_b.shape
        )
    )
    assert np.array_equal(X_curves_a, X_curves_b), (
        "X_curves differs across same-seed runs (seed={})".format(seed)
    )

    # Excluded-invalid count and exhaustion flag are reproduced exactly.
    assert n_excluded_a == n_excluded_b, (
        "n_excluded_invalid differs across same-seed runs: {} vs {}".format(
            n_excluded_a, n_excluded_b
        )
    )
    assert exhausted_a == exhausted_b, (
        "exhausted differs across same-seed runs: {} vs {}".format(
            exhausted_a, exhausted_b
        )
    )


# Feature: validation-experiments, Property 6: Gaussian noise is zero-mean, shape-preserving, and seed-reproducible
#
# Validates: Requirements 2.4
#
# ``add_gaussian_noise(curves, noise_std, seed)`` SHALL add zero-mean Gaussian
# noise with standard deviation ``noise_std`` independently per time point to
# each of the 3 channels, preserve the ``(3, 7801)`` curve shape, leave the
# caller's array unmutated, and produce identical noise for a repeated seed
# (Requirement 2.4).
#
# The base curve set is all zeros of the production shape ``(3, pysim.NUM_RESULTS)``
# == ``(3, 7801)``. Because the base is zero, the noised output *is* the added
# noise, so the per-channel sample mean and sample std are measured directly:
#
#   * Shape preserved: ``output.shape == curves.shape == (3, 7801)``.
#   * Zero-mean: each channel's sample mean is within ``6 * noise_std / sqrt(N)``
#     of 0 (6 standard errors of the mean; with N=7801 the chance of a genuine
#     zero-mean draw exceeding this is ~1e-9 per channel, so the bound is
#     comfortably non-flaky across 100 examples x 3 channels).
#   * Correct std: each channel's sample std is within 10% of ``noise_std``
#     (the relative standard error of the sample std is ~1/sqrt(2N) ~= 0.8%,
#     so 10% is a wide, non-flaky tolerance).
#   * Seed-reproducible: a second call with the same ``(curves, noise_std, seed)``
#     returns a byte-identical array (``np.array_equal``).
#   * No mutation: the caller's input array is unchanged after the call.
#
# ``noise_std`` is drawn from a sensible positive range [1e-3, 0.5]; seeds are
# generated as non-negative ints.
@settings(max_examples=100, deadline=None)
@given(
    noise_std=st.floats(
        min_value=1e-3,
        max_value=0.5,
        allow_nan=False,
        allow_infinity=False,
    ),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_gaussian_noise_is_zero_mean_shape_preserving_and_seed_reproducible(noise_std, seed):
    """Gaussian noise preserves shape, is ~zero-mean with correct std, and is reproducible."""
    # Production curve shape: 3 channels x pysim.NUM_RESULTS (7801) time points.
    base = np.zeros((3, pysim.NUM_RESULTS), dtype=np.float64)
    assert base.shape == (3, 7801)
    base_before = base.copy()

    noised = validate_recovery.add_gaussian_noise(base, noise_std, seed)

    # --- Shape preserved -------------------------------------------------------
    assert noised.shape == base.shape == (3, 7801), (
        "output shape {} != input shape {}".format(noised.shape, base.shape)
    )

    # --- No mutation of the caller's array -------------------------------------
    assert np.array_equal(base, base_before), "input array was mutated in place"

    # --- Zero-mean and correct std, measured per channel -----------------------
    # Base is zero, so ``noised`` equals the added noise itself.
    n_samples = noised.shape[1]
    mean_tol = 6.0 * noise_std / np.sqrt(n_samples)  # 6 standard errors of the mean
    std_tol = 0.10 * noise_std                        # 10% relative tolerance on std
    for ch in range(noised.shape[0]):
        channel = noised[ch]
        sample_mean = float(np.mean(channel))
        sample_std = float(np.std(channel))
        assert abs(sample_mean) <= mean_tol, (
            "channel {} mean {:.3e} exceeds tolerance {:.3e} "
            "(noise_std={:.3e})".format(ch, sample_mean, mean_tol, noise_std)
        )
        assert abs(sample_std - noise_std) <= std_tol, (
            "channel {} std {:.3e} differs from noise_std {:.3e} "
            "by more than 10%".format(ch, sample_std, noise_std)
        )

    # --- Seed-reproducible: identical inputs -> byte-identical output ----------
    noised_again = validate_recovery.add_gaussian_noise(base, noise_std, seed)
    assert np.array_equal(noised, noised_again), (
        "repeated call with the same (curves, noise_std, seed) produced "
        "different noise (seed={}, noise_std={!r})".format(seed, noise_std)
    )


# ---------------------------------------------------------------------------
# Task 4.7 — Example test: the predict+refine pipeline (estimate_parameters)
#
# Validates: Requirements 2.2, 2.5
#
# Requirement 2.2: ground-truth curves are generated by invoking the
# DNA_Walker_Simulator (here via ``generate_ground_truth``, which forward-
# simulates LHS draws through the REAL ``pysim``).
# Requirement 2.5: when curves for a retained sample are available, the
# Recovery_Experiment estimates the Parameter_Set by running the
# Predict_Refine_Pipeline (``estimate_parameters``).
#
# This is a plain example test (NOT Hypothesis): on ONE synthetic curve set it
# asserts that ``estimate_parameters`` returns a dict with exactly the seven
# ``pysim.PARAM_NAMES`` keys, each mapping to a finite ``float``.
#
# To keep the test fast and torch-free, a LIGHTWEIGHT FAKE predictor is used in
# place of the heavy CNN/Transformer. It exposes the same minimal Inverse_Model
# interface the pipeline relies on — ``predict(X) -> (1, 7)`` and
# ``get_param_names()`` — and predicts the midpoint of the Configured_Ranges
# (an in-range, physically sensible 7-vector). A SMALL ``maxiter`` and
# ``multistart=0`` keep the single refinement cheap; with ``multistart=0`` the
# pipeline performs no stochastic jitter, so no seed is required.
# ---------------------------------------------------------------------------
class _MidpointFakePredictor:
    """A torch-free stand-in Inverse_Model predicting the midpoint of ranges.

    Mirrors the minimal interface ``estimate_parameters`` consumes:

      * ``predict(X) -> np.ndarray`` of shape ``(1, 7)`` in physical-parameter
        space, ordered to ``pysim.PARAM_NAMES``; here the prediction is the
        midpoint ``(min + max) / 2`` of each parameter's Configured_Range, which
        is guaranteed in-range and independent of the input curves.
      * ``get_param_names() -> list[str]`` naming the seven trainable params.

    Using a fake avoids importing ``torch`` or loading any trained checkpoint.
    """

    def __init__(self, ranges):
        # Midpoint per parameter, in canonical pysim.PARAM_NAMES order.
        self._midpoint = np.array(
            [
                (ranges[name][0] + ranges[name][1]) / 2.0
                for name in pysim.PARAM_NAMES
            ],
            dtype=np.float64,
        )

    def predict(self, X):  # noqa: N803 (X matches the predictor convention)
        # Return the in-range midpoint as a (1, 7) batch regardless of X.
        return self._midpoint.reshape(1, -1)

    def get_param_names(self):
        return list(pysim.PARAM_NAMES)


def test_estimate_parameters_returns_seven_finite_params():
    """estimate_parameters yields exactly the 7 PARAM_NAMES keys, all finite floats."""
    ranges = validation_common.load_configured_ranges()

    # --- Requirement 2.2: one synthetic curve set from the REAL simulator ------
    # Draw a single valid ground-truth sample; a generous draw budget makes the
    # retention reliable for this deterministic example.
    Y_true, X_curves, _n_excluded, exhausted = validate_recovery.generate_ground_truth(
        1, ranges, seed=12345, max_total_draws=50
    )
    assert not exhausted and X_curves.shape[0] >= 1, (
        "could not generate a valid synthetic curve set for the example test"
    )
    curves = X_curves[0]
    assert curves.shape == (3, pysim.NUM_RESULTS)

    # --- Requirement 2.5: run the Predict_Refine_Pipeline ----------------------
    # Lightweight fake predictor + small maxiter + multistart=0 keep this fast;
    # with multistart=0 no stochastic jitter occurs, so no seed is needed.
    predictor = _MidpointFakePredictor(ranges)
    estimated = validate_recovery.estimate_parameters(
        predictor,
        curves,
        ranges,
        method="Powell",
        maxiter=5,
        multistart=0,
        seed=None,
    )

    # --- Assertions: exactly the 7 PARAM_NAMES keys, all finite floats ---------
    assert isinstance(estimated, dict)
    assert set(estimated.keys()) == set(pysim.PARAM_NAMES), (
        "estimated keys {} != PARAM_NAMES {}".format(
            sorted(estimated.keys()), sorted(pysim.PARAM_NAMES)
        )
    )
    assert len(estimated) == len(pysim.PARAM_NAMES) == 7

    for name in pysim.PARAM_NAMES:
        value = estimated[name]
        assert isinstance(value, float), (
            "param {!r} value {!r} is not a float".format(name, value)
        )
        assert np.isfinite(value), (
            "param {!r} value {!r} is not finite".format(name, value)
        )


# ===========================================================================
# Task 4.11 — Edge-case and JSON-schema example tests for the Recovery_Experiment
#
# Validates: Requirements 2.1, 2.9, 2.11, 2.13
#
# These are plain example/edge-case tests (NOT Hypothesis). They exercise the
# ``run_recovery`` orchestration (task 4.10) and the ``generate_ground_truth``
# redraw loop (task 4.1) for the four behaviours below:
#
#   * Req 2.1  — sample-count bound rejection: n_samples outside [1, 1_000_000]
#                raises ``ValueError`` (before any heavy work).
#   * Req 2.11 — redraw-until-retained: with an adequate draw budget,
#                ``generate_ground_truth`` keeps drawing until valid samples are
#                retained, and the retained/excluded counts are consistent.
#   * Req 2.13 — draw-budget exhaustion: when every drawn sample is invalid the
#                experiment stops before metrics and writes
#                ``all_samples_excluded: true`` with zero retained samples and an
#                empty ``parameters`` block.
#   * Req 2.9  — JSON schema: a tiny successful run writes ``recovery_metrics.json``
#                carrying the required top-level and per-parameter keys.
#
# All tests use a lightweight torch-free fake predictor (``_MidpointFakePredictor``
# defined above) injected via ``run_recovery(predictor=...)`` and a ``tmp_path``
# Results_Directory, so no ``torch`` is loaded and no trained checkpoint is read.
# ===========================================================================


# --- Req 2.1: sample-count bound rejection ---------------------------------
#
# Requirement 2.1: the Recovery_Experiment draws a configurable number of
# ground-truth Parameter_Sets where the count is an integer between 1 and
# 1,000,000 inclusive. ``run_recovery`` validates ``n_samples`` first (step 1),
# raising ``ValueError`` before any sampling, prediction, or refinement — so the
# injected fake predictor is never invoked and ``tmp_path`` stays empty.
@pytest.mark.parametrize("bad_n", [0, 1_000_001])
def test_run_recovery_rejects_out_of_range_sample_count(bad_n, tmp_path):
    """run_recovery raises ValueError for n_samples outside [1, 1_000_000]."""
    ranges = validation_common.load_configured_ranges()
    predictor = _MidpointFakePredictor(ranges)

    with pytest.raises(ValueError):
        validate_recovery.run_recovery(
            n_samples=bad_n,
            seed=42,
            predictor=predictor,
            results_dir=str(tmp_path),
        )

    # The error is raised before any heavy work, so nothing is written.
    assert not (tmp_path / "recovery_metrics.json").exists()


# --- Req 2.11: redraw-until-retained ---------------------------------------
#
# Requirement 2.11: the experiment keeps drawing Parameter_Sets, up to the draw
# budget, until at least one sample is retained. Exercised directly via
# ``generate_ground_truth`` against the REAL simulator with a small ``n`` and an
# adequate ``max_total_draws``: it must retain valid samples and the
# retained/excluded counts must partition the consumed draws sensibly.
def test_generate_ground_truth_redraws_until_samples_retained():
    """generate_ground_truth retains valid samples with consistent exclusion counts."""
    ranges = validation_common.load_configured_ranges()

    n = 2
    max_total_draws = 40  # generous budget so retention is reliable
    Y_true, X_curves, n_excluded_invalid, exhausted = (
        validate_recovery.generate_ground_truth(n, ranges, seed=20240517,
                                                 max_total_draws=max_total_draws)
    )

    retained = int(Y_true.shape[0])

    # At least one sample retained, never more than requested.
    assert retained >= 1
    assert retained <= n

    # Y_true / X_curves stay row-aligned and correctly shaped.
    assert X_curves.shape[0] == retained
    assert Y_true.shape[1] == _NUM_PARAMS

    # Exclusion count is a sensible non-negative integer and, together with the
    # retained count, never exceeds the consumed draw budget.
    assert n_excluded_invalid >= 0
    assert retained + n_excluded_invalid <= max_total_draws

    # With a generous budget the request is satisfied, so it is not exhausted and
    # exactly n samples are retained.
    assert exhausted is False
    assert retained == n


# --- Req 2.13: draw-budget exhaustion / all_samples_excluded ----------------
#
# Requirement 2.13: if the draw budget is reached with no sample retained, the
# experiment stops BEFORE computing metrics and reports the all-samples-excluded
# condition. Exhaustion is forced deterministically by monkeypatching
# ``pysim.run_simulation`` to always return an invalid result (``dt_used < 0``),
# so every drawn sample is excluded and zero are retained — no real (heavy)
# simulation runs. A fake predictor is injected but, because the run stops before
# the estimation step, it is never invoked.
def test_run_recovery_reports_all_samples_excluded_on_exhaustion(tmp_path, monkeypatch):
    """run_recovery writes all_samples_excluded=true when every draw is invalid."""
    ranges = validation_common.load_configured_ranges()
    predictor = _MidpointFakePredictor(ranges)

    # Force every forward simulation to be invalid (dt_used < 0) so the redraw
    # loop exhausts its budget with zero retained samples (Requirement 2.13).
    invalid_signals = np.zeros((3, pysim.NUM_RESULTS), dtype=np.float64)

    def _always_invalid(params, fixed_params=None):
        return invalid_signals, -1.0

    monkeypatch.setattr(pysim, "run_simulation", _always_invalid)

    metrics = validate_recovery.run_recovery(
        n_samples=2,
        seed=42,
        multistart=0,
        max_total_draws=6,
        predictor=predictor,
        results_dir=str(tmp_path),
    )

    # Returned metrics report the terminal all-samples-excluded condition.
    assert metrics["all_samples_excluded"] is True
    assert metrics["retained_sample_count"] == 0
    assert metrics["parameters"] == {}
    # Every consumed draw was excluded as invalid.
    assert metrics["excluded_sample_count"] >= 1

    # The JSON file on disk carries the same terminal report.
    json_path = tmp_path / "recovery_metrics.json"
    assert json_path.exists()
    with open(str(json_path), "r") as fh:
        on_disk = json.load(fh)
    assert on_disk["all_samples_excluded"] is True
    assert on_disk["retained_sample_count"] == 0
    assert on_disk["parameters"] == {}


# --- Req 2.9: recovery_metrics.json schema ---------------------------------
#
# Requirement 2.9: the experiment writes the recovery metrics as a JSON file
# carrying, for each of the 7 parameters, the relative-error mean/median/std/
# min/max/p5/p95, the per-parameter R², the zero-truth-excluded count, plus the
# retained/excluded sample counts. A tiny successful run (n_samples=2, injected
# fake predictor, small draw budget, multistart=0, small maxiter) writes the file
# and it is read back with ``json.load`` to assert the schema.
_REQUIRED_TOP_LEVEL_KEYS = {
    "experiment",
    "model",
    "seed",
    "config",
    "retained_sample_count",
    "excluded_sample_count",
    "all_samples_excluded",
    "parameters",
}
_REQUIRED_REL_ERROR_STATS = {"mean", "median", "std", "min", "max", "p5", "p95"}


def test_recovery_metrics_json_has_required_schema(tmp_path):
    """A tiny successful run writes recovery_metrics.json with the required keys."""
    ranges = validation_common.load_configured_ranges()
    predictor = _MidpointFakePredictor(ranges)

    metrics = validate_recovery.run_recovery(
        n_samples=2,
        model="cnn",
        seed=4242,
        method="Powell",
        maxiter=3,          # small refinement so the real pipeline stays fast
        multistart=0,        # no jitter -> no per-sample estimation seed needed
        max_total_draws=40,  # generous budget so 2 valid samples are retained
        predictor=predictor,
        results_dir=str(tmp_path),
    )

    # A successful run retained samples and is not the terminal exclusion case.
    assert metrics["all_samples_excluded"] is False
    assert metrics["retained_sample_count"] >= 1

    # Read the file back from disk with json.load and validate the schema there.
    json_path = tmp_path / "recovery_metrics.json"
    assert json_path.exists()
    with open(str(json_path), "r") as fh:
        data = json.load(fh)

    # --- Required top-level keys (Requirement 2.9) -----------------------------
    assert _REQUIRED_TOP_LEVEL_KEYS.issubset(set(data.keys())), (
        "missing top-level keys: {}".format(
            _REQUIRED_TOP_LEVEL_KEYS - set(data.keys())
        )
    )
    assert isinstance(data["retained_sample_count"], int)
    assert isinstance(data["excluded_sample_count"], int)
    assert data["all_samples_excluded"] is False

    # --- Per-parameter keys for each of the 7 PARAM_NAMES (Requirement 2.9) ----
    params_block = data["parameters"]
    assert set(params_block.keys()) == set(pysim.PARAM_NAMES), (
        "parameters block keys {} != PARAM_NAMES {}".format(
            sorted(params_block.keys()), sorted(pysim.PARAM_NAMES)
        )
    )

    for name in pysim.PARAM_NAMES:
        entry = params_block[name]
        # Each parameter carries rel_error, r_squared, zero_truth_excluded_count.
        assert {"rel_error", "r_squared", "zero_truth_excluded_count"}.issubset(
            set(entry.keys())
        ), "parameter {!r} missing required keys (got {})".format(
            name, sorted(entry.keys())
        )

        # rel_error carries the 7 distribution stats.
        rel_error = entry["rel_error"]
        assert rel_error is not None, (
            "rel_error for {!r} is None; expected the 7 distribution stats".format(name)
        )
        assert _REQUIRED_REL_ERROR_STATS.issubset(set(rel_error.keys())), (
            "parameter {!r} rel_error missing stats: {}".format(
                name, _REQUIRED_REL_ERROR_STATS - set(rel_error.keys())
            )
        )

        # zero_truth_excluded_count is an int; r_squared is present (may be float).
        assert isinstance(entry["zero_truth_excluded_count"], int)
        assert "r_squared" in entry


# ===========================================================================
# Task 4.12 — SMOKE test: recovery scatter figure output
#
# Validates: Requirements 2.8, 6.4, 6.6
#
# This is a SMOKE test (NOT Hypothesis). It verifies the two figure guarantees
# of the Recovery_Experiment's true-vs-predicted scatter plots:
#
#   * Requirement 2.8 / 6.4 — the experiment writes a per-parameter
#     true-vs-predicted scatter plot as a non-empty image file in the
#     Results_Directory, rendered headlessly via the non-interactive ``Agg``
#     backend.
#   * Requirement 6.6 — each figure axis is labeled with its quantity and unit
#     so the figure is interpretable without the source code.
#
# Two complementary checks are used:
#
#   (A) DIRECT check of ``validate_recovery.plot_scatter`` on small synthetic
#       truth/estimate arrays. Because ``plot_scatter`` closes its Figure (so the
#       rendered Axes can't be re-inspected afterwards), we use the same approach
#       as the 8.9 figure test: temporarily wrap
#       ``matplotlib.axes.Axes.set_xlabel`` / ``set_ylabel`` to RECORD the label
#       strings as they are set, then assert the recorded x/y labels are
#       non-empty and name the parameter (quantity) plus a unit token. The PNG is
#       also asserted to exist and be non-empty.
#
#   (B) END-TO-END via ``run_recovery`` (Requirement 2.8) with the torch-free
#       injected ``_MidpointFakePredictor`` (n_samples=2, multistart=0, a small
#       draw budget, and a ``tmp_path`` Results_Directory): all 7
#       ``recovery_scatter_<param>.png`` files must exist and be non-empty.
# ===========================================================================


def test_plot_scatter_writes_nonempty_png_with_labeled_axes(tmp_path, monkeypatch):
    """plot_scatter writes a non-empty PNG whose axes name the quantity + unit."""
    import matplotlib

    matplotlib.use("Agg")  # headless rendering, matches the helper (Req 6.4)
    from matplotlib.axes import Axes

    # --- Record axis-label strings as plot_scatter sets them (it closes the
    #     Figure afterwards, so capture them at set time — same as task 8.9). ---
    recorded = {"x": [], "y": []}
    orig_set_xlabel = Axes.set_xlabel
    orig_set_ylabel = Axes.set_ylabel

    def record_xlabel(self, label, *args, **kwargs):
        recorded["x"].append(label)
        return orig_set_xlabel(self, label, *args, **kwargs)

    def record_ylabel(self, label, *args, **kwargs):
        recorded["y"].append(label)
        return orig_set_ylabel(self, label, *args, **kwargs)

    monkeypatch.setattr(Axes, "set_xlabel", record_xlabel)
    monkeypatch.setattr(Axes, "set_ylabel", record_ylabel)

    # --- Small synthetic truth/estimate Parameter_Sets in full (m, 7) form -----
    # Column i corresponds positionally to pysim.PARAM_NAMES[i]; plot_scatter
    # selects the requested parameter's column. Values are arbitrary finite
    # numbers (plotting does not require them to be in-range).
    rng = np.random.default_rng(4242)
    m = 6
    Y_true = rng.uniform(0.5, 5.0, size=(m, _NUM_PARAMS)).astype(np.float64)
    # Estimate = truth plus a small perturbation so the scatter spreads off y=x.
    Y_est = Y_true + rng.normal(0.0, 0.1, size=(m, _NUM_PARAMS))

    param_name = "E_b"  # its PARAM_UNITS string carries the "kBT" unit token
    out_path = tmp_path / "recovery_scatter_{}.png".format(param_name)

    returned = validate_recovery.plot_scatter(
        Y_true, Y_est, param_name, str(out_path)
    )

    # --- The helper wrote a real, non-empty PNG to the requested path ----------
    assert returned == str(out_path)
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    # --- An x-axis and a y-axis label were set, both non-empty (Req 6.6) -------
    assert recorded["x"] and recorded["y"]
    xlabel = recorded["x"][-1]
    ylabel = recorded["y"][-1]
    assert isinstance(xlabel, str) and xlabel.strip()
    assert isinstance(ylabel, str) and ylabel.strip()

    # --- Both axes name the parameter (quantity) and its unit (Req 6.6) --------
    # The label format is "True E_b [<unit>]" / "Predicted E_b [<unit>]" where
    # <unit> is PARAM_UNITS[param_name] (which contains the "kBT" unit token).
    unit_string = validate_recovery.PARAM_UNITS[param_name]
    for label in (xlabel, ylabel):
        assert param_name in label, (
            "axis label {!r} does not name the parameter {!r}".format(
                label, param_name
            )
        )
        assert unit_string in label, (
            "axis label {!r} does not include the quantity+unit {!r}".format(
                label, unit_string
            )
        )
        assert "kbt" in label.lower(), (
            "axis label {!r} does not include a unit token".format(label)
        )


def test_run_recovery_writes_all_seven_scatter_pngs(tmp_path):
    """run_recovery writes all 7 recovery_scatter_<param>.png files, non-empty."""
    ranges = validation_common.load_configured_ranges()
    predictor = _MidpointFakePredictor(ranges)

    metrics = validate_recovery.run_recovery(
        n_samples=2,
        seed=42,
        method="Powell",
        maxiter=3,           # small refinement so the real pipeline stays fast
        multistart=0,        # no jitter -> no per-sample estimation seed needed
        max_total_draws=40,  # generous budget so 2 valid samples are retained
        predictor=predictor,
        results_dir=str(tmp_path),
    )

    # A successful run retained samples (not the terminal all-excluded case), so
    # the 7 scatter plots are produced (Requirement 2.8).
    assert metrics["all_samples_excluded"] is False
    assert metrics["retained_sample_count"] >= 1

    # One recovery_scatter_<param>.png per parameter, each a non-empty image file
    # written into the Results_Directory (Requirements 2.8, 6.4).
    for name in pysim.PARAM_NAMES:
        scatter_path = tmp_path / "recovery_scatter_{}.png".format(name)
        assert scatter_path.exists(), (
            "missing scatter figure for parameter {!r}".format(name)
        )
        assert scatter_path.stat().st_size > 0, (
            "scatter figure for parameter {!r} is empty".format(name)
        )
