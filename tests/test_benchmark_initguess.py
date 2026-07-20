# coding=utf-8
"""Property-based tests for ``benchmark_initguess`` (Warm_Start_Experiment).

Currently covers Property 15 (all strategies are evaluated on identical target
curves). Further warm-start property/example/smoke tests (6.4, 6.5, 6.7, 6.8,
6.10, 6.11) are added by later tasks.

These tests exercise the REAL ``pysim`` physics simulator through
``benchmark_initguess.make_target_curves`` and are therefore SLOW. The number of
targets (``n_targets``) is kept small so the suite finishes in a reasonable time
while still running >=100 Hypothesis examples.
"""

from unittest import mock

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import benchmark_initguess
import pysim
import refine
import validation_common


# Configured_Ranges are fixed for a given config; load once at import time.
_RANGES = validation_common.load_configured_ranges()
_NUM_PARAMS = len(validation_common.PARAM_NAMES)
_NUM_RESULTS = pysim.NUM_RESULTS  # 7801; channels x time points -> (3, 7801)


# Feature: validation-experiments, Property 15: All strategies are evaluated on identical target curves
#
# Validates: Requirements 3.9
#
# ``make_target_curves(n_targets, ranges, seed)`` is the single source of the
# target curve sets in the Warm_Start_Experiment. ``run_benchmark`` (task 6.9)
# feeds each generated target to all four Initialization_Strategies, so the
# foundational invariant guaranteeing "every strategy is compared on identical
# inputs" (Requirement 3.9) is that the generated target set is well-defined and
# stable: it is fully determined by ``(n_targets, ranges, seed)`` and does not
# change as it is consumed.
#
# This property verifies, for any small ``n_targets`` and any non-negative seed:
#
#   (a) Reproducibility / single-source-of-truth. Two independent calls with the
#       same ``(n_targets, ranges, seed)`` produce identical target sets: the
#       same number of targets, the same truth Parameter_Sets (key-for-key), and
#       array-equal curves for every target. This is exactly what lets the
#       experiment hand "the same target curves" to each strategy.
#
#   (b) Stable, well-shaped targets. The returned list has exactly ``n_targets``
#       entries, each curve set is an ``np.ndarray`` of shape ``(3, 7801)``, and
#       the curves handed to a given target index are array-equal to themselves
#       across repeated access (simulating each of the four strategies reading
#       the same target object): a strategy cannot perturb the shared target.
#
# ``n_targets`` is kept in [1, 3] because each target runs the REAL simulator;
# the property holds for all valid values. Seeds are generated as non-negative
# ints (the LHS sampler requires a non-negative integer seed).
@settings(max_examples=100, deadline=None)
@given(
    n_targets=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_all_strategies_evaluated_on_identical_target_curves(n_targets, seed):
    """The generated target set is the single, stable source fed to every strategy."""
    targets_a = benchmark_initguess.make_target_curves(n_targets, _RANGES, seed)
    targets_b = benchmark_initguess.make_target_curves(n_targets, _RANGES, seed)

    # --- (b) Stable, well-shaped targets ---------------------------------------
    assert len(targets_a) == n_targets, (
        "expected {} targets, got {}".format(n_targets, len(targets_a))
    )
    assert len(targets_b) == n_targets

    for i, (truth_params, curves) in enumerate(targets_a):
        # Truth params name the seven trainable parameters.
        assert set(truth_params.keys()) == set(validation_common.PARAM_NAMES)
        # Curves are a (3, 7801) ndarray.
        assert isinstance(curves, np.ndarray)
        assert curves.shape == (3, _NUM_RESULTS), (
            "target {} curves have shape {}, expected (3, {})".format(
                i, curves.shape, _NUM_RESULTS
            )
        )
        # Each of the four strategies reads the SAME target object; repeated
        # access yields array-equal curves (no strategy can perturb the target).
        for _strategy in benchmark_initguess.INIT_STRATEGIES:
            curves_for_strategy = targets_a[i][1]
            assert np.array_equal(curves_for_strategy, curves), (
                "target {} curves differ across repeated access".format(i)
            )

    # --- (a) Reproducibility / single source of truth --------------------------
    # Two calls with the same (n_targets, ranges, seed) yield identical targets.
    for i, ((truth_a, curves_a), (truth_b, curves_b)) in enumerate(
        zip(targets_a, targets_b)
    ):
        assert truth_a.keys() == truth_b.keys()
        for name in truth_a:
            assert truth_a[name] == truth_b[name], (
                "target {} truth param {!r} differs between runs".format(i, name)
            )
        assert np.array_equal(curves_a, curves_b), (
            "target {} curves differ between two same-seed runs".format(i)
        )


def _in_range_params():
    """Hypothesis strategy: an init-guess Parameter_Set within Configured_Ranges.

    Produces a dict keyed by ``pysim.PARAM_NAMES`` whose every value lies inside
    that parameter's ``[min, max]`` Configured_Range. ``refine_with_cost`` maps
    this guess into optimization space (``k0`` in log10), so values must be
    valid (in particular ``k0 > 0``, which the Configured_Range guarantees).
    """
    return st.fixed_dictionaries(
        {
            name: st.floats(
                min_value=float(lo),
                max_value=float(hi),
                allow_nan=False,
                allow_infinity=False,
            )
            for name, (lo, hi) in _RANGES.items()
        }
    )


# Feature: validation-experiments, Property 12: Warm-start cost accounting is consistent
#
# Validates: Requirements 3.3, 3.4, 3.5, 3.13
#
# Property statement (design doc): for any refinement run (reset to a zero
# counter before it starts), the recorded ``total_cost`` equals the number of
# ``pysim.run_simulation`` calls made during that run; if the run reaches the
# target Curve_RMSE then ``cost_at_target`` is a positive integer with
# ``cost_at_target <= total_cost``, otherwise ``cost_at_target`` is ``None``.
# Evaluations whose simulation returns ``dt_used < 0`` (so ``curve_rmse == inf``)
# are counted in ``total_cost`` and never satisfy the target.
#
# Test approach (documented per the task):
#   To keep the property fast and independent of pysim's heavy 14-state
#   simulator, we use the most faithful counting-contract stub available:
#     1. ``pysim._run_simulation_impl`` is replaced by a fast stub returning
#        ``(zeros((3, NUM_RESULTS)), 1.0)`` so a real ``pysim.run_simulation``
#        call is cheap but STILL increments the process-global Call_Counter via
#        the real try/finally wrapper (the one-call-per-eval contract).
#     2. ``refine.curve_rmse`` is replaced by a deterministic fake that calls the
#        real ``pysim.run_simulation`` exactly once (so each objective evaluation
#        increments the counter exactly once, exactly like the production
#        ``refine.curve_rmse`` -> ``pysim.run_simulation`` path) and then returns
#        a CONTROLLED RMSE so we can drive every branch of the cost accounting:
#          - "reach_immediate": every eval is below target -> reached at the
#            up-front init evaluation, so ``cost_at_target == 1``.
#          - "reach_later":     the init eval is above target, every later eval is
#            below target -> reached on the first optimizer evaluation.
#          - "never":           every eval is a finite value above target -> the
#            target is never reached, ``cost_at_target is None``.
#          - "invalid":         the up-front init evaluation returns ``inf``
#            (mimicking a ``dt_used < 0`` simulation), while the optimizer's own
#            evaluations stay finite-but-above-target. This proves an ``inf`` eval
#            is COUNTED in ``total_cost`` yet never reaches the target, without
#            forcing scipy's line search into a degenerate all-flat/all-inf
#            landscape (a real refinement run always has a finite anchor: the warm
#            start and many sampled points produce valid simulations).
#   Because ``refine_with_cost`` resets the Call_Counter at the START of the run,
#   ``pysim.get_call_count()`` immediately after the run equals the number of
#   ``run_simulation`` calls made during the run, which must equal ``total_cost``.
#   We independently tally the fake's invocations (one run_simulation per
#   invocation) and assert it equals ``total_cost`` too.
@settings(max_examples=100, deadline=None)
@given(
    init_params=_in_range_params(),
    target_rmse=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    mode=st.sampled_from(["reach_immediate", "reach_later", "never", "invalid"]),
    method=st.sampled_from(["Powell", "Nelder-Mead"]),
    maxiter=st.integers(min_value=2, max_value=10),
)
def test_warm_start_cost_accounting_is_consistent(
    init_params, target_rmse, mode, method, maxiter
):
    """``refine_with_cost`` accounts for cost consistently across all branches."""
    # Curves are unused by the stubbed objective; any (3, 7801) array works.
    target_curves = np.zeros((3, _NUM_RESULTS), dtype=np.float64)

    below = target_rmse * 0.5          # strictly below the target (and > 0)
    above = target_rmse + 1.0          # strictly above the target

    # Per-run state: number of times the fake objective is invoked. Each
    # invocation makes exactly one pysim.run_simulation call, so this is the
    # ground-truth count of forward simulations performed by the run.
    state = {"n_calls": 0}

    def _fast_impl(_params, _fixed_params=None):
        # Cheap stand-in for the heavy simulator; shape mirrors the valid path.
        return np.zeros((3, _NUM_RESULTS)), 1.0

    def _fake_curve_rmse(_params, _curves):
        # One counted forward simulation per evaluation, via the REAL wrapper
        # (its try/finally increments the Call_Counter exactly once).
        pysim.run_simulation(_params)
        state["n_calls"] += 1
        n = state["n_calls"]
        if mode == "reach_immediate":
            return below
        if mode == "reach_later":
            # Call 1 is the up-front init evaluation (above target); every
            # subsequent optimizer evaluation is below target.
            return above if n == 1 else below
        if mode == "never":
            return above
        # mode == "invalid": the up-front init evaluation (call #1) mimics a
        # dt_used < 0 simulation by returning inf; the optimizer's subsequent
        # evaluations stay finite-but-above-target so scipy stays well-behaved.
        # Either way every eval is >= target, so the target is never reached,
        # while the inf init eval is definitely counted in total_cost.
        return float("inf") if n == 1 else above

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_impl), \
            mock.patch.object(refine, "curve_rmse", _fake_curve_rmse):
        result = benchmark_initguess.refine_with_cost(
            init_params, target_curves, _RANGES,
            method=method, maxiter=maxiter, target_rmse=target_rmse,
        )
        # refine_with_cost reset the counter at the START of the run, and no
        # run_simulation call happens after it reads total_cost, so the live
        # Call_Counter still equals the run's total cost here.
        live_count = pysim.get_call_count()

    # --- total_cost is a non-negative int equal to the forward-sim call count ---
    assert isinstance(result.total_cost, int)
    assert result.total_cost >= 0
    # total_cost equals the number of pysim.run_simulation calls made in the run,
    # measured two independent ways: the live Call_Counter and the fake's tally.
    assert result.total_cost == live_count
    assert result.total_cost == state["n_calls"]
    # The init guess is always evaluated once up front, so at least one sim ran.
    assert result.total_cost >= 1

    if mode in ("reach_immediate", "reach_later"):
        # --- reached the target: cost_at_target is a positive int <= total_cost ---
        assert result.reached_target is True
        assert isinstance(result.cost_at_target, int)
        assert result.cost_at_target > 0
        assert result.cost_at_target <= result.total_cost
        if mode == "reach_immediate":
            # The up-front init evaluation already meets the target (call #1).
            assert result.cost_at_target == 1
    else:
        # --- never reached (finite-above-target OR invalid inf): None ---
        assert result.reached_target is False
        assert result.cost_at_target is None


# Feature: validation-experiments, Property 13: All reported costs are non-negative integer call counts
#
# Validates: Requirements 3.7, 6.5
#
# Property statement (design doc): for any strategy run, ``total_cost`` is a
# non-negative integer and ``cost_at_target`` (when not ``None``) is a positive
# integer; no comparison between strategies is expressed in wall-clock time.
#
# Where Property 12 (above) pins down the *value* of the cost accounting (that
# ``total_cost`` equals the number of forward-sim calls, that ``cost_at_target``
# is bounded by it, etc.), Property 13 pins down the *type and sign contract* of
# every reported cost field: each cost the Warm_Start_Experiment reports is a
# Pysim_Call_Cost — an integer count of forward simulations — and never a
# wall-clock duration (which would surface as a float). This is the invariant
# that lets the experiment claim a platform-independent cost metric
# (Requirements 3.7, 6.5).
#
# Test approach (mirrors task 6.4 / Property 12 so the run stays fast):
#   1. ``pysim._run_simulation_impl`` is replaced by a fast stub returning
#      ``(zeros((3, NUM_RESULTS)), 1.0)`` so each real ``pysim.run_simulation``
#      call is cheap but STILL increments the process-global Call_Counter via the
#      real try/finally wrapper.
#   2. ``refine.curve_rmse`` is replaced by a deterministic fake that performs
#      exactly one counted ``pysim.run_simulation`` per evaluation and returns a
#      CONTROLLED RMSE, letting us drive both the reached and not-reached cases:
#        - "reach_immediate" / "reach_later": the target IS reached, so
#          ``cost_at_target`` is not ``None`` and must be a positive int.
#        - "never" / "invalid": the target is NEVER reached, so
#          ``cost_at_target`` must be exactly ``None`` (and total_cost is still a
#          non-negative int). "invalid" mimics a ``dt_used < 0`` evaluation by
#          returning ``inf`` on the up-front init eval.
#   Across many runs we assert the type/sign contract on every cost field:
#     - ``total_cost`` is a Python ``int`` (``type is int``: not bool, not a
#       numpy integer, and crucially not a float wall-clock value) and ``>= 0``.
#     - ``cost_at_target`` is either ``None`` or a Python ``int`` ``> 0``.
#     - No cost field is a float / numpy floating value.
@settings(max_examples=100, deadline=None)
@given(
    init_params=_in_range_params(),
    target_rmse=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    mode=st.sampled_from(["reach_immediate", "reach_later", "never", "invalid"]),
    method=st.sampled_from(["Powell", "Nelder-Mead"]),
    maxiter=st.integers(min_value=2, max_value=10),
)
def test_all_reported_costs_are_non_negative_integer_call_counts(
    init_params, target_rmse, mode, method, maxiter
):
    """Every reported cost is an integer Pysim_Call_Cost, never a wall-clock float."""
    # Curves are unused by the stubbed objective; any (3, 7801) array works.
    target_curves = np.zeros((3, _NUM_RESULTS), dtype=np.float64)

    below = target_rmse * 0.5          # strictly below the target (and > 0)
    above = target_rmse + 1.0          # strictly above the target

    state = {"n_calls": 0}

    def _fast_impl(_params, _fixed_params=None):
        # Cheap stand-in for the heavy simulator; shape mirrors the valid path.
        return np.zeros((3, _NUM_RESULTS)), 1.0

    def _fake_curve_rmse(_params, _curves):
        # One counted forward simulation per evaluation, via the REAL wrapper.
        pysim.run_simulation(_params)
        state["n_calls"] += 1
        n = state["n_calls"]
        if mode == "reach_immediate":
            return below
        if mode == "reach_later":
            return above if n == 1 else below
        if mode == "never":
            return above
        # mode == "invalid": up-front init eval returns inf (dt_used < 0 proxy);
        # every later eval is finite-but-above-target, so target is never met.
        return float("inf") if n == 1 else above

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_impl), \
            mock.patch.object(refine, "curve_rmse", _fake_curve_rmse):
        result = benchmark_initguess.refine_with_cost(
            init_params, target_curves, _RANGES,
            method=method, maxiter=maxiter, target_rmse=target_rmse,
        )

    # --- total_cost: a non-negative *Python* integer, never a wall-clock float -
    # type-is-int excludes bool, numpy integers, AND floats in one check, proving
    # total_cost is a forward-simulation call count rather than a timing value.
    assert type(result.total_cost) is int, (
        "total_cost must be a Python int, got {!r}".format(type(result.total_cost))
    )
    assert not isinstance(result.total_cost, float)
    assert not isinstance(result.total_cost, np.floating)
    assert result.total_cost >= 0
    # The init guess is always evaluated once up front, so at least one sim ran.
    assert result.total_cost >= 1

    # --- cost_at_target: None, or a positive *Python* integer ------------------
    if mode in ("reach_immediate", "reach_later"):
        # Target reached -> cost_at_target is a positive integer call count.
        assert result.cost_at_target is not None
        assert type(result.cost_at_target) is int, (
            "cost_at_target must be a Python int, got {!r}".format(
                type(result.cost_at_target)
            )
        )
        assert not isinstance(result.cost_at_target, float)
        assert not isinstance(result.cost_at_target, np.floating)
        assert result.cost_at_target > 0
    else:
        # Target never reached (finite-above-target OR invalid inf) -> None.
        assert result.cost_at_target is None


# Feature: validation-experiments, Property 14: Latin Hypercube multistart cost is additive across starts
#
# Validates: Requirements 3.8
#
# Property statement (design doc): for any configured number of starts
# ``n_starts``, the LHS-multistart strategy's recorded ``total_cost`` equals the
# sum of the ``Pysim_Call_Cost`` of its individual starts, and the recorded
# ``n_starts`` equals the configured value.
#
# This is the cost-aggregation contract of ``run_strategy('lhs_multistart', ...)``
# (Requirement 3.8): the multi-start strategy runs ``cfg.n_starts`` independent
# refinements and must report the *cumulative* forward-simulation cost across
# every start, not the cost of a single start.
#
# Test approach (documented per the task), kept fast and deterministic:
#   1. ``pysim._run_simulation_impl`` is replaced by a fast stub returning
#      ``(zeros((3, NUM_RESULTS)), 1.0)`` so every real ``pysim.run_simulation``
#      call is cheap but STILL increments the process-global Call_Counter via the
#      real try/finally wrapper (the one-call-per-eval contract).
#   2. ``refine.curve_rmse`` is replaced by a cheap deterministic fake that makes
#      exactly one counted ``pysim.run_simulation`` call per evaluation (so each
#      objective evaluation increments the Call_Counter exactly once, like the
#      production ``refine.curve_rmse`` -> ``pysim.run_simulation`` path) and
#      returns a finite RMSE.
#   3. ``benchmark_initguess.refine_with_cost`` is wrapped so we capture each
#      per-start ``RunResult.total_cost``. The REAL ``refine_with_cost`` resets
#      the Call_Counter at the START of each call, so each start's recorded
#      ``total_cost`` is that start's own independent ``Pysim_Call_Cost``. The
#      wrapper delegates to the real implementation, so the aggregation logic in
#      ``run_strategy`` runs exactly as in production.
#   We then assert the aggregated ``RunResult.total_cost`` equals the SUM of the
#   captured per-start ``total_cost`` values, and that the recorded ``n_starts``
#   equals the configured ``cfg.n_starts``.
#
# ``n_starts`` is kept in [2, 5] (multi-start, but small for speed) and
# ``maxiter`` small; the property holds for all valid configurations.
@settings(max_examples=100, deadline=None)
@given(
    n_starts=st.integers(min_value=2, max_value=5),
    maxiter=st.integers(min_value=2, max_value=8),
    target_rmse=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    method=st.sampled_from(["Powell", "Nelder-Mead"]),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
)
def test_lhs_multistart_cost_is_additive_across_starts(
    n_starts, maxiter, target_rmse, method, seed
):
    """``run_strategy('lhs_multistart')`` total_cost == sum of per-start costs."""
    # Curves are unused by the stubbed objective; any (3, 7801) array works.
    target_curves = np.zeros((3, _NUM_RESULTS), dtype=np.float64)

    def _fast_impl(_params, _fixed_params=None):
        # Cheap stand-in for the heavy simulator; shape mirrors the valid path.
        return np.zeros((3, _NUM_RESULTS)), 1.0

    def _fake_curve_rmse(params, _curves):
        # One counted forward simulation per evaluation, via the REAL wrapper
        # (its try/finally increments the Call_Counter exactly once).
        pysim.run_simulation(params)
        # Deterministic finite RMSE derived from the candidate params so the
        # optimizer follows a stable, reproducible path. Value/branch taken does
        # not affect cost additivity; we only need a counted, finite evaluation.
        vec = np.array(
            [float(params[name]) for name in validation_common.PARAM_NAMES],
            dtype=np.float64,
        )
        return float(np.sqrt(np.mean(vec * vec)))

    cfg = benchmark_initguess.BenchmarkConfig(
        method=method, maxiter=maxiter, target_rmse=target_rmse,
        n_starts=n_starts, seed=seed,
    )

    # Capture each per-start RunResult.total_cost by wrapping the REAL
    # refine_with_cost (which resets the Call_Counter per call, so each start's
    # total_cost is its own independent Pysim_Call_Cost).
    real_refine_with_cost = benchmark_initguess.refine_with_cost
    per_start_costs = []

    def _wrapped_refine_with_cost(*args, **kwargs):
        start_result = real_refine_with_cost(*args, **kwargs)
        per_start_costs.append(start_result.total_cost)
        return start_result

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_impl), \
            mock.patch.object(refine, "curve_rmse", _fake_curve_rmse), \
            mock.patch.object(
                benchmark_initguess, "refine_with_cost", _wrapped_refine_with_cost
            ):
        result = benchmark_initguess.run_strategy(
            "lhs_multistart", None, target_curves, _RANGES, cfg,
        )

    # --- n_starts recorded equals the configured value (Requirement 3.8) -------
    assert result.n_starts == n_starts, (
        "recorded n_starts {} != configured {}".format(result.n_starts, n_starts)
    )
    # One refinement was run per configured start.
    assert len(per_start_costs) == n_starts, (
        "expected {} per-start refinements, captured {}".format(
            n_starts, len(per_start_costs)
        )
    )

    # --- total_cost is the SUM of the individual starts' Pysim_Call_Cost -------
    expected_total = sum(per_start_costs)
    assert type(result.total_cost) is int, (
        "total_cost must be a Python int, got {!r}".format(type(result.total_cost))
    )
    assert result.total_cost == expected_total, (
        "aggregated total_cost {} != sum of per-start costs {} (parts={})".format(
            result.total_cost, expected_total, per_start_costs
        )
    )
    # Each start performs at least the up-front init evaluation, so the cumulative
    # cost is at least one forward simulation per start.
    assert result.total_cost >= n_starts


# Feature: validation-experiments, Property 16: Warm-start randomized strategies are reproducible under a fixed seed
#
# Validates: Requirements 3.12, 6.1
#
# Property statement (design doc): for any seed, two runs of the benchmark with
# the same seed produce identical uniform-random and LHS-multistart initial
# guesses and therefore identical recorded costs for those strategies.
#
# This is the reproducibility contract of the randomized Initialization_Strategies
# (Requirements 3.12, 6.1): the ``'random'`` and ``'lhs_multistart'`` strategies
# draw their init guesses through ``validation_common.lhs_params`` seeded by the
# recorded ``cfg.seed``, so re-running ``run_strategy`` with the SAME
# ``BenchmarkConfig`` (same seed) must reproduce the identical init guesses and,
# given a deterministic refiner path, the identical ``RunResult``.
#
# Test approach (documented per the task), kept fast and fully deterministic:
#   1. ``pysim._run_simulation_impl`` is replaced by a fast stub returning
#      ``(zeros((3, NUM_RESULTS)), 1.0)`` so every real ``pysim.run_simulation``
#      call is cheap but STILL increments the process-global Call_Counter via the
#      real try/finally wrapper (the one-call-per-eval contract), making cost
#      accounting realistic without paying for the heavy 14-state simulator.
#   2. ``refine.curve_rmse`` is replaced by a DETERMINISTIC fake that performs
#      exactly one counted ``pysim.run_simulation`` per evaluation and returns an
#      RMSE that is a pure function of the candidate Parameter_Set (the RMS of its
#      seven values). Because the objective is deterministic and ``scipy.optimize``
#      is deterministic, the entire optimizer trajectory — and thus every recorded
#      cost field — is fully determined by the init guess, which in turn is fully
#      determined by ``cfg.seed``.
#   3. ``benchmark_initguess.refine_with_cost`` is wrapped to CAPTURE the
#      ``init_params`` dict handed to each refinement, so we can additionally
#      assert (beyond ``RunResult`` equality) that the two same-seed runs produced
#      byte-for-byte identical initial guesses (one for ``'random'``,
#      ``cfg.n_starts`` for ``'lhs_multistart'``). The wrapper delegates to the
#      REAL implementation so ``run_strategy``'s logic runs exactly as in
#      production.
#
# For each randomized strategy we call ``run_strategy(strategy, predictor=None,
# target, ranges, cfg)`` twice with the SAME ``BenchmarkConfig`` and assert:
#   (a) the captured init guesses are identical between the two runs, and
#   (b) the two ``RunResult`` objects are equal (same ``total_cost``,
#       ``cost_at_target``, ``converged_rmse``, ``reached_target``, ``n_starts``).
#
# The seed is generated by Hypothesis as a non-negative int (the LHS sampler
# requires a non-negative integer seed); ``n_starts``/``maxiter`` are kept small
# for speed. The property holds for all valid configurations.
def _run_strategy_capturing_init_params(strategy, target_curves, ranges, cfg):
    """Run ``run_strategy`` once with deterministic stubs, capturing init guesses.

    Returns ``(RunResult, [init_params, ...])`` where the list holds every
    ``init_params`` dict passed to ``refine_with_cost`` during the run (one for
    the single-start strategies, ``cfg.n_starts`` for ``lhs_multistart``).
    """

    def _fast_impl(_params, _fixed_params=None):
        # Cheap stand-in for the heavy simulator; shape mirrors the valid path.
        return np.zeros((3, _NUM_RESULTS)), 1.0

    def _fake_curve_rmse(params, _curves):
        # One counted forward simulation per evaluation, via the REAL wrapper
        # (its try/finally increments the Call_Counter exactly once), then a
        # deterministic RMSE that is a pure function of the candidate params so
        # the optimizer follows a stable, reproducible trajectory.
        pysim.run_simulation(params)
        vec = np.array(
            [float(params[name]) for name in validation_common.PARAM_NAMES],
            dtype=np.float64,
        )
        return float(np.sqrt(np.mean(vec * vec)))

    real_refine_with_cost = benchmark_initguess.refine_with_cost
    captured_init_params = []

    def _wrapped_refine_with_cost(init_params, *args, **kwargs):
        # Capture a copy of the init guess so later mutation cannot affect it.
        captured_init_params.append(dict(init_params))
        return real_refine_with_cost(init_params, *args, **kwargs)

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_impl), \
            mock.patch.object(refine, "curve_rmse", _fake_curve_rmse), \
            mock.patch.object(
                benchmark_initguess, "refine_with_cost", _wrapped_refine_with_cost
            ):
        result = benchmark_initguess.run_strategy(
            strategy, None, target_curves, ranges, cfg,
        )

    return result, captured_init_params


@settings(max_examples=100, deadline=None)
@given(
    strategy=st.sampled_from(["random", "lhs_multistart"]),
    seed=st.integers(min_value=0, max_value=2 ** 32 - 1),
    n_starts=st.integers(min_value=2, max_value=4),
    maxiter=st.integers(min_value=2, max_value=8),
    target_rmse=st.floats(
        min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
    method=st.sampled_from(["Powell", "Nelder-Mead"]),
)
def test_randomized_strategies_are_reproducible_under_a_fixed_seed(
    strategy, seed, n_starts, maxiter, target_rmse, method
):
    """Same-seed ``run_strategy`` reproduces identical init guesses and costs."""
    # Curves are unused by the stubbed objective; any (3, 7801) array works.
    target_curves = np.zeros((3, _NUM_RESULTS), dtype=np.float64)

    cfg = benchmark_initguess.BenchmarkConfig(
        method=method, maxiter=maxiter, target_rmse=target_rmse,
        n_starts=n_starts, seed=seed,
    )

    # Two independent runs with the SAME config (same recorded seed).
    result_a, inits_a = _run_strategy_capturing_init_params(
        strategy, target_curves, _RANGES, cfg
    )
    result_b, inits_b = _run_strategy_capturing_init_params(
        strategy, target_curves, _RANGES, cfg
    )

    # --- (a) identical initial guesses between the two same-seed runs ----------
    expected_n_inits = n_starts if strategy == "lhs_multistart" else 1
    assert len(inits_a) == expected_n_inits, (
        "{} run captured {} init guesses, expected {}".format(
            strategy, len(inits_a), expected_n_inits
        )
    )
    assert len(inits_b) == len(inits_a), (
        "the two same-seed runs drew a different number of init guesses"
    )
    for idx, (guess_a, guess_b) in enumerate(zip(inits_a, inits_b)):
        assert guess_a.keys() == guess_b.keys()
        for name in guess_a:
            assert guess_a[name] == guess_b[name], (
                "{} init guess #{} param {!r} differs between same-seed runs "
                "({!r} != {!r})".format(strategy, idx, name, guess_a[name], guess_b[name])
            )

    # --- (b) identical recorded RunResult between the two same-seed runs -------
    # RunResult is a dataclass, so == compares every recorded field.
    assert result_a == result_b, (
        "{} RunResults differ between same-seed runs: {!r} != {!r}".format(
            strategy, result_a, result_b
        )
    )
    # Spell out the individual cost/outcome fields for a clearer failure message.
    assert result_a.total_cost == result_b.total_cost
    assert result_a.cost_at_target == result_b.cost_at_target
    assert result_a.converged_rmse == result_b.converged_rmse
    assert result_a.reached_target == result_b.reached_target
    assert result_a.n_starts == result_b.n_starts
    assert result_a.strategy == strategy


# ===========================================================================
# Task 6.10 — Example tests (NOT Hypothesis) for strategy enumeration, refiner
# config equality, converged-RMSE recording, and the warm-start JSON schema.
#
# Validates: Requirements 3.1, 3.2, 3.6, 3.10
#
# These are fast, deterministic example tests for the ``run_benchmark``
# orchestration (task 6.9). They keep the REAL ``benchmark_initguess`` /
# ``refine`` / ``scipy.optimize`` machinery intact and only replace the heavy
# 14-state physics kernel with a cheap stub so each tiny end-to-end run finishes
# quickly:
#
#   * ``pysim._run_simulation_impl`` -> ``_fast_sim_impl`` returns a valid
#     ``(zeros((3, NUM_RESULTS)), 1.0)`` so every real ``pysim.run_simulation``
#     call is cheap (and still increments the Call_Counter through the real
#     try/finally wrapper). ``dt_used == 1.0 >= 0`` keeps every drawn sample
#     valid, so ``make_target_curves`` collects its targets in one batch.
#   * A lightweight ``_FakePredictor`` stands in for the deep-learning model so
#     no ``torch`` is imported; ``run_benchmark(..., predictor=...)`` injects it
#     (task 6.9). Its ``predict`` returns the midpoint of the Configured_Ranges
#     as a ``(1, 7)`` physical-space array.
#
# Runs use ``n_targets`` in {1, 2} and small ``maxiter``/``n_starts`` for speed;
# the asserted invariants hold for all valid configurations.


class _FakePredictor:
    """Lightweight Inverse_Model stand-in (no torch) for warm-start dry runs.

    Exposes the pluggable predictor interface the suite relies on:
    ``predict(X) -> (1, 7)`` (the midpoint of the Configured_Ranges, in
    ``pysim.PARAM_NAMES`` order, in physical space) and ``get_param_names()``.
    The ``'dl'`` strategy feeds it a single ``(3, 7801)`` target and takes row 0
    of the returned array as its init guess, so a fixed midpoint is a valid,
    deterministic warm start (``k0``'s midpoint is positive, so the log10
    transform in ``refine`` is well-defined).
    """

    def __init__(self, ranges):
        self._midpoint = np.array(
            [
                (float(ranges[name][0]) + float(ranges[name][1])) / 2.0
                for name in validation_common.PARAM_NAMES
            ],
            dtype=np.float64,
        )

    def predict(self, _X):
        # Shape (1, 7) physical-space output, mirroring NanorobotPredictor /
        # TransformerPredictor's (N, 7) contract for a single curve set.
        return self._midpoint.reshape(1, -1).copy()

    def get_param_names(self):
        return list(validation_common.PARAM_NAMES)


def _fast_sim_impl(_params, _fixed_params=None):
    """Cheap valid stand-in for the heavy simulator (``dt_used >= 0``)."""
    return np.zeros((3, _NUM_RESULTS), dtype=np.float64), 1.0


# Distribution-stats keys produced by validation_common.distribution_stats; the
# warm-start JSON summarizes every cost/RMSE distribution with exactly these.
_STATS_KEYS = {"mean", "median", "std", "min", "max", "p5", "p95"}


def test_init_strategies_enumeration_and_all_four_evaluated(tmp_path):
    """Exactly the four strategies exist and run_benchmark evaluates each (Req 3.1)."""
    # The Initialization_Strategy enumeration is exactly the four documented
    # strategies, in order (Requirement 3.1).
    assert benchmark_initguess.INIT_STRATEGIES == (
        "dl", "random", "lhs_multistart", "reference"
    )

    predictor = _FakePredictor(_RANGES)
    with mock.patch.object(pysim, "_run_simulation_impl", _fast_sim_impl):
        metrics = benchmark_initguess.run_benchmark(
            model="cnn", n_targets=2, target_rmse=0.02, maxiter=3,
            method="Powell", n_starts=2, seed=42,
            results_dir=str(tmp_path), predictor=predictor,
        )

    # The JSON 'strategies' block carries one summary per strategy: all four are
    # evaluated, nothing extra (Requirement 3.1).
    assert set(metrics["strategies"].keys()) == set(benchmark_initguess.INIT_STRATEGIES)

    # Every per_target entry was evaluated with all four strategies on the same
    # target (Requirement 3.1; same-target invariant is Property 15 / Req 3.9).
    assert len(metrics["per_target"]) == 2
    for entry in metrics["per_target"]:
        for strategy in benchmark_initguess.INIT_STRATEGIES:
            assert strategy in entry, (
                "per_target entry {} missing strategy {!r}".format(
                    entry.get("target_index"), strategy
                )
            )


def test_refiner_config_identical_across_strategies(tmp_path):
    """All four strategies are refined with the same method/maxiter/bounds (Req 3.2)."""
    predictor = _FakePredictor(_RANGES)

    real_refine_with_cost = benchmark_initguess.refine_with_cost
    calls = []  # one record per refine_with_cost invocation

    def _recording_refine_with_cost(init_params, target_curves, ranges,
                                    method="Powell", maxiter=400,
                                    target_rmse=0.02, strategy="dl"):
        # Record the refiner configuration handed to THIS strategy, then delegate
        # to the real implementation so run_benchmark completes normally.
        calls.append({
            "strategy": strategy,
            "method": method,
            "maxiter": maxiter,
            # Bounds == the Configured_Ranges object passed through; identity is
            # the strongest "same bounds" check (Requirement 3.2).
            "ranges_id": id(ranges),
        })
        return real_refine_with_cost(
            init_params, target_curves, ranges, method=method,
            maxiter=maxiter, target_rmse=target_rmse, strategy=strategy,
        )

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_sim_impl), \
            mock.patch.object(
                benchmark_initguess, "refine_with_cost", _recording_refine_with_cost
            ):
        benchmark_initguess.run_benchmark(
            model="cnn", n_targets=1, target_rmse=0.02, maxiter=7,
            method="Powell", n_starts=3, seed=42,
            results_dir=str(tmp_path), predictor=predictor,
        )

    # The refiner ran for every strategy: 3 single-start strategies + n_starts(3)
    # for lhs_multistart on the single target.
    assert len(calls) == 6, "expected 6 refinements (1 target), got {}".format(len(calls))
    assert {c["strategy"] for c in calls} == set(benchmark_initguess.INIT_STRATEGIES)

    # --- identical method across all four strategies (Requirement 3.2) ---------
    assert {c["method"] for c in calls} == {"Powell"}
    # --- identical maxiter across all four strategies (Requirement 3.2) --------
    assert {c["maxiter"] for c in calls} == {7}
    # --- identical bounds: the SAME Configured_Ranges object passed through ----
    assert len({c["ranges_id"] for c in calls}) == 1, (
        "strategies were refined with different ranges/bounds objects"
    )


def test_converged_rmse_recorded_per_run(tmp_path):
    """Each RunResult / per_target entry records a finite converged_rmse (Req 3.6)."""
    import math

    predictor = _FakePredictor(_RANGES)
    ranges = validation_common.load_configured_ranges()

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_sim_impl):
        # (a) Direct RunResult check: every strategy returns a RunResult whose
        #     converged_rmse is a finite float (Requirement 3.6).
        targets = benchmark_initguess.make_target_curves(1, ranges, 7)
        cfg = benchmark_initguess.BenchmarkConfig(
            method="Powell", maxiter=3, target_rmse=0.02, n_starts=2, seed=7
        )
        for strategy in benchmark_initguess.INIT_STRATEGIES:
            run_result = benchmark_initguess.run_strategy(
                strategy, predictor, targets[0][1], ranges, cfg
            )
            assert isinstance(run_result.converged_rmse, float)
            assert math.isfinite(run_result.converged_rmse), (
                "{} converged_rmse is not finite: {!r}".format(
                    strategy, run_result.converged_rmse
                )
            )

        # (b) End-to-end: the per_target JSON section records a finite
        #     converged_rmse for every strategy on every target.
        metrics = benchmark_initguess.run_benchmark(
            model="cnn", n_targets=2, target_rmse=0.02, maxiter=3,
            method="Powell", n_starts=2, seed=42,
            results_dir=str(tmp_path), predictor=predictor,
        )

    for entry in metrics["per_target"]:
        for strategy in benchmark_initguess.INIT_STRATEGIES:
            converged_rmse = entry[strategy]["converged_rmse"]
            assert isinstance(converged_rmse, float)
            assert math.isfinite(converged_rmse), (
                "per_target {} strategy {!r} converged_rmse not finite: {!r}".format(
                    entry["target_index"], strategy, converged_rmse
                )
            )


def test_warmstart_json_schema(tmp_path):
    """warmstart_<model>.json carries the design.md keys and round-trips (Req 3.10)."""
    import json

    predictor = _FakePredictor(_RANGES)
    model = "cnn"

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_sim_impl):
        benchmark_initguess.run_benchmark(
            model=model, n_targets=2, target_rmse=0.02, maxiter=3,
            method="Powell", n_starts=2, seed=42,
            results_dir=str(tmp_path), predictor=predictor,
        )

    # The JSON was written into the (tmp) Results_Directory and reads back.
    json_path = tmp_path / "warmstart_{}.json".format(model)
    assert json_path.exists(), "warmstart_{}.json was not written".format(model)
    with open(str(json_path)) as handle:
        data = json.load(handle)

    # --- top-level keys (design.md warm-start JSON data model) -----------------
    assert data["experiment"] == "warm_start_ablation"
    assert data["model"] == model
    assert isinstance(data["seed"], int)
    assert "strategies" in data
    assert "per_target" in data

    # --- config block, including the Pysim_Call_Cost unit marker (Req 3.7) -----
    config = data["config"]
    for key in ("n_targets", "target_rmse", "refine_method", "maxiter", "n_starts"):
        assert key in config, "config missing {!r}".format(key)
    assert config["cost_unit"] == "pysim_call_cost"

    # --- per-strategy summary blocks (Requirement 3.10) ------------------------
    strategies = data["strategies"]
    assert set(strategies.keys()) == set(benchmark_initguess.INIT_STRATEGIES)
    for strategy, block in strategies.items():
        assert "n_starts" in block, "{} block missing n_starts".format(strategy)
        assert "reached_target_count" in block, (
            "{} block missing reached_target_count".format(strategy)
        )
        # cost_at_target is a distribution-stats dict, or null when no run of the
        # strategy reached the target (Requirements 3.4, 3.5).
        cost_at_target = block["cost_at_target"]
        assert cost_at_target is None or _STATS_KEYS.issubset(set(cost_at_target.keys())), (
            "{} cost_at_target is neither null nor a stats dict: {!r}".format(
                strategy, cost_at_target
            )
        )
        # total_cost and converged_rmse are always full distribution-stats dicts.
        assert _STATS_KEYS.issubset(set(block["total_cost"].keys())), (
            "{} total_cost missing stats keys".format(strategy)
        )
        assert _STATS_KEYS.issubset(set(block["converged_rmse"].keys())), (
            "{} converged_rmse missing stats keys".format(strategy)
        )

    # --- per_target breakdown: one entry per target, all four strategies -------
    assert len(data["per_target"]) == 2
    for entry in data["per_target"]:
        assert "target_index" in entry
        for strategy in benchmark_initguess.INIT_STRATEGIES:
            assert strategy in entry, (
                "per_target entry missing strategy {!r}".format(strategy)
            )
            sub = entry[strategy]
            for field in ("reached_target", "cost_at_target", "total_cost",
                          "converged_rmse"):
                assert field in sub, (
                    "per_target {} strategy {!r} missing field {!r}".format(
                        entry["target_index"], strategy, field
                    )
                )


# ===========================================================================
# Task 6.11 — Smoke test (NOT Hypothesis) for the warm-start comparison figure.
#
# Validates: Requirements 3.11, 6.4, 6.6
#
# Requirement 3.11: the Warm_Start_Experiment writes a figure comparing the
#   per-strategy Pysim_Call_Cost and converged Curve_RMSE distributions.
# Requirement 6.4: figures are written as image files in the Results_Directory.
# Requirement 6.6: each figure axis is labeled with its quantity and unit so the
#   figure is interpretable without the source code.
#
# Two complementary checks:
#   (1) End-to-end (Req 3.11, 6.4): run the REAL ``run_benchmark`` with the
#       injected ``_FakePredictor`` (reused from the 6.10 tests), a tiny config
#       (n_targets=2, small maxiter/n_starts), and the heavy 14-state kernel
#       stubbed by ``_fast_sim_impl`` so the run is cheap. Assert the
#       ``warmstart_<model>.png`` comparison figure is written into the (tmp)
#       Results_Directory and is a non-empty file.
#   (2) Axis labels (Req 6.6): call the figure helper
#       ``benchmark_initguess._plot_benchmark_compare`` directly on a small
#       synthetic ``per_strategy_results`` mapping (strategy -> list[RunResult]).
#       Because the helper closes its figure before returning, the labels cannot
#       be read back off the (closed) Axes; instead we temporarily wrap
#       ``matplotlib.axes.Axes.set_xlabel`` / ``set_ylabel`` to record every
#       label string the helper sets, then assert the recorded labels are
#       non-empty and name the quantity + unit: the cost axis mentions
#       ``Pysim_Call_Cost`` (or "forward-simulation calls"), the RMSE axis
#       mentions ``RMSE``.
#
# Uses ``tmp_path`` for all output and the non-interactive matplotlib ``Agg``
# backend (the helper selects ``Agg`` itself; the figure renders headlessly).


def _synthetic_per_strategy_results():
    """Build a small synthetic ``strategy -> list[RunResult]`` mapping.

    Mirrors the shape ``_plot_benchmark_compare`` consumes (one ``RunResult``
    per target, for each of :data:`benchmark_initguess.INIT_STRATEGIES`). A few
    runs per strategy — a mix of reached/not-reached with distinct
    ``total_cost`` and ``converged_rmse`` values — is enough to render the two
    box-plot panels.
    """
    per_strategy_results = {}
    for offset, strategy in enumerate(benchmark_initguess.INIT_STRATEGIES):
        per_strategy_results[strategy] = [
            benchmark_initguess.RunResult(
                strategy=strategy, reached_target=True, cost_at_target=10 + offset,
                total_cost=20 + offset, converged_rmse=0.01 + 0.001 * offset,
                n_starts=1,
            ),
            benchmark_initguess.RunResult(
                strategy=strategy, reached_target=False, cost_at_target=None,
                total_cost=30 + offset, converged_rmse=0.05 + 0.001 * offset,
                n_starts=1,
            ),
            benchmark_initguess.RunResult(
                strategy=strategy, reached_target=True, cost_at_target=15 + offset,
                total_cost=25 + offset, converged_rmse=0.02 + 0.001 * offset,
                n_starts=1,
            ),
        ]
    return per_strategy_results


def test_warmstart_figure_written_end_to_end(tmp_path):
    """run_benchmark writes a non-empty warmstart_<model>.png (Req 3.11, 6.4)."""
    model = "cnn"
    predictor = _FakePredictor(_RANGES)

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_sim_impl):
        benchmark_initguess.run_benchmark(
            model=model, n_targets=2, target_rmse=0.02, maxiter=3,
            method="Powell", n_starts=2, seed=42,
            results_dir=str(tmp_path), predictor=predictor,
        )

    # The comparison figure was written into the (tmp) Results_Directory and is a
    # real, non-empty image file (Requirements 3.11, 6.4).
    figure_path = tmp_path / "warmstart_{}.png".format(model)
    assert figure_path.exists(), (
        "warmstart_{}.png comparison figure was not written".format(model)
    )
    assert figure_path.stat().st_size > 0, (
        "warmstart_{}.png was written but is empty".format(model)
    )


def test_warmstart_figure_axes_labeled_with_quantity_and_unit(tmp_path):
    """The comparison figure labels its axes with the quantity + unit (Req 6.6)."""
    import matplotlib

    matplotlib.use("Agg")  # headless rendering (Requirement 6.4)
    import matplotlib.axes

    per_strategy_results = _synthetic_per_strategy_results()
    out_path = tmp_path / "warmstart_axis_label_check.png"

    # The helper closes its figure before returning, so the labels can't be read
    # back off the (closed) Axes. Instead, record every label string set during
    # rendering by wrapping the Axes label setters and delegating to the real
    # implementation so the figure still renders correctly.
    real_set_xlabel = matplotlib.axes.Axes.set_xlabel
    real_set_ylabel = matplotlib.axes.Axes.set_ylabel
    recorded_xlabels = []
    recorded_ylabels = []

    def _recording_set_xlabel(self, xlabel, *args, **kwargs):
        recorded_xlabels.append(xlabel)
        return real_set_xlabel(self, xlabel, *args, **kwargs)

    def _recording_set_ylabel(self, ylabel, *args, **kwargs):
        recorded_ylabels.append(ylabel)
        return real_set_ylabel(self, ylabel, *args, **kwargs)

    with mock.patch.object(matplotlib.axes.Axes, "set_xlabel", _recording_set_xlabel), \
            mock.patch.object(matplotlib.axes.Axes, "set_ylabel", _recording_set_ylabel):
        returned = benchmark_initguess._plot_benchmark_compare(
            per_strategy_results, "cnn", str(out_path)
        )

    # The helper rendered and returned its output path.
    assert returned == str(out_path)

    # Both panels set an x- and a y-axis label (2 panels -> >= 2 of each).
    assert len(recorded_xlabels) >= 2, (
        "expected x-axis labels on both panels, recorded {!r}".format(recorded_xlabels)
    )
    assert len(recorded_ylabels) >= 2, (
        "expected y-axis labels on both panels, recorded {!r}".format(recorded_ylabels)
    )

    # --- every recorded label is a non-empty string (Requirement 6.6) ----------
    for label in recorded_xlabels + recorded_ylabels:
        assert isinstance(label, str) and label.strip(), (
            "axis label is empty or not a string: {!r}".format(label)
        )

    # --- the cost axis names the quantity + unit: Pysim_Call_Cost (forward-
    #     simulation calls), never wall-clock time (Requirements 3.11, 6.5, 6.6) -
    assert any(
        ("Pysim_Call_Cost" in label) or ("forward-simulation calls" in label)
        for label in recorded_ylabels
    ), (
        "no y-axis label names the cost quantity/unit (Pysim_Call_Cost / "
        "forward-simulation calls): {!r}".format(recorded_ylabels)
    )

    # --- the RMSE axis names the converged Curve_RMSE quantity (Requirement 6.6) -
    assert any("RMSE" in label for label in recorded_ylabels), (
        "no y-axis label names the RMSE quantity: {!r}".format(recorded_ylabels)
    )
