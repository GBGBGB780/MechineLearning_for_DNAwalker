# coding=utf-8
"""
benchmark_initguess.py — Warm-start vs. baseline ablation (Component 4).

This script compares four Initialization_Strategies for the local physics
Refiner — deep-learning prediction, a single uniform-random draw, Latin
Hypercube multi-start, and the documented Reference_Parameters — under an
identical refinement routine, measuring efficiency in ``Pysim_Call_Cost`` (the
number of forward-simulation calls read from the ``pysim`` Call_Counter) rather
than wall-clock time (Requirements 3.1-3.13, 6.5).

Scope grows task-by-task. Currently implemented:
  - ``INIT_STRATEGIES``     : the four Initialization_Strategy identifiers
  - ``REFERENCE_PARAMETERS``: the documented Reference_Parameters Parameter_Set
  - ``RunResult``           : per-run cost/outcome record (dataclass)
  - ``make_target_curves``  : LHS + pysim generation of valid target curves
    (computation-only, Requirements 3.1, 5.1) — task 6.1
  - ``refine_with_cost``    : cost-instrumented refinement (task 6.3)
  - ``BenchmarkConfig``     : per-run config shared by all strategies (task 6.6)
  - ``run_strategy``        : per-strategy init-guess + multistart cost
    aggregation (Requirements 3.1, 3.3, 3.8, 3.12) — task 6.6
  - ``run_benchmark``       : orchestration of all four strategies on the same
    targets; writes ``warmstart_<model>.json`` + comparison figure; argparse CLI
    (Requirements 3.7, 3.9, 3.10, 3.11, 6.3, 6.4, 6.5) — task 6.9
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import minimize

# Make the repo-root kernel/helpers importable regardless of the caller's cwd
# (same pattern as validate_recovery.py).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pysim  # noqa: E402
import refine  # noqa: E402
import validation_common  # noqa: E402

# Number of time points per channel in a simulated curve set (pysim contract).
_NUM_RESULTS = pysim.NUM_RESULTS  # 7801
_NUM_PARAMS = len(validation_common.PARAM_NAMES)  # 7

# The four Initialization_Strategies evaluated by the Warm_Start_Experiment
# (Requirement 3.1): deep-learning prediction, a single uniform-random draw,
# Latin Hypercube multi-start, and the documented Reference_Parameters.
INIT_STRATEGIES = ('dl', 'random', 'lhs_multistart', 'reference')

# The documented Reference_Parameters Parameter_Set (Glossary / Requirement 3.1),
# keyed by pysim.PARAM_NAMES.
REFERENCE_PARAMETERS = {
    'E_b': -1.2,
    'E_b_azo_trans': -1.0,
    'E_b_azo_cis': -0.1,
    'k_mig': 0.05,
    'k0': 8e-6,
    'drt_z': 0.5,
    'drt_s': 0.05,
}


@dataclass
class RunResult:
    """Outcome and cost record for a single refinement run.

    Captures everything the Warm_Start_Experiment reports per run, all costs in
    units of ``Pysim_Call_Cost`` (forward-simulation calls), never wall-clock
    time (Requirements 3.4, 3.5, 3.6, 3.7, 3.8).

    Attributes:
        strategy: The Initialization_Strategy identifier (one of
            :data:`INIT_STRATEGIES`).
        reached_target: ``True`` when the run's running-best ``curve_rmse``
            reached the configured target Curve_RMSE (Requirement 3.4), else
            ``False`` (Requirement 3.5).
        cost_at_target: ``Pysim_Call_Cost`` read from the Call_Counter the first
            time the target was reached, or ``None`` when the target was never
            reached (Requirements 3.4, 3.5).
        total_cost: Total ``Pysim_Call_Cost`` expended by the run
            (Requirement 3.5); for ``lhs_multistart`` this aggregates every
            start (Requirement 3.8).
        converged_rmse: The converged Curve_RMSE for the run (Requirement 3.6).
        n_starts: The number of refinement starts recorded for the run; 1 for
            single-start strategies and the configured count for
            ``lhs_multistart`` (Requirement 3.8).
    """

    strategy: str
    reached_target: bool
    cost_at_target: Optional[int]
    total_cost: int
    converged_rmse: float
    n_starts: int = 1


def _derive_batch_seed(base_seed, batch_index):
    """Derive a deterministic per-batch integer seed from the base seed.

    Uses :class:`numpy.random.SeedSequence` keyed on ``(base_seed, batch_index)``
    so that (a) successive redraw batches get distinct, well-mixed seeds and
    (b) two runs with the *same* ``base_seed`` derive the *same* sequence of
    per-batch seeds. This makes the redraw-on-invalid loop reproducible under a
    fixed seed (Requirements 3.12, 6.1) and mirrors the helper used by
    ``validate_recovery.generate_ground_truth``.

    Args:
        base_seed: The validated base integer seed for target generation.
        batch_index: Zero-based index of the draw batch.

    Returns:
        int: A deterministic 32-bit seed for this batch's LHS draw.
    """
    seq = np.random.SeedSequence([int(base_seed), int(batch_index)])
    return int(seq.generate_state(1)[0])


def make_target_curves(n_targets, ranges, seed):
    """Generate ``n_targets`` valid target curve sets via LHS + ``pysim``.

    Draws Parameter_Sets via :func:`validation_common.lhs_params`, forward-
    simulates each through :func:`pysim.run_simulation`, and keeps only the
    samples whose simulation is valid (``dt_used >= 0``). Invalid samples
    (``dt_used < 0``) are discarded and additional batches are drawn until
    ``n_targets`` valid targets have been collected (redraw-on-invalid). Every
    target curve is generated purely by the DNA_Walker_Simulator, so the
    experiment is computation-only (Requirements 3.1, 5.1).

    Sampling proceeds in deterministic batches whose seeds are derived from
    ``seed`` via :func:`_derive_batch_seed`, so repeating the call with the same
    ``seed`` reproduces the identical set of targets (Requirements 3.12, 6.1).

    Args:
        n_targets: Number of valid target curve sets to generate (a positive
            integer).
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` keyed by
            ``pysim.PARAM_NAMES``.
        seed: Base integer seed for sampling. Validated via
            :func:`validation_common.require_seed`; ``None``/non-int raises
            :class:`validation_common.MissingSeedError` before any draw
            (Requirement 6.2).

    Returns:
        list[tuple[dict, np.ndarray]]: A list of ``n_targets`` ``(truth_params,
        curves)`` pairs, where ``truth_params`` is a dict keyed by
        ``pysim.PARAM_NAMES`` and ``curves`` is an ``np.ndarray`` of shape
        ``(3, 7801)``.
    """
    # Refuse the stochastic step without a valid seed (Requirement 6.2),
    # before any drawing occurs.
    base_seed = validation_common.require_seed(seed, "make_target_curves")

    n_targets = int(n_targets)

    targets = []
    batch_index = 0

    while len(targets) < n_targets:
        # Draw a batch sized to the remaining shortfall; this keeps the draw
        # count tight while still redrawing as many batches as needed.
        batch_size = n_targets - len(targets)

        batch_seed = _derive_batch_seed(base_seed, batch_index)
        params_batch = validation_common.lhs_params(batch_size, ranges, batch_seed)
        batch_index += 1

        for row in params_batch:
            signals, dt_used = pysim.run_simulation(row)
            if dt_used >= 0:
                truth_params = {
                    name: float(row[i])
                    for i, name in enumerate(validation_common.PARAM_NAMES)
                }
                curves = np.asarray(signals, dtype=np.float64)
                targets.append((truth_params, curves))
                if len(targets) >= n_targets:
                    break

    return targets


def refine_with_cost(init_params, target_curves, ranges, method='Powell',
                     maxiter=400, target_rmse=0.02, strategy='dl'):
    """Refine ``init_params`` while measuring cost in ``Pysim_Call_Cost``.

    Runs ``scipy.optimize.minimize`` with the **same configuration as**
    :func:`refine.refine` — identical method options (Powell uses ``bounds`` +
    ``xtol``/``ftol``; Nelder-Mead uses ``xatol``/``fatol``), identical
    ``maxiter``, the same ``bounds`` = Configured_Ranges, ``k0`` optimized in
    log10 space via ``refine.LOG_PARAMS``, the same
    ``refine._to_opt_space``/``refine._from_opt_space`` transforms, and the same
    out-of-bounds soft-penalty objective (clip to bounds, add
    ``sum(|x - clip(x)|) * 10``). The optimizer is driven here, rather than via
    :func:`refine.refine` directly, so the experiment has per-evaluation
    visibility into the running-best Curve_RMSE — which it needs to detect the
    first target crossing (Requirement 3.4) and to read the Call_Counter at that
    instant (Requirement 3.3). The actual RMSE is computed by
    :func:`refine.curve_rmse` so the numerics match the production refiner
    exactly.

    Cost accounting (all in units of ``Pysim_Call_Cost``):

      - The Call_Counter is reset to 0 at the **start** of the run via
        :func:`pysim.reset_call_count` (Requirement 3.3).
      - Like :func:`refine.refine`, the initial guess is evaluated once up front,
        and every subsequent objective evaluation calls
        :func:`refine.curve_rmse` exactly once, which calls
        :func:`pysim.run_simulation` exactly once and increments the counter.
      - After each evaluation the running-best Curve_RMSE is updated; the
        **first** time the running-best is ``<= target_rmse`` the current
        :func:`pysim.get_call_count` is recorded as ``cost_at_target``
        (Requirement 3.4).
      - Evaluations whose simulation is invalid (``dt_used < 0``) are still
        counted by the Call_Counter, and :func:`refine.curve_rmse` returns
        ``inf`` for them, so they can never satisfy the target
        (Requirement 3.13).
      - At the end, ``total_cost`` is read from :func:`pysim.get_call_count`
        (Requirement 3.5) and ``converged_rmse`` is the best Curve_RMSE found
        across the whole run (Requirement 3.6).

    Args:
        init_params: Initial-guess Parameter_Set as a dict keyed by
            ``pysim.PARAM_NAMES``.
        target_curves: Target curve set of shape ``(3, 7801)`` to fit.
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` used as the
            refinement bounds, keyed by ``pysim.PARAM_NAMES``.
        method: Refiner method, ``'Powell'`` (default) or ``'Nelder-Mead'``;
            mirrors :func:`refine.refine` option handling.
        maxiter: Maximum optimizer iterations (Requirement 3.2); a positive int.
        target_rmse: Configured target Curve_RMSE; a finite real ``> 0``
            (Requirement 3.4).
        strategy: Initialization_Strategy label recorded on the returned
            :class:`RunResult`. Callers (e.g. :func:`run_strategy`, task 6.6)
            set/override this to the strategy under test.

    Returns:
        RunResult: With ``reached_target`` (whether the target was ever met),
        ``cost_at_target`` (the ``Pysim_Call_Cost`` at first crossing, or
        ``None``), ``total_cost`` (total ``Pysim_Call_Cost`` expended),
        ``converged_rmse`` (best Curve_RMSE found), and ``n_starts == 1``.
    """
    # Reset the Call_Counter to 0 at the START of the run (Requirement 3.3) so
    # total_cost reflects only the forward simulations performed by this run.
    pysim.reset_call_count()

    # Mirror refine.refine's optimization-space setup exactly: bounds from the
    # Configured_Ranges (k0 in log10 space via refine.LOG_PARAMS), the initial
    # vector, and per-coordinate lo/hi for the soft-penalty clip.
    bounds = refine._opt_bounds(ranges)
    x0 = refine._to_opt_space(init_params, ranges)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    # Running-best Curve_RMSE and the Call_Counter reading at first target
    # crossing. Held in a dict so the nested objective can mutate them.
    state = {
        'best_rmse': float('inf'),
        'cost_at_target': None,
    }

    def _track(rmse):
        """Update the running-best and record cost at first target crossing."""
        if rmse < state['best_rmse']:
            state['best_rmse'] = rmse
        # First time the running-best meets the target, snapshot the cost.
        # An invalid simulation yields rmse == inf (Requirement 3.13), which can
        # never be <= target_rmse, so it never triggers a crossing.
        if state['cost_at_target'] is None and state['best_rmse'] <= target_rmse:
            state['cost_at_target'] = pysim.get_call_count()

    # Evaluate the initial guess once up front, mirroring refine.refine's
    # init_rmse computation. This is a real (counted) forward simulation and
    # seeds the running-best, so a warm start that already meets the target is
    # captured with a small cost_at_target.
    init_rmse = refine.curve_rmse(init_params, target_curves)
    _track(init_rmse)

    def objective(x):
        # Same out-of-bounds soft penalty as refine.refine: clip into bounds,
        # penalize the excursion, evaluate Curve_RMSE on the clipped params.
        xc = np.clip(x, lo, hi)
        penalty = np.sum(np.abs(x - xc)) * 10.0
        params = refine._from_opt_space(xc)
        rmse = refine.curve_rmse(params, target_curves)  # one counted pysim call
        _track(rmse)
        return rmse + penalty

    # Identical option handling to refine.refine: Nelder-Mead uses xatol/fatol
    # and no bounds arg; Powell uses xtol/ftol and passes bounds.
    opts = {'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False}
    if method == 'Powell':
        opts = {'maxiter': maxiter, 'xtol': 1e-5, 'ftol': 1e-6, 'disp': False}

    minimize(objective, x0, method=method,
             bounds=list(zip(lo, hi)) if method == 'Powell' else None,
             options=opts)

    total_cost = pysim.get_call_count()
    converged_rmse = state['best_rmse']
    reached_target = state['cost_at_target'] is not None

    return RunResult(
        strategy=strategy,
        reached_target=reached_target,
        cost_at_target=state['cost_at_target'],
        total_cost=int(total_cost),
        converged_rmse=float(converged_rmse),
        n_starts=1,
    )


@dataclass
class BenchmarkConfig:
    """Per-run configuration shared by every Initialization_Strategy.

    A small, documented config object carrying everything :func:`run_strategy`
    needs to drive :func:`refine_with_cost` identically across the four
    strategies (Requirement 3.2) plus the seed for the randomized strategies
    (Requirement 3.12) and the LHS multi-start count (Requirement 3.8).

    :func:`run_strategy` also accepts any object exposing these attributes, or a
    plain ``dict`` carrying these keys, via :func:`_cfg_get`; this dataclass is
    simply the canonical shape.

    Attributes:
        method: Refiner method passed to :func:`refine_with_cost`
            (``'Powell'`` default or ``'Nelder-Mead'``); identical across
            strategies (Requirement 3.2).
        maxiter: Maximum optimizer iterations; identical across strategies and a
            configurable positive int defaulting to 400 (Requirement 3.2).
        target_rmse: Configured target Curve_RMSE; a finite real ``> 0``
            defaulting to 0.02 (Requirement 3.4), identical across strategies.
        n_starts: Number of Latin Hypercube starts for ``'lhs_multistart'``; a
            configurable positive int defaulting to 10 (Requirement 3.8). Ignored
            by the single-start strategies.
        seed: Fixed, recorded integer seed for the ``'random'`` and
            ``'lhs_multistart'`` strategies (Requirement 3.12); validated via
            :func:`validation_common.require_seed`.
    """

    method: str = 'Powell'
    maxiter: int = 400
    target_rmse: float = 0.02
    n_starts: int = 10
    seed: int = 42


def _cfg_get(cfg, key, default=None):
    """Read ``key`` from a :class:`BenchmarkConfig`, namespace, or dict ``cfg``.

    Lets :func:`run_strategy` accept any of the config shapes the task allows: a
    :class:`BenchmarkConfig` (or any object with the attribute) via
    ``getattr``, or a plain mapping via ``__getitem__``.

    Args:
        cfg: The configuration carrier (dataclass/namespace/object or dict).
        key: The configuration field name to read.
        default: Value returned when the field is absent.

    Returns:
        The configured value for ``key``, or ``default`` if not present.
    """
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _vec_to_params(vec):
    """Convert a physical 7-vector to a Parameter_Set dict keyed by name.

    Mirrors the production ``predict_refine.py`` convention
    (``{pysim.PARAM_NAMES[i]: float(params[i])}``) so the init guess handed to
    :func:`refine_with_cost` is keyed by ``pysim.PARAM_NAMES`` in canonical
    order.

    Args:
        vec: A length-7 array-like in ``pysim.PARAM_NAMES`` order.

    Returns:
        dict: ``{param_name: float(value)}`` for the seven trainable params.
    """
    vec = np.asarray(vec, dtype=np.float64).ravel()
    return {
        name: float(vec[i])
        for i, name in enumerate(validation_common.PARAM_NAMES)
    }


def run_strategy(strategy, predictor, target, ranges, cfg):
    """Build the per-strategy init guess and refine it, measuring cost.

    Produces the initial guess for the requested Initialization_Strategy
    (Requirement 3.1), then drives :func:`refine_with_cost` with the **same**
    refiner method, ``maxiter``, bounds (``ranges`` = Configured_Ranges) and
    ``target_rmse`` for every strategy (Requirement 3.2). The Call_Counter is
    reset per refinement run inside :func:`refine_with_cost` (Requirement 3.3).

    Strategy init guesses:

      - ``'dl'``: feed the target curve set to ``predictor.predict`` and take
        row 0 of the returned physical ``(N, 7)`` array (``predict`` accepts a
        single ``(3, T)`` curve set; see ``train_cnn/predict_refine.py``), then
        refine from that 7-vector. ``n_starts == 1``.
      - ``'random'``: a single uniform-random draw within the Configured_Ranges
        via :func:`validation_common.lhs_params` (``n == 1``), gated by
        :func:`validation_common.require_seed` on ``cfg.seed``
        (Requirements 3.1, 3.12). ``n_starts == 1``.
      - ``'lhs_multistart'``: draw ``cfg.n_starts`` LHS starts within the
        Configured_Ranges (gated by :func:`validation_common.require_seed`),
        refine from each, and **aggregate** the result across **all** starts
        (Requirement 3.8): ``total_cost`` is the sum of every start's
        ``Pysim_Call_Cost`` (each :func:`refine_with_cost` call resets the
        counter, so summing reconstructs the cumulative cost), ``converged_rmse``
        is the best (minimum) across starts, ``reached_target`` is true if *any*
        start reached, and ``cost_at_target`` is the **cumulative** cost at the
        first start that reached the target — i.e. the sum of the prior starts'
        ``total_cost`` plus that start's own ``cost_at_target`` (or ``None`` if
        none reached). ``n_starts == cfg.n_starts``.
      - ``'reference'``: refine from the documented :data:`REFERENCE_PARAMETERS`.
        ``n_starts == 1``.

    Args:
        strategy: One of :data:`INIT_STRATEGIES`.
        predictor: An Inverse_Model predictor exposing ``predict(X) -> (N, 7)``;
            used only by the ``'dl'`` strategy (may be ``None`` otherwise).
        target: Target curve set of shape ``(3, 7801)`` to fit; the same target
            is used across strategies by the caller (Requirement 3.9).
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` used both for
            sampling init guesses and as the refinement bounds.
        cfg: A :class:`BenchmarkConfig` (or any object/dict exposing ``method``,
            ``maxiter``, ``target_rmse``, ``n_starts``, ``seed``).

    Returns:
        RunResult: Labeled with ``strategy``; ``n_starts == 1`` for the
        single-start strategies and ``cfg.n_starts`` for ``'lhs_multistart'``.

    Raises:
        ValueError: If ``strategy`` is not one of :data:`INIT_STRATEGIES`.
        validation_common.MissingSeedError: If a randomized strategy is run
            without a valid integer ``cfg.seed``.
    """
    if strategy not in INIT_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {strategy!r}; expected one of {INIT_STRATEGIES}."
        )

    method = _cfg_get(cfg, 'method', 'Powell')
    maxiter = _cfg_get(cfg, 'maxiter', 400)
    target_rmse = _cfg_get(cfg, 'target_rmse', 0.02)

    if strategy == 'dl':
        # DL warm start: predict on the (3, 7801) target, take the first row of
        # the (N, 7) physical-space output as the init guess (predict_refine.py).
        preds = np.asarray(predictor.predict(target), dtype=np.float64)
        init_params = _vec_to_params(preds[0])
        return refine_with_cost(
            init_params, target, ranges, method=method,
            maxiter=maxiter, target_rmse=target_rmse, strategy='dl',
        )

    if strategy == 'reference':
        # Documented Reference_Parameters as the init guess (no randomness).
        return refine_with_cost(
            dict(REFERENCE_PARAMETERS), target, ranges, method=method,
            maxiter=maxiter, target_rmse=target_rmse, strategy='reference',
        )

    if strategy == 'random':
        # Single uniform-random draw within Configured_Ranges. Refuse to draw
        # without a recorded seed (Requirement 3.12).
        seed = validation_common.require_seed(
            _cfg_get(cfg, 'seed'), "run_strategy:random")
        draw = validation_common.lhs_params(1, ranges, seed)[0]
        init_params = _vec_to_params(draw)
        return refine_with_cost(
            init_params, target, ranges, method=method,
            maxiter=maxiter, target_rmse=target_rmse, strategy='random',
        )

    # strategy == 'lhs_multistart'
    # Refuse to draw without a recorded seed (Requirement 3.12).
    seed = validation_common.require_seed(
        _cfg_get(cfg, 'seed'), "run_strategy:lhs_multistart")
    n_starts = int(_cfg_get(cfg, 'n_starts', 10))

    starts = validation_common.lhs_params(n_starts, ranges, seed)

    aggregate_total_cost = 0          # sum of every start's Pysim_Call_Cost (3.8)
    best_rmse = float('inf')          # best converged Curve_RMSE across starts
    cost_at_target = None             # cumulative cost at first start to reach
    reached_target = False

    for row in starts:
        init_params = _vec_to_params(row)
        # refine_with_cost resets the Call_Counter per call, so each result's
        # total_cost is this start's own cost; we sum to get the cumulative.
        result = refine_with_cost(
            init_params, target, ranges, method=method,
            maxiter=maxiter, target_rmse=target_rmse, strategy='lhs_multistart',
        )

        # Record the cumulative cost at the FIRST start that reached the target,
        # expressed relative to the running cumulative sum across starts.
        if result.reached_target and cost_at_target is None:
            cost_at_target = aggregate_total_cost + result.cost_at_target
            reached_target = True

        aggregate_total_cost += result.total_cost

        if result.converged_rmse < best_rmse:
            best_rmse = result.converged_rmse

    return RunResult(
        strategy='lhs_multistart',
        reached_target=reached_target,
        cost_at_target=cost_at_target,
        total_cost=int(aggregate_total_cost),
        converged_rmse=float(best_rmse),
        n_starts=n_starts,
    )


# ---------------------------------------------------------------------------
# task 6.9: run_benchmark(...) — orchestrate all four strategies on the same
#   targets; write warmstart_<model>.json + comparison figure; argparse CLI.
# ---------------------------------------------------------------------------

# Output artifact names; the model tag is filled in by run_benchmark so the CNN
# and Transformer studies write side-by-side without clobbering each other.
_JSON_NAME_FMT = "warmstart_{model}.json"
_FIGURE_NAME_FMT = "warmstart_{model}.png"


def _summarize_strategy(results):
    """Summarize one strategy's per-target :class:`RunResult` list for the JSON.

    Builds the per-strategy block of the warm-start JSON data model (design.md):
    the recorded ``n_starts``, the reached-target count and rate, and the
    :func:`validation_common.distribution_stats` distributions of
    ``cost_at_target`` (over the runs that actually reached the target),
    ``total_cost``, and ``converged_rmse`` across all targets. All costs are
    integer ``Pysim_Call_Cost`` values (Requirement 3.7); the summary statistics
    are the standard mean/median/std/min/max/p5/p95 dict.

    Args:
        results: List of :class:`RunResult`, one per target, for a single
            Initialization_Strategy.

    Returns:
        dict: ``{n_starts, reached_target_count, reached_target_rate,
        cost_at_target, total_cost, converged_rmse}``. ``cost_at_target`` is
        ``None`` when no run reached the target (no values to summarize);
        otherwise it is a ``distribution_stats`` dict.
    """
    n_targets = len(results)
    n_starts = results[0].n_starts if results else 1

    reached = [r for r in results if r.reached_target]
    reached_count = len(reached)

    # cost_at_target distribution is only defined over runs that reached the
    # target (Requirement 3.4/3.5: it is None for runs that never reached).
    cost_at_target_values = [r.cost_at_target for r in reached]
    cost_at_target_stats = (
        validation_common.distribution_stats(cost_at_target_values)
        if cost_at_target_values else None
    )

    total_cost_values = [r.total_cost for r in results]
    converged_rmse_values = [r.converged_rmse for r in results]

    return {
        "n_starts": int(n_starts),
        "reached_target_count": int(reached_count),
        "reached_target_rate": (reached_count / n_targets) if n_targets else 0.0,
        "cost_at_target": cost_at_target_stats,
        "total_cost": validation_common.distribution_stats(total_cost_values),
        "converged_rmse": validation_common.distribution_stats(converged_rmse_values),
    }


def _plot_benchmark_compare(per_strategy_results, model, out_path):
    """Write the labeled per-strategy comparison figure (Requirements 3.11, 6.6).

    Renders two side-by-side panels comparing the four Initialization_Strategies:

      - Left panel: the per-strategy ``total_cost`` distribution (box plot),
        with the y-axis labeled in ``Pysim_Call_Cost`` (forward-simulation
        calls) — never wall-clock time (Requirements 3.7, 3.11, 6.5).
      - Right panel: the per-strategy converged ``Curve_RMSE`` distribution
        (box plot), with the y-axis labeled as the dimensionless Curve_RMSE.

    Both axes name their quantity and unit so the figure is interpretable
    without the source code (Requirement 6.6). The non-interactive ``Agg``
    backend is selected so the figure renders headlessly (Requirement 6.4).

    Args:
        per_strategy_results: Mapping ``strategy -> list[RunResult]`` (one entry
            per target) for each of :data:`INIT_STRATEGIES`.
        model: The Inverse_Model tag (``'cnn'``/``'transformer'``) for the title.
        out_path: Destination PNG path.

    Returns:
        str: ``out_path``.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless rendering (Requirement 6.4)
    import matplotlib.pyplot as plt

    strategies = list(INIT_STRATEGIES)
    cost_data = [
        [r.total_cost for r in per_strategy_results[s]] for s in strategies
    ]
    rmse_data = [
        [r.converged_rmse for r in per_strategy_results[s]] for s in strategies
    ]

    fig, (ax_cost, ax_rmse) = plt.subplots(1, 2, figsize=(12.0, 5.0))

    positions = range(1, len(strategies) + 1)

    ax_cost.boxplot(cost_data, positions=list(positions), showmeans=True)
    ax_cost.set_xticks(list(positions))
    ax_cost.set_xticklabels(strategies, rotation=20, ha="right")
    ax_cost.set_xlabel("Initialization strategy")
    # Axis label names the quantity + unit (Requirement 6.6): cost is measured in
    # forward-simulation calls (Pysim_Call_Cost), not wall-clock time (Req 6.5).
    ax_cost.set_ylabel("Pysim_Call_Cost (forward-simulation calls)")
    ax_cost.set_title("Total optimization cost per strategy")

    ax_rmse.boxplot(rmse_data, positions=list(positions), showmeans=True)
    ax_rmse.set_xticks(list(positions))
    ax_rmse.set_xticklabels(strategies, rotation=20, ha="right")
    ax_rmse.set_xlabel("Initialization strategy")
    # Curve_RMSE is a dimensionless RMSE over normalized fluorescence curves.
    ax_rmse.set_ylabel("Converged Curve_RMSE (dimensionless fluorescence-fraction RMSE)")
    ax_rmse.set_title("Converged Curve_RMSE per strategy")

    fig.suptitle(
        f"Warm-start ablation ({model}): Pysim_Call_Cost and Curve_RMSE by strategy",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_benchmark(model='cnn', n_targets=50, target_rmse=0.02, maxiter=400,
                  method='Powell', n_starts=10, seed=42,
                  results_dir='results/validation', predictor=None,
                  predictor_overrides=None):
    """Run the Warm_Start_Experiment over all four strategies and write outputs.

    Generates ``n_targets`` valid target curve sets once via
    :func:`make_target_curves` and evaluates **all four**
    Initialization_Strategies (:data:`INIT_STRATEGIES`) on **each** target
    (Requirement 3.9: every strategy sees identical inputs). For each target a
    :class:`BenchmarkConfig` carrying the shared ``method``/``maxiter``/
    ``target_rmse``/``n_starts`` and a per-target seed drives
    :func:`run_strategy`, so the refiner configuration is identical across
    strategies (Requirement 3.2).

    Each target is evaluated with a deterministic per-target seed derived from
    ``seed`` via :func:`_derive_batch_seed`, so the randomized strategies
    (``'random'`` / ``'lhs_multistart'``) draw different (but fully reproducible)
    initial guesses per target while the whole run is reproducible from the base
    ``seed`` (Requirements 3.12, 6.1).

    The per-strategy distributions of ``cost_at_target``, ``total_cost``,
    ``converged_rmse``, and the reached-target count/rate are summarized with
    :func:`validation_common.distribution_stats` and written to
    ``warmstart_<model>.json`` following the design.md warm-start JSON data model
    (Requirements 3.7, 3.10). All cost fields are integer ``Pysim_Call_Cost``
    values (Requirement 3.7). A labeled comparison figure
    ``warmstart_<model>.png`` (per-strategy Pysim_Call_Cost and Curve_RMSE
    distributions) is written alongside it (Requirements 3.11, 6.4).

    The deep-learning predictor used by the ``'dl'`` strategy is built lazily via
    :func:`validation_common.make_predictor` — only when not supplied — so that
    the heavy ``torch`` import is incurred only when an actual DL run is needed.

    Args:
        model: Inverse_Model to warm-start from: ``'cnn'`` or ``'transformer'``
            (Requirement 3.1). Also tags the output artifacts.
        n_targets: Number of target curve sets to evaluate every strategy on; a
            positive int defaulting to 50 (Requirement 3.9).
        target_rmse: Configured target Curve_RMSE; a finite real ``> 0``
            defaulting to 0.02 (Requirement 3.4).
        maxiter: Maximum refiner iterations, identical across strategies; a
            positive int defaulting to 400 (Requirement 3.2).
        method: Refiner method (``'Powell'`` default or ``'Nelder-Mead'``),
            identical across strategies (Requirement 3.2).
        n_starts: Number of Latin Hypercube starts for ``'lhs_multistart'``; a
            positive int defaulting to 10 (Requirement 3.8).
        seed: Fixed, recorded base integer seed for sampling targets and for the
            randomized strategies (Requirements 3.12, 6.1). Validated via
            :func:`validation_common.require_seed`.
        results_dir: Results_Directory for the JSON + figure artifacts.
        predictor: Optional pre-built Inverse_Model predictor exposing
            ``predict(X) -> (N, 7)``. When ``None`` (the default) it is built via
            :func:`validation_common.make_predictor(model, **predictor_overrides)`.
            Supplying it lets callers inject a custom/fake predictor (e.g. for a
            lightweight dry run) without importing ``torch``.
        predictor_overrides: Optional dict of constructor keyword overrides
            forwarded to :func:`validation_common.make_predictor` (e.g.
            ``model_path``/``config`` overrides) when ``predictor`` is ``None``.

    Returns:
        dict: The full metrics object written to ``warmstart_<model>.json``.
    """
    # Validate the base seed up front (Requirement 6.2) before any work; the
    # randomized strategies and target sampling are all stochastic steps.
    base_seed = validation_common.require_seed(seed, "run_benchmark")

    ranges = validation_common.load_configured_ranges()

    # Build the DL predictor lazily (only when not injected) so importing /
    # dry-running this module never requires torch unless a real DL run is asked.
    if predictor is None:
        predictor = validation_common.make_predictor(
            model, **(predictor_overrides or {})
        )

    # Generate the target curve sets ONCE; the same set is fed to every strategy
    # (Requirement 3.9). Sampling uses the base seed (reproducible, Req. 6.1).
    targets = make_target_curves(n_targets, ranges, base_seed)

    # Collect per-strategy RunResult lists (one entry per target) and the
    # per-target breakdown for the JSON's "per_target" section.
    per_strategy_results = {s: [] for s in INIT_STRATEGIES}
    per_target = []

    for target_index, (truth_params, curves) in enumerate(targets):
        # Per-target seed: distinct per target yet reproducible from base_seed
        # (Requirements 3.12, 6.1). Randomized strategies read cfg.seed.
        target_seed = _derive_batch_seed(base_seed, target_index)
        cfg = BenchmarkConfig(
            method=method,
            maxiter=maxiter,
            target_rmse=target_rmse,
            n_starts=n_starts,
            seed=target_seed,
        )

        target_entry = {"target_index": target_index}
        for strategy in INIT_STRATEGIES:
            # Same target `curves` handed to every strategy (Requirement 3.9).
            result = run_strategy(strategy, predictor, curves, ranges, cfg)
            per_strategy_results[strategy].append(result)
            target_entry[strategy] = {
                "reached_target": bool(result.reached_target),
                "cost_at_target": result.cost_at_target,
                "total_cost": int(result.total_cost),
                "converged_rmse": float(result.converged_rmse),
            }
        per_target.append(target_entry)

    # Assemble the metrics object per the design.md warm-start JSON data model.
    metrics = {
        "experiment": "warm_start_ablation",
        "model": model,
        "seed": base_seed,
        "config": {
            "n_targets": int(n_targets),
            "target_rmse": float(target_rmse),
            "refine_method": method,
            "maxiter": int(maxiter),
            "n_starts": int(n_starts),
            "cost_unit": "pysim_call_cost",  # costs are Pysim_Call_Cost (Req 3.7, 6.5)
        },
        "strategies": {
            strategy: _summarize_strategy(per_strategy_results[strategy])
            for strategy in INIT_STRATEGIES
        },
        "per_target": per_target,
    }

    # Write JSON (Requirements 3.10, 6.3) and the labeled comparison figure
    # (Requirements 3.11, 6.4) into the Results_Directory.
    json_path = os.path.join(results_dir, _JSON_NAME_FMT.format(model=model))
    validation_common.write_json(json_path, metrics)

    figure_path = os.path.join(results_dir, _FIGURE_NAME_FMT.format(model=model))
    _plot_benchmark_compare(per_strategy_results, model, figure_path)

    return metrics


def main(argv=None):
    """CLI entry point for the Warm_Start_Experiment (Requirements 3.1-3.13).

    Flags:
      ``--model``        Inverse_Model to warm-start from (``cnn``/``transformer``)
      ``--targets``      number of target curve sets (default: 50)
      ``--target-rmse``  target Curve_RMSE (default: 0.02)
      ``--maxiter``      max refiner iterations (default: 400)
      ``--method``       refiner method (``Powell``/``Nelder-Mead``)
      ``--starts``       LHS multi-start count (default: 10)
      ``--seed``         fixed, recorded base seed (default: 42)
      ``--results-dir``  Results_Directory (default: results/validation)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Warm-start vs. baseline ablation: evaluate four initialization "
            "strategies under an identical refiner on the same target curves, "
            "reporting cost in Pysim_Call_Cost (forward-simulation calls)."
        )
    )
    parser.add_argument(
        "--model", default="cnn", choices=("cnn", "transformer"),
        help="Inverse model to warm-start from (default: cnn).",
    )
    parser.add_argument(
        "--targets", type=int, default=50,
        help="Number of target curve sets evaluated by every strategy (default: 50).",
    )
    parser.add_argument(
        "--target-rmse", type=float, default=0.02,
        help="Target Curve_RMSE that defines convergence (default: 0.02).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=400,
        help="Maximum refiner iterations, identical across strategies (default: 400).",
    )
    parser.add_argument(
        "--method", default="Powell", choices=("Powell", "Nelder-Mead"),
        help="Refiner method, identical across strategies (default: Powell).",
    )
    parser.add_argument(
        "--starts", type=int, default=10,
        help="Latin Hypercube multi-start count (default: 10).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Fixed, recorded base random seed (default: 42).",
    )
    parser.add_argument(
        "--results-dir", default="results/validation",
        help="Results_Directory for the JSON and figure (default: results/validation).",
    )
    args = parser.parse_args(argv)

    metrics = run_benchmark(
        model=args.model,
        n_targets=args.targets,
        target_rmse=args.target_rmse,
        maxiter=args.maxiter,
        method=args.method,
        n_starts=args.starts,
        seed=args.seed,
        results_dir=args.results_dir,
    )

    json_path = os.path.join(
        args.results_dir, _JSON_NAME_FMT.format(model=args.model))
    figure_path = os.path.join(
        args.results_dir, _FIGURE_NAME_FMT.format(model=args.model))
    print(f"Wrote {json_path}")
    print(f"Wrote {figure_path}")
    for strategy in INIT_STRATEGIES:
        block = metrics["strategies"][strategy]
        print(
            f"  {strategy}: reached={block['reached_target_count']}/"
            f"{args.targets} total_cost_mean={block['total_cost']['mean']:.1f}"
        )
    return metrics


if __name__ == "__main__":
    main()
