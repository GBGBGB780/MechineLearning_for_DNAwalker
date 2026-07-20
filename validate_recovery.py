# coding=utf-8
"""
validate_recovery.py — Synthetic parameter-recovery experiment (Component 3).

This script draws ground-truth Parameter_Sets via Latin Hypercube sampling,
forward-simulates them through the DNA_Walker_Simulator (``pysim``), and (in
later tasks) runs the predict+refine pipeline to measure how accurately the
seven physical parameters are recovered from the three fluorescence curves.

Scope grows task-by-task. Currently implemented:
  - ``generate_ground_truth`` : LHS draw + forward-simulate with
    redraw-on-invalid up to a draw budget (Requirements 2.2, 2.3, 2.10, 2.11,
    2.13).
  - ``add_gaussian_noise``    : zero-mean per-time-point Gaussian noise added
    to the three channels of a curve set (Requirement 2.4).
  - ``estimate_parameters``   : the predict+refine pipeline on one curve set —
    DL predict -> refine (+ optional multistart), returning the estimated
    Parameter_Set dict (Requirement 2.5).
  - ``compute_recovery_metrics`` : per-parameter relative-error distribution,
    summary stats, and R² (Requirements 2.6, 2.7, 2.9, 2.12).
  - ``plot_scatter``          : one true-vs-predicted scatter per parameter
    (Requirements 2.8, 6.6).
  - ``run_recovery`` + CLI    : the end-to-end orchestration wiring sampling ->
    optional noise -> estimation -> metrics -> JSON + 7 scatter plots, plus the
    ``argparse`` command-line entry point (Requirements 2.1, 2.9, 2.11, 2.13,
    5.1, 6.3).
"""

import argparse
import os
import sys

import numpy as np

# Make the repo-root kernel/helpers importable regardless of the caller's cwd.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pysim  # noqa: E402
import refine  # noqa: E402
import validation_common  # noqa: E402

# Number of time points per channel in a simulated curve set (pysim contract).
_NUM_RESULTS = pysim.NUM_RESULTS  # 7801
_NUM_PARAMS = len(validation_common.PARAM_NAMES)  # 7


def _derive_batch_seed(base_seed, batch_index):
    """Derive a deterministic per-batch integer seed from the base seed.

    Uses :class:`numpy.random.SeedSequence` keyed on ``(base_seed, batch_index)``
    so that (a) successive batches within one run get distinct, well-mixed seeds
    and (b) two runs with the *same* ``base_seed`` derive the *same* sequence of
    per-batch seeds. This is what makes the redraw-on-invalid loop reproducible
    under a fixed seed (Requirements 2.10, 6.1).

    Args:
        base_seed: The validated base integer seed for the experiment.
        batch_index: Zero-based index of the draw batch.

    Returns:
        int: A deterministic 32-bit seed for this batch's LHS draw.
    """
    seq = np.random.SeedSequence([int(base_seed), int(batch_index)])
    return int(seq.generate_state(1)[0])


def _derive_sample_seed(base_seed, tag, sample_index):
    """Derive a deterministic per-sample integer seed for a stochastic substep.

    Used by :func:`run_recovery` to give each retained sample its own, fully
    reproducible seed for the two per-sample stochastic substeps — Gaussian
    noise injection (``tag=1``) and the multi-start estimation jitter
    (``tag=2``) — without those substeps sharing or colliding with each other or
    with the sampling seeds derived by :func:`_derive_batch_seed`. Keying the
    :class:`numpy.random.SeedSequence` on ``(base_seed, tag, sample_index)``
    makes a repeated :func:`run_recovery` call with the same ``seed`` reproduce
    identical noise and jitter for every sample (Requirements 2.10, 6.1).

    Args:
        base_seed: The validated base integer seed for the experiment.
        tag: A small integer identifying the substep (``1`` = noise,
            ``2`` = estimation jitter).
        sample_index: Zero-based index of the retained sample.

    Returns:
        int: A deterministic 32-bit seed for this sample's substep.
    """
    seq = np.random.SeedSequence([int(base_seed), int(tag), int(sample_index)])
    return int(seq.generate_state(1)[0])


def generate_ground_truth(n, ranges, seed, max_total_draws=None):
    """Draw and forward-simulate ground-truth samples, redrawing on invalid.

    Draws Parameter_Sets via :func:`validation_common.lhs_params`, forward-
    simulates each one through :func:`pysim.run_simulation`, and keeps the
    samples whose simulation is valid (``dt_used >= 0``). Samples that come back
    invalid (``dt_used < 0``) are excluded and counted (Requirement 2.3).

    Sampling proceeds in deterministic batches: a batch of up to ``n`` draws is
    taken, each draw is simulated, and retained/excluded accordingly. If fewer
    than ``n`` valid samples have been retained after a batch, additional
    batches are drawn — each with its own deterministically derived seed — until
    either ``n`` samples are retained or the cumulative number of draws reaches
    ``max_total_draws`` (Requirements 2.11, 2.13). Because every batch seed is
    derived from ``seed`` via :func:`_derive_batch_seed`, repeating the call with
    the same ``seed`` reproduces identical retained samples (Requirements 2.10,
    6.1).

    Args:
        n: Target number of valid samples to retain (a positive integer).
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` keyed by
            ``pysim.PARAM_NAMES``.
        seed: Base integer seed for sampling. Validated via
            :func:`validation_common.require_seed`; ``None``/non-int raises
            :class:`validation_common.MissingSeedError` before any draw.
        max_total_draws: Maximum cumulative number of Parameter_Sets to draw
            (the draw budget). Defaults to ``10 * n`` when ``None``
            (Requirement 2.11). The 4.10 caller passes this explicitly.

    Returns:
        tuple: ``(Y_true, X_curves, n_excluded_invalid, exhausted)`` where

          * ``Y_true``  — ``np.ndarray`` of shape ``(m, 7)`` of retained
            Parameter_Sets, ``m <= n``;
          * ``X_curves`` — ``np.ndarray`` of shape ``(m, 3, 7801)`` of the
            corresponding valid simulated curves;
          * ``n_excluded_invalid`` — ``int`` count of draws excluded for
            returning ``dt_used < 0``;
          * ``exhausted`` — ``bool`` that is ``True`` when the draw budget was
            reached before ``n`` samples could be retained (Requirement 2.13).
    """
    # Refuse the stochastic step without a valid seed (Requirement 6.2),
    # before any drawing occurs.
    base_seed = validation_common.require_seed(seed, "generate_ground_truth")

    n = int(n)
    if max_total_draws is None:
        max_total_draws = 10 * n  # Requirement 2.11 default
    max_total_draws = int(max_total_draws)

    retained_Y = []
    retained_X = []
    n_excluded_invalid = 0
    total_drawn = 0
    batch_index = 0

    while len(retained_Y) < n and total_drawn < max_total_draws:
        # Draw up to `n` per batch, but never exceed the remaining draw budget.
        remaining_budget = max_total_draws - total_drawn
        batch_size = min(n, remaining_budget)
        if batch_size <= 0:
            break

        batch_seed = _derive_batch_seed(base_seed, batch_index)
        params_batch = validation_common.lhs_params(batch_size, ranges, batch_seed)
        batch_index += 1

        for row in params_batch:
            total_drawn += 1
            signals, dt_used = pysim.run_simulation(row)
            if dt_used >= 0:
                retained_Y.append(np.asarray(row, dtype=np.float64))
                retained_X.append(np.asarray(signals, dtype=np.float64))
                if len(retained_Y) >= n:
                    break
            else:
                n_excluded_invalid += 1

    if retained_Y:
        Y_true = np.vstack(retained_Y)
        X_curves = np.stack(retained_X, axis=0)
    else:
        # No sample retained: return correctly shaped empty arrays.
        Y_true = np.empty((0, _NUM_PARAMS), dtype=np.float64)
        X_curves = np.empty((0, 3, _NUM_RESULTS), dtype=np.float64)

    # Budget exhausted before retaining the requested count (Requirement 2.13).
    exhausted = len(retained_Y) < n

    return Y_true, X_curves, n_excluded_invalid, exhausted


def add_gaussian_noise(curves, noise_std, seed):
    """Add zero-mean Gaussian noise independently per time point to each channel.

    Adds independent zero-mean Gaussian noise with standard deviation
    ``noise_std`` to every element of ``curves`` — that is, an independent draw
    per time point for each of the 3 channels (Requirement 2.4). The added noise
    is reproducible: repeating the call with the same ``seed`` yields byte-
    identical noise.

    The input is never mutated in place; a new array is returned.

    Args:
        curves: The simulated curve set to perturb, shape ``(3, 7801)``
            (3 channels × ``pysim.NUM_RESULTS`` time points). Array-like; it is
            converted to a ``float64`` copy before noise is added.
        noise_std: Standard deviation of the Gaussian noise. Must be a finite
            real value strictly greater than 0.0 (Requirement 2.4).
        seed: Integer seed for reproducible noise. Validated via
            :func:`validation_common.require_seed`; ``None``/non-int raises
            :class:`validation_common.MissingSeedError` before any sampling.

    Returns:
        np.ndarray: A new array of the same shape as ``curves`` with the noise
        added (``float64``).

    Raises:
        validation_common.MissingSeedError: If ``seed`` is ``None`` or not an
            ``int`` (raised before sampling).
        ValueError: If ``noise_std`` is non-finite or ``<= 0``.
    """
    # Refuse the stochastic step without a valid seed (Requirement 6.2),
    # before any sampling occurs.
    noise_seed = validation_common.require_seed(seed, "add_gaussian_noise")

    # Reject a degenerate / non-finite standard deviation (Requirement 2.4).
    std = float(noise_std)
    if not np.isfinite(std) or std <= 0.0:
        raise ValueError(
            "noise_std must be a finite real value > 0 "
            f"(got {noise_std!r})."
        )

    # Work on a fresh float64 copy so the caller's array is never mutated.
    base = np.array(curves, dtype=np.float64, copy=True)

    rng = np.random.default_rng(noise_seed)
    noise = rng.normal(loc=0.0, scale=std, size=base.shape)

    return base + noise


def estimate_parameters(predictor, curves, ranges, method='Powell', maxiter=400,
                        multistart=8, seed=None):
    """Estimate a Parameter_Set from one curve set via the Predict_Refine_Pipeline.

    Mirrors the production pipeline in ``train_cnn/predict_refine.py``: an
    Inverse_Model produces a deep-learning prediction, which seeds a local
    physics refinement (:func:`refine.refine`); an optional multi-start then
    jitters the deep-learning prediction and refines from each jittered start,
    keeping whichever refined result has the lowest Curve_RMSE (Requirement 2.5).

    Steps:
      1. **DL predict.** ``predictor.predict(curves)`` returns an ``(N, 7)``
         physical-space array (the predictor accepts a single ``(3, T)`` curve
         set or a batch ``(N, 3, T)``). The first row is taken as this sample's
         7-vector and turned into a ``dict`` keyed by ``pysim.PARAM_NAMES`` — the
         same ``params -> pdict`` construction used by ``predict_refine.py``.
      2. **Refine (main start).** ``refine.refine(pdict, curves, ranges,
         method=method, maxiter=maxiter)`` refines from the DL prediction and
         returns ``(refined_params, refined_rmse, init_rmse)``; the refined
         params/RMSE are taken as the running best.
      3. **Multi-start (optional).** When ``multistart > 0``, an RNG seeded via
         :func:`validation_common.require_seed` jitters the DL prediction by
         ``rng.normal(0, 0.1) * (hi - lo)`` per parameter, clipped to each
         parameter's bounds, and refines from that start. A start that achieves a
         strictly lower refined Curve_RMSE replaces the running best. This
         matches ``predict_refine.py``'s multistart loop.

    The seed guard is only consulted when ``multistart > 0`` because the
    multi-start jitter is the sole stochastic step here — DL prediction and the
    deterministic refinement need no seed (Requirement 6.2).

    Args:
        predictor: An Inverse_Model exposing ``predict(X) -> (N, 7)`` (e.g. the
            objects built by :func:`validation_common.make_predictor`).
        curves: One simulated curve set of shape ``(3, 7801)`` to invert.
        ranges: Configured_Ranges as ``{param_name: (min, max)}`` keyed by
            ``pysim.PARAM_NAMES``; used as refinement bounds and to scale the
            multi-start jitter.
        method: Refiner method passed through to :func:`refine.refine`
            (``'Powell'`` or ``'Nelder-Mead'``). Defaults to ``'Powell'``.
        maxiter: Maximum refiner iterations per start. Defaults to ``400``.
        multistart: Number of additional jittered starts to try beyond the main
            DL-seeded start. ``0`` (or ``None``) disables multi-start. Defaults
            to ``8``.
        seed: Integer seed for the multi-start jitter RNG. Required (validated
            via :func:`validation_common.require_seed`) only when
            ``multistart > 0``; ``None``/non-int then raises
            :class:`validation_common.MissingSeedError` before any jitter.

    Returns:
        dict: The estimated Parameter_Set — a 7-key dict keyed by
        ``pysim.PARAM_NAMES`` — corresponding to the refinement with the lowest
        Curve_RMSE across the main start and all multi-starts.

    Raises:
        validation_common.MissingSeedError: If ``multistart > 0`` and ``seed`` is
            ``None`` or not an ``int`` (raised before any jitter is drawn).
    """
    # 1. DL prediction -> physical 7-vector -> dict keyed by pysim.PARAM_NAMES
    #    (predict returns (N, 7); take the first row for this single curve set).
    params = predictor.predict(curves)[0]
    pdict = {pysim.PARAM_NAMES[i]: float(params[i]) for i in range(len(params))}

    # 2. Refine from the DL prediction (main start). refine returns
    #    (refined_params, refined_rmse, init_rmse); best == lowest curve RMSE.
    best_params, best_rmse, _ = refine.refine(
        pdict, curves, ranges, method=method, maxiter=maxiter, verbose=False
    )

    # 3. Optional multi-start: jitter the DL prediction, refine each, keep best.
    if multistart and multistart > 0:
        # The jitter is the only stochastic step -> require a seed here.
        jitter_seed = validation_common.require_seed(seed, "estimate_parameters")
        rng = np.random.default_rng(jitter_seed)
        for _ in range(int(multistart)):
            start = {}
            for nm in pysim.PARAM_NAMES:
                lo, hi = ranges[nm]
                start[nm] = float(
                    np.clip(pdict[nm] + rng.normal(0, 0.1) * (hi - lo), lo, hi)
                )
            rp, rr, _ = refine.refine(
                start, curves, ranges, method=method, maxiter=maxiter, verbose=False
            )
            if rr < best_rmse:
                best_rmse, best_params = rr, rp

    return best_params


def _as_param_array(Y):
    """Coerce ``Y`` into an ``(m, 7)`` float64 array keyed positionally to params.

    Accepts either:
      * an array-like of shape ``(m, 7)`` whose column ``i`` already corresponds
        to ``pysim.PARAM_NAMES[i]`` (the canonical/array form), or
      * a sequence of ``dict`` Parameter_Sets (e.g. the dicts returned by
        :func:`estimate_parameters`), each keyed by ``pysim.PARAM_NAMES``; these
        are stacked row-wise into the same positional ``(m, 7)`` layout.

    Picking the positional array form (column ``i`` == ``PARAM_NAMES[i]``) is the
    documented contract of :func:`compute_recovery_metrics`; this helper just lets
    the caller pass whichever of the two equivalent shapes is convenient.

    Args:
        Y: Either an ``(m, 7)`` array-like or a sequence of 7-key dicts keyed by
            ``pysim.PARAM_NAMES``.

    Returns:
        np.ndarray: A ``(m, 7)`` ``float64`` array; ``(0, 7)`` for an empty input.

    Raises:
        ValueError: If an array-like input does not have exactly 7 columns, or a
            dict row is missing one of ``pysim.PARAM_NAMES``.
    """
    # Sequence of dicts -> stack into positional rows keyed by PARAM_NAMES.
    if isinstance(Y, (list, tuple)) and len(Y) > 0 and isinstance(Y[0], dict):
        rows = []
        for d in Y:
            try:
                rows.append([float(d[name]) for name in validation_common.PARAM_NAMES])
            except KeyError as exc:
                raise ValueError(
                    f"Parameter_Set dict missing key {exc!s}; expected all of "
                    f"{validation_common.PARAM_NAMES}."
                ) from exc
        return np.asarray(rows, dtype=np.float64)

    arr = np.asarray(Y, dtype=np.float64)
    if arr.size == 0:
        # Normalize any empty input to the canonical (0, 7) shape.
        return arr.reshape((0, _NUM_PARAMS))
    if arr.ndim != 2 or arr.shape[1] != _NUM_PARAMS:
        raise ValueError(
            f"Expected an (m, {_NUM_PARAMS}) array keyed positionally to "
            f"pysim.PARAM_NAMES; got shape {arr.shape}."
        )
    return arr


def compute_recovery_metrics(Y_true, Y_est):
    """Compute per-parameter recovery metrics from truth/estimate Parameter_Sets.

    Given the ground-truth and estimated Parameter_Sets for ``m`` retained
    samples, this computes — independently for each of the seven parameters —
    the relative-error distribution, its summary statistics, and the coefficient
    of determination R² (Requirements 2.6, 2.7, 2.9, 2.12).

    **Input form.** ``Y_true`` and ``Y_est`` are taken in *array form*: each is
    an ``(m, 7)`` array whose column ``i`` corresponds positionally to
    ``pysim.PARAM_NAMES[i]``. For convenience a sequence of 7-key ``dict``
    Parameter_Sets (keyed by ``pysim.PARAM_NAMES`` — e.g. the dicts returned by
    :func:`estimate_parameters`) is also accepted and is coerced to the same
    positional layout via :func:`_as_param_array`. The two inputs must describe
    the same number of samples.

    **Per parameter, the computation is:**

      1. **Relative error** via :func:`validation_common.relative_error` applied
         to the whole estimate/truth columns, giving one value per sample
         (Requirement 2.6).
      2. **Zero-truth exclusion** (Requirement 2.12): samples whose ground-truth
         value is *exactly* 0 are dropped from that parameter's relative-error
         distribution and counted as ``zero_truth_excluded_count``. (Such samples
         make the relative error non-finite, since the divisor is ``abs(0)``.)
      3. **Non-finite exclusion**: any remaining non-finite relative error (e.g.
         from a non-finite estimate) is also dropped, so only finite relative
         errors are summarized.
      4. **Distribution statistics** via :func:`validation_common.distribution_stats`
         over the retained finite relative errors — the seven stats
         ``{mean, median, std, min, max, p5, p95}`` (Requirement 2.9). When no
         finite relative error survives the exclusions for a parameter,
         ``rel_error`` is reported as ``None``.
      5. **R²** via :func:`validation_common.r_squared` between the *full* truth
         and estimate columns across all ``m`` retained samples (Requirement
         2.7); the zero-truth/non-finite filtering above applies only to the
         relative-error distribution, not to R². R² is ``None`` when there are no
         samples (``m == 0``).

    Args:
        Y_true: Ground-truth Parameter_Sets — an ``(m, 7)`` array keyed
            positionally to ``pysim.PARAM_NAMES`` (or a sequence of 7-key dicts).
        Y_est: Estimated Parameter_Sets in the same form and sample order as
            ``Y_true``.

    Returns:
        dict: A metrics dict shaped per the design's ``recovery_metrics.json``::

            {
              "parameters": {
                "<param_name>": {
                  "rel_error": {           # or None when no error is retained
                    "mean": float, "median": float, "std": float,
                    "min": float, "max": float, "p5": float, "p95": float
                  },
                  "r_squared": float | None,
                  "zero_truth_excluded_count": int
                },
                ...  # one entry per pysim.PARAM_NAMES, in canonical order
              }
            }

        The top-level retained/excluded sample counts and the
        ``all_samples_excluded`` flag described in the design are *not* added
        here; the :func:`run_recovery` orchestration (task 4.10) wraps this dict
        and adds them.

    Raises:
        ValueError: If the inputs cannot be coerced to matching ``(m, 7)`` arrays
            (mismatched sample counts, wrong column count, or a dict row missing
            a parameter key).
    """
    true_arr = _as_param_array(Y_true)
    est_arr = _as_param_array(Y_est)

    if true_arr.shape[0] != est_arr.shape[0]:
        raise ValueError(
            "Y_true and Y_est must describe the same number of samples; got "
            f"{true_arr.shape[0]} and {est_arr.shape[0]}."
        )

    m = true_arr.shape[0]
    parameters = {}

    for i, name in enumerate(validation_common.PARAM_NAMES):
        truth_col = true_arr[:, i]
        est_col = est_arr[:, i]

        # Relative error per sample (Requirement 2.6). truth == 0 positions come
        # back non-finite (inf/nan) by relative_error's documented contract.
        rel = np.asarray(
            validation_common.relative_error(est_col, truth_col), dtype=np.float64
        ).ravel()

        # Zero-truth exclusion + count (Requirement 2.12).
        zero_truth_mask = (truth_col == 0.0)
        zero_truth_excluded_count = int(np.count_nonzero(zero_truth_mask))

        # Keep only finite relative errors (drops zero-truth and any other
        # non-finite entry) for the distribution.
        retained_rel = rel[np.isfinite(rel)]

        if retained_rel.size > 0:
            rel_error_stats = validation_common.distribution_stats(retained_rel)
        else:
            # No finite relative error survived the exclusions for this param.
            rel_error_stats = None

        # R² across all retained samples (Requirement 2.7), over the full
        # columns (independent of the zero-truth/non-finite filtering above).
        r2 = validation_common.r_squared(truth_col, est_col) if m > 0 else None

        parameters[name] = {
            "rel_error": rel_error_stats,
            "r_squared": r2,
            "zero_truth_excluded_count": zero_truth_excluded_count,
        }

    return {"parameters": parameters}


# Per-parameter quantity + unit annotations for figure axis labels
# (Requirement 6.6: each axis names its quantity and unit so the figure is
# interpretable without the source code). Energies are in units of kBT and the
# directionality factors are dimensionless [0, 1] (see configfile.ini comments);
# k_mig / k0 are kinetic rate constants. The text is short enough to fit on an
# axis while still naming the physical quantity.
PARAM_UNITS = {
    'E_b': 'base-pairing binding energy (kBT)',
    'E_b_azo_trans': 'trans-azobenzene hairpin energy (kBT)',
    'E_b_azo_cis': 'cis-azobenzene hairpin energy (kBT)',
    'k_mig': 'branch-migration rate (1/s)',
    'k0': 'Arrhenius base dissociation rate (1/s)',
    'drt_z': 'Z-direction directionality (dimensionless)',
    'drt_s': 'S-direction directionality (dimensionless)',
}


def _param_column(values, param_name):
    """Extract the 1-D per-parameter value array for ``param_name`` from ``values``.

    Accepts either of the two equivalent input forms documented on
    :func:`plot_scatter`:

      * a **1-D** (or column ``(m, 1)``) array already holding this parameter's
        per-sample values, returned as-is (flattened); or
      * a **full ``(m, 7)``** Parameter_Set array whose column ``i`` corresponds
        positionally to ``pysim.PARAM_NAMES[i]`` — the column for ``param_name``
        is selected.

    Args:
        values: A 1-D per-parameter array or a full ``(m, 7)`` array.
        param_name: One of ``pysim.PARAM_NAMES`` (used to select the column when
            ``values`` is the full ``(m, 7)`` form).

    Returns:
        np.ndarray: A 1-D ``float64`` array of this parameter's per-sample values.

    Raises:
        ValueError: If ``param_name`` is not a known parameter, or ``values`` is
            a 2-D array whose column count is neither 1 nor 7.
    """
    if param_name not in validation_common.PARAM_NAMES:
        raise ValueError(
            f"Unknown param_name {param_name!r}; expected one of "
            f"{validation_common.PARAM_NAMES}."
        )

    arr = np.asarray(values, dtype=np.float64)

    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[1] == 1:
            return arr.ravel()
        if arr.shape[1] == _NUM_PARAMS:
            col = validation_common.PARAM_NAMES.index(param_name)
            return arr[:, col]
    raise ValueError(
        f"Expected a 1-D per-parameter array or a full (m, {_NUM_PARAMS}) array; "
        f"got shape {arr.shape}."
    )


def plot_scatter(Y_true, Y_est, param_name, out_path):
    """Write one true-vs-predicted scatter plot for a single parameter.

    Renders a publication-style scatter of the estimated (predicted) values
    against the ground-truth (true) values for one of the seven physical
    parameters, with a ``y = x`` reference line marking perfect recovery
    (Requirement 2.8). Both axes name the parameter's quantity and unit so the
    figure is interpretable without the source code (Requirement 6.6), and the
    title names the parameter. The non-interactive ``Agg`` backend is selected so
    the figure renders headlessly (Requirement 6.4).

    **Input form (chosen signature).** ``Y_true`` and ``Y_est`` may each be
    given as either:

      * the **per-parameter 1-D array** for ``param_name`` (length ``m``), or
      * the **full ``(m, 7)`` Parameter_Set array** whose column ``i`` is
        ``pysim.PARAM_NAMES[i]`` — in which case the column for ``param_name`` is
        selected.

    This lets :func:`run_recovery` (task 4.10) call ``plot_scatter`` once per
    parameter by passing the full ``(m, 7)`` truth/estimate arrays plus the
    parameter name, while callers that already hold a single column can pass it
    directly. Both inputs must describe the same number of samples.

    Args:
        Y_true: Ground-truth values — a 1-D per-parameter array or a full
            ``(m, 7)`` array (see *Input form*).
        Y_est: Estimated/predicted values in the same form and sample order as
            ``Y_true``.
        param_name: The parameter to plot; one of ``pysim.PARAM_NAMES``. Used for
            the axis labels (via :data:`PARAM_UNITS`) and the title.
        out_path: Destination PNG path. Its parent directory is created if it
            does not yet exist.

    Returns:
        str: ``out_path`` (the path the figure was written to).

    Raises:
        ValueError: If ``param_name`` is unknown, the inputs cannot be reduced to
            matching 1-D per-parameter arrays, or there are no samples to plot.
    """
    true_vals = _param_column(Y_true, param_name)
    est_vals = _param_column(Y_est, param_name)

    if true_vals.shape[0] != est_vals.shape[0]:
        raise ValueError(
            "Y_true and Y_est must describe the same number of samples; got "
            f"{true_vals.shape[0]} and {est_vals.shape[0]}."
        )
    if true_vals.size == 0:
        raise ValueError(
            f"No samples to plot for parameter {param_name!r}."
        )

    import matplotlib

    matplotlib.use("Agg")  # headless rendering (Requirement 6.4)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 6.0))

    ax.scatter(true_vals, est_vals, s=18, alpha=0.6, edgecolors="none",
               label="samples")

    # y = x reference line spanning the combined data range (perfect recovery).
    combined = np.concatenate([true_vals, est_vals])
    lo = float(np.min(combined))
    hi = float(np.max(combined))
    if lo == hi:
        # Degenerate (all values identical): pad so the reference line is visible.
        pad = abs(lo) * 0.05 if lo != 0.0 else 1.0
        lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color="crimson", linestyle="--", linewidth=1.5,
            label="y = x (perfect recovery)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    # Axis labels name the quantity + unit (Requirement 6.6).
    unit = PARAM_UNITS.get(param_name, param_name)
    ax.set_xlabel(f"True {param_name} [{unit}]")
    ax.set_ylabel(f"Predicted {param_name} [{unit}]")
    ax.set_title(f"Parameter recovery: {param_name} (true vs. predicted)")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# task 4.10: run_recovery(...) — orchestrate sampling -> optional noise ->
#   estimation -> metrics -> 7 scatter plots -> recovery_metrics.json; argparse
#   CLI (Requirements 2.1, 2.9, 2.11, 2.13, 5.1, 6.3).
# ---------------------------------------------------------------------------

# Output artifact names for the Recovery_Experiment.
_JSON_NAME = "recovery_metrics.json"
_SCATTER_NAME_FMT = "recovery_scatter_{param}.png"

# Allowed inclusive bounds for the configurable sample count (Requirement 2.1).
_MIN_SAMPLES = 1
_MAX_SAMPLES = 1_000_000


def run_recovery(n_samples=1000, model='cnn', noise_std=None, seed=42,
                 method='Powell', maxiter=400, multistart=8,
                 max_total_draws=None, results_dir='results/validation',
                 predictor=None, predictor_overrides=None):
    """Run the Recovery_Experiment end-to-end and write JSON + 7 scatter plots.

    Orchestrates the synthetic parameter-recovery study: draw ground-truth
    Parameter_Sets via LHS and forward-simulate them, optionally perturb each
    retained sample's curves with Gaussian noise, estimate the Parameter_Set via
    the Predict_Refine_Pipeline, compute per-parameter recovery metrics, and
    write ``recovery_metrics.json`` plus one true-vs-predicted scatter plot per
    parameter into the Results_Directory (Requirements 2.1-2.13, 5.1, 6.3).

    Pipeline:

      1. **Validate the sample count** to be an integer in ``[1, 1_000_000]``
         (Requirement 2.1); a value outside that range raises ``ValueError``.
      2. **Default the draw budget** to ``10 * n_samples`` when ``max_total_draws``
         is ``None`` (Requirement 2.11).
      3. **Validate the seed** up front via
         :func:`validation_common.require_seed` so the experiment refuses to run
         any stochastic step without a recorded seed (Requirements 6.2, 2.10).
      4. **Sample + simulate** via :func:`generate_ground_truth` (LHS draw,
         forward-simulate, redraw-on-invalid up to the budget). If the budget is
         exhausted with **zero** retained samples, stop **before** metrics, write
         ``recovery_metrics.json`` with ``all_samples_excluded: true`` and return
         it (Requirement 2.13).
      5. **Build the predictor** via :func:`validation_common.make_predictor`
         (unless one is injected — see ``predictor``).
      6. **Per retained sample**: optionally add Gaussian noise (only when
         ``noise_std`` is finite and ``> 0`` — Requirement 2.4) before estimation,
         then run :func:`estimate_parameters`, collecting the estimated 7-vectors
         into ``Y_est`` of shape ``(m, 7)``.
      7. **Metrics** via :func:`compute_recovery_metrics` (Requirements 2.6, 2.7,
         2.9, 2.12), wrapped with the top-level fields of the design's
         ``recovery_metrics.json`` data model.
      8. **Write** the JSON via :func:`validation_common.write_json` and the 7
         scatter plots via :func:`plot_scatter` (Requirements 2.8, 6.3, 6.6).

    Reproducibility (Requirements 2.10, 6.1): the sampling seed, the per-sample
    noise seed, and the per-sample estimation-jitter seed are all derived
    deterministically from ``seed`` (via :func:`_derive_batch_seed` inside
    ``generate_ground_truth`` and :func:`_derive_sample_seed` here), so a
    repeated run with the same ``seed`` reproduces identical retained samples,
    noise, and estimates.

    Args:
        n_samples: Requested number of ground-truth samples to retain; an integer
            in ``[1, 1_000_000]`` (default ``1000``) (Requirement 2.1).
        model: Inverse_Model used for estimation: ``'cnn'`` or ``'transformer'``
            (also recorded in the JSON). Ignored when ``predictor`` is injected.
        noise_std: Standard deviation of the optional Gaussian curve noise. Noise
            is enabled only when this is a finite real value ``> 0``
            (Requirement 2.4); ``None`` / non-positive / non-finite disables it.
        seed: Fixed, recorded base integer seed for all stochastic steps
            (Requirements 2.10, 6.1, 6.2). Validated via
            :func:`validation_common.require_seed`.
        method: Refiner method passed through to :func:`estimate_parameters`
            (``'Powell'`` default or ``'Nelder-Mead'``).
        maxiter: Maximum refiner iterations per start (default ``400``).
        multistart: Number of additional jittered estimation starts beyond the
            DL-seeded start (default ``8``); ``0`` disables multi-start.
        max_total_draws: Maximum cumulative number of Parameter_Sets to draw.
            Defaults to ``10 * n_samples`` when ``None`` (Requirement 2.11).
        results_dir: Results_Directory for the JSON + scatter artifacts.
        predictor: Optional pre-built Inverse_Model predictor exposing
            ``predict(X) -> (N, 7)``. When ``None`` (the default) it is built via
            :func:`validation_common.make_predictor(model, **predictor_overrides)`.
            Supplying it lets callers inject a custom/fake predictor (e.g. for a
            lightweight dry run) without importing ``torch``.
        predictor_overrides: Optional dict of constructor keyword overrides
            forwarded to :func:`validation_common.make_predictor` when
            ``predictor`` is ``None``.

    Returns:
        dict: The full metrics object written to ``recovery_metrics.json``.

    Raises:
        ValueError: If ``n_samples`` is not an integer in ``[1, 1_000_000]``.
        validation_common.MissingSeedError: If ``seed`` is ``None`` or not an
            ``int`` (raised before any stochastic step).
    """
    # 1. Validate the requested sample count (Requirement 2.1).
    n_samples = int(n_samples)
    if n_samples < _MIN_SAMPLES or n_samples > _MAX_SAMPLES:
        raise ValueError(
            f"n_samples must be an integer in [{_MIN_SAMPLES}, {_MAX_SAMPLES}]; "
            f"got {n_samples}."
        )

    # 2. Default the draw budget to 10 * n_samples (Requirement 2.11).
    if max_total_draws is None:
        max_total_draws = 10 * n_samples
    max_total_draws = int(max_total_draws)

    # 3. Refuse any stochastic step without a recorded seed (Requirements 6.2,
    #    2.10), before any drawing occurs.
    base_seed = validation_common.require_seed(seed, "run_recovery")

    # Noise is enabled only for a finite, strictly-positive std (Requirement 2.4).
    noise_enabled = noise_std is not None and np.isfinite(noise_std) and noise_std > 0.0

    ranges = validation_common.load_configured_ranges()

    # 4. Sample + forward-simulate with redraw-on-invalid up to the budget
    #    (Requirements 2.2, 2.3, 2.11). generate_ground_truth re-validates the
    #    seed and derives per-batch seeds from it.
    Y_true, X_curves, n_excluded_invalid, exhausted = generate_ground_truth(
        n_samples, ranges, base_seed, max_total_draws
    )

    retained_count = int(Y_true.shape[0])

    # Shared config block recorded in the JSON regardless of the outcome.
    config_block = {
        "requested_samples": n_samples,
        "max_total_draws": max_total_draws,
        "noise_enabled": bool(noise_enabled),
        "noise_std": (float(noise_std) if noise_enabled else None),
        "refine_method": method,
        "maxiter": int(maxiter),
        "multistart": int(multistart),
    }

    # 4b. Budget exhausted with ZERO retained samples: stop BEFORE metrics and
    #     report the all-samples-excluded condition (Requirement 2.13).
    if retained_count == 0:
        metrics = {
            "experiment": "synthetic_parameter_recovery",
            "model": model,
            "seed": base_seed,
            "config": config_block,
            "retained_sample_count": 0,
            "excluded_sample_count": int(n_excluded_invalid),
            "all_samples_excluded": True,
            "parameters": {},
        }
        json_path = os.path.join(results_dir, _JSON_NAME)
        validation_common.write_json(json_path, metrics)
        return metrics

    # 5. Build the DL predictor lazily (only when not injected) so a dry run can
    #    pass a fake predictor without importing torch.
    if predictor is None:
        predictor = validation_common.make_predictor(
            model, **(predictor_overrides or {})
        )

    # 6. Per retained sample: optional noise -> estimate -> collect 7-vectors.
    est_rows = []
    for i in range(retained_count):
        curves = X_curves[i]

        if noise_enabled:
            # Per-sample, reproducible noise seed (Requirements 2.4, 2.10, 6.1).
            noise_seed = _derive_sample_seed(base_seed, 1, i)
            curves = add_gaussian_noise(curves, noise_std, noise_seed)

        # Per-sample, reproducible estimation-jitter seed (multistart only;
        # estimate_parameters consults it solely when multistart > 0).
        est_seed = _derive_sample_seed(base_seed, 2, i)
        est_pdict = estimate_parameters(
            predictor, curves, ranges, method=method, maxiter=maxiter,
            multistart=multistart, seed=est_seed,
        )
        est_rows.append(
            [float(est_pdict[name]) for name in validation_common.PARAM_NAMES]
        )

    Y_est = np.asarray(est_rows, dtype=np.float64)  # (m, 7)

    # 7. Per-parameter recovery metrics (Requirements 2.6, 2.7, 2.9, 2.12), then
    #    wrap with the top-level fields of the recovery_metrics.json data model.
    core = compute_recovery_metrics(Y_true, Y_est)

    metrics = {
        "experiment": "synthetic_parameter_recovery",
        "model": model,
        "seed": base_seed,
        "config": config_block,
        "retained_sample_count": retained_count,
        "excluded_sample_count": int(n_excluded_invalid),
        "all_samples_excluded": False,
        "parameters": core["parameters"],
    }

    # 8. Write the JSON (Requirements 2.9, 6.3) and the 7 scatter plots
    #    (Requirements 2.8, 6.6) into the Results_Directory.
    json_path = os.path.join(results_dir, _JSON_NAME)
    validation_common.write_json(json_path, metrics)

    for name in validation_common.PARAM_NAMES:
        scatter_path = os.path.join(
            results_dir, _SCATTER_NAME_FMT.format(param=name))
        plot_scatter(Y_true, Y_est, name, scatter_path)

    return metrics


def main(argv=None):
    """CLI entry point for the Recovery_Experiment (Requirements 2.1-2.13).

    Flags:
      ``--model``       Inverse_Model used for estimation (``cnn``/``transformer``)
      ``--samples``     number of ground-truth samples (default: 1000)
      ``--seed``        fixed, recorded base seed (default: 42)
      ``--noise-std``   optional Gaussian curve-noise std (enabled when > 0)
      ``--maxiter``     max refiner iterations per start (default: 400)
      ``--multistart``  additional jittered estimation starts (default: 8)
      ``--results-dir`` Results_Directory (default: results/validation)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic parameter-recovery experiment: draw ground-truth "
            "Parameter_Sets via Latin Hypercube sampling, forward-simulate them, "
            "optionally add Gaussian noise, recover them via the predict+refine "
            "pipeline, and report per-parameter relative-error statistics and R²."
        )
    )
    parser.add_argument(
        "--model", default="cnn", choices=("cnn", "transformer"),
        help="Inverse model used for estimation (default: cnn).",
    )
    parser.add_argument(
        "--samples", type=int, default=1000,
        help="Number of ground-truth samples to draw, in [1, 1000000] (default: 1000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Fixed, recorded base random seed (default: 42).",
    )
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="Optional Gaussian curve-noise standard deviation; noise is enabled "
             "only when this is finite and > 0 (default: disabled).",
    )
    parser.add_argument(
        "--maxiter", type=int, default=400,
        help="Maximum refiner iterations per start (default: 400).",
    )
    parser.add_argument(
        "--multistart", type=int, default=8,
        help="Additional jittered estimation starts beyond the DL start (default: 8).",
    )
    parser.add_argument(
        "--results-dir", default="results/validation",
        help="Results_Directory for the JSON and scatter plots (default: results/validation).",
    )
    args = parser.parse_args(argv)

    metrics = run_recovery(
        n_samples=args.samples,
        model=args.model,
        noise_std=args.noise_std,
        seed=args.seed,
        method='Powell',
        maxiter=args.maxiter,
        multistart=args.multistart,
        results_dir=args.results_dir,
    )

    json_path = os.path.join(args.results_dir, _JSON_NAME)
    print(f"Wrote {json_path}")
    if metrics.get("all_samples_excluded"):
        print("  all_samples_excluded=true: every drawn sample was invalid; "
              "no metrics or scatter plots were produced.")
    else:
        print(f"  retained={metrics['retained_sample_count']} "
              f"excluded={metrics['excluded_sample_count']}")
        for name in validation_common.PARAM_NAMES:
            block = metrics["parameters"][name]
            scatter_path = os.path.join(
                args.results_dir, _SCATTER_NAME_FMT.format(param=name))
            print(f"  {name}: r_squared={block['r_squared']}  -> {scatter_path}")
    return metrics


if __name__ == "__main__":
    main()
