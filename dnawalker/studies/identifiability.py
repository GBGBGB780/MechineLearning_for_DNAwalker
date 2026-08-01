# coding=utf-8
"""
dnawalker.studies.identifiability — 参数可辨识性 / 敏感性分析 (纯计算, 无需外部数据)
dnawalker.studies.identifiability — Parameter identifiability / sensitivity analysis.

本脚本使用**与目标曲线和已训练 checkpoint 无关**的局部敏感性分析，判断哪些参数
不会被荧光曲线充分约束，并通过 Fisher 信息与 profile likelihood 量化病态程度。

This script uses target-curve- and checkpoint-independent local sensitivity
analysis to identify parameters that the fluorescence curves do not constrain,
then quantifies ill-conditioning with Fisher information and profile likelihood.

两种分析 / Two analyses:

  1. Jacobian + Fisher 信息矩阵 (FIM) / Jacobian + Fisher Information Matrix
     - J = d(curve)/d(theta) 在优化空间 (k0 取 log10, 与 refine.py 一致) 用中心差分估计。
       J = d(curve)/d(theta) by central finite differences in the same optimization space
       refine.py uses (k0 in log10).
     - 每参数敏感度 = J 的列范数 ||J[:, i]||: 越小 = 曲线越不依赖该参数 = 越不可辨识。
       Per-parameter sensitivity = column norm ||J[:, i]||: smaller = less identifiable.
     - FIM = J^T J (等噪声假设)。其条件数衡量整体病态程度; 最小特征值对应的特征向量
       给出"不可辨识方向"(参数的哪种线性组合最不受曲线约束)。
       FIM = J^T J; its condition number quantifies overall ill-conditioning, and the
       eigenvector of the smallest eigenvalue is the least-identifiable direction.

  2. Profile likelihood (可选) / Profile likelihood (optional)
     - 固定某参数在其范围内扫一遍, 每个取值下用 refine 优化其余 6 个参数, 记录最优曲线
       RMSE。谷底平坦 = 该参数不可辨识 (大范围取值都能拟合出几乎一样的曲线)。
       Fix one parameter on a grid; at each value refine the other 6 and record the best
       curve RMSE. A flat valley means that parameter is non-identifiable.

Usage:
    python -m dnawalker.studies.identifiability
    python -m dnawalker.studies.identifiability --points 5
    python -m dnawalker.studies.identifiability --profile
    python -m dnawalker.studies.identifiability --profile --profile-grid 7
"""

import argparse
import os

import numpy as np

from dnawalker.physics import simulator as pysim
from dnawalker.physics import refinement as refine
from dnawalker.paths import RESULTS_DIR
from . import protocol as validation_common

PARAM_NAMES = pysim.PARAM_NAMES
_DEFAULT_RESULTS_DIR = os.fspath(
    RESULTS_DIR / "validation" / "identifiability"
)

# Documented reference parameters (common configuration / warm-start record).
REFERENCE_PARAMETERS = {
    'E_b': -1.2,
    'E_b_azo_trans': -1.0,
    'E_b_azo_cis': -0.1,
    'k_mig': 0.05,
    'k0': 8e-6,
    'drt_z': 0.5,
    'drt_s': 0.05,
}

# Output artifact names (under the Results_Directory).
_JSON_NAME = "identifiability_metrics.json"
_FIGURE_NAME = "identifiability_sensitivity.png"


def _simulate_flat(opt_vec):
    """Forward-simulate an opt-space vector; return the flattened (3*T,) curve.

    Returns ``None`` for an invalid simulation (non-finite or negative
    ``dt_used``), so callers can skip finite-difference points that fall in the
    infeasible region.
    """
    params = refine._from_opt_space(opt_vec)
    signals, dt = pysim.run_simulation(params)
    if not np.isfinite(dt) or dt < 0 or not np.all(np.isfinite(signals)):
        return None
    return np.asarray(signals, dtype=np.float64).reshape(-1)


def compute_jacobian(center_params, ranges, rel_step=1e-3, workers=1):
    """Central-difference Jacobian J = d(curve)/d(theta) at ``center_params``.

    The differentiation happens in the **optimization space** used by
    ``refine.py`` (``k0`` in log10), so the per-parameter sensitivities are
    directly comparable to how the refiner "sees" each parameter. The step for
    coordinate ``i`` is ``rel_step`` times that coordinate's opt-space range
    width, which makes the step scale-appropriate for every parameter.

    Args:
        center_params: dict Parameter_Set (keyed by pysim.PARAM_NAMES) to
            linearize around.
        ranges: Configured_Ranges {name: (min, max)}.
        rel_step: finite-difference step as a fraction of each coordinate's
            opt-space range width.
        workers: Process count for the ``2*7`` independent neighbor simulations.
            ``1`` (default) runs them serially — bit-for-bit the historical path.
            ``>1`` fans the neighbor points across a ``ProcessPoolExecutor``; the
            assembled ``J`` is identical because each neighbor simulation is pure
            and order-independent (only the base point gates success).

    Returns:
        (J, ok): J is an (M, 7) array (M = 3*NUM_RESULTS) or None if the center
        or too many neighbors are infeasible; ok is a bool success flag.
    """
    rel_step = validation_common.require_finite_real(
        rel_step, "rel_step", minimum=0.0, strict_minimum=True
    )
    workers = validation_common.require_int(workers, "workers", minimum=1)

    bounds = refine._opt_bounds(ranges)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    x0 = refine._to_opt_space(center_params, ranges)

    base = _simulate_flat(x0)
    if base is None:
        return None, False

    n = len(PARAM_NAMES)
    J = np.zeros((base.shape[0], n), dtype=np.float64)

    # Build the 2*n neighbor points (+h / -h per coordinate) up front so the
    # simulations can optionally run in parallel; keep per-column step metadata
    # for the assembly below (which is unchanged, incl. one-sided fallback).
    plus_vecs, minus_vecs, steps = [], [], []
    for i in range(n):
        h = rel_step * (hi[i] - lo[i])
        if h <= 0:
            h = rel_step * (abs(x0[i]) + 1.0)
        xp = x0.copy(); xp[i] = min(x0[i] + h, hi[i])
        xm = x0.copy(); xm[i] = max(x0[i] - h, lo[i])
        plus_vecs.append(xp)
        minus_vecs.append(xm)
        steps.append((xp[i], xm[i]))

    all_vecs = plus_vecs + minus_vecs
    if workers and workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=workers) as ex:
            sims = list(ex.map(_simulate_flat, all_vecs))
    else:
        sims = [_simulate_flat(v) for v in all_vecs]
    cps = sims[:n]
    cms = sims[n:]

    for i in range(n):
        cp, cm = cps[i], cms[i]
        xpi, xmi = steps[i]
        if cp is None or cm is None:
            # Fall back to a one-sided difference against the base point.
            if cp is not None:
                J[:, i] = (cp - base) / (xpi - x0[i])
            elif cm is not None:
                J[:, i] = (base - cm) / (x0[i] - xmi)
            else:
                # Both neighbors infeasible: leave this column at 0 (i.e. the
                # curve is effectively flat / unreachable in this direction).
                J[:, i] = 0.0
            continue

        denom = (xpi - xmi)
        J[:, i] = (cp - cm) / denom if denom != 0 else 0.0

    return J, True


def fisher_analysis(J):
    """Per-parameter sensitivity + Fisher Information Matrix diagnostics.

    Args:
        J: (M, 7) Jacobian d(curve)/d(theta) in optimization space.

    Returns:
        dict with per-parameter column-norm sensitivities (raw + normalized),
        the FIM eigenvalues, its condition number, and the least-identifiable
        direction (eigenvector of the smallest eigenvalue) as param weights.
    """
    J = np.asarray(J, dtype=np.float64)
    if (J.ndim != 2 or J.shape[0] == 0
            or J.shape[1] != len(PARAM_NAMES)
            or not np.all(np.isfinite(J))):
        raise ValueError(
            "J must be a non-empty finite matrix with "
            f"{len(PARAM_NAMES)} columns, got {J.shape}"
        )

    col_norm = np.linalg.norm(J, axis=0)          # ||J[:, i]|| per parameter
    max_norm = float(np.max(col_norm)) if col_norm.size else 0.0
    normalized = (col_norm / max_norm) if max_norm > 0 else col_norm

    fim = J.T @ J                                  # (7, 7), symmetric PSD
    # Symmetric eigendecomposition (FIM is symmetric positive semi-definite).
    raw_eigvals, eigvecs = np.linalg.eigh(fim)
    eigvals = np.clip(raw_eigvals, 0.0, None)      # numerical floor at 0 (for reporting)

    # Condition number from the MAGNITUDES of the raw eigenvalues. A rank-deficient
    # FIM can produce a tiny NEGATIVE smallest eigenvalue from rounding; clipping it
    # to 0 and then excluding it (min over strictly-positive values) would silently
    # report the ratio against the *second* smallest eigenvalue and thus UNDERSTATE
    # the ill-conditioning. Using |eigval| keeps that near-singular direction in the
    # denominator so the condition number faithfully reflects the near-singularity.
    abs_eigvals = np.abs(raw_eigvals)
    max_abs = float(np.max(abs_eigvals)) if abs_eigvals.size else 0.0
    min_abs = float(np.min(abs_eigvals)) if abs_eigvals.size else 0.0
    cond = float(max_abs / min_abs) if min_abs > 0 else float('inf')

    # Eigenvector of the smallest eigenvalue = least-constrained direction:
    # the linear combination of parameters the curves barely respond to. eigh
    # returns eigenvalues ascending, so index 0 is the smallest (clipping is
    # monotonic and does not change the ordering).
    least_dir = eigvecs[:, int(np.argmin(raw_eigvals))]

    return {
        "sensitivity": {
            PARAM_NAMES[i]: float(col_norm[i]) for i in range(len(PARAM_NAMES))
        },
        "sensitivity_normalized": {
            PARAM_NAMES[i]: float(normalized[i]) for i in range(len(PARAM_NAMES))
        },
        "fim_eigenvalues": [float(v) for v in eigvals],
        "fim_condition_number": cond,
        "least_identifiable_direction": {
            PARAM_NAMES[i]: float(least_dir[i]) for i in range(len(PARAM_NAMES))
        },
    }


def averaged_sensitivity(ranges, n_points=1, seed=42, rel_step=1e-3, workers=1):
    """Average per-parameter sensitivity over several feasible reference points.

    The reference Parameter_Set is always included as the first point; any
    additional points are drawn by Latin Hypercube across the Configured_Ranges
    (feasible ones only). Averaging the per-point column norms makes the
    sensitivity ranking robust to the particular linearization point.

    Returns:
        (mean_norms dict, per_point list, fisher_at_reference dict)
    """
    n_points = validation_common.require_int(
        n_points, "n_points", minimum=1
    )
    rel_step = validation_common.require_finite_real(
        rel_step, "rel_step", minimum=0.0, strict_minimum=True
    )
    workers = validation_common.require_int(workers, "workers", minimum=1)
    seed = validation_common.require_seed(seed, "averaged_sensitivity")

    centers = [dict(REFERENCE_PARAMETERS)]
    if n_points > 1:
        draws = validation_common.lhs_params(n_points - 1, ranges, seed)
        for row in draws:
            centers.append(
                {PARAM_NAMES[i]: float(row[i]) for i in range(len(PARAM_NAMES))}
            )

    per_point_norms = []
    fisher_ref = None
    for idx, center in enumerate(centers):
        J, ok = compute_jacobian(center, ranges, rel_step=rel_step, workers=workers)
        if not ok:
            continue
        fa = fisher_analysis(J)
        if idx == 0:
            fisher_ref = fa  # keep the reference-point FIM diagnostics
        per_point_norms.append(
            np.array([fa["sensitivity"][p] for p in PARAM_NAMES], dtype=np.float64)
        )

    if not per_point_norms:
        raise RuntimeError("No feasible reference point produced a valid Jacobian.")

    mean_norm = np.mean(np.vstack(per_point_norms), axis=0)
    max_mean = float(np.max(mean_norm)) if mean_norm.size else 0.0
    mean_dict = {
        PARAM_NAMES[i]: {
            "sensitivity": float(mean_norm[i]),
            "sensitivity_normalized": float(mean_norm[i] / max_mean) if max_mean > 0 else 0.0,
        }
        for i in range(len(PARAM_NAMES))
    }
    return mean_dict, len(per_point_norms), fisher_ref


def _refine_fixing_one(fixed_idx, fixed_val_opt, exp_curves, ranges,
                       maxiter=150):
    """Refine the other 6 parameters with parameter ``fixed_idx`` held fixed.

    Used by the profile-likelihood scan: optimizes a 6-dim opt-space vector
    (all params except ``fixed_idx``) to minimize curve RMSE against
    ``exp_curves``, keeping ``fixed_idx`` pinned at ``fixed_val_opt``.

    This profile helper intentionally uses Nelder-Mead with a clip-and-penalize
    objective. Powell's bounded line search can fail when a free coordinate
    lands exactly on a bound. The optimizer is therefore fixed rather than
    exposing a ``method`` argument that would be ignored.

    Returns the best curve RMSE found (a float; ``inf`` if all infeasible).
    """
    from scipy.optimize import minimize

    fixed_idx = validation_common.require_int(
        fixed_idx, "fixed_idx", minimum=0, maximum=len(PARAM_NAMES) - 1
    )
    fixed_val_opt = validation_common.require_finite_real(
        fixed_val_opt, "fixed_val_opt"
    )
    maxiter = validation_common.require_int(maxiter, "maxiter", minimum=1)

    bounds = refine._opt_bounds(ranges)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)
    if not lo[fixed_idx] <= fixed_val_opt <= hi[fixed_idx]:
        raise ValueError(
            f"fixed_val_opt is outside parameter {PARAM_NAMES[fixed_idx]!r} "
            f"bounds [{lo[fixed_idx]}, {hi[fixed_idx]}]"
        )

    free_idx = [i for i in range(len(PARAM_NAMES)) if i != fixed_idx]

    # Start the free coordinates at the midpoint of their opt-space range.
    x0_full = (lo + hi) / 2.0
    x0_full[fixed_idx] = fixed_val_opt
    x0_free = x0_full[free_idx]
    lo_free, hi_free = lo[free_idx], hi[free_idx]

    best = {"rmse": float("inf")}

    def objective(xf):
        xfc = np.clip(xf, lo_free, hi_free)
        penalty = float(np.sum(np.abs(xf - xfc)) * 10.0)
        full = x0_full.copy()
        full[free_idx] = xfc
        full[fixed_idx] = fixed_val_opt
        rmse = refine.curve_rmse(refine._from_opt_space(full), exp_curves)
        # Track the best feasible (un-penalized) curve RMSE seen.
        if penalty == 0.0 and rmse < best["rmse"]:
            best["rmse"] = rmse
        return rmse + penalty

    # Nelder-Mead: unbounded simplex (clip handled in the objective), so no
    # bounded line-search degeneracy. Initial-guess RMSE seeds the running best.
    objective(x0_free)
    minimize(objective, x0_free, method="Nelder-Mead",
             options={'maxiter': maxiter, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False})
    return best["rmse"]


def profile_likelihood(ranges, grid=7, maxiter=150):
    """Profile-likelihood scan for every parameter around the reference curve.

    The target is the reference Parameter_Set's own simulated curve, so a
    perfectly identifiable parameter would show a sharp RMSE minimum at its
    reference value; a flat profile (low RMSE across the whole range) means the
    parameter is non-identifiable — many values fit the curve equally well.

    Returns:
        dict {param: {"grid": [...physical values...], "rmse": [...],
                      "flatness": max_rmse - min_rmse}}.
    """
    grid = validation_common.require_int(grid, "grid", minimum=2)
    maxiter = validation_common.require_int(maxiter, "maxiter", minimum=1)

    bounds = refine._opt_bounds(ranges)
    lo = np.array([b[0] for b in bounds], dtype=np.float64)
    hi = np.array([b[1] for b in bounds], dtype=np.float64)

    # Reference target curve (computation-only).
    ref_signals, dt = pysim.run_simulation(REFERENCE_PARAMETERS)
    if (not np.isfinite(dt) or dt < 0
            or np.asarray(ref_signals).shape != (3, pysim.NUM_RESULTS)
            or not np.all(np.isfinite(ref_signals))):
        raise RuntimeError("Reference parameters produced an invalid simulation.")
    target = np.asarray(ref_signals, dtype=np.float64)

    out = {}
    for i, name in enumerate(PARAM_NAMES):
        grid_opt = np.linspace(lo[i], hi[i], grid)
        rmses = []
        phys_vals = []
        for g in grid_opt:
            rmse = _refine_fixing_one(i, float(g), target, ranges,
                                      maxiter=maxiter)
            rmses.append(float(rmse))
            # Physical value for this grid point (k0 back from log10).
            full = (lo + hi) / 2.0
            full[i] = g
            phys_vals.append(refine._from_opt_space(full)[name])
        finite = [r for r in rmses if np.isfinite(r)]
        flatness = (max(finite) - min(finite)) if finite else float('inf')
        out[name] = {
            "grid": [float(v) for v in phys_vals],
            "rmse": rmses,
            "flatness": float(flatness),
        }
    return out


def _plot_sensitivity(mean_sensitivity, out_path):
    """Bar chart of normalized per-parameter sensitivity (labeled axes)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(PARAM_NAMES)
    vals = [mean_sensitivity[n]["sensitivity_normalized"] for n in names]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = ["tab:green" if v >= 0.5 else ("tab:orange" if v >= 0.15 else "tab:red")
              for v in vals]
    ax.bar(range(len(names)), vals, color=colors)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_xlabel("Physical parameter")
    ax.set_ylabel("Normalized curve sensitivity  ||dCurve/dtheta||  (dimensionless, max=1)")
    ax.set_title("Local parameter sensitivity of the fluorescence curves\n"
                 "(low bar = curve barely depends on the parameter = non-identifiable)")
    ax.axhline(0.15, color="gray", linestyle="--", linewidth=1,
               label="non-identifiable threshold (~0.15)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_analysis(points=1, seed=42, rel_step=1e-3, do_profile=False,
                 profile_grid=7, profile_maxiter=150,
                 results_dir=_DEFAULT_RESULTS_DIR, workers=1):
    """Run the identifiability analysis end-to-end and write JSON + figure."""
    points = validation_common.require_int(points, "points", minimum=1)
    workers = validation_common.require_int(workers, "workers", minimum=1)
    rel_step = validation_common.require_finite_real(
        rel_step, "rel_step", minimum=0.0, strict_minimum=True
    )
    if not isinstance(do_profile, (bool, np.bool_)):
        raise ValueError(f"do_profile must be a boolean, got {do_profile!r}")
    do_profile = bool(do_profile)
    if do_profile:
        profile_grid = validation_common.require_int(
            profile_grid, "profile_grid", minimum=2
        )
        profile_maxiter = validation_common.require_int(
            profile_maxiter, "profile_maxiter", minimum=1
        )

    ranges = validation_common.load_configured_ranges()

    mean_sensitivity, n_used, fisher_ref = averaged_sensitivity(
        ranges, n_points=points, seed=seed, rel_step=rel_step, workers=workers
    )

    metrics = {
        "experiment": "parameter_identifiability",
        "seed": seed,
        "config": {
            "jacobian_points": points,
            "jacobian_points_used": n_used,
            "rel_step": rel_step,
            "profile": bool(do_profile),
            "profile_grid": profile_grid if do_profile else None,
            "profile_maxiter": profile_maxiter if do_profile else None,
            "profile_method": "Nelder-Mead" if do_profile else None,
        },
        "mean_sensitivity": mean_sensitivity,
        "fisher_at_reference": fisher_ref,
    }

    if do_profile:
        metrics["profile_likelihood"] = profile_likelihood(
            ranges, grid=profile_grid, maxiter=profile_maxiter
        )

    validation_common.write_json(os.path.join(results_dir, _JSON_NAME), metrics)
    _plot_sensitivity(mean_sensitivity, os.path.join(results_dir, _FIGURE_NAME))
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Parameter identifiability / sensitivity analysis: Jacobian + Fisher "
            "Information (and optional profile likelihood), explaining which "
            "parameters the fluorescence curves do not constrain."
        )
    )
    parser.add_argument("--points", type=int, default=1,
                        help="Reference points to average the Jacobian over "
                             "(reference + LHS draws; default 1 = reference only).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Fixed seed for the LHS reference points (default: 42).")
    parser.add_argument("--rel-step", type=float, default=1e-3,
                        help="Finite-difference step as a fraction of each "
                             "parameter's opt-space range (default: 1e-3).")
    parser.add_argument("--profile", action="store_true",
                        help="Also run the (slower) profile-likelihood scan.")
    parser.add_argument("--profile-grid", type=int, default=7,
                        help="Grid points per parameter for profile likelihood (default: 7).")
    parser.add_argument("--profile-maxiter", type=int, default=150,
                        help="Refiner maxiter per profile grid point (default: 150).")
    parser.add_argument(
        "--results-dir", default=_DEFAULT_RESULTS_DIR,
        help=(
            "Results_Directory "
            "(default: repository results/validation/identifiability)."
        ),
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Processes for the Jacobian's independent neighbor "
                             "simulations (default 1 = serial; >1 parallelizes "
                             "the 2*7 finite-difference sims per reference point). "
                             "NOTE: pysim is fast (matrix-power accelerated), so "
                             "process-pool overhead can outweigh the gain here; "
                             "worth it mainly for a costlier simulator or many "
                             "--points. Results are bit-identical to serial.")
    args = parser.parse_args(argv)

    metrics = run_analysis(
        points=args.points, seed=args.seed, rel_step=args.rel_step,
        do_profile=args.profile, profile_grid=args.profile_grid,
        profile_maxiter=args.profile_maxiter, results_dir=args.results_dir,
        workers=args.workers,
    )

    print(f"Wrote {os.path.join(args.results_dir, _JSON_NAME)}")
    print(f"Wrote {os.path.join(args.results_dir, _FIGURE_NAME)}")
    print("\nNormalized curve sensitivity (low = non-identifiable):")
    ms = metrics["mean_sensitivity"]
    for name in sorted(PARAM_NAMES, key=lambda p: -ms[p]["sensitivity_normalized"]):
        v = ms[name]["sensitivity_normalized"]
        tag = "identifiable" if v >= 0.5 else ("weak" if v >= 0.15 else "NON-identifiable")
        print(f"  {name:16s} {v:6.3f}  [{tag}]")
    fr = metrics.get("fisher_at_reference") or {}
    if "fim_condition_number" in fr:
        print(f"\nFIM condition number (reference point): {fr['fim_condition_number']:.3e}")
    if args.profile:
        print("\nProfile-likelihood flatness (small = flat valley = non-identifiable):")
        pl = metrics["profile_likelihood"]
        for name in sorted(PARAM_NAMES, key=lambda p: pl[p]["flatness"]):
            print(f"  {name:16s} flatness(maxRMSE-minRMSE)={pl[name]['flatness']:.5f}")
    return metrics


def cli(argv=None):
    """Console-script wrapper returning a process exit status."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
