# coding=utf-8
"""
multiseed_retrain.py — Multi-seed retraining study (Multi_Seed_Experiment).

This script retrains both inverse models (CNN and Transformer) across several
recorded random seeds, evaluates each retrained model on a fixed held-out test
set, and reports the per-seed test-set metric together with its across-seed mean
and sample standard deviation. Reporting metrics as mean +/- std over multiple
seeds validates that the observed "CNN beats Transformer" result is statistically
robust rather than a single-run artifact (Requirement 4).

Scope of this file grows task-by-task. Currently implemented:
  - ``aggregate``  : across-seed mean / sample-std aggregation (Requirements 4.4, 4.9)

Also implemented (task 8.5):
  - ``write_seed_override_ini`` : per-seed override INI writer (Requirements 4.1, 4.2)
  - ``TrainOutcome``            : per-seed training outcome record (Requirement 4.8)
  - ``train_one``               : seed-override training, continue-on-failure (Req. 4.1, 4.2, 4.8)
  - ``eval_one``                : fixed-split evaluation -> curve_rmse_mean (Requirements 4.3, 4.8)

Also implemented (task 8.6):
  - ``run_multiseed`` : orchestration over both models x all seeds, writing
    ``multiseed_metrics.json`` + the labeled ``multiseed_compare.png`` figure
    (Requirements 4.4, 4.5, 4.6, 4.7, 4.9, 6.3, 6.4), plus an ``argparse`` CLI.
"""

import argparse
import configparser
import contextlib
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass

import numpy as np

# Make the repo root importable regardless of the caller's cwd, so the shared
# suite helpers (validation_common) and the kernel (pysim) resolve cleanly.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Shared suite helpers (repo root). make_predictor builds the pluggable
# Inverse_Model predictor used by eval_one (Requirement 5.3); write_json creates
# the Results_Directory and serializes the metrics dict (Requirement 6.3).
from validation_common import make_predictor, write_json  # noqa: E402

# 5 distinct, recorded seeds for the study (Requirement 4.1, 4.2). Defined here
# so the module exposes the canonical seed list even before run_multiseed lands.
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

# ---------------------------------------------------------------------------
# Repo layout anchors (resolved relative to this file so the orchestration is
# independent of the caller's working directory).
# ---------------------------------------------------------------------------
_CNN_DIR = os.path.join(_THIS_DIR, "train_cnn")
_TRANSFORMER_DIR = os.path.join(_THIS_DIR, "train_transformer")
_ROOT_CONFIG = os.path.join(_THIS_DIR, "configfile.ini")
_TRANSFORMER_CONFIG = os.path.join(_TRANSFORMER_DIR, "config_transformer.ini")

# Override INIs are written here so per-seed runs do not clobber each other and
# the artifacts stay grouped with the suite's other validation outputs.
_OVERRIDES_DIR = os.path.join(_THIS_DIR, "results", "validation", "overrides")


def aggregate(per_seed_values):
    """Aggregate a per-seed metric across seeds (Requirements 4.4, 4.9).

    Input shape
    -----------
    ``per_seed_values`` is a list of ``(value, ok)`` pairs, one per seed:

      - ``value`` (float | None): the seed's test-set metric (e.g. curve_rmse_mean);
        may be ``None`` for a failed seed.
      - ``ok`` (bool): ``True`` if the train+eval run for that seed completed
        successfully, ``False`` otherwise.

    Only successful seeds (``ok`` truthy) contribute to the aggregates; failed
    seeds are ignored here (they are still retained in the per-seed report by the
    orchestrator, see Requirement 4.8). The value of a failed seed is never read,
    so ``None`` values on failed seeds are safe.

    Aggregation rules
    -----------------
      - ``n_success``           = number of successful seeds.
      - ``mean``                = mean over successful seeds, or ``None`` when no
                                  seed succeeded.
      - ``std``                 = sample standard deviation (``ddof=1``) over
                                  successful seeds when at least 2 succeed,
                                  otherwise ``None``.
      - ``std_available``       = ``True`` iff at least 2 seeds succeeded.
      - ``insufficient_seeds``  = ``True`` iff fewer than 2 seeds succeeded
                                  (the std cannot be computed, Requirement 4.9).

    Args:
        per_seed_values: iterable of ``(value, ok)`` pairs.

    Returns:
        dict with keys ``mean``, ``std``, ``std_available``,
        ``insufficient_seeds`` and ``n_success``.
    """
    successes = [value for value, ok in per_seed_values if ok]
    n_success = len(successes)

    if n_success == 0:
        # No seed succeeded: nothing to report beyond the flags.
        mean = None
    else:
        mean = float(np.mean(np.asarray(successes, dtype=float)))

    if n_success >= 2:
        # Sample standard deviation (ddof=1) over the successful seeds.
        std = float(np.std(np.asarray(successes, dtype=float), ddof=1))
        std_available = True
        insufficient_seeds = False
    else:
        # Fewer than 2 successes: sample std is undefined (Requirement 4.9).
        std = None
        std_available = False
        insufficient_seeds = True

    return {
        "mean": mean,
        "std": std,
        "std_available": std_available,
        "insufficient_seeds": insufficient_seeds,
        "n_success": n_success,
    }


# ===========================================================================
# Task 8.5 — seed-override training & evaluation (continue-on-failure)
# ===========================================================================
#
# These helpers vary the training seed by layering a tiny override INI on top of
# the base config(s) — exactly the ``--config`` / ``--transformer-config``
# mechanism the existing train scripts already accept (see train_cnn/train_mlp.py
# and train_transformer/train_transformer.py). Per-seed model/scaler output paths
# keep the seeds from clobbering each other's checkpoints. Training is invoked as
# an isolated subprocess so each seed gets a fresh process/RNG state and so a
# crash in one seed cannot take down the orchestrator (continue-on-failure,
# Requirement 4.8). Evaluation reuses the existing ``eval_testset.get_test_split``
# on the FIXED held-out split (driven by ``test_seed``, independent of the
# training seed — Requirement 4.3) plus the same curve-RMSE metric as
# ``eval_testset.evaluate``.


def _normalize_base(model):
    """Normalize a model selector to ``'cnn'`` or ``'transformer'``.

    Args:
        model: Model selector, case-insensitive (``'cnn'`` / ``'transformer'``).

    Returns:
        str: The canonical lowercase base name.

    Raises:
        ValueError: If ``model`` is neither ``'cnn'`` nor ``'transformer'``.
    """
    key = str(model).strip().lower()
    if key not in ("cnn", "transformer"):
        raise ValueError(f"Unknown model {model!r}; expected 'cnn' or 'transformer'.")
    return key


def _cnn_seed_filenames(seed):
    """Per-seed bare filenames for the CNN (resolved under ``<cwd>/results``).

    The CNN config (``Config.get_model_save_path`` etc.) joins ``"results"`` with
    a bare filename, so the override only needs the filenames; the directory is
    fixed to ``results`` relative to the trainer's working directory.
    """
    return {
        "model_save_path": f"best_mlp_model.seed{seed}.pth",
        "x_scaler_file": f"x_scaler.seed{seed}.pkl",
        "y_scaler_file": f"y_scaler.seed{seed}.pkl",
    }


def _cnn_model_path(seed):
    """Absolute path of the CNN per-seed checkpoint (under ``train_cnn/results``)."""
    return os.path.join(_CNN_DIR, "results", f"best_mlp_model.seed{seed}.pth")


def _transformer_seed_relpaths(seed):
    """Per-seed paths for the transformer, relative to its config dir.

    ``TransformerConfig`` resolves these against ``train_transformer/`` (the
    config-file directory), matching the default ``results/...`` convention.
    """
    return {
        "model_save_path": f"results/best_transformer_model.seed{seed}.pth",
        "y_scaler_file": f"results/transformer_y_scaler.seed{seed}.pkl",
    }


def _transformer_model_path(seed):
    """Absolute path of the transformer per-seed checkpoint."""
    return os.path.join(
        _TRANSFORMER_DIR, "results", f"best_transformer_model.seed{seed}.pth"
    )


def write_seed_override_ini(seed, base="cnn"):
    """Write a tiny override INI varying the training seed and output paths.

    The override is layered on top of the base config via the train scripts'
    existing ``--config`` (and, for the transformer, ``--transformer-config``)
    mechanism. It always sets ``[TRAINING] random_seed=<seed>`` (consumed by the
    parent :class:`config_loader.Config`, which drives both the CNN training and
    the transformer's data split) and per-seed model/scaler output paths so
    different seeds never overwrite each other's checkpoints.

    Layout per ``base``:

    * ``base='cnn'`` — a single ``[TRAINING]`` section carrying ``random_seed``
      and the per-seed ``model_save_path`` / ``x_scaler_file`` / ``y_scaler_file``
      (bare filenames; the CNN config places them under ``results/``).
    * ``base='transformer'`` — ``[TRAINING] random_seed`` (read by the parent
      config) **and** a ``[TRANSFORMER]`` section with the per-seed
      ``model_save_path`` / ``y_scaler_file`` (read by ``TransformerConfig``).
      The same file is passed to both ``--config`` and ``--transformer-config``;
      each loader picks up only the section it understands.

    Args:
        seed: The integer training seed to inject.
        base: ``'cnn'`` or ``'transformer'`` (case-insensitive).

    Returns:
        str: Absolute path to the written override INI
        (``results/validation/overrides/<base>_seed<seed>.ini``).
    """
    base = _normalize_base(base)
    os.makedirs(_OVERRIDES_DIR, exist_ok=True)

    parser = configparser.ConfigParser()
    # Preserve the case of option names (configparser lowercases by default).
    parser.optionxform = str

    if base == "cnn":
        names = _cnn_seed_filenames(seed)
        parser["TRAINING"] = {
            "random_seed": str(seed),
            "model_save_path": names["model_save_path"],
            "x_scaler_file": names["x_scaler_file"],
            "y_scaler_file": names["y_scaler_file"],
        }
    else:  # transformer
        rel = _transformer_seed_relpaths(seed)
        # random_seed is read from the PARENT config; the per-seed output paths
        # live in the TRANSFORMER section read by TransformerConfig.
        parser["TRAINING"] = {"random_seed": str(seed)}
        parser["TRANSFORMER"] = {
            "model_save_path": rel["model_save_path"],
            "y_scaler_file": rel["y_scaler_file"],
        }

    out_path = os.path.join(_OVERRIDES_DIR, f"{base}_seed{seed}.ini")
    with open(out_path, "w", encoding="utf-8") as f:
        parser.write(f)
    return out_path


@dataclass
class TrainOutcome:
    """Result of a single seed's training run (Requirement 4.8).

    Attributes:
        seed: The training seed used for this run.
        model_path: Absolute path to the per-seed checkpoint on success, else
            ``None``.
        ok: ``True`` if training completed and produced a checkpoint.
        error: A short error description when ``ok`` is ``False``, else ``None``.
    """

    seed: int
    model_path: "str | None"
    ok: bool
    error: "str | None"


def _stderr_tail(text, limit=2000):
    """Return the trailing ``limit`` characters of captured stderr (or None)."""
    if not text:
        return None
    text = text.strip()
    return text[-limit:] if len(text) > limit else text


def train_one(model, seed):
    """Retrain one model for one seed, capturing failures (continue-on-failure).

    Builds the per-seed override INI, then invokes the existing trainer as an
    isolated subprocess via its ``--config`` mechanism:

    * CNN — ``python train_mlp.py --config <override>`` (cwd ``train_cnn/``);
    * Transformer — ``python train_transformer.py --config <override>
      --transformer-config <override>`` (cwd ``train_transformer/``).

    Any exception, a non-zero exit code, or a missing checkpoint is captured and
    reported as ``TrainOutcome(ok=False, error=...)`` rather than raised, so the
    orchestrator can continue with the remaining seeds (Requirement 4.8). On
    success ``model_path`` points at the per-seed checkpoint (Requirements 4.1,
    4.2).

    Args:
        model: ``'cnn'`` or ``'transformer'``.
        seed: Integer training seed.

    Returns:
        TrainOutcome: The per-seed training outcome.
    """
    try:
        base = _normalize_base(model)
        override = write_seed_override_ini(seed, base)

        if base == "cnn":
            cmd = [sys.executable, "train_mlp.py", "--config", override]
            cwd = _CNN_DIR
            model_path = _cnn_model_path(seed)
        else:  # transformer
            cmd = [
                sys.executable,
                "train_transformer.py",
                "--config", override,
                "--transformer-config", override,
            ]
            cwd = _TRANSFORMER_DIR
            model_path = _transformer_model_path(seed)

        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True
        )

        if result.returncode != 0:
            return TrainOutcome(
                seed=seed,
                model_path=None,
                ok=False,
                error=(
                    f"trainer exited with code {result.returncode}: "
                    f"{_stderr_tail(result.stderr) or '<no stderr>'}"
                ),
            )

        if not os.path.exists(model_path):
            return TrainOutcome(
                seed=seed,
                model_path=None,
                ok=False,
                error=f"checkpoint not found after training: {model_path}",
            )

        return TrainOutcome(seed=seed, model_path=model_path, ok=True, error=None)

    except Exception as exc:  # continue-on-failure (Requirement 4.8)
        return TrainOutcome(seed=seed, model_path=None, ok=False, error=repr(exc))


@contextlib.contextmanager
def _pushd(target_dir):
    """Temporarily change the working directory (restored on exit).

    The CNN config resolves its y_scaler path relative to the working directory,
    so evaluation must run from ``train_cnn/`` for the per-seed scaler to load.
    """
    prev = os.getcwd()
    os.chdir(target_dir)
    try:
        yield
    finally:
        os.chdir(prev)


def _load_eval_module(base):
    """Load the model's ``eval_testset.py`` under a unique module name.

    Both ``train_cnn/eval_testset.py`` and ``train_transformer/eval_testset.py``
    define a module literally named ``eval_testset``; loading them under unique
    names via :mod:`importlib.util` avoids a ``sys.modules`` collision. Each
    module's top-level code inserts the directories it needs onto ``sys.path``
    before importing its (torch-backed) predictor, so the load is self-contained.
    """
    if base == "cnn":
        mod_name = "_multiseed_cnn_eval_testset"
        path = os.path.join(_CNN_DIR, "eval_testset.py")
    else:  # transformer
        mod_name = "_multiseed_transformer_eval_testset"
        path = os.path.join(_TRANSFORMER_DIR, "eval_testset.py")

    if mod_name in sys.modules:
        return sys.modules[mod_name]

    # Ensure the model dir (and repo root) are importable for the module's own
    # ``from inference_* import ...`` / ``import pysim`` statements.
    for p in (_THIS_DIR, _CNN_DIR if base == "cnn" else _TRANSFORMER_DIR):
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def eval_one(model, seed, model_path, test_seed):
    """Evaluate a retrained model on the FIXED held-out test split.

    Reuses the existing ``eval_testset.get_test_split`` with an explicit
    ``test_seed`` so the split is identical across seeds and both architectures
    and independent of the training seed (Requirement 4.3), then computes the
    same per-sample curve RMSE as ``eval_testset.evaluate`` (predict -> forward
    simulate -> RMSE vs. the sample's true curve), returning its mean
    (``curve_rmse_mean``). Any failure is captured and reported as ``None`` so
    the study continues with the remaining seeds (Requirement 4.8).

    Args:
        model: ``'cnn'`` or ``'transformer'``.
        seed: The training seed whose checkpoint/scaler is being evaluated
            (used to locate the per-seed override and output paths).
        model_path: Absolute path to the per-seed checkpoint to evaluate.
        test_seed: Seed driving the fixed held-out split (Requirement 4.3).

    Returns:
        float | None: The mean curve RMSE over valid test samples, or ``None``
        on any failure / when no valid samples remain.
    """
    try:
        import numpy as _np  # local alias; numpy already imported at module top
        import pysim

        base = _normalize_base(model)
        override = write_seed_override_ini(seed, base)
        eval_mod = _load_eval_module(base)

        # Evaluate from the model's directory so cwd-relative scaler paths (CNN)
        # resolve to the per-seed scaler written during training.
        eval_dir = _CNN_DIR if base == "cnn" else _TRANSFORMER_DIR
        with _pushd(eval_dir):
            if base == "cnn":
                predictor = make_predictor(
                    "cnn", model_path=model_path, config_override_file=override
                )
                X_test, Y_test = eval_mod.get_test_split(
                    predictor.config, test_seed=test_seed
                )
            else:  # transformer
                predictor = make_predictor(
                    "transformer",
                    model_path=model_path,
                    parent_config_override_file=override,
                    transformer_config_override_file=override,
                )
                X_test, Y_test = eval_mod.get_test_split(
                    predictor.parent_config, test_seed=test_seed
                )

            n = X_test.shape[0]
            preds = predictor.predict(X_test)  # (n, 7) physical space

            # Same metric as eval_testset.evaluate: per-sample curve RMSE across
            # the 3 channels, averaged over valid samples.
            rmse_list = []
            for i in range(n):
                pdict = {pysim.PARAM_NAMES[j]: float(preds[i, j]) for j in range(7)}
                sig, dt = pysim.run_simulation(pdict)
                if dt < 0 or not _np.all(_np.isfinite(sig)):
                    continue  # invalid simulation (Requirement-consistent skip)
                if _np.max(_np.abs(sig)) > 5.0:
                    continue  # catastrophic / non-physical prediction
                true_curve = X_test[i]
                rmse = _np.sqrt(_np.mean((sig - true_curve) ** 2, axis=1))
                rmse_list.append(rmse)

            if not rmse_list:
                return None  # no valid samples -> treat as a failed evaluation

            rmse_arr = _np.array(rmse_list)            # (m, 3)
            per_sample_mean = rmse_arr.mean(axis=1)    # (m,)
            return float(per_sample_mean.mean())

    except Exception:  # continue-on-failure (Requirement 4.8)
        return None


# ===========================================================================
# Task 8.6 — orchestration, multiseed_metrics.json, comparison figure, CLI
# ===========================================================================
#
# run_multiseed ties together the task 8.3/8.5 building blocks: for each model
# and each seed it retrains (train_one) and, on success, evaluates on the FIXED
# held-out split (eval_one), recording a per-seed entry that always retains the
# failure indication (Requirement 4.5, 4.8). The per-seed metrics are aggregated
# with aggregate() (mean + sample-std with the < 2-success degenerate handling,
# Requirements 4.4, 4.9), serialized to multiseed_metrics.json (Requirement 4.6,
# 6.3), and visualized as the labeled multiseed_compare.png CNN-vs-Transformer
# comparison (Requirement 4.7, 6.4). A failed seed never aborts the study; it is
# recorded and the orchestration continues (Requirement 4.8).

# Canonical names for the two inverse models compared by the study, and the
# metric reported per seed (Requirement 4.3 -> curve_rmse_mean from eval_one).
_MODELS = ("cnn", "transformer")
_METRIC_NAME = "curve_rmse_mean"

# Output artifact names (live under the Results_Directory, Requirement 4.6, 4.7).
_JSON_NAME = "multiseed_metrics.json"
_FIGURE_NAME = "multiseed_compare.png"


def _run_one_model(model, seeds, test_seed, skip_train):
    """Run the full per-seed train+eval sweep for a single model.

    For each seed: retrain via :func:`train_one` (unless ``skip_train`` is set,
    in which case the existing per-seed checkpoint is reused), then — when a
    checkpoint is available — evaluate on the FIXED held-out split via
    :func:`eval_one`. Every seed yields a per-seed entry that always carries its
    success flag and any error message, so failures are retained in the report
    while never aborting the sweep (continue-on-failure, Requirement 4.8).

    Args:
        model: ``'cnn'`` or ``'transformer'``.
        seeds: Iterable of integer training seeds.
        test_seed: Seed driving the fixed held-out split (Requirement 4.3).
        skip_train: When ``True``, skip :func:`train_one` and assume the per-seed
            checkpoint already exists (useful for re-evaluation only).

    Returns:
        dict: The per-model report with keys ``per_seed`` (list of entries) plus
        the aggregate keys (``n_success``, ``mean``, ``std``, ``std_available``,
        ``insufficient_seeds``) — matching the multiseed JSON data model.
    """
    base = _normalize_base(model)
    per_seed = []
    agg_inputs = []  # (value, ok) pairs fed to aggregate()

    for seed in seeds:
        if skip_train:
            # Re-eval mode: assume the per-seed checkpoint already exists; do not
            # invoke the (expensive) trainer. Locate the canonical checkpoint and
            # treat a missing file as a failed seed (recorded, not raised).
            model_path = (
                _cnn_model_path(seed)
                if base == "cnn"
                else _transformer_model_path(seed)
            )
            if os.path.exists(model_path):
                outcome = TrainOutcome(
                    seed=seed, model_path=model_path, ok=True, error=None
                )
            else:
                outcome = TrainOutcome(
                    seed=seed,
                    model_path=None,
                    ok=False,
                    error=(
                        "skip_train=True but checkpoint not found: "
                        f"{model_path}"
                    ),
                )
        else:
            outcome = train_one(base, seed)

        if not outcome.ok:
            # Training failed (or checkpoint missing under skip_train): record the
            # seed with its failure and move on (Requirement 4.8).
            per_seed.append(
                {
                    "seed": seed,
                    "ok": False,
                    _METRIC_NAME: None,
                    "error": outcome.error,
                }
            )
            agg_inputs.append((None, False))
            continue

        # Training ok -> evaluate on the fixed held-out split (Requirement 4.3).
        metric = eval_one(base, seed, outcome.model_path, test_seed)
        if metric is None:
            # Evaluation failed / produced no valid samples: a failed seed.
            per_seed.append(
                {
                    "seed": seed,
                    "ok": False,
                    _METRIC_NAME: None,
                    "error": "evaluation produced no metric",
                }
            )
            agg_inputs.append((None, False))
        else:
            per_seed.append(
                {
                    "seed": seed,
                    "ok": True,
                    _METRIC_NAME: metric,
                    "error": None,
                }
            )
            agg_inputs.append((metric, True))

    # Aggregate over the successful seeds (mean + sample-std, with the
    # < 2-success degenerate handling, Requirements 4.4, 4.9).
    agg = aggregate(agg_inputs)
    return {
        "per_seed": per_seed,
        "n_success": agg["n_success"],
        "mean": agg["mean"],
        "std": agg["std"],
        "std_available": agg["std_available"],
        "insufficient_seeds": agg["insufficient_seeds"],
    }


def _plot_multiseed_compare(models_report, seeds, out_path):
    """Write the labeled CNN-vs-Transformer comparison figure (Requirement 4.7).

    Draws one group per model showing the per-seed test-set metric distribution
    (a strip/scatter of the successful seeds' ``curve_rmse_mean``) overlaid with
    the across-seed mean and, where available, a +/- sample-std error bar. Axes
    name the quantity and unit so the figure is interpretable without the source
    (Requirement 6.6): the y-axis is labeled "Test-set curve RMSE (dimensionless
    fluorescence-fraction RMSE)" and the x-axis names the two models. Uses the
    non-interactive ``Agg`` backend so it renders headless (Requirement 6.4).

    Args:
        models_report: Mapping ``model -> per-model report`` (as produced by
            :func:`_run_one_model`).
        seeds: The seed list (used only for the title/annotation).
        out_path: Destination PNG path.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless rendering (Requirement 6.4)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    # Plot whichever models are present in the report (supports single-model
    # subset runs as well as the full CNN-vs-Transformer comparison).
    present = [m for m in _MODELS if m in models_report]
    present += [m for m in models_report if m not in present]

    labels = []
    for x_pos, model in enumerate(present):
        report = models_report[model]
        labels.append(model.upper())

        # Successful seeds' per-seed metric values for this model.
        values = [
            entry[_METRIC_NAME]
            for entry in report["per_seed"]
            if entry["ok"] and entry[_METRIC_NAME] is not None
        ]

        if values:
            # Strip plot: jitter the points horizontally around the model's slot
            # so overlapping seeds remain visible.
            jitter = (np.linspace(-0.12, 0.12, num=len(values))
                      if len(values) > 1 else [0.0])
            ax.scatter(
                [x_pos + j for j in jitter],
                values,
                color="tab:blue" if model == "cnn" else "tab:orange",
                alpha=0.8,
                zorder=3,
                label=f"{model.upper()} per-seed",
            )

        # Mean +/- sample-std marker (error bar omitted when std unavailable).
        mean = report["mean"]
        if mean is not None:
            yerr = report["std"] if report["std_available"] else None
            ax.errorbar(
                x_pos,
                mean,
                yerr=yerr,
                fmt="D",
                color="black",
                markersize=8,
                capsize=6,
                zorder=4,
                label="mean +/- std" if x_pos == 0 else None,
            )

    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Inverse model")
    # Axis label names the quantity and its unit (Requirement 6.6). The metric is
    # a dimensionless RMSE over normalized fluorescence-fraction curves.
    ax.set_ylabel("Test-set curve RMSE (dimensionless fluorescence-fraction RMSE)")
    ax.set_title(
        "Multi-seed test-set curve RMSE: CNN vs. Transformer\n"
        f"(seeds={list(seeds)})"
    )
    # Only draw a legend when at least one labeled artist exists (avoids a noisy
    # warning in the degenerate all-seeds-failed case).
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    ax.margins(x=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_multiseed(
    seeds=DEFAULT_SEEDS,
    test_seed=42,
    results_dir="results/validation",
    skip_train=False,
    models=None,
):
    """Orchestrate the multi-seed retraining study for the inverse models.

    For each model in ``models`` (default both ``'cnn'`` and ``'transformer'``)
    and each seed in ``seeds``: retrain via :func:`train_one` (skipped when
    ``skip_train`` is ``True``, in which case the existing per-seed checkpoint is
    reused for re-evaluation) and, on success, evaluate on the FIXED held-out
    split (driven by ``test_seed``, identical across seeds and models,
    Requirement 4.3) via :func:`eval_one`. Each seed produces a per-seed entry
    that always retains its failure indication; a failed seed is recorded and
    never aborts the sweep (continue-on-failure, Requirement 4.8).

    The per-seed metrics are aggregated with :func:`aggregate` (mean + sample
    standard deviation with the ``< 2``-success degenerate handling, Requirements
    4.4, 4.9), written to ``multiseed_metrics.json`` (Requirements 4.6, 6.3), and
    visualized as the labeled ``multiseed_compare.png`` CNN-vs-Transformer figure
    (Requirements 4.7, 6.4).

    Args:
        seeds: Iterable of 5 distinct, recorded training seeds (Requirements 4.1,
            4.2). Defaults to :data:`DEFAULT_SEEDS`.
        test_seed: Seed driving the fixed held-out split (Requirement 4.3).
        results_dir: Results_Directory for the JSON + figure artifacts.
        skip_train: When ``True``, skip :func:`train_one` and assume the per-seed
            checkpoints already exist — useful for re-evaluating an existing set
            of checkpoints without paying the (expensive) retraining cost. A
            missing checkpoint is recorded as a failed seed rather than raised.
        models: Optional subset of models to run (``('cnn',)`` /
            ``('transformer',)`` / both). Defaults to both. Running a single
            model in its own process lets the CNN and Transformer sweeps proceed
            in parallel; their per-model JSONs can be merged afterward with
            :func:`merge_results`.

    Returns:
        dict: The full metrics object that was written to
        ``multiseed_metrics.json``.
    """
    seeds = list(seeds)
    models = tuple(_normalize_base(m) for m in (models if models else _MODELS))

    models_report = {
        model: _run_one_model(model, seeds, test_seed, skip_train)
        for model in models
    }

    metrics = {
        "experiment": "multiseed_retraining",
        "metric": _METRIC_NAME,
        "test_seed": test_seed,
        "seeds": seeds,
        "models": models_report,
    }

    # Write JSON (Requirement 4.6, 6.3) and the comparison figure (Req. 4.7, 6.4).
    json_path = os.path.join(results_dir, _JSON_NAME)
    write_json(json_path, metrics)

    figure_path = os.path.join(results_dir, _FIGURE_NAME)
    _plot_multiseed_compare(models_report, seeds, figure_path)

    return metrics


def merge_results(part_dirs, results_dir="results/validation"):
    """Merge per-model multiseed JSONs (from parallel runs) into the final output.

    Each directory in ``part_dirs`` holds a ``multiseed_metrics.json`` produced by
    a single-model :func:`run_multiseed` (e.g. one CNN process and one Transformer
    process run in parallel). This combines their ``models`` blocks into one
    metrics object, writes the canonical ``multiseed_metrics.json`` +
    ``multiseed_compare.png`` into ``results_dir``, and returns the merged dict.
    The ``seeds`` / ``test_seed`` are taken from the first part (they must match
    across parts for a valid comparison).

    Args:
        part_dirs: Iterable of directories each containing a per-model
            ``multiseed_metrics.json``.
        results_dir: Destination Results_Directory for the merged artifacts.

    Returns:
        dict: The merged metrics object.
    """
    import json

    merged_models = {}
    seeds = None
    test_seed = None
    for d in part_dirs:
        with open(os.path.join(d, _JSON_NAME), encoding="utf-8") as f:
            part = json.load(f)
        if seeds is None:
            seeds = part["seeds"]
            test_seed = part["test_seed"]
        merged_models.update(part["models"])

    # Order the merged models canonically (cnn, transformer) when present.
    ordered = {m: merged_models[m] for m in _MODELS if m in merged_models}
    ordered.update({m: v for m, v in merged_models.items() if m not in ordered})

    metrics = {
        "experiment": "multiseed_retraining",
        "metric": _METRIC_NAME,
        "test_seed": test_seed,
        "seeds": seeds,
        "models": ordered,
    }
    write_json(os.path.join(results_dir, _JSON_NAME), metrics)
    _plot_multiseed_compare(ordered, seeds, os.path.join(results_dir, _FIGURE_NAME))
    return metrics


def _parse_seeds(text):
    """Parse a comma-separated ``--seeds`` CLI value into a tuple of ints."""
    return tuple(int(part) for part in str(text).split(",") if part.strip() != "")


def main(argv=None):
    """CLI entry point for the multi-seed retraining study (Requirement 4.4-4.9).

    Flags:
      ``--seeds``        comma-separated training seeds (default: ``DEFAULT_SEEDS``)
      ``--test-seed``    fixed held-out split seed (default: ``42``)
      ``--results-dir``  Results_Directory (default: ``results/validation``)
      ``--skip-train``   reuse existing per-seed checkpoints (re-eval only)
    """
    parser = argparse.ArgumentParser(
        description=(
            "Multi-seed retraining study: retrain CNN and Transformer across "
            "several seeds, evaluate each on a fixed held-out test set, and "
            "report per-seed metrics with their across-seed mean +/- std."
        )
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help=(
            "Comma-separated list of training seeds "
            f"(default: {','.join(str(s) for s in DEFAULT_SEEDS)})."
        ),
    )
    parser.add_argument(
        "--test-seed",
        type=int,
        default=42,
        help="Seed driving the fixed held-out test split (default: 42).",
    )
    parser.add_argument(
        "--results-dir",
        default="results/validation",
        help="Results_Directory for the JSON and figure (default: results/validation).",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Skip retraining and reuse existing per-seed checkpoints "
            "(re-evaluation only; missing checkpoints are recorded as failures)."
        ),
    )
    parser.add_argument(
        "--models",
        type=lambda s: tuple(p.strip().lower() for p in s.split(",") if p.strip()),
        default=None,
        help=(
            "Comma-separated subset of models to run (cnn / transformer / both). "
            "Default runs both. Run a single model per process to parallelize the "
            "CNN and Transformer sweeps, then merge with --merge."
        ),
    )
    args = parser.parse_args(argv)

    metrics = run_multiseed(
        seeds=args.seeds,
        test_seed=args.test_seed,
        results_dir=args.results_dir,
        skip_train=args.skip_train,
        models=args.models,
    )

    json_path = os.path.join(args.results_dir, _JSON_NAME)
    figure_path = os.path.join(args.results_dir, _FIGURE_NAME)
    print(f"Wrote {json_path}")
    print(f"Wrote {figure_path}")
    for model in metrics["models"]:
        report = metrics["models"][model]
        print(
            f"  {model}: n_success={report['n_success']} "
            f"mean={report['mean']} std={report['std']}"
        )
    return metrics


if __name__ == "__main__":
    main()
