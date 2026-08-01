# coding=utf-8
"""Canonical multi-seed retraining study and command-line implementation.

This module retrains both inverse models (CNN and Transformer) across several
recorded random seeds, evaluates each retrained model on a fixed held-out test
set, and reports the per-seed test-set metric together with its across-seed mean
and sample standard deviation. The completed experiment measures whether a
single-checkpoint model ranking remains stable across training seeds; the code
does not assume that outcome in advance.

The canonical implementation lives in :mod:`dnawalker.studies.multiseed`;
all training subprocesses use package module entry points.
"""

import argparse
import importlib
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import replace
from numbers import Real
from pathlib import Path

from dnawalker.config import Config
from dnawalker.paths import DEFAULT_CONFIG, REPO_ROOT, RESULTS_DIR
from dnawalker.shared.artifacts import sha256_file
from .constants import (
    EXPERIMENT_NAME,
    FIGURE_NAME,
    JSON_NAME,
    METRIC_NAME,
    MODEL_NAMES,
)
from .evaluation import EvalOutcome
from .reporting import (
    load_merged_metrics,
    plot_multiseed_compare,
)
from .runtime import (
    TrainOutcome,
    build_model_specs,
    checkpoint_split_seed,
    execute_training,
    normalize_model,
    stderr_tail,
    write_override_file,
)
from .statistics import aggregate
from dnawalker.shared.parameters import vector_to_param_dict
from ..protocol import make_predictor, require_seed, write_json

# Repository layout anchors are imported from ``dnawalker.paths`` so this
# orchestration is independent of the caller's working directory.
_THIS_DIR = REPO_ROOT

# Five distinct, recorded seeds for the study (Requirement 4.1, 4.2).
DEFAULT_SEEDS = (42, 43, 44, 45, 46)

_MODEL_SPECS = build_model_specs(REPO_ROOT)
_ROOT_CONFIG = os.fspath(DEFAULT_CONFIG)

# Override INIs are written here so per-seed runs do not clobber each other and
# the artifacts stay grouped with the suite's other validation outputs.
_DEFAULT_RESULTS_DIR = os.fspath(
    RESULTS_DIR / "validation" / "multiseed"
)
_OVERRIDES_DIR = os.path.join(_DEFAULT_RESULTS_DIR, "overrides")
_DATASET_PATH = None
_SPLIT_MANIFEST_PATH = None
_TRAIN_SUBSET_SIZE = None
_LOGS_DIR = os.path.join(_DEFAULT_RESULTS_DIR, "logs")
_ALLOW_OVERWRITE = False


@contextmanager
def _configured_runtime(
        *, dataset_path=None, artifacts_dir=None, overrides_dir=None,
        logs_dir=None, allow_overwrite=False, split_manifest_path=None,
        train_subset_size=None):
    """Temporarily bind one process to an isolated release namespace."""
    global _MODEL_SPECS, _OVERRIDES_DIR, _DATASET_PATH, _LOGS_DIR
    global _ALLOW_OVERWRITE, _SPLIT_MANIFEST_PATH, _TRAIN_SUBSET_SIZE

    previous = (
        _MODEL_SPECS,
        _OVERRIDES_DIR,
        _DATASET_PATH,
        _SPLIT_MANIFEST_PATH,
        _TRAIN_SUBSET_SIZE,
        _LOGS_DIR,
        _ALLOW_OVERWRITE,
    )
    try:
        _MODEL_SPECS = build_model_specs(
            REPO_ROOT, artifact_root=artifacts_dir
        )
        _OVERRIDES_DIR = str(Path(overrides_dir).resolve())
        _DATASET_PATH = (
            str(Path(dataset_path).resolve())
            if dataset_path is not None
            else None
        )
        _SPLIT_MANIFEST_PATH = (
            str(Path(split_manifest_path).resolve())
            if split_manifest_path is not None
            else None
        )
        _TRAIN_SUBSET_SIZE = (
            int(train_subset_size)
            if train_subset_size is not None
            else None
        )
        _LOGS_DIR = str(Path(logs_dir).resolve())
        _ALLOW_OVERWRITE = bool(allow_overwrite)
        yield
    finally:
        (
            _MODEL_SPECS,
            _OVERRIDES_DIR,
            _DATASET_PATH,
            _SPLIT_MANIFEST_PATH,
            _TRAIN_SUBSET_SIZE,
            _LOGS_DIR,
            _ALLOW_OVERWRITE,
        ) = previous


# ===========================================================================
# Task 8.5 — seed-override training & evaluation (continue-on-failure)
# ===========================================================================
#
# These helpers vary the training seed by layering a tiny override INI on top of
# the base config(s) accepted by the canonical ``python -m`` training entry
# points. Per-seed model/scaler output paths keep checkpoints isolated. Training
# runs in a fresh subprocess so each seed gets independent process/RNG state and
# one failed seed cannot stop the whole study (Requirement 4.8).


def _normalize_base(model):
    """Backward-compatible alias for :func:`multiseed.runtime.normalize_model`."""
    return normalize_model(model)


def _cnn_model_path(seed):
    """Absolute path of the CNN per-seed checkpoint in ``artifacts/``."""
    return _MODEL_SPECS["cnn"].checkpoint_path(seed)


def _transformer_model_path(seed):
    """Absolute path of the transformer per-seed checkpoint."""
    return _MODEL_SPECS["transformer"].checkpoint_path(seed)


def write_seed_override_ini(seed, base="cnn", split_seed=None):
    """Write a tiny override INI varying the training seed and output paths.

    The override is layered on top of the base config via the train scripts'
    existing ``--config`` (and, for the transformer, ``--transformer-config``)
    mechanism. It sets ``[TRAINING] random_seed=<seed>`` for model stochasticity,
    a separate fixed ``split_seed``, and per-seed model/scaler output paths so
    different seeds never overwrite each other's checkpoints.

    Layout per ``base``:

    * ``base='cnn'`` — a single ``[TRAINING]`` section carrying
      ``random_seed``, ``split_seed``, and the per-seed ``model_save_path`` /
      ``y_scaler_file`` (bare filenames; the CNN config places them under
      ``artifacts/models/cnn``).
    * ``base='transformer'`` — ``[TRAINING] random_seed`` and ``split_seed``
      (read by the parent config) **and** a ``[TRANSFORMER]`` section with the per-seed
      ``model_save_path`` / ``y_scaler_file`` (read by ``TransformerConfig``).
      The same file is passed to both ``--config`` and ``--transformer-config``;
      each loader picks up only the section it understands.

    Args:
        seed: The integer training seed to inject.
        base: ``'cnn'`` or ``'transformer'`` (case-insensitive).
        split_seed: Fixed dataset-membership seed. Defaults to the root config.

    Returns:
        str: Absolute path to the written override INI
        (``results/validation/multiseed/overrides/<base>_seed<seed>.ini``).
    """
    base = _normalize_base(base)
    if split_seed is None:
        split_seed = Config(_ROOT_CONFIG).get_split_seed()
    return write_override_file(
        _MODEL_SPECS[base],
        seed=seed,
        split_seed=split_seed,
        override_dir=_OVERRIDES_DIR,
        dataset_path=_DATASET_PATH,
        split_manifest_path=_SPLIT_MANIFEST_PATH,
        train_subset_size=_TRAIN_SUBSET_SIZE,
    )


def _checkpoint_split_seed(path):
    """Read split provenance from a safe tensor checkpoint, or return ``None``."""
    return checkpoint_split_seed(path)


def _stderr_tail(text, limit=2000):
    """Return the trailing ``limit`` characters of captured stderr (or None)."""
    return stderr_tail(text, limit=limit)


def train_one(model, seed, split_seed=None):
    """Retrain one model for one seed, capturing failures (continue-on-failure).

    Builds the per-seed override INI, then invokes the canonical trainer as an
    isolated package-module subprocess:

    * CNN — ``python -m dnawalker.cnn.train --config <override>``;
    * Transformer — ``python -m dnawalker.transformer.train --config
      <override> --transformer-config <override>``.

    Both commands run from the repository root. Any exception, a non-zero exit
    code, or a missing checkpoint is captured and reported as ``TrainOutcome(ok=False, error=...)`` rather than raised, so the
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
        spec = _MODEL_SPECS[base]
        protected_paths = (
            spec.checkpoint_path(seed),
            spec.scaler_path(seed),
        )
        existing = [path for path in protected_paths if os.path.exists(path)]
        if existing and not _ALLOW_OVERWRITE:
            return TrainOutcome(
                seed=seed,
                model_path=None,
                ok=False,
                error=(
                    "refusing to overwrite existing release artifacts: "
                    + ", ".join(existing)
                ),
            )
        override = write_seed_override_ini(
            seed, base, split_seed=split_seed
        )
        log_path = os.path.join(_LOGS_DIR, f"{base}_seed{seed}.log")
        return execute_training(
            spec,
            seed=seed,
            override_path=override,
            python_executable=sys.executable,
            log_path=log_path,
        )
    except Exception as exc:  # continue-on-failure (Requirement 4.8)
        return TrainOutcome(seed=seed, model_path=None, ok=False, error=repr(exc))


def _load_eval_module(base):
    """Import the model-specific evaluation adapter by its package name."""
    module_name = (
        "dnawalker.cnn.evaluate"
        if base == "cnn"
        else "dnawalker.transformer.evaluate"
    )
    return importlib.import_module(module_name)


def _validate_evaluation_provenance(predictor, config):
    """Lazily validate hashes without expanding the facade's import graph."""
    from dnawalker.shared.evaluation import evaluation_provenance

    return evaluation_provenance(predictor, config)


def _require_current_checkpoint_provenance(predictor, seed, test_seed):
    """Reject legacy or partial checkpoints from scientific aggregation."""
    required_values = {
        "model_seed": getattr(
            predictor, "checkpoint_model_seed", None
        ),
        "split_seed": getattr(
            predictor, "checkpoint_split_seed", None
        ),
        "dataset_sha256": getattr(
            predictor, "checkpoint_dataset_sha256", None
        ),
        "y_scaler_sha256": getattr(
            predictor, "checkpoint_y_scaler_sha256", None
        ),
        "epoch": getattr(predictor, "checkpoint_epoch", None),
        "val_mse": getattr(predictor, "checkpoint_val_mse", None),
    }
    missing = [
        name for name, value in required_values.items()
        if value is None
    ]
    if not getattr(
        predictor, "checkpoint_param_names_present", False
    ):
        missing.append("param_names")
    if missing:
        raise ValueError(
            "checkpoint lacks current provenance: "
            + ", ".join(sorted(missing))
        )

    if required_values["model_seed"] != seed:
        raise ValueError(
            "checkpoint model_seed does not match run seed: "
            f"{required_values['model_seed']} vs {seed}"
        )
    if required_values["split_seed"] != test_seed:
        raise ValueError(
            "checkpoint split_seed does not match test seed: "
            f"{required_values['split_seed']} vs {test_seed}"
        )
    epoch = required_values["epoch"]
    if (isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch <= 0):
        raise ValueError(
            f"checkpoint epoch must be a positive integer, got {epoch!r}"
        )
    val_mse = required_values["val_mse"]
    if (isinstance(val_mse, bool)
            or not isinstance(val_mse, Real)
            or not math.isfinite(float(val_mse))):
        raise ValueError(
            f"checkpoint val_mse must be finite, got {val_mse!r}"
        )


def _evaluation_result(
        outcome, return_outcome, provenance=None):
    """Return a detailed outcome or the backward-compatible scalar metric."""
    if provenance is not None and outcome.provenance is None:
        outcome = replace(outcome, provenance=dict(provenance))
    return outcome if return_outcome else outcome.curve_rmse_mean


def eval_one(model, seed, model_path, test_seed, *, return_outcome=False):
    """Evaluate a retrained model on the FIXED held-out test split.

    Reuses the existing ``eval_testset.get_test_split`` with an explicit
    ``test_seed`` so the split is identical across seeds and both architectures
    and independent of the training seed (Requirement 4.3), then computes the
    same per-sample curve RMSE as ``eval_testset.evaluate`` (predict -> forward
    simulate -> RMSE vs. the sample's true curve), returning its mean
    (``curve_rmse_mean``). Invalid and catastrophic simulations are counted
    explicitly so a low metric cannot hide poor test-sample coverage. Any
    failure is captured so the study continues with the remaining seeds
    (Requirement 4.8).

    Args:
        model: ``'cnn'`` or ``'transformer'``.
        seed: The training seed whose checkpoint/scaler is being evaluated
            (used to locate the per-seed override and output paths).
        model_path: Absolute path to the per-seed checkpoint to evaluate.
        test_seed: Seed driving the fixed held-out split (Requirement 4.3).
        return_outcome: Return :class:`EvalOutcome` with sample counts instead
            of the backward-compatible scalar metric.

    Returns:
        float | None | EvalOutcome: By default, the mean curve RMSE over valid
        test samples or ``None`` on failure. With ``return_outcome=True``, a
        validated detailed result including all test-sample counts.
    """
    n_test_samples = 0
    n_valid = 0
    n_invalid = 0
    n_extreme = 0
    provenance = None
    try:
        import numpy as _np
        from dnawalker.physics import simulator as pysim

        base = _normalize_base(model)
        override = write_seed_override_ini(seed, base, split_seed=test_seed)
        eval_mod = _load_eval_module(base)

        if base == "cnn":
            predictor = make_predictor(
                "cnn", model_path=model_path, config_override_file=override
            )
            _require_current_checkpoint_provenance(
                predictor, seed, test_seed
            )
            provenance = _validate_evaluation_provenance(
                predictor, predictor.config
            )
            X_test, _ = eval_mod.get_test_split(
                predictor.config, test_seed=test_seed
            )
        else:  # transformer
            predictor = make_predictor(
                "transformer",
                model_path=model_path,
                parent_config_override_file=override,
                transformer_config_override_file=override,
            )
            _require_current_checkpoint_provenance(
                predictor, seed, test_seed
            )
            provenance = _validate_evaluation_provenance(
                predictor, predictor.parent_config
            )
            X_test, _ = eval_mod.get_test_split(
                predictor.parent_config, test_seed=test_seed
            )

        X_test = _np.asarray(X_test)
        n = int(X_test.shape[0])
        n_test_samples = n
        prediction_names = predictor.get_param_names()
        preds = _np.asarray(
            predictor.predict(X_test), dtype=_np.float64
        )
        expected_shape = (n, len(prediction_names))
        if preds.shape != expected_shape:
            outcome = EvalOutcome(
                curve_rmse_mean=None,
                n_test_samples=n,
                n_valid=0,
                n_invalid=n,
                n_extreme=0,
                error=(
                    "predictor output shape mismatch: "
                    f"expected {expected_shape}, got {preds.shape}"
                ),
            )
            return _evaluation_result(outcome, return_outcome, provenance)
        if not _np.all(_np.isfinite(preds)):
            outcome = EvalOutcome(
                curve_rmse_mean=None,
                n_test_samples=n,
                n_valid=0,
                n_invalid=n,
                n_extreme=0,
                error="predictor produced NaN or Inf",
            )
            return _evaluation_result(outcome, return_outcome, provenance)

        # Same metric as eval_testset.evaluate: per-sample curve RMSE across
        # the 3 channels, averaged over valid samples.
        rmse_list = []
        for i in range(n):
            try:
                pdict = vector_to_param_dict(
                    preds[i],
                    prediction_names,
                )
                sig, dt = pysim.run_simulation(pdict)
                sig = _np.asarray(sig, dtype=_np.float64)
                if (not _np.isfinite(dt)
                        or dt < 0
                        or sig.shape != X_test[i].shape
                        or not _np.all(_np.isfinite(sig))):
                    n_invalid += 1
                    continue
                if _np.max(_np.abs(sig)) > 5.0:
                    n_extreme += 1
                    continue
                true_curve = X_test[i]
                rmse = _np.sqrt(
                    _np.mean((sig - true_curve) ** 2, axis=1)
                )
                if not _np.all(_np.isfinite(rmse)):
                    n_invalid += 1
                    continue
            except Exception:
                n_invalid += 1
                continue
            rmse_list.append(rmse)
            n_valid += 1

        if not rmse_list:
            outcome = EvalOutcome(
                curve_rmse_mean=None,
                n_test_samples=n,
                n_valid=n_valid,
                n_invalid=n_invalid,
                n_extreme=n_extreme,
                error="evaluation produced no valid test samples",
            )
            return _evaluation_result(outcome, return_outcome, provenance)

        rmse_arr = _np.array(rmse_list)            # (m, 3)
        per_sample_mean = rmse_arr.mean(axis=1)    # (m,)
        outcome = EvalOutcome(
            curve_rmse_mean=float(per_sample_mean.mean()),
            n_test_samples=n,
            n_valid=n_valid,
            n_invalid=n_invalid,
            n_extreme=n_extreme,
            error=None,
        )
        return _evaluation_result(outcome, return_outcome, provenance)

    except Exception as exc:  # continue-on-failure (Requirement 4.8)
        accounted = n_valid + n_invalid + n_extreme
        n_invalid += max(n_test_samples - accounted, 0)
        outcome = EvalOutcome(
            curve_rmse_mean=None,
            n_test_samples=n_test_samples,
            n_valid=n_valid,
            n_invalid=n_invalid,
            n_extreme=n_extreme,
            error=f"{type(exc).__name__}: {exc}",
        )
        return _evaluation_result(outcome, return_outcome, provenance)


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
_MODELS = MODEL_NAMES
_METRIC_NAME = METRIC_NAME

# Output artifact names (live under the Results_Directory, Requirement 4.6, 4.7).
_JSON_NAME = JSON_NAME
_FIGURE_NAME = FIGURE_NAME


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
    unevaluated_counts = {
        "n_test_samples": None,
        "n_valid": None,
        "n_invalid": None,
        "n_extreme": None,
        "provenance": None,
    }

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
            checkpoint_split = (
                _checkpoint_split_seed(model_path)
                if os.path.exists(model_path)
                else None
            )
            if checkpoint_split == test_seed:
                outcome = TrainOutcome(
                    seed=seed, model_path=model_path, ok=True, error=None
                )
            elif os.path.exists(model_path):
                outcome = TrainOutcome(
                    seed=seed,
                    model_path=None,
                    ok=False,
                    error=(
                        "checkpoint has missing or incompatible split_seed "
                        f"(expected {test_seed}, found {checkpoint_split}); "
                        "retrain this seed"
                    ),
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
            outcome = train_one(base, seed, split_seed=test_seed)

        if not outcome.ok:
            # Training failed (or checkpoint missing under skip_train): record the
            # seed with its failure and move on (Requirement 4.8).
            per_seed.append(
                {
                    "seed": seed,
                    "ok": False,
                    _METRIC_NAME: None,
                    "error": outcome.error,
                    "training_log": outcome.log_path,
                    **unevaluated_counts,
                }
            )
            agg_inputs.append((None, False))
            continue

        # Training ok -> evaluate on the fixed held-out split (Requirement 4.3).
        try:
            evaluation = eval_one(
                base,
                seed,
                outcome.model_path,
                test_seed,
                return_outcome=True,
            )
        except Exception as exc:
            evaluation = EvalOutcome(
                curve_rmse_mean=None,
                n_test_samples=0,
                n_valid=0,
                n_invalid=0,
                n_extreme=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        if not isinstance(evaluation, EvalOutcome):
            evaluation = EvalOutcome(
                curve_rmse_mean=None,
                n_test_samples=0,
                n_valid=0,
                n_invalid=0,
                n_extreme=0,
                error=(
                    "evaluation did not return the required EvalOutcome "
                    f"(got {type(evaluation).__name__})"
                ),
            )
        metric = evaluation.curve_rmse_mean
        if not evaluation.ok:
            # Evaluation failed / produced no valid samples: a failed seed.
            per_seed.append(
                {
                    "seed": seed,
                    "ok": False,
                    _METRIC_NAME: None,
                    "error": evaluation.error,
                    "training_log": outcome.log_path,
                    **evaluation.count_fields(),
                    "provenance": evaluation.provenance,
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
                    "training_log": outcome.log_path,
                    **evaluation.count_fields(),
                    "provenance": evaluation.provenance,
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
    """Compatibility wrapper around :mod:`multiseed.reporting`."""
    return plot_multiseed_compare(models_report, seeds, out_path)


def run_multiseed(
    seeds=DEFAULT_SEEDS,
    test_seed=42,
    results_dir=_DEFAULT_RESULTS_DIR,
    skip_train=False,
    models=None,
    dataset_path=None,
    artifacts_dir=None,
    overrides_dir=None,
    logs_dir=None,
    allow_overwrite=False,
    split_manifest_path=None,
    train_subset_size=None,
):
    """Run an isolated multi-seed train/evaluate sweep.

    ``dataset_path`` and ``artifacts_dir`` form the formal-release boundary.
    When training with either one supplied, both are required. Seeded
    checkpoints/scalers are written below ``artifacts_dir/<model>`` and the
    generated overrides/logs default below ``results_dir``. Existing target
    artifacts are rejected unless ``allow_overwrite`` is explicitly true.
    """
    seeds = [require_seed(seed, "run_multiseed:seeds") for seed in seeds]
    if not seeds:
        raise ValueError("seeds must contain at least one training seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be distinct, got {seeds}")
    test_seed = require_seed(test_seed, "run_multiseed:test_seed")
    if not isinstance(skip_train, bool):
        raise ValueError(f"skip_train must be a boolean, got {skip_train!r}")
    if not isinstance(allow_overwrite, bool):
        raise ValueError(
            "allow_overwrite must be a boolean, "
            f"got {allow_overwrite!r}"
        )
    if (split_manifest_path is None) != (train_subset_size is None):
        raise ValueError(
            "split_manifest_path and train_subset_size must be provided together"
        )
    if train_subset_size is not None:
        if (
            isinstance(train_subset_size, bool)
            or not isinstance(train_subset_size, int)
            or train_subset_size <= 0
        ):
            raise ValueError(
                "train_subset_size must be a positive integer, got "
                f"{train_subset_size!r}"
            )

    selected_models = _MODELS if models is None else tuple(models)
    if not selected_models:
        raise ValueError("models must contain at least one model")
    models = tuple(_normalize_base(model) for model in selected_models)
    if len(set(models)) != len(models):
        raise ValueError(f"models must be distinct, got {models}")

    explicit_release_paths = dataset_path is not None or artifacts_dir is not None
    if not skip_train and explicit_release_paths:
        if dataset_path is None or artifacts_dir is None:
            raise ValueError(
                "formal training requires both dataset_path and artifacts_dir"
            )

    resolved_dataset = None
    if dataset_path is not None:
        candidate = Path(dataset_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"release dataset not found: {candidate}"
            )
        resolved_dataset = str(candidate)
    resolved_manifest = None
    if split_manifest_path is not None:
        candidate = Path(split_manifest_path).resolve()
        if not candidate.is_file():
            raise FileNotFoundError(
                f"split manifest not found: {candidate}"
            )
        resolved_manifest = str(candidate)

    results_dir = str(Path(results_dir).resolve())
    resolved_artifacts = (
        str(Path(artifacts_dir).resolve())
        if artifacts_dir is not None
        else None
    )
    resolved_overrides = str(
        Path(overrides_dir or os.path.join(results_dir, "overrides")).resolve()
    )
    resolved_logs = str(
        Path(logs_dir or os.path.join(results_dir, "logs")).resolve()
    )

    with _configured_runtime(
        dataset_path=resolved_dataset,
        artifacts_dir=resolved_artifacts,
        overrides_dir=resolved_overrides,
        logs_dir=resolved_logs,
        allow_overwrite=allow_overwrite,
        split_manifest_path=resolved_manifest,
        train_subset_size=train_subset_size,
    ):
        if (not skip_train and artifacts_dir is not None
                and not allow_overwrite):
            collisions = []
            for model in models:
                spec = _MODEL_SPECS[model]
                for seed in seeds:
                    for path in (
                        spec.checkpoint_path(seed), spec.scaler_path(seed)
                    ):
                        if os.path.exists(path):
                            collisions.append(path)
            if collisions:
                raise FileExistsError(
                    "refusing to overwrite existing release artifacts: "
                    + ", ".join(collisions)
                )

        models_report = {
            model: _run_one_model(model, seeds, test_seed, skip_train)
            for model in models
        }

        metrics = {
            "experiment": EXPERIMENT_NAME,
            "metric": _METRIC_NAME,
            "test_seed": test_seed,
            "seeds": seeds,
            "models": models_report,
        }
        if resolved_dataset is not None:
            metrics["run_provenance"] = {
                "dataset_path": resolved_dataset,
                "dataset_sha256": sha256_file(resolved_dataset),
                "artifacts_dir": (
                    resolved_artifacts
                    if resolved_artifacts is not None
                    else str(
                        Path(_MODEL_SPECS["cnn"].artifact_dir).parent
                    )
                ),
                "overrides_dir": resolved_overrides,
                "logs_dir": resolved_logs,
            }
            if resolved_manifest is not None:
                metrics["run_provenance"].update({
                    "split_manifest_path": resolved_manifest,
                    "split_manifest_sha256": sha256_file(
                        resolved_manifest
                    ),
                    "train_subset_size": int(train_subset_size),
                })

        json_path = os.path.join(results_dir, _JSON_NAME)
        write_json(json_path, metrics)

        figure_path = os.path.join(results_dir, _FIGURE_NAME)
        _plot_multiseed_compare(models_report, seeds, figure_path)

    return metrics


def merge_results(part_dirs, results_dir=_DEFAULT_RESULTS_DIR):
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
    metrics = load_merged_metrics(part_dirs)
    write_json(os.path.join(results_dir, _JSON_NAME), metrics)
    _plot_multiseed_compare(
        metrics["models"],
        metrics["seeds"],
        os.path.join(results_dir, _FIGURE_NAME),
    )
    return metrics


def _parse_seeds(text):
    """Parse a comma-separated ``--seeds`` CLI value into a tuple of ints."""
    return tuple(int(part) for part in str(text).split(",") if part.strip() != "")


def main(argv=None):
    """CLI entry point for the multi-seed retraining study (Requirement 4.4-4.9).

    Flags:
      ``--seeds``        comma-separated training seeds (default: ``DEFAULT_SEEDS``)
      ``--test-seed``    fixed held-out split seed (default: ``42``)
      ``--results-dir``  Results_Directory (default: ``results/validation/multiseed``)
      ``--skip-train``   reuse existing per-seed checkpoints (re-eval only)
      ``--merge``        merge compatible single-model result directories
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
        default=_DEFAULT_RESULTS_DIR,
        help=(
            "Results_Directory for the JSON and figure "
            "(default: repository results/validation/multiseed/multiseed)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "Exact release dataset .npz. Formal training requires this together "
            "with --artifacts-dir."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "Isolated model-artifact root; checkpoints/scalers are written "
            "under <dir>/cnn and <dir>/transformer."
        ),
    )
    parser.add_argument(
        "--overrides-dir",
        default=None,
        help="Directory for generated per-seed INIs (default: <results-dir>/overrides).",
    )
    parser.add_argument(
        "--logs-dir",
        default=None,
        help="Directory for captured trainer logs (default: <results-dir>/logs).",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help=(
            "Explicitly allow replacement of existing seeded artifacts. "
            "Never use for a formal immutable release run."
        ),
    )
    parser.add_argument(
        "--split-manifest",
        default=None,
        help=(
            "Explicit activity-stratified split manifest. Requires "
            "--train-subset-size."
        ),
    )
    parser.add_argument(
        "--train-subset-size",
        type=int,
        default=None,
        help="Training rows selected from --split-manifest.",
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
    parser.add_argument(
        "--merge",
        nargs="+",
        metavar="PART_DIR",
        default=None,
        help=(
            "Merge compatible single-model result directories instead of "
            "training. Writes the combined JSON and figure to --results-dir."
        ),
    )
    args = parser.parse_args(argv)

    if args.merge:
        metrics = merge_results(args.merge, results_dir=args.results_dir)
    else:
        metrics = run_multiseed(
            seeds=args.seeds,
            test_seed=args.test_seed,
            results_dir=args.results_dir,
            skip_train=args.skip_train,
            models=args.models,
            dataset_path=args.dataset,
            artifacts_dir=args.artifacts_dir,
            overrides_dir=args.overrides_dir,
            logs_dir=args.logs_dir,
            allow_overwrite=args.allow_overwrite,
            split_manifest_path=args.split_manifest,
            train_subset_size=args.train_subset_size,
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


def cli(argv=None):
    """Console-script wrapper returning a process exit status."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
