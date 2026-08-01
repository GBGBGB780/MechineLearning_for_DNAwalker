"""Reporting and result-merging helpers for multi-seed experiments."""

import json
import math
import os
from numbers import Integral, Real

import numpy as np

from ..protocol import PARAM_NAMES
from .constants import (
    EXPERIMENT_NAME,
    JSON_NAME,
    METRIC_NAME,
    MODEL_NAMES,
)
from .statistics import aggregate


def _reject_json_constant(value):
    """Reject Python's non-standard NaN/Infinity JSON extensions."""
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _validated_seeds(values, name):
    """Return a non-empty, distinct list of non-negative integer seeds."""
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty JSON array")
    seeds = []
    for value in values:
        if (isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0):
            raise ValueError(
                f"{name} must contain non-negative integer seeds, got {value!r}"
            )
        seeds.append(int(value))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{name} must contain distinct seeds, got {seeds}")
    return seeds


def _same_optional_float(actual, expected):
    if expected is None:
        return actual is None
    return (
        isinstance(actual, Real)
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and math.isclose(
            float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-15
        )
    )


def _validate_provenance(
        provenance, *, model, expected_seed, test_seed, required):
    """Validate and return the dataset hash for one evaluated seed."""
    if provenance is None:
        if required:
            raise ValueError(
                f"{model} seed {expected_seed} successful metric lacks "
                "evaluation provenance"
            )
        return None
    if not isinstance(provenance, dict):
        raise ValueError(
            f"{model} seed {expected_seed} provenance must be an object or null"
        )

    required_keys = {
        "param_names",
        "split_seed",
        "checkpoint_model_seed",
        "checkpoint_split_seed",
        "checkpoint_dataset_sha256",
        "checkpoint_y_scaler_sha256",
        "checkpoint_epoch",
        "checkpoint_val_mse",
        "checkpoint_param_names_present",
        "dataset_path",
        "dataset_sha256",
        "checkpoint_path",
        "checkpoint_sha256",
        "y_scaler_path",
        "y_scaler_sha256",
        "device",
    }
    missing = sorted(required_keys.difference(provenance))
    if missing:
        raise ValueError(
            f"{model} seed {expected_seed} provenance lacks fields: {missing}"
        )

    param_names = provenance["param_names"]
    if (
        not isinstance(param_names, list)
        or any(
            not isinstance(name, str) or not name.strip()
            for name in param_names
        )
        or [name.lower() for name in param_names]
        != [name.lower() for name in PARAM_NAMES]
    ):
        raise ValueError(
            f"{model} seed {expected_seed} provenance parameter order mismatch"
        )

    seed_fields = {
        "checkpoint_model_seed": expected_seed,
        "split_seed": test_seed,
        "checkpoint_split_seed": test_seed,
    }
    for name, expected in seed_fields.items():
        value = provenance[name]
        if (isinstance(value, bool)
                or not isinstance(value, Integral)
                or int(value) != expected):
            raise ValueError(
                f"{model} seed {expected_seed} provenance {name} mismatch"
            )

    epoch = provenance["checkpoint_epoch"]
    if (isinstance(epoch, bool)
            or not isinstance(epoch, Integral)
            or epoch <= 0):
        raise ValueError(
            f"{model} seed {expected_seed} has invalid provenance "
            "checkpoint_epoch"
        )
    val_mse = provenance["checkpoint_val_mse"]
    if (isinstance(val_mse, bool)
            or not isinstance(val_mse, Real)
            or not math.isfinite(float(val_mse))):
        raise ValueError(
            f"{model} seed {expected_seed} has invalid provenance "
            "checkpoint_val_mse"
        )
    if provenance["checkpoint_param_names_present"] is not True:
        raise ValueError(
            f"{model} seed {expected_seed} provenance lacks checkpoint "
            "parameter metadata"
        )

    hash_names = (
        "checkpoint_dataset_sha256",
        "checkpoint_y_scaler_sha256",
        "dataset_sha256",
        "checkpoint_sha256",
        "y_scaler_sha256",
    )
    normalized_hashes = {}
    for name in hash_names:
        value = provenance[name]
        if (not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value.lower())):
            raise ValueError(
                f"{model} seed {expected_seed} has invalid provenance {name}"
            )
        normalized_hashes[name] = value.lower()

    if (normalized_hashes["checkpoint_dataset_sha256"]
            != normalized_hashes["dataset_sha256"]):
        raise ValueError(
            f"{model} seed {expected_seed} provenance dataset hash mismatch"
        )
    if (normalized_hashes["checkpoint_y_scaler_sha256"]
            != normalized_hashes["y_scaler_sha256"]):
        raise ValueError(
            f"{model} seed {expected_seed} provenance scaler hash mismatch"
        )

    for name in ("dataset_path", "checkpoint_path", "y_scaler_path", "device"):
        value = provenance[name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{model} seed {expected_seed} has invalid provenance {name}"
            )

    return normalized_hashes["dataset_sha256"]


def _validate_model_report(model, report, seeds, test_seed):
    """Validate per-seed rows and verify every stored aggregate field."""
    if model not in MODEL_NAMES:
        raise ValueError(f"Unknown model block in merged results: {model!r}")
    if not isinstance(report, dict):
        raise ValueError(f"{model} report must be an object")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, list) or len(per_seed) != len(seeds):
        raise ValueError(
            f"{model} per_seed must have one entry for each configured seed"
        )

    pairs = []
    dataset_sha256 = None
    for expected_seed, entry in zip(seeds, per_seed):
        if not isinstance(entry, dict) or entry.get("seed") != expected_seed:
            raise ValueError(
                f"{model} per_seed entries do not match configured seed order"
            )
        ok = entry.get("ok")
        if not isinstance(ok, bool):
            raise ValueError(f"{model} seed {expected_seed} has invalid ok flag")
        count_names = (
            "n_test_samples",
            "n_valid",
            "n_invalid",
            "n_extreme",
        )
        missing_counts = [name for name in count_names if name not in entry]
        if missing_counts:
            raise ValueError(
                f"{model} seed {expected_seed} lacks sample counts: "
                f"{missing_counts}"
            )
        raw_counts = [entry[name] for name in count_names]
        counts_absent = all(value is None for value in raw_counts)
        if not counts_absent:
            if any(value is None for value in raw_counts):
                raise ValueError(
                    f"{model} seed {expected_seed} has partial sample counts"
                )
            for name, value in zip(count_names, raw_counts):
                if (isinstance(value, bool)
                        or not isinstance(value, Integral)
                        or value < 0):
                    raise ValueError(
                        f"{model} seed {expected_seed} has invalid "
                        f"{name}: {value!r}"
                    )
            n_test, n_valid, n_invalid, n_extreme = (
                int(value) for value in raw_counts
            )
            if n_valid + n_invalid + n_extreme != n_test:
                raise ValueError(
                    f"{model} seed {expected_seed} sample counts do not sum "
                    "to n_test_samples"
                )

        if "provenance" not in entry:
            raise ValueError(
                f"{model} seed {expected_seed} lacks provenance field"
            )
        provenance = entry["provenance"]
        if counts_absent and provenance is not None:
            raise ValueError(
                f"{model} seed {expected_seed} was not evaluated but carries "
                "provenance"
            )
        row_dataset_sha256 = _validate_provenance(
            provenance,
            model=model,
            expected_seed=expected_seed,
            test_seed=test_seed,
            required=ok,
        )
        if row_dataset_sha256 is not None:
            if dataset_sha256 is None:
                dataset_sha256 = row_dataset_sha256
            elif row_dataset_sha256 != dataset_sha256:
                raise ValueError(
                    f"{model} evaluated seeds use different dataset hashes"
                )

        value = entry.get(METRIC_NAME)
        if ok:
            if counts_absent or int(entry["n_valid"]) == 0:
                raise ValueError(
                    f"{model} seed {expected_seed} successful metric has "
                    "no valid evaluated samples"
                )
            if (isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))):
                raise ValueError(
                    f"{model} seed {expected_seed} has no finite successful metric"
                )
            if entry.get("error") is not None:
                raise ValueError(
                    f"{model} seed {expected_seed} is successful but has an error"
                )
            pairs.append((float(value), True))
        else:
            if value is not None:
                raise ValueError(
                    f"{model} seed {expected_seed} failed but carries a metric"
                )
            error = entry.get("error")
            if not isinstance(error, str) or not error.strip():
                raise ValueError(
                    f"{model} seed {expected_seed} failed without an error"
                )
            pairs.append((None, False))

    expected = aggregate(pairs)
    for key in ("n_success", "std_available", "insufficient_seeds"):
        if report.get(key) != expected[key]:
            raise ValueError(
                f"{model} aggregate field {key!r} does not match per_seed rows"
            )
    for key in ("mean", "std"):
        if not _same_optional_float(report.get(key), expected[key]):
            raise ValueError(
                f"{model} aggregate field {key!r} does not match per_seed rows"
            )
    return dataset_sha256


def plot_multiseed_compare(models_report, seeds, out_path):
    """Write the labeled CNN-vs-Transformer comparison figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    present = [name for name in MODEL_NAMES if name in models_report]
    present += [name for name in models_report if name not in present]

    labels = []
    for x_pos, model in enumerate(present):
        report = models_report[model]
        labels.append(model.upper())
        values = [
            entry[METRIC_NAME]
            for entry in report["per_seed"]
            if entry["ok"] and entry[METRIC_NAME] is not None
        ]
        if values:
            jitter = (
                np.linspace(-0.12, 0.12, num=len(values))
                if len(values) > 1
                else [0.0]
            )
            ax.scatter(
                [x_pos + value for value in jitter],
                values,
                color="tab:blue" if model == "cnn" else "tab:orange",
                alpha=0.8,
                zorder=3,
                label=f"{model.upper()} per-seed",
            )

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
    ax.set_ylabel(
        "Test-set curve RMSE (dimensionless fluorescence-fraction RMSE)"
    )
    ax.set_title(
        "Multi-seed test-set curve RMSE: CNN vs. Transformer\n"
        f"(seeds={list(seeds)})"
    )
    handles, _ = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best", fontsize=8)
    ax.margins(x=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def load_merged_metrics(part_dirs):
    """Load and validate compatible per-model result parts."""
    directories = list(part_dirs)
    if not directories:
        raise ValueError("part_dirs must contain at least one result directory")

    merged_models = {}
    seeds = None
    test_seed = None
    dataset_sha256 = None
    part_run_provenance = []
    for directory in directories:
        path = os.path.join(directory, JSON_NAME)
        with open(path, encoding="utf-8") as handle:
            part = json.load(handle, parse_constant=_reject_json_constant)
        if not isinstance(part, dict):
            raise ValueError(f"Result part in {directory} must be a JSON object")
        run_provenance = part.get("run_provenance")
        if run_provenance is not None:
            if not isinstance(run_provenance, dict):
                raise ValueError(
                    f"run_provenance in {directory} must be an object"
                )
            run_dataset_hash = run_provenance.get("dataset_sha256")
            if (not isinstance(run_dataset_hash, str)
                    or len(run_dataset_hash) != 64):
                raise ValueError(
                    f"run_provenance dataset hash in {directory} is invalid"
                )
            part_run_provenance.append({
                "result_dir": os.path.abspath(directory),
                **run_provenance,
            })
        if part.get("experiment") != EXPERIMENT_NAME:
            raise ValueError(
                f"Unexpected experiment in {directory}: "
                f"{part.get('experiment')!r}"
            )
        if part.get("metric") != METRIC_NAME:
            raise ValueError(
                f"Unexpected metric in {directory}: {part.get('metric')!r}"
            )
        part_seeds = _validated_seeds(
            part.get("seeds"), f"seeds in {directory}"
        )
        part_test_seed = part.get("test_seed")
        if (isinstance(part_test_seed, bool)
                or not isinstance(part_test_seed, Integral)
                or part_test_seed < 0):
            raise ValueError(
                f"test_seed in {directory} must be a non-negative integer"
            )
        part_test_seed = int(part_test_seed)
        part_models = part.get("models")
        if not isinstance(part_models, dict) or not part_models:
            raise ValueError(f"models in {directory} must be a non-empty object")
        for model, report in part_models.items():
            report_dataset_sha256 = _validate_model_report(
                model, report, part_seeds, part_test_seed
            )
            if report_dataset_sha256 is not None:
                if dataset_sha256 is None:
                    dataset_sha256 = report_dataset_sha256
                elif report_dataset_sha256 != dataset_sha256:
                    raise ValueError(
                        "Cannot merge multiseed results evaluated against "
                        "different dataset hashes"
                    )
        if (run_provenance is not None and dataset_sha256 is not None
                and run_dataset_hash != dataset_sha256):
            raise ValueError(
                "run_provenance dataset hash does not match evaluated "
                f"artifacts in {directory}"
            )

        if seeds is None:
            seeds = part_seeds
            test_seed = part_test_seed
        elif part_seeds != seeds or part_test_seed != test_seed:
            raise ValueError(
                "Cannot merge multiseed results with different seeds/test_seed: "
                f"expected seeds={seeds}, test_seed={test_seed}; "
                f"got seeds={part_seeds}, test_seed={part_test_seed} "
                f"in {directory}"
            )
        overlap = set(merged_models).intersection(part_models)
        if overlap:
            raise ValueError(
                f"Duplicate model blocks while merging: {sorted(overlap)}"
            )
        merged_models.update(part_models)

    ordered = {
        model: merged_models[model]
        for model in MODEL_NAMES
        if model in merged_models
    }
    ordered.update(
        (model, value)
        for model, value in merged_models.items()
        if model not in ordered
    )
    merged = {
        "experiment": EXPERIMENT_NAME,
        "metric": METRIC_NAME,
        "test_seed": test_seed,
        "seeds": seeds,
        "models": ordered,
    }
    if part_run_provenance:
        merged["run_provenance"] = {
            "dataset_sha256": dataset_sha256,
            "parts": part_run_provenance,
        }
    return merged
