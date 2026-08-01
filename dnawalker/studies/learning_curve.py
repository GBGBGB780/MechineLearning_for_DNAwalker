"""Nested 8k/16k/24k learning-curve comparison for CNN and Transformer."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from dnawalker.config import Config
from dnawalker.data.splits import (
    load_explicit_split,
    write_activity_split_manifest,
)
from dnawalker.paths import REPO_ROOT, RESULTS_DIR
from dnawalker.shared.artifacts import sha256_file
from .protocol import write_json
from .multiseed.constants import JSON_NAME, METRIC_NAME, MODEL_NAMES
from .multiseed.reporting import load_merged_metrics
from .multiseed.runner import DEFAULT_SEEDS, merge_results, run_multiseed


EXPERIMENT_NAME = "nested_activity_stratified_learning_curve"
SUMMARY_JSON = "learning_curve_metrics.json"
SUMMARY_FIGURE = "learning_curve_compare.png"
SUMMARY_REPORT = "learning_curve_report.md"
DEFAULT_TRAIN_SIZES = (8_000, 16_000, 24_000)
DEFAULT_VAL_SIZE = 3_000
DEFAULT_TEST_SIZE = 3_000
DEFAULT_EQUIVALENCE_MARGIN = 0.001
DEFAULT_DISTRIBUTION_BINS = 20
DEFAULT_RESULTS_DIR = RESULTS_DIR / "learning_curve" / "nested_30k"


def _parse_ints(text):
    return tuple(int(part.strip()) for part in str(text).split(",") if part.strip())


def _generator_bin_edges():
    edges = np.asarray(Config().get_bin_edges(), dtype=np.float64)
    edges[-1] = np.inf
    return edges


def prepare_manifest(
    dataset_path,
    artifacts_dir,
    *,
    train_sizes=DEFAULT_TRAIN_SIZES,
    val_size=DEFAULT_VAL_SIZE,
    test_size=DEFAULT_TEST_SIZE,
    split_seed=42,
):
    """Create and validate the fixed activity-stratified split manifest."""
    dataset_path = str(Path(dataset_path).resolve())
    artifacts_dir = Path(artifacts_dir).resolve()
    manifest_path = artifacts_dir / "split_manifest.npz"
    if not manifest_path.exists():
        write_activity_split_manifest(
            dataset_path,
            manifest_path,
            _generator_bin_edges(),
            train_sizes=train_sizes,
            val_size=val_size,
            test_size=test_size,
            split_seed=split_seed,
        )

    with np.load(dataset_path, allow_pickle=False) as dataset:
        n_samples = int(dataset["X"].shape[0])
    for train_size in train_sizes:
        load_explicit_split(
            manifest_path,
            dataset_path,
            n_samples=n_samples,
            split_seed=split_seed,
            train_size=train_size,
        )
    return str(manifest_path)


def _part_paths(results_dir, artifacts_dir, train_size, model):
    """Return separated result and binary-artifact roots for one grid cell."""
    results_root = Path(results_dir).resolve()
    artifacts_root = Path(artifacts_dir).resolve()
    part = results_root / f"train_{train_size}" / "parts" / model
    artifacts = artifacts_root / "models" / f"train_{train_size}"
    return part, artifacts


def _part_is_complete(part_dir, model, seeds, train_size, manifest_sha256):
    path = Path(part_dir) / JSON_NAME
    if not path.is_file():
        return False
    try:
        with path.open(encoding="utf-8") as handle:
            metrics = json.load(handle)
        if metrics.get("seeds") != list(seeds):
            return False
        report = metrics["models"][model]
        if report.get("n_success") != len(seeds):
            return False
        for entry in report["per_seed"]:
            provenance = entry.get("provenance")
            if not isinstance(provenance, dict):
                return False
            if (
                not entry["ok"]
                or provenance.get("train_subset_size") != train_size
                or provenance.get("split_manifest_sha256")
                != manifest_sha256
            ):
                return False
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False
    return True


def run_training_grid(
    dataset_path,
    manifest_path,
    results_dir,
    artifacts_dir,
    *,
    train_sizes=DEFAULT_TRAIN_SIZES,
    seeds=DEFAULT_SEEDS,
    split_seed=42,
    models=MODEL_NAMES,
    allow_overwrite=False,
):
    """Run every requested size/model/seed and strictly merge model parts."""
    dataset_path = str(Path(dataset_path).resolve())
    manifest_path = str(Path(manifest_path).resolve())
    results_dir = Path(results_dir).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    manifest_hash = sha256_file(manifest_path)

    merged_dirs = {}
    for train_size in train_sizes:
        part_dirs = []
        for model in models:
            part_dir, model_artifacts_dir = _part_paths(
                results_dir, artifacts_dir, train_size, model
            )
            if not _part_is_complete(
                part_dir,
                model,
                seeds,
                train_size,
                manifest_hash,
            ):
                run_multiseed(
                    seeds=seeds,
                    test_seed=split_seed,
                    results_dir=str(part_dir),
                    models=(model,),
                    dataset_path=dataset_path,
                    artifacts_dir=str(model_artifacts_dir),
                    allow_overwrite=allow_overwrite,
                    split_manifest_path=manifest_path,
                    train_subset_size=train_size,
                )
            part_dirs.append(str(part_dir))

        if set(models) == set(MODEL_NAMES):
            merged_dir = (
                results_dir / f"train_{train_size}" / "merged"
            )
            merge_results(part_dirs, results_dir=str(merged_dir))
            merged_dirs[int(train_size)] = str(merged_dir)
    return merged_dirs


def paired_statistics(cnn_values, transformer_values, margin):
    """Compute paired architecture differences and a conservative decision."""
    cnn = np.asarray(cnn_values, dtype=np.float64)
    transformer = np.asarray(transformer_values, dtype=np.float64)
    if cnn.shape != transformer.shape or cnn.ndim != 1 or cnn.size < 2:
        raise ValueError("paired statistics require aligned vectors of length >= 2")
    if not np.all(np.isfinite(cnn)) or not np.all(np.isfinite(transformer)):
        raise ValueError("paired metrics must be finite")
    if not math.isfinite(float(margin)) or margin <= 0:
        raise ValueError("equivalence margin must be finite and positive")

    differences = transformer - cnn
    n = int(differences.size)
    mean = float(differences.mean())
    sample_std = float(differences.std(ddof=1))
    standard_error = sample_std / math.sqrt(n)
    if standard_error == 0:
        ci_low = ci_high = mean
        p_value = 0.0 if mean != 0 else 1.0
    else:
        critical = float(stats.t.ppf(0.975, df=n - 1))
        ci_low = mean - critical * standard_error
        ci_high = mean + critical * standard_error
        p_value = float(stats.ttest_1samp(differences, 0.0).pvalue)

    if ci_high < -margin:
        decision = "transformer_meaningfully_lower"
    elif ci_low > margin:
        decision = "cnn_meaningfully_lower"
    elif ci_low >= -margin and ci_high <= margin:
        decision = "practically_equivalent"
    else:
        decision = "inconclusive"

    return {
        "difference_definition": "transformer_minus_cnn",
        "per_seed_differences": differences.tolist(),
        "mean_difference": mean,
        "sample_std": sample_std,
        "standard_error": standard_error,
        "ci95": [float(ci_low), float(ci_high)],
        "paired_t_pvalue_two_sided": p_value,
        "equivalence_margin": float(margin),
        "decision": decision,
        "transformer_lower_seed_count": int((differences < 0).sum()),
        "cnn_lower_seed_count": int((differences > 0).sum()),
        "tie_seed_count": int((differences == 0).sum()),
    }


def _partition_summary(manifest_path):
    with np.load(manifest_path, allow_pickle=False) as manifest:
        bins = np.asarray(manifest["activity_bin_ids"], dtype=np.int64)
        train_sizes = np.asarray(manifest["train_sizes"], dtype=np.int64)
        val = np.asarray(manifest["val_indices"], dtype=np.int64)
        test = np.asarray(manifest["test_indices"], dtype=np.int64)
        n_bins = int(np.asarray(manifest["bin_edges"]).size - 1)
        output = {
            "master_activity_bin_counts": np.bincount(
                bins, minlength=n_bins
            ).tolist(),
            "validation_size": int(val.size),
            "validation_activity_bin_counts": np.bincount(
                bins[val], minlength=n_bins
            ).tolist(),
            "test_size": int(test.size),
            "test_activity_bin_counts": np.bincount(
                bins[test], minlength=n_bins
            ).tolist(),
            "training": {},
        }
        previous = set()
        for size in train_sizes:
            indices = np.asarray(
                manifest[f"train_{int(size)}"], dtype=np.int64
            )
            current = set(indices.tolist())
            if not previous.issubset(current):
                raise ValueError("training subsets are not nested")
            previous = current
            output["training"][str(int(size))] = {
                "size": int(indices.size),
                "activity_bin_counts": np.bincount(
                    bins[indices], minlength=n_bins
                ).tolist(),
            }
    return output


def _parameter_distribution_summary(
    dataset_path,
    manifest_path,
    *,
    histogram_bins=DEFAULT_DISTRIBUTION_BINS,
):
    """Audit retained parameter coverage after simulation/activity filtering."""
    if histogram_bins < 2:
        raise ValueError("histogram_bins must be at least 2")

    with np.load(dataset_path, allow_pickle=False) as dataset:
        labels = np.asarray(dataset["Y"], dtype=np.float64)
        raw_names = np.asarray(dataset["parameter_names"]).tolist()
    names = [
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in raw_names
    ]
    if (
        labels.ndim != 2
        or labels.shape[0] < 2
        or labels.shape[1] != len(names)
        or not np.all(np.isfinite(labels))
    ):
        raise ValueError("dataset labels must be a finite two-dimensional array")

    configured_ranges = {
        name.lower(): bounds
        for name, bounds in Config().get_param_ranges().items()
    }
    try:
        lower = np.asarray(
            [configured_ranges[name.lower()][0] for name in names],
            dtype=np.float64,
        )
        upper = np.asarray(
            [configured_ranges[name.lower()][1] for name in names],
            dtype=np.float64,
        )
    except KeyError as exc:
        raise ValueError(
            f"dataset parameter has no configured range: {exc.args[0]}"
        ) from exc
    if np.any(upper <= lower):
        raise ValueError("configured parameter ranges must be non-degenerate")

    normalized = (labels - lower) / (upper - lower)
    tolerance = 1e-12
    if np.any(normalized < -tolerance) or np.any(normalized > 1.0 + tolerance):
        raise ValueError("dataset labels fall outside configured parameter ranges")
    normalized = np.clip(normalized, 0.0, 1.0)
    n_samples = int(normalized.shape[0])
    expected_bin_count = n_samples / histogram_bins
    unit_edges = np.linspace(0.0, 1.0, histogram_bins + 1)
    stratum_upper = np.nextafter(1.0, 0.0)

    marginals = []
    occupied_counts = []
    for column, name in enumerate(names):
        values = normalized[:, column]
        counts = np.histogram(values, bins=unit_edges)[0]
        strata = np.floor(
            np.minimum(values, stratum_upper) * n_samples
        ).astype(np.int64)
        occupied = int(np.unique(strata).size)
        occupied_counts.append(occupied)
        uniform_test = stats.kstest(values, "uniform")
        marginals.append({
            "parameter": name,
            "normalized_min": float(values.min()),
            "normalized_max": float(values.max()),
            "normalized_mean": float(values.mean()),
            "normalized_std": float(values.std()),
            "histogram_bin_count": int(histogram_bins),
            "histogram_counts": counts.tolist(),
            "histogram_max_relative_deviation": float(
                np.max(np.abs(counts - expected_bin_count))
                / expected_bin_count
            ),
            "occupied_lhs_strata": occupied,
            "total_lhs_strata": n_samples,
            "uniform_ks_statistic": float(uniform_test.statistic),
            "uniform_ks_pvalue": float(uniform_test.pvalue),
        })

    correlations = np.corrcoef(normalized, rowvar=False)
    if not np.all(np.isfinite(correlations)):
        raise ValueError("parameter correlations are not finite")
    upper_triangle = np.triu(np.abs(correlations), k=1)
    strongest_flat = int(np.argmax(upper_triangle))
    strongest_pair = np.unravel_index(strongest_flat, correlations.shape)

    split_reports = {}
    with np.load(manifest_path, allow_pickle=False) as manifest:
        partition_keys = [
            ("validation", "val_indices"),
            ("test", "test_indices"),
        ]
        partition_keys.extend(
            (f"train_{int(size)}", f"train_{int(size)}")
            for size in np.asarray(manifest["train_sizes"], dtype=np.int64)
        )
        for label, key in partition_keys:
            indices = np.asarray(manifest[key], dtype=np.int64)
            selected = normalized[indices]
            mean_differences = np.abs(
                selected.mean(axis=0) - normalized.mean(axis=0)
            )
            ks_values = np.asarray([
                stats.ks_2samp(
                    selected[:, column],
                    normalized[:, column],
                ).statistic
                for column in range(normalized.shape[1])
            ])
            split_reports[label] = {
                "size": int(indices.size),
                "max_normalized_mean_difference": float(
                    mean_differences.max()
                ),
                "max_mean_difference_parameter": names[
                    int(np.argmax(mean_differences))
                ],
                "max_two_sample_ks_statistic": float(ks_values.max()),
                "max_ks_parameter": names[int(np.argmax(ks_values))],
            }

    return {
        "retained_sample_count": n_samples,
        "unique_parameter_rows": int(np.unique(labels, axis=0).shape[0]),
        "candidate_sampling": "latin_hypercube",
        "retained_rows_are_filtered_and_activity_balanced": True,
        "strict_lhs_stratification_preserved_after_filtering": bool(
            all(count == n_samples for count in occupied_counts)
        ),
        "strongest_absolute_pearson_correlation": {
            "parameters": [
                names[int(strongest_pair[0])],
                names[int(strongest_pair[1])],
            ],
            "correlation": float(correlations[strongest_pair]),
        },
        "marginals": marginals,
        "split_representativeness": split_reports,
        "interpretation": (
            "Activity-bin balance is exact, but LHS candidate sampling does "
            "not guarantee uniform retained parameter marginals after physical "
            "validity filtering and activity-quota selection."
        ),
    }


def _load_reference_results():
    references = {}
    audit_dir = Path(REPO_ROOT) / "docs" / "audit"
    legacy = {}
    for model in MODEL_NAMES:
        path = audit_dir / f"heldout_{model}_cpu.json"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                data = json.load(handle)
            legacy[model] = float(data["curve_rmse_mean"])
    if len(legacy) == 2:
        references["legacy_single_checkpoint"] = {
            "source": "docs/audit/heldout_<model>_cpu.json",
            "models": legacy,
            "transformer_minus_cnn": legacy["transformer"] - legacy["cnn"],
            "interpretation": "compatibility_diagnostic_only",
        }

    current_path = (
        Path(REPO_ROOT)
        / "results"
        / "releases"
        / "retrain-3a5a494-ds557506e93079"
        / "multiseed"
        / JSON_NAME
    )
    if current_path.is_file():
        with current_path.open(encoding="utf-8") as handle:
            current = json.load(handle)
        models = current.get("models", {})
        if all(model in models for model in MODEL_NAMES):
            current_means = {
                model: float(models[model]["mean"])
                for model in MODEL_NAMES
            }
            references["current_10k_dataset_8k_train"] = {
                "source": str(current_path.relative_to(REPO_ROOT)),
                "models": current_means,
                "transformer_minus_cnn": (
                    current_means["transformer"] - current_means["cnn"]
                ),
                "dataset_is_not_nested_with_30k_master": True,
            }
    return references


def _validate_run_provenance(entry, train_size, manifest_hash, dataset_hash):
    provenance = entry.get("provenance")
    required = {
        "train_subset_size": train_size,
        "checkpoint_train_subset_size": train_size,
        "split_manifest_sha256": manifest_hash,
        "checkpoint_split_manifest_sha256": manifest_hash,
        "dataset_sha256": dataset_hash,
        "checkpoint_dataset_sha256": dataset_hash,
    }
    for name, expected in required.items():
        if provenance.get(name) != expected:
            raise ValueError(
                f"learning-curve provenance mismatch for {name}: "
                f"{provenance.get(name)!r} vs {expected!r}"
            )


def _recovery_evidence(records):
    """Summarize numeric rows restored after temporary artifacts were removed."""
    recovered = []
    unavailable_hashes = 0
    for train_size, model, entry in records:
        provenance = entry["provenance"]
        source = provenance.get("recovered_from_session_log")
        if source is None:
            continue
        hash_status = provenance.get(
            "checkpoint_sha256_status",
            "original_checkpoint_hash_recorded",
        )
        if hash_status != "original_checkpoint_hash_recorded":
            unavailable_hashes += 1
        recovered.append({
            "train_size": int(train_size),
            "model": model,
            "seed": int(entry["seed"]),
            "source": source,
            "source_lines": list(
                provenance.get("recovery_source_lines", [])
            ),
            "source_sha256": provenance.get("recovery_source_sha256"),
            "checkpoint_sha256_status": hash_status,
        })
    return {
        "recovered_result_row_count": len(recovered),
        "checkpoint_hash_unavailable_row_count": unavailable_hashes,
        "records": recovered,
        "interpretation": (
            "Recovered rows preserve recorded metrics, counts, and provenance. "
            "A row without its original checkpoint SHA-256 cannot independently "
            "verify the checkpoint binary that produced that evaluation."
        ),
    }


def summarize(
    dataset_path,
    manifest_path,
    results_dir,
    artifacts_dir,
    *,
    train_sizes=DEFAULT_TRAIN_SIZES,
    seeds=DEFAULT_SEEDS,
    split_seed=42,
    equivalence_margin=DEFAULT_EQUIVALENCE_MARGIN,
):
    """Validate all strict results, compute paired statistics, and write report."""
    dataset_path = str(Path(dataset_path).resolve())
    manifest_path = str(Path(manifest_path).resolve())
    results_dir = Path(results_dir).resolve()
    artifacts_dir = Path(artifacts_dir).resolve()
    dataset_hash = sha256_file(dataset_path)
    manifest_hash = sha256_file(manifest_path)

    size_reports = {}
    evaluated_rows = []
    for train_size in train_sizes:
        part_dirs = [
            str(
                _part_paths(
                    results_dir,
                    artifacts_dir,
                    train_size,
                    model,
                )[0]
            )
            for model in MODEL_NAMES
        ]
        metrics = load_merged_metrics(part_dirs)
        if metrics["seeds"] != list(seeds):
            raise ValueError(
                f"seed mismatch at train_size={train_size}: {metrics['seeds']}"
            )

        values = {}
        models = {}
        for model in MODEL_NAMES:
            report = metrics["models"][model]
            if report["n_success"] != len(seeds):
                raise ValueError(
                    f"{model} train_size={train_size} is incomplete"
                )
            per_seed_values = []
            count_totals = {
                "n_test_samples": 0,
                "n_valid": 0,
                "n_invalid": 0,
                "n_extreme": 0,
            }
            for entry in report["per_seed"]:
                evaluated_rows.append((train_size, model, entry))
                if entry["n_test_samples"] != DEFAULT_TEST_SIZE:
                    raise ValueError(
                        f"unexpected test size in {model}: "
                        f"{entry['n_test_samples']}"
                    )
                _validate_run_provenance(
                    entry, train_size, manifest_hash, dataset_hash
                )
                per_seed_values.append(float(entry[METRIC_NAME]))
                for name in count_totals:
                    count_totals[name] += int(entry[name])
            values[model] = per_seed_values
            models[model] = {
                "per_seed": per_seed_values,
                "mean": float(report["mean"]),
                "sample_std": float(report["std"]),
                "count_totals": count_totals,
            }

        size_reports[str(train_size)] = {
            "train_size": int(train_size),
            "models": models,
            "paired_comparison": paired_statistics(
                values["cnn"],
                values["transformer"],
                equivalence_margin,
            ),
        }

    summary = {
        "schema_version": 1,
        "experiment": EXPERIMENT_NAME,
        "metric": METRIC_NAME,
        "metric_units": "dimensionless_fluorescence_fraction_rmse",
        "dataset_path": dataset_path,
        "dataset_sha256": dataset_hash,
        "split_manifest_path": manifest_path,
        "split_manifest_sha256": manifest_hash,
        "split_seed": int(split_seed),
        "model_seeds": list(seeds),
        "protocol": {
            "train_sizes": list(train_sizes),
            "validation_size": DEFAULT_VAL_SIZE,
            "test_size": DEFAULT_TEST_SIZE,
            "activity_stratified": True,
            "nested_training_membership": True,
            "fixed_validation_membership": True,
            "fixed_test_membership": True,
            "equivalence_margin": equivalence_margin,
        },
        "partitions": _partition_summary(manifest_path),
        "parameter_distribution_audit": _parameter_distribution_summary(
            dataset_path,
            manifest_path,
        ),
        "results_by_train_size": size_reports,
        "reference_results": _load_reference_results(),
        "evidence_recovery": _recovery_evidence(evaluated_rows),
    }
    output_dir = results_dir
    write_json(output_dir / SUMMARY_JSON, summary)
    _plot_summary(summary, output_dir / SUMMARY_FIGURE)
    _write_markdown_report(summary, output_dir / SUMMARY_REPORT)
    return summary


def _plot_summary(summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [
        int(value)
        for value in summary["protocol"]["train_sizes"]
    ]
    fig, (metric_ax, difference_ax) = plt.subplots(
        1, 2, figsize=(11.0, 4.6)
    )
    colors = {"cnn": "tab:blue", "transformer": "tab:orange"}
    for model in MODEL_NAMES:
        means = [
            summary["results_by_train_size"][str(size)]["models"][model]["mean"]
            for size in sizes
        ]
        stds = [
            summary["results_by_train_size"][str(size)]["models"][model][
                "sample_std"
            ]
            for size in sizes
        ]
        metric_ax.errorbar(
            sizes,
            means,
            yerr=stds,
            marker="o",
            capsize=5,
            label=model.upper(),
            color=colors[model],
        )
    metric_ax.set_xlabel("Training samples")
    metric_ax.set_ylabel("Test curve RMSE")
    metric_ax.set_title("Nested learning curve (mean +/- seed SD)")
    metric_ax.legend()

    differences = [
        summary["results_by_train_size"][str(size)]["paired_comparison"][
            "mean_difference"
        ]
        for size in sizes
    ]
    lows = [
        summary["results_by_train_size"][str(size)]["paired_comparison"][
            "ci95"
        ][0]
        for size in sizes
    ]
    highs = [
        summary["results_by_train_size"][str(size)]["paired_comparison"][
            "ci95"
        ][1]
        for size in sizes
    ]
    difference_ax.errorbar(
        sizes,
        differences,
        yerr=[
            np.asarray(differences) - np.asarray(lows),
            np.asarray(highs) - np.asarray(differences),
        ],
        marker="o",
        capsize=5,
        color="black",
    )
    margin = summary["protocol"]["equivalence_margin"]
    difference_ax.axhline(0.0, color="0.4", linewidth=1)
    difference_ax.axhspan(
        -margin, margin, color="tab:green", alpha=0.15
    )
    difference_ax.set_xlabel("Training samples")
    difference_ax.set_ylabel("Transformer - CNN RMSE")
    difference_ax.set_title("Paired difference (95% CI)")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_markdown_report(summary, output_path):
    lines = [
        "# Nested learning-curve result",
        "",
        "| Train | CNN mean +/- SD | Transformer mean +/- SD | "
        "T-CNN difference (95% CI) | p | Decision |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for raw_size in summary["protocol"]["train_sizes"]:
        size = str(raw_size)
        result = summary["results_by_train_size"][size]
        cnn = result["models"]["cnn"]
        transformer = result["models"]["transformer"]
        paired = result["paired_comparison"]
        lines.append(
            f"| {int(raw_size):,} | {cnn['mean']:.8f} +/- "
            f"{cnn['sample_std']:.8f} | {transformer['mean']:.8f} +/- "
            f"{transformer['sample_std']:.8f} | "
            f"{paired['mean_difference']:+.8f} "
            f"[{paired['ci95'][0]:+.8f}, {paired['ci95'][1]:+.8f}] | "
            f"{paired['paired_t_pvalue_two_sided']:.4g} | "
            f"{paired['decision']} |"
        )
    lines.extend([
        "",
        "Difference is Transformer RMSE minus CNN RMSE. The predeclared "
        f"practical-equivalence margin is +/-"
        f"{summary['protocol']['equivalence_margin']:.4f} RMSE.",
        "",
        f"Dataset SHA-256: `{summary['dataset_sha256']}`",
        "",
        f"Split manifest SHA-256: `{summary['split_manifest_sha256']}`",
    ])
    recovery = summary.get("evidence_recovery", {})
    recovered_count = int(recovery.get("recovered_result_row_count", 0))
    if recovered_count:
        unavailable_count = int(
            recovery.get("checkpoint_hash_unavailable_row_count", 0)
        )
        lines.extend([
            "",
            "Evidence-recovery note: "
            f"{recovered_count} result rows were restored from an append-only "
            "session log after temporary artifacts were removed. "
            f"{unavailable_count} rows lack the original checkpoint SHA-256; "
            "those checkpoint binaries are not independently verifiable.",
        ])
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed activity-stratified 8k/16k/24k learning curve."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help=(
            "Study artifact root for the manifest, checkpoints, and scalers "
            "(default: the dataset's parent directory)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_RESULTS_DIR),
        help=(
            "Study output root for JSON, figures, reports, overrides, and logs "
            "(default: repository results/learning_curve/nested_30k)."
        ),
    )
    parser.add_argument(
        "--train-sizes",
        type=_parse_ints,
        default=DEFAULT_TRAIN_SIZES,
    )
    parser.add_argument(
        "--seeds", type=_parse_ints, default=DEFAULT_SEEDS
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=lambda value: tuple(
            part.strip().lower()
            for part in value.split(",")
            if part.strip()
        ),
        default=MODEL_NAMES,
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=DEFAULT_EQUIVALENCE_MARGIN,
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.prepare_only and args.summarize_only:
        parser.error("--prepare-only and --summarize-only are mutually exclusive")
    if not args.train_sizes or sorted(set(args.train_sizes)) != list(
        args.train_sizes
    ):
        parser.error("--train-sizes must be distinct and increasing")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds must be non-empty and distinct")
    if any(model not in MODEL_NAMES for model in args.models):
        parser.error("--models accepts only cnn and transformer")

    artifacts_dir = (
        Path(args.artifacts_dir).resolve()
        if args.artifacts_dir is not None
        else Path(args.dataset).resolve().parent
    )
    results_dir = Path(args.results_dir).resolve()

    manifest = prepare_manifest(
        args.dataset,
        artifacts_dir,
        train_sizes=args.train_sizes,
        split_seed=args.split_seed,
    )
    if args.prepare_only:
        print(f"Wrote {manifest}")
        return 0

    if not args.summarize_only:
        run_training_grid(
            args.dataset,
            manifest,
            results_dir,
            artifacts_dir,
            train_sizes=args.train_sizes,
            seeds=args.seeds,
            split_seed=args.split_seed,
            models=args.models,
            allow_overwrite=args.allow_overwrite,
        )

    if set(args.models) == set(MODEL_NAMES):
        summary = summarize(
            args.dataset,
            manifest,
            results_dir,
            artifacts_dir,
            train_sizes=args.train_sizes,
            seeds=args.seeds,
            split_seed=args.split_seed,
            equivalence_margin=args.equivalence_margin,
        )
        print(
            f"Wrote {results_dir / SUMMARY_JSON}"
        )
        for size, result in summary["results_by_train_size"].items():
            paired = result["paired_comparison"]
            print(
                f"  train={size}: difference="
                f"{paired['mean_difference']:+.8f}, "
                f"CI={paired['ci95']}, decision={paired['decision']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
