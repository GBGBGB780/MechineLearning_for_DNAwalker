"""Summarize experimental-fit stability across refinement RNG seeds."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dnawalker.paths import REPO_ROOT
from dnawalker.shared.artifacts import sha256_file
from .protocol import write_json


MODEL_NAMES = ("cnn", "transformer")
DATASET_NAMES = ("original", "generalization")
METRIC_NAMES = ("fam_rmse", "tye_rmse", "cy5_rmse", "avg_rmse")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
JSON_NAME = "fit_robustness.json"
FIGURE_NAME = "fit_robustness.png"


def _parse_seeds(text):
    seeds = tuple(int(part.strip()) for part in str(text).split(","))
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise argparse.ArgumentTypeError(
            "seeds must be unique non-negative comma-separated integers"
        )
    return seeds


def _relative_path(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _finite_metric(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{label} must be finite, got {value!r}")
    return float(value)


def _statistics(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_std": (
            float(array.std(ddof=1)) if array.size > 1 else None
        ),
        "median": float(np.median(array)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _run_signature(payload):
    settings = dict(payload["evaluation_settings"])
    settings.pop("seed", None)
    return {
        "schema_version": payload.get("schema_version"),
        "experiment": payload.get("experiment"),
        "model": payload.get("model"),
        "checkpoint_selection": payload.get("checkpoint_selection"),
        "evaluation_settings_without_seed": settings,
        "experimental_inputs": payload.get("experimental_inputs"),
        "provenance": payload.get("provenance"),
    }


def load_model_runs(model, directory, seeds):
    """Load and validate one model's strict per-refinement-seed JSON files."""
    if model not in MODEL_NAMES:
        raise ValueError(f"unsupported model: {model}")
    directory = Path(directory)
    runs = []
    expected_signature = None

    for seed in seeds:
        path = directory / f"rmse_dual.refine_seed{seed}_mps.json"
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != 2:
            raise ValueError(f"{path} is not a strict schema-v2 result")
        if payload.get("model") != model:
            raise ValueError(f"{path} model mismatch: {payload.get('model')!r}")
        settings = payload.get("evaluation_settings")
        if not isinstance(settings, dict) or settings.get("seed") != seed:
            raise ValueError(f"{path} refinement seed mismatch")
        provenance = payload.get("provenance")
        if (
            not isinstance(provenance, dict)
            or provenance.get("release_schema") != "current"
            or provenance.get("device") != "mps"
        ):
            raise ValueError(f"{path} lacks current MPS provenance")

        signature = _run_signature(payload)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(f"{path} invariant run metadata mismatch")

        datasets = {}
        for dataset in DATASET_NAMES:
            report = payload.get(dataset)
            if not isinstance(report, dict):
                raise ValueError(f"{path} missing {dataset} report")
            datasets[dataset] = {
                metric: _finite_metric(
                    report.get(metric), f"{path}:{dataset}:{metric}"
                )
                for metric in METRIC_NAMES
            }
        runs.append({
            "refinement_seed": seed,
            "source_json": _relative_path(path),
            "source_json_sha256": sha256_file(path),
            **datasets,
            "combined_mean_avg_rmse": _finite_metric(
                payload.get("mean_avg_rmse"),
                f"{path}:mean_avg_rmse",
            ),
        })

    aggregates = {
        dataset: {
            metric: _statistics([run[dataset][metric] for run in runs])
            for metric in METRIC_NAMES
        }
        for dataset in DATASET_NAMES
    }
    aggregates["combined_mean_avg_rmse"] = _statistics(
        [run["combined_mean_avg_rmse"] for run in runs]
    )
    return {
        "checkpoint_selection": expected_signature["checkpoint_selection"],
        "evaluation_settings": {
            **expected_signature["evaluation_settings_without_seed"],
            "refinement_seeds": list(seeds),
        },
        "experimental_inputs": expected_signature["experimental_inputs"],
        "provenance": expected_signature["provenance"],
        "runs": runs,
        "aggregate": aggregates,
    }


def _plot_summary(summary, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = summary["refinement_seeds"]
    colors = {"cnn": "tab:blue", "transformer": "tab:orange"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), sharey=True)
    for axis, dataset in zip(axes, DATASET_NAMES):
        for model in MODEL_NAMES:
            values = [
                run[dataset]["avg_rmse"]
                for run in summary["models"][model]["runs"]
            ]
            median = summary["models"][model]["aggregate"][dataset][
                "avg_rmse"
            ]["median"]
            axis.plot(
                seeds,
                values,
                marker="o",
                color=colors[model],
                label=model.upper(),
            )
            axis.axhline(
                median,
                color=colors[model],
                linestyle="--",
                alpha=0.45,
            )
        axis.set_title(f"{dataset.capitalize()} experimental dataset")
        axis.set_xlabel("Refinement RNG seed")
        axis.set_xticks(seeds)
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Refined three-channel mean RMSE")
    axes[1].legend(loc="best")
    fig.suptitle(
        "Physics-refinement stability for validation-selected 24k checkpoints"
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize(cnn_dir, transformer_dir, output_dir, seeds=DEFAULT_SEEDS):
    """Validate per-seed fits and write the combined robustness evidence."""
    seeds = tuple(seeds)
    models = {
        "cnn": load_model_runs("cnn", cnn_dir, seeds),
        "transformer": load_model_runs(
            "transformer", transformer_dir, seeds
        ),
    }
    if (
        models["cnn"]["experimental_inputs"]
        != models["transformer"]["experimental_inputs"]
    ):
        raise ValueError("CNN and Transformer experimental inputs differ")
    summary = {
        "schema_version": 1,
        "experiment": "experimental_fit_refinement_robustness",
        "refinement_seeds": list(seeds),
        "models": models,
        "interpretation_boundary": (
            "This evaluates optimizer-start sensitivity for one checkpoint per "
            "architecture selected by validation MSE. It is not a multi-model-"
            "seed architecture comparison."
        ),
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / JSON_NAME, summary)
    _plot_summary(summary, output_dir / FIGURE_NAME)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize experimental-fit refinement robustness."
    )
    parser.add_argument("--cnn-dir", required=True)
    parser.add_argument("--transformer-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default=DEFAULT_SEEDS,
        help="Comma-separated refinement RNG seeds (default: 0,1,2,3,4).",
    )
    args = parser.parse_args(argv)
    summarize(
        args.cnn_dir,
        args.transformer_dir,
        args.output_dir,
        seeds=args.seeds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
