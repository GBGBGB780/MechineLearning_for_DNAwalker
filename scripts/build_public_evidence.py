#!/usr/bin/env python3
"""Build the tracked, path-sanitized public evidence snapshot."""

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "evidence"

JSON_SOURCES = {
    "current_10k_manifest.json": (
        ROOT
        / "results"
        / "releases"
        / "retrain-3a5a494-ds557506e93079"
        / "manifest.json"
    ),
    "fit_robustness.json": (
        ROOT
        / "results"
        / "evaluation"
        / "comparisons"
        / "train24k_refinement_robustness"
        / "fit_robustness.json"
    ),
    "identifiability_metrics.json": (
        ROOT
        / "results"
        / "validation"
        / "identifiability"
        / "identifiability_metrics.json"
    ),
}

LEARNING_CURVE_SOURCE = (
    ROOT
    / "results"
    / "learning_curve"
    / "nested_30k"
    / "learning_curve_metrics.json"
)

IMAGE_SOURCES = {
    "current_10k_comparison.png": (
        ROOT
        / "results"
        / "releases"
        / "retrain-3a5a494-ds557506e93079"
        / "multiseed"
        / "multiseed_compare.png"
    ),
    "nested_learning_curve.png": (
        ROOT
        / "results"
        / "learning_curve"
        / "nested_30k"
        / "learning_curve_compare.png"
    ),
    "cnn_experimental_fit.png": (
        ROOT
        / "results"
        / "evaluation"
        / "cnn"
        / "train24k_seed43"
        / "dual_fit.refine_seed0_mps.png"
    ),
    "transformer_experimental_fit.png": (
        ROOT
        / "results"
        / "evaluation"
        / "transformer"
        / "train24k_seed46"
        / "dual_fit.refine_seed0_mps.png"
    ),
    "refinement_robustness.png": (
        ROOT
        / "results"
        / "evaluation"
        / "comparisons"
        / "train24k_refinement_robustness"
        / "fit_robustness.png"
    ),
    "identifiability_sensitivity.png": (
        ROOT
        / "results"
        / "validation"
        / "identifiability"
        / "identifiability_sensitivity.png"
    ),
    "signal_spectrum.png": (
        ROOT
        / "results"
        / "validation"
        / "signal"
        / "evidence_spectrum.png"
    ),
    "signal_autocorrelation.png": (
        ROOT
        / "results"
        / "validation"
        / "signal"
        / "evidence_autocorr.png"
    ),
}

PRIVATE_MARKERS = (
    "/Users/",
    "/private/tmp",
    ".codex/sessions",
)
STUDENT_ID_PATTERN = re.compile(r"\bA\d{7}[A-Z]\b")


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_constant)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path, payload):
    text = json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def _public_learning_curve(payload):
    recovery = payload["evidence_recovery"]
    records = recovery["records"]
    source_hashes = sorted({record["source_sha256"] for record in records})
    statuses = [
        {
            "train_size": record["train_size"],
            "model": record["model"],
            "seed": record["seed"],
            "checkpoint_sha256_status": record[
                "checkpoint_sha256_status"
            ],
        }
        for record in records
    ]
    reference_results = dict(payload["reference_results"])
    reference_results["current_10k_dataset_8k_train"] = {
        **reference_results["current_10k_dataset_8k_train"],
        "source": "docs/evidence/current_10k_manifest.json",
    }
    return {
        "schema_version": payload["schema_version"],
        "experiment": payload["experiment"],
        "metric": payload["metric"],
        "metric_units": payload["metric_units"],
        "dataset_path": (
            "artifacts/studies/nested_learning_curve_30k/"
            "dnawalker_training_30k_seed42.npz"
        ),
        "dataset_sha256": payload["dataset_sha256"],
        "split_manifest_path": (
            "artifacts/studies/nested_learning_curve_30k/"
            "split_manifest.npz"
        ),
        "split_manifest_sha256": payload["split_manifest_sha256"],
        "split_seed": payload["split_seed"],
        "model_seeds": payload["model_seeds"],
        "protocol": payload["protocol"],
        "partitions": payload["partitions"],
        "parameter_distribution_audit": payload[
            "parameter_distribution_audit"
        ],
        "results_by_train_size": payload["results_by_train_size"],
        "reference_results": reference_results,
        "evidence_recovery": {
            "recovered_result_row_count": recovery[
                "recovered_result_row_count"
            ],
            "checkpoint_hash_unavailable_row_count": recovery[
                "checkpoint_hash_unavailable_row_count"
            ],
            "source_description": (
                "append-only local session log; not distributed"
            ),
            "source_sha256": source_hashes,
            "records": statuses,
            "interpretation": recovery["interpretation"],
        },
        "private_source_json_sha256": _sha256(LEARNING_CURVE_SOURCE),
    }


def _public_current_10k(payload):
    public = dict(payload)
    configuration = dict(public["configuration"])
    configuration.pop("cnn_ablation", None)
    configuration.pop("transformer_ablation", None)
    public["configuration"] = configuration
    return public


def _assert_public_text(path):
    text = Path(path).read_text(encoding="utf-8")
    marker = next((item for item in PRIVATE_MARKERS if item in text), None)
    if marker is not None:
        raise ValueError(f"{path} contains private marker {marker!r}")
    if STUDENT_ID_PATTERN.search(text):
        raise ValueError(f"{path} contains a student-ID-like value")


def build(output_dir=OUTPUT_DIR):
    """Build curated evidence and return the generated paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    for name, source in JSON_SOURCES.items():
        payload = _read_json(source)
        if name == "current_10k_manifest.json":
            payload = _public_current_10k(payload)
        destination = output_dir / name
        _write_json(destination, payload)
        _assert_public_text(destination)
        generated.append(destination)

    learning_curve = _public_learning_curve(
        _read_json(LEARNING_CURVE_SOURCE)
    )
    learning_curve_path = output_dir / "nested_learning_curve.json"
    _write_json(learning_curve_path, learning_curve)
    _assert_public_text(learning_curve_path)
    generated.append(learning_curve_path)

    for name, source in IMAGE_SOURCES.items():
        destination = output_dir / name
        shutil.copyfile(source, destination)
        generated.append(destination)

    checksums = output_dir / "SHA256SUMS"
    checksums.write_text(
        "".join(
            f"{_sha256(path)}  {path.name}\n"
            for path in sorted(generated, key=lambda item: item.name)
        ),
        encoding="ascii",
    )
    generated.append(checksums)
    return generated


def main():
    generated = build()
    print(f"Built {len(generated)} public evidence files in {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
