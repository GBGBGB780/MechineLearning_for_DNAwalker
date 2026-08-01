import json

import pytest

from dnawalker.studies import fit_robustness


def _payload(model, seed, *, checkpoint_hash=None):
    offset = 0.001 if model == "transformer" else 0.0
    return {
        "schema_version": 2,
        "experiment": "dual_experimental_curve_fit",
        "model": model,
        "checkpoint_selection": "minimum_validation_mse_seeds_42_46",
        "evaluation_settings": {
            "ensemble": 20,
            "noise_std": 0.005,
            "refine_enabled": True,
            "refinement_method": "Powell",
            "maxiter": 500,
            "multistart": 8,
            "seed": seed,
        },
        "experimental_inputs": {
            "original": {"path": "original.xlsx", "sha256": "1" * 64},
            "generalization": {
                "path": "generalization.xlsx",
                "sha256": "2" * 64,
            },
        },
        "original": {
            "fam_rmse": 0.01 + offset + seed * 0.001,
            "tye_rmse": 0.02 + offset,
            "cy5_rmse": 0.03 + offset,
            "avg_rmse": 0.02 + offset + seed * 0.001,
        },
        "generalization": {
            "fam_rmse": 0.02 + offset + seed * 0.001,
            "tye_rmse": 0.03 + offset,
            "cy5_rmse": 0.04 + offset,
            "avg_rmse": 0.03 + offset + seed * 0.001,
        },
        "mean_avg_rmse": 0.025 + offset + seed * 0.001,
        "provenance": {
            "release_schema": "current",
            "device": "mps",
            "checkpoint_sha256": (
                checkpoint_hash or (model[0] * 64)
            ),
            "dataset_sha256": "d" * 64,
            "y_scaler_sha256": "s" * 64,
        },
    }


def _write_runs(directory, model, seeds=(0, 1, 2)):
    directory.mkdir(parents=True)
    for seed in seeds:
        path = directory / f"rmse_dual.refine_seed{seed}_mps.json"
        path.write_text(
            json.dumps(_payload(model, seed)),
            encoding="utf-8",
        )


def test_load_model_runs_validates_and_aggregates(tmp_path):
    directory = tmp_path / "cnn"
    _write_runs(directory, "cnn")

    result = fit_robustness.load_model_runs(
        "cnn", directory, seeds=(0, 1, 2)
    )

    assert [run["refinement_seed"] for run in result["runs"]] == [0, 1, 2]
    stats = result["aggregate"]["original"]["avg_rmse"]
    assert stats["n"] == 3
    assert stats["mean"] == pytest.approx(0.021)
    assert stats["median"] == pytest.approx(0.021)
    assert stats["min"] == pytest.approx(0.020)
    assert stats["max"] == pytest.approx(0.022)
    assert result["evaluation_settings"]["refinement_seeds"] == [0, 1, 2]


def test_load_model_runs_rejects_invariant_metadata_mismatch(tmp_path):
    directory = tmp_path / "cnn"
    _write_runs(directory, "cnn", seeds=(0, 1))
    path = directory / "rmse_dual.refine_seed1_mps.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["checkpoint_sha256"] = "x" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="invariant run metadata mismatch"):
        fit_robustness.load_model_runs("cnn", directory, seeds=(0, 1))


def test_load_model_runs_rejects_non_mps_or_wrong_seed(tmp_path):
    directory = tmp_path / "transformer"
    _write_runs(directory, "transformer", seeds=(0,))
    path = directory / "rmse_dual.refine_seed0_mps.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["provenance"]["device"] = "cpu"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="current MPS provenance"):
        fit_robustness.load_model_runs(
            "transformer", directory, seeds=(0,)
        )


def test_summarize_writes_machine_json_and_english_figure(tmp_path):
    cnn_dir = tmp_path / "cnn"
    transformer_dir = tmp_path / "transformer"
    output_dir = tmp_path / "summary"
    _write_runs(cnn_dir, "cnn")
    _write_runs(transformer_dir, "transformer")

    summary = fit_robustness.summarize(
        cnn_dir,
        transformer_dir,
        output_dir,
        seeds=(0, 1, 2),
    )

    assert summary["refinement_seeds"] == [0, 1, 2]
    assert set(summary["models"]) == {"cnn", "transformer"}
    assert (output_dir / fit_robustness.JSON_NAME).is_file()
    assert (output_dir / fit_robustness.FIGURE_NAME).is_file()
