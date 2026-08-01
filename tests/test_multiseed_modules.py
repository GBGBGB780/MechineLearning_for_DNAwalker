"""Unit tests for the refactored multi-seed package boundaries."""

import configparser
import json
from types import SimpleNamespace

import pytest

from dnawalker.studies.multiseed import runner as multiseed_retrain
from dnawalker.studies.multiseed.reporting import load_merged_metrics
from dnawalker.studies.multiseed.runtime import (
    ModelRunSpec,
    checkpoint_split_seed,
    execute_training,
    write_override_file,
)
from dnawalker.studies.multiseed.statistics import aggregate


def _provenance(seed, split_seed=42):
    """Return a complete synthetic artifact binding for merge validation."""
    return {
        "param_names": [
            "e_b",
            "e_b_azo_trans",
            "e_b_azo_cis",
            "k_mig",
            "k0",
            "drt_z",
            "drt_s",
        ],
        "split_seed": split_seed,
        "checkpoint_model_seed": seed,
        "checkpoint_split_seed": split_seed,
        "checkpoint_dataset_sha256": "1" * 64,
        "checkpoint_y_scaler_sha256": "3" * 64,
        "checkpoint_epoch": 1,
        "checkpoint_val_mse": 0.01,
        "checkpoint_param_names_present": True,
        "device": "cpu",
        "inference_batch_size": 16,
        "dataset_path": "/artifacts/training_dataset.npz",
        "dataset_sha256": "1" * 64,
        "checkpoint_path": f"/artifacts/model.seed{seed}.pth",
        "checkpoint_sha256": "2" * 64,
        "y_scaler_path": f"/artifacts/y_scaler.seed{seed}.pkl",
        "y_scaler_sha256": "3" * 64,
    }


def _model_report(seeds, values):
    per_seed = [
        {
            "seed": seed,
            "ok": value is not None,
            "curve_rmse_mean": value,
            "error": None if value is not None else "failed",
            "n_test_samples": 10 if value is not None else None,
            "n_valid": 10 if value is not None else None,
            "n_invalid": 0 if value is not None else None,
            "n_extreme": 0 if value is not None else None,
            "provenance": (
                _provenance(seed) if value is not None else None
            ),
        }
        for seed, value in zip(seeds, values)
    ]
    stats = aggregate([
        (value, value is not None) for value in values
    ])
    return {"per_seed": per_seed, **stats}


def test_canonical_runner_exposes_core_types_and_statistics():
    assert multiseed_retrain.aggregate is aggregate
    assert (
        multiseed_retrain.TrainOutcome.__module__
        == "dnawalker.studies.multiseed.runtime"
    )


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_model_specs_build_seeded_commands_and_outputs(model):
    spec = multiseed_retrain._MODEL_SPECS[model]
    command = spec.training_command("/python", "/override.ini")

    assert command[:5] == [
        "/python",
        "-m",
        spec.trainer_script,
        "--config",
        "/override.ini",
    ]
    assert spec.checkpoint_path(43).endswith(
        f"artifacts/models/{model}/" + (
            "best_mlp_model.seed43.pth"
            if model == "cnn"
            else "best_transformer_model.seed43.pth"
        )
    )
    if model == "transformer":
        assert command[-2:] == ["--transformer-config", "/override.ini"]
    else:
        assert "--transformer-config" not in command


def test_model_run_spec_rejects_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model run spec"):
        ModelRunSpec(
            name="unknown",
            train_dir=tmp_path,
            artifact_dir=tmp_path / "artifacts",
            trainer_script="dnawalker.cnn.train",
            checkpoint_template="model.seed{seed}.pth",
        )


def test_model_run_spec_default_artifacts_stay_out_of_results(tmp_path):
    spec = ModelRunSpec(
        name="cnn",
        train_dir=tmp_path,
        trainer_script="dnawalker.cnn.train",
        checkpoint_template="model.seed{seed}.pth",
    )

    assert spec.artifact_dir == (
        tmp_path / "artifacts" / "models" / "cnn"
    ).resolve()
    assert spec.checkpoint_path(42) == str(
        (
            tmp_path
            / "artifacts"
            / "models"
            / "cnn"
            / "model.seed42.pth"
        ).resolve()
    )


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_override_writer_keeps_model_seed_separate_from_split_seed(
        model, tmp_path):
    spec = multiseed_retrain._MODEL_SPECS[model]
    path = write_override_file(
        spec,
        seed=46,
        split_seed=42,
        override_dir=tmp_path,
    )

    parser = configparser.ConfigParser()
    parser.read(path)
    assert parser["TRAINING"]["random_seed"] == "46"
    assert parser["TRAINING"]["split_seed"] == "42"
    output_section = "TRAINING" if model == "cnn" else "TRANSFORMER"
    assert parser[output_section]["model_save_path"].endswith("seed46.pth")
    assert not parser[output_section]["model_save_path"].startswith("results/")


def test_execute_training_uses_spec_and_detects_checkpoint(tmp_path):
    spec = ModelRunSpec(
        name="cnn",
        train_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        trainer_script="dnawalker.cnn.train",
        checkpoint_template="model.seed{seed}.pth",
    )
    calls = []

    def successful_runner(command, **kwargs):
        calls.append((command, kwargs))
        checkpoint = tmp_path / "artifacts" / "model.seed7.pth"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b"weights")
        return SimpleNamespace(returncode=0, stderr="")

    outcome = execute_training(
        spec,
        seed=7,
        override_path="/override.ini",
        python_executable="/python",
        runner=successful_runner,
    )

    assert outcome.ok is True
    assert outcome.model_path == spec.checkpoint_path(7)
    assert calls[0][0] == [
        "/python",
        "-m",
        "dnawalker.cnn.train",
        "--config",
        "/override.ini",
    ]
    assert calls[0][1]["cwd"] == str(tmp_path)


def test_execute_training_preserves_subprocess_failure_context(tmp_path):
    spec = ModelRunSpec(
        name="transformer",
        train_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        trainer_script="dnawalker.transformer.train",
        checkpoint_template="model.seed{seed}.pth",
    )

    def failed_runner(_command, **_kwargs):
        return SimpleNamespace(returncode=3, stderr="training failed")

    outcome = execute_training(
        spec,
        seed=8,
        override_path="/override.ini",
        python_executable="/python",
        runner=failed_runner,
    )

    assert outcome.ok is False
    assert outcome.model_path is None
    assert "code 3" in outcome.error
    assert "training failed" in outcome.error


def test_checkpoint_split_seed_treats_malformed_checkpoint_as_unusable(tmp_path):
    checkpoint = tmp_path / "broken.pth"
    checkpoint.write_bytes(b"not-a-pytorch-checkpoint")

    assert checkpoint_split_seed(checkpoint) is None


def test_checkpoint_split_seed_rejects_lossy_numeric_coercion(
        tmp_path, monkeypatch):
    checkpoint = tmp_path / "seed.pth"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "torch.load",
        lambda *_args, **_kwargs: {"split_seed": 42.5},
    )

    assert checkpoint_split_seed(checkpoint) is None


def test_load_merged_metrics_orders_models_canonically(tmp_path):
    transformer_dir = tmp_path / "transformer"
    cnn_dir = tmp_path / "cnn"
    transformer_dir.mkdir()
    cnn_dir.mkdir()
    template = {
        "experiment": "multiseed_retraining",
        "metric": "curve_rmse_mean",
        "test_seed": 42,
        "seeds": [42, 43],
    }
    (transformer_dir / "multiseed_metrics.json").write_text(
        json.dumps({
            **template,
            "models": {
                "transformer": _model_report([42, 43], [0.2, 0.21])
            },
        }),
        encoding="utf-8",
    )
    (cnn_dir / "multiseed_metrics.json").write_text(
        json.dumps({
            **template,
            "models": {"cnn": _model_report([42, 43], [0.1, 0.11])},
        }),
        encoding="utf-8",
    )

    merged = load_merged_metrics([transformer_dir, cnn_dir])
    assert list(merged["models"]) == ["cnn", "transformer"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (
            lambda data: data["models"]["cnn"].update(mean=999.0),
            "aggregate field",
        ),
        (
            lambda data: data["models"]["cnn"]["per_seed"][0].update(
                curve_rmse_mean=float("nan")
            ),
            "non-finite JSON",
        ),
        (
            lambda data: data["models"]["cnn"]["per_seed"][0].update(seed=99),
            "seed order",
        ),
        (
            lambda data: data["models"]["cnn"]["per_seed"][0].update(
                n_invalid=1
            ),
            "do not sum",
        ),
        (
            lambda data: data["models"]["cnn"]["per_seed"][0]["provenance"].update(
                param_names=[
                    "e_b_azo_trans",
                    "e_b",
                    "e_b_azo_cis",
                    "k_mig",
                    "k0",
                    "drt_z",
                    "drt_s",
                ]
            ),
            "parameter order",
        ),
    ],
)
def test_load_merged_metrics_rejects_corrupt_model_reports(
        tmp_path, mutate, match):
    part_dir = tmp_path / "part"
    part_dir.mkdir()
    data = {
        "experiment": "multiseed_retraining",
        "metric": "curve_rmse_mean",
        "test_seed": 42,
        "seeds": [42, 43],
        "models": {"cnn": _model_report([42, 43], [0.1, 0.2])},
    }
    mutate(data)
    (part_dir / "multiseed_metrics.json").write_text(
        json.dumps(data), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=match):
        load_merged_metrics([part_dir])


def test_cli_merge_mode_does_not_start_training(monkeypatch, tmp_path):
    calls = []
    expected = {
        "models": {},
        "seeds": [42],
        "test_seed": 42,
    }

    def fake_merge(part_dirs, results_dir):
        calls.append((part_dirs, results_dir))
        return expected

    def unexpected_run(**_kwargs):
        raise AssertionError("run_multiseed must not run in merge mode")

    monkeypatch.setattr(multiseed_retrain, "merge_results", fake_merge)
    monkeypatch.setattr(multiseed_retrain, "run_multiseed", unexpected_run)

    result = multiseed_retrain.main([
        "--merge", "cnn-part", "transformer-part",
        "--results-dir", str(tmp_path),
    ])

    assert result is expected
    assert calls == [(["cnn-part", "transformer-part"], str(tmp_path))]


def test_package_eval_modules_have_distinct_qualified_names():
    cnn = multiseed_retrain._load_eval_module("cnn")
    transformer = multiseed_retrain._load_eval_module("transformer")

    assert cnn.__name__ == "dnawalker.cnn.evaluate"
    assert transformer.__name__ == "dnawalker.transformer.evaluate"
