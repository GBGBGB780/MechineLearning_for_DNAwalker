import json
from pathlib import Path


_AUDIT_DIR = Path(__file__).resolve().parents[1] / "docs" / "audit"


def _reject_constant(value):
    raise ValueError(f"non-standard JSON constant: {value}")


def _load(name):
    with (_AUDIT_DIR / name).open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=_reject_constant)


def test_versioned_audit_results_are_strict_and_provenance_bound():
    for model in ("cnn", "transformer"):
        cpu = _load(f"heldout_{model}_cpu.json")
        mps = _load(f"heldout_{model}_mps.json")

        assert cpu["n_samples"] == mps["n_samples"] == 1000
        assert cpu["provenance"]["device"] == "cpu"
        assert mps["provenance"]["device"] == "mps"
        assert cpu["provenance"]["split_seed"] == 42
        for key in (
            "checkpoint_model_seed",
            "checkpoint_split_seed",
            "checkpoint_dataset_sha256",
            "checkpoint_y_scaler_sha256",
        ):
            assert cpu["provenance"][key] is None
            assert mps["provenance"][key] is None
        if model == "cnn":
            assert cpu["provenance"]["checkpoint_epoch"] is None
            assert cpu["provenance"]["checkpoint_val_mse"] is None
            assert not cpu["provenance"]["checkpoint_param_names_present"]
        else:
            assert cpu["provenance"]["checkpoint_epoch"] == 264
            assert cpu["provenance"]["checkpoint_val_mse"] > 0
            assert cpu["provenance"]["checkpoint_param_names_present"]
        assert (
            cpu["provenance"]["checkpoint_epoch"]
            == mps["provenance"]["checkpoint_epoch"]
        )
        assert (
            cpu["provenance"]["checkpoint_val_mse"]
            == mps["provenance"]["checkpoint_val_mse"]
        )
        assert (
            cpu["provenance"]["checkpoint_param_names_present"]
            == mps["provenance"]["checkpoint_param_names_present"]
        )
        for artifact in ("dataset", "checkpoint", "y_scaler"):
            digest_key = f"{artifact}_sha256"
            assert cpu["provenance"][digest_key] == mps["provenance"][digest_key]
            assert len(cpu["provenance"][digest_key]) == 64


def test_versioned_cpu_mps_metrics_remain_within_audited_tolerance():
    for model in ("cnn", "transformer"):
        cpu = _load(f"heldout_{model}_cpu.json")
        mps = _load(f"heldout_{model}_mps.json")

        for key, cpu_value in cpu.items():
            if isinstance(cpu_value, float):
                assert abs(cpu_value - mps[key]) <= 2e-7, (model, key)
