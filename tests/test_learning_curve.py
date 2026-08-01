"""Tests for the fixed activity-stratified nested learning-curve protocol."""

import configparser
import json

import numpy as np
import pytest

from dnawalker.physics import pysim
from dnawalker.data.splits import (
    build_nested_activity_splits,
    load_explicit_split,
    write_activity_split_manifest,
)
from dnawalker.data.generate import MAX_VALS, MIN_VALS
from dnawalker.studies import learning_curve
from dnawalker.shared import evaluation as testset
from dnawalker.studies.multiseed.runtime import (
    ModelRunSpec,
    write_override_file,
)


EDGES = np.array([0.01, 0.05, 0.1, 0.2, 0.4, np.inf])
SCORES = (0.02, 0.07, 0.15, 0.3, 0.6)


def _curves(rows_per_bin=6):
    curves = np.zeros((rows_per_bin * len(SCORES), 3, 4), dtype=np.float32)
    for bin_index, score in enumerate(SCORES):
        start = bin_index * rows_per_bin
        curves[start:start + rows_per_bin, 0, -1] = score
    return curves


def _dataset(path, rows_per_bin=6):
    curves = _curves(rows_per_bin)
    labels = np.arange(
        curves.shape[0] * len(pysim.PARAM_NAMES), dtype=np.float64
    ).reshape(curves.shape[0], len(pysim.PARAM_NAMES))
    np.savez_compressed(
        path,
        X=curves,
        Y=labels,
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )
    return curves, labels


def test_nested_activity_splits_are_balanced_fixed_and_nested():
    split = build_nested_activity_splits(
        _curves(),
        EDGES,
        train_sizes=(5, 10, 15),
        val_size=5,
        test_size=5,
        split_seed=42,
    )
    bins = split["activity_bin_ids"]

    assert np.bincount(
        bins[split["test_indices"]], minlength=5
    ).tolist() == [1] * 5
    assert np.bincount(
        bins[split["val_indices"]], minlength=5
    ).tolist() == [1] * 5
    assert np.bincount(
        bins[split["train_indices"][15]], minlength=5
    ).tolist() == [3] * 5
    assert set(split["train_indices"][5]).issubset(
        split["train_indices"][10]
    )
    assert set(split["train_indices"][10]).issubset(
        split["train_indices"][15]
    )
    assert set(split["test_indices"]).isdisjoint(split["val_indices"])

    repeated = build_nested_activity_splits(
        _curves(),
        EDGES,
        train_sizes=(5, 10, 15),
        val_size=5,
        test_size=5,
        split_seed=42,
    )
    for name in ("test_indices", "val_indices"):
        np.testing.assert_array_equal(split[name], repeated[name])
    for size in (5, 10, 15):
        np.testing.assert_array_equal(
            split["train_indices"][size],
            repeated["train_indices"][size],
        )


def test_manifest_is_bound_to_dataset_and_rejects_overlap_or_cleanup(tmp_path):
    dataset = tmp_path / "master.npz"
    curves, _ = _dataset(dataset)
    manifest = tmp_path / "split.npz"
    write_activity_split_manifest(
        dataset,
        manifest,
        EDGES,
        train_sizes=(5, 10, 15),
        val_size=5,
        test_size=5,
        split_seed=7,
    )

    loaded = load_explicit_split(
        manifest,
        dataset,
        n_samples=curves.shape[0],
        split_seed=7,
        train_size=10,
        valid_mask=np.ones(curves.shape[0], dtype=bool),
    )
    assert loaded.train.size == 10
    assert loaded.val.size == 5
    assert loaded.test.size == 5

    with np.load(manifest, allow_pickle=False) as original:
        payload = {
            name: np.array(original[name], copy=True)
            for name in original.files
        }

    overlap_payload = dict(payload)
    overlap_payload["train_15"] = payload["train_15"].copy()
    train_10_members = set(payload["train_10"].tolist())
    replace_position = next(
        position
        for position, index in enumerate(payload["train_15"])
        if index not in train_10_members
    )
    overlap_payload["train_15"][replace_position] = payload["test_indices"][0]
    overlap_manifest = tmp_path / "overlap.npz"
    np.savez_compressed(overlap_manifest, **overlap_payload)
    with pytest.raises(ValueError, match="partitions overlap"):
        load_explicit_split(
            overlap_manifest,
            dataset,
            n_samples=curves.shape[0],
            split_seed=7,
            train_size=10,
        )

    nonnested_payload = dict(payload)
    nonnested_payload["train_10"] = payload["train_10"].copy()
    required_member = payload["train_5"][0]
    replace_position = int(np.flatnonzero(
        nonnested_payload["train_10"] == required_member
    )[0])
    replacement = next(
        index
        for index in payload["train_15"]
        if index not in train_10_members
    )
    nonnested_payload["train_10"][replace_position] = replacement
    nonnested_manifest = tmp_path / "nonnested.npz"
    np.savez_compressed(nonnested_manifest, **nonnested_payload)
    with pytest.raises(ValueError, match="not nested"):
        load_explicit_split(
            nonnested_manifest,
            dataset,
            n_samples=curves.shape[0],
            split_seed=7,
            train_size=10,
        )

    mask = np.ones(curves.shape[0], dtype=bool)
    mask[loaded.test[0]] = False
    with pytest.raises(ValueError, match="fail training cleanup"):
        load_explicit_split(
            manifest,
            dataset,
            n_samples=curves.shape[0],
            split_seed=7,
            train_size=10,
            valid_mask=mask,
        )

    with np.load(dataset, allow_pickle=False) as original:
        np.savez_compressed(
            dataset,
            X=original["X"],
            Y=original["Y"] + 1.0,
            parameter_names=original["parameter_names"],
        )
    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        load_explicit_split(
            manifest,
            dataset,
            n_samples=curves.shape[0],
            split_seed=7,
            train_size=10,
        )


def test_multiseed_override_records_explicit_split_contract(tmp_path):
    spec = ModelRunSpec(
        name="cnn",
        train_dir=tmp_path,
        artifact_dir=tmp_path / "artifacts",
        trainer_script="dnawalker.cnn.train",
        checkpoint_template="model.seed{seed}.pth",
    )
    manifest = tmp_path / "split.npz"
    manifest.write_bytes(b"split")
    path = write_override_file(
        spec,
        seed=43,
        split_seed=42,
        override_dir=tmp_path / "overrides",
        split_manifest_path=manifest,
        train_subset_size=16_000,
    )
    parser = configparser.ConfigParser()
    parser.read(path)
    assert parser["TRAINING"]["split_manifest_file"] == str(
        manifest.resolve()
    )
    assert parser["TRAINING"]["train_subset_size"] == "16000"


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_learning_curve_separates_results_from_model_artifacts(
    tmp_path, model
):
    results_dir = tmp_path / "results" / "learning_curve" / "nested_30k"
    artifacts_dir = (
        tmp_path / "artifacts" / "studies" / "nested_learning_curve_30k"
    )

    part_dir, model_artifacts = learning_curve._part_paths(
        results_dir,
        artifacts_dir,
        24_000,
        model,
    )

    assert part_dir == (
        results_dir.resolve() / "train_24000" / "parts" / model
    )
    assert model_artifacts == (
        artifacts_dir.resolve() / "models" / "train_24000"
    )
    assert results_dir.resolve() not in model_artifacts.parents
    assert artifacts_dir.resolve() not in part_dir.parents


def test_incomplete_part_with_null_provenance_is_resumable(tmp_path):
    part_dir = tmp_path / "part"
    part_dir.mkdir()
    (part_dir / "multiseed_metrics.json").write_text(
        json.dumps({
            "seeds": [42],
            "models": {
                "cnn": {
                    "n_success": 1,
                    "per_seed": [{
                        "seed": 42,
                        "ok": True,
                        "provenance": None,
                    }],
                },
            },
        }),
        encoding="utf-8",
    )

    assert learning_curve._part_is_complete(
        part_dir,
        "cnn",
        (42,),
        8_000,
        "manifest-hash",
    ) is False


def test_recovery_evidence_preserves_checkpoint_hash_limitation():
    records = [
        (
            8_000,
            "cnn",
            {
                "seed": 42,
                "provenance": {
                    "recovered_from_session_log": "session.jsonl",
                    "recovery_source_lines": [10, 20],
                    "recovery_source_sha256": "source-hash",
                },
            },
        ),
        (
            8_000,
            "transformer",
            {
                "seed": 43,
                "provenance": {
                    "recovered_from_session_log": "session.jsonl",
                    "checkpoint_sha256_status": (
                        "sentinel_not_original_checkpoint_hash"
                    ),
                },
            },
        ),
        (16_000, "cnn", {"seed": 42, "provenance": {}}),
    ]

    evidence = learning_curve._recovery_evidence(records)

    assert evidence["recovered_result_row_count"] == 2
    assert evidence["checkpoint_hash_unavailable_row_count"] == 1
    assert evidence["records"][0]["checkpoint_sha256_status"] == (
        "original_checkpoint_hash_recorded"
    )
    assert evidence["records"][1]["model"] == "transformer"


def test_testset_uses_manifest_test_membership(tmp_path):
    dataset = tmp_path / "master.npz"
    curves, labels = _dataset(dataset)
    manifest = tmp_path / "split.npz"
    write_activity_split_manifest(
        dataset,
        manifest,
        EDGES,
        train_sizes=(5, 10, 15),
        val_size=5,
        test_size=5,
        split_seed=42,
    )
    explicit = load_explicit_split(
        manifest,
        dataset,
        n_samples=curves.shape[0],
        split_seed=42,
        train_size=10,
    )

    class Config:
        @staticmethod
        def get_dataset_file():
            return str(dataset)

        @staticmethod
        def get_trainable_param_names():
            return list(pysim.PARAM_NAMES)

        @staticmethod
        def get_num_curves():
            return 3

        @staticmethod
        def get_seq_length():
            return 4

        @staticmethod
        def get_log_transform_params():
            return []

        @staticmethod
        def get_log_epsilon():
            return 1e-9

        @staticmethod
        def get_amplitude_filter_enabled():
            return False

        @staticmethod
        def get_safe_threshold():
            return 1e20

        @staticmethod
        def get_nan_replacement_value():
            return -1e30

        @staticmethod
        def get_split_seed():
            return 42

        @staticmethod
        def get_split_manifest_file():
            return str(manifest)

        @staticmethod
        def get_train_subset_size():
            return 10

    actual_curves, actual_labels = testset.get_test_split(
        Config(), test_seed=42
    )
    np.testing.assert_array_equal(actual_curves, curves[explicit.test])
    np.testing.assert_array_equal(actual_labels, labels[explicit.test])


def test_paired_statistics_uses_predeclared_practical_margin():
    equivalent = learning_curve.paired_statistics(
        [0.0200, 0.0201, 0.0199, 0.0202, 0.0198],
        [0.0201, 0.0202, 0.0200, 0.0203, 0.0199],
        0.001,
    )
    assert equivalent["decision"] == "practically_equivalent"
    assert equivalent["mean_difference"] == pytest.approx(0.0001)

    transformer_wins = learning_curve.paired_statistics(
        [0.0230, 0.0231, 0.0229, 0.0232, 0.0228],
        [0.0200, 0.0201, 0.0199, 0.0202, 0.0198],
        0.001,
    )
    assert (
        transformer_wins["decision"]
        == "transformer_meaningfully_lower"
    )


def test_parameter_distribution_audit_distinguishes_candidate_lhs_from_retained(
    tmp_path,
):
    n_samples = 20
    unit = np.empty((n_samples, len(pysim.PARAM_NAMES)), dtype=np.float64)
    base = (np.arange(n_samples, dtype=np.float64) + 0.5) / n_samples
    for column in range(unit.shape[1]):
        unit[:, column] = np.roll(base, column * 3)
    labels = MIN_VALS + (MAX_VALS - MIN_VALS) * unit
    dataset = tmp_path / "dataset.npz"
    np.savez_compressed(
        dataset,
        X=np.zeros((n_samples, 3, 4), dtype=np.float32),
        Y=labels,
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )
    manifest = tmp_path / "manifest.npz"
    np.savez_compressed(
        manifest,
        train_sizes=np.asarray([10], dtype=np.int64),
        train_10=np.arange(10, dtype=np.int64),
        val_indices=np.arange(10, 15, dtype=np.int64),
        test_indices=np.arange(15, 20, dtype=np.int64),
    )

    audit = learning_curve._parameter_distribution_summary(
        dataset,
        manifest,
        histogram_bins=5,
    )

    assert audit["retained_sample_count"] == n_samples
    assert audit["unique_parameter_rows"] == n_samples
    assert audit["candidate_sampling"] == "latin_hypercube"
    assert audit["strict_lhs_stratification_preserved_after_filtering"] is True
    assert len(audit["marginals"]) == len(pysim.PARAM_NAMES)
    assert audit["split_representativeness"]["test"]["size"] == 5
