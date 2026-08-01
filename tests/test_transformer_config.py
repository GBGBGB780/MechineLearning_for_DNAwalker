# coding=utf-8
"""Validation tests for Transformer-specific configuration."""

import os

import pytest

from dnawalker.paths import ARTIFACTS_DIR
from dnawalker.transformer.config import (
    TransformerConfig,
    load_configs,
)
from dnawalker.transformer.model import build_transformer_model

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CONFIG = os.path.join(_ROOT, "configs", "transformer.ini")


def test_canonical_transformer_config_is_valid():
    cfg = TransformerConfig(_CONFIG)
    assert cfg.get_d_model() % cfg.get_n_heads() == 0
    assert cfg.get_patch_size() > 0


def test_canonical_transformer_parameter_count():
    parent_config, transformer_config = load_configs()
    model = build_transformer_model(parent_config, transformer_config)

    assert sum(parameter.numel() for parameter in model.parameters()) == 3_243_271
    assert all(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("stride", "0", "integer hyperparameters"),
        ("n_heads", "7", "divisible|整除"),
        ("dropout", "1.0", "dropout"),
        ("learning_rate", "inf", "learning_rate"),
        ("weight_decay", "nan", "weight_decay"),
        ("warmup_ratio", "1.0", "warmup_ratio"),
        ("scheduler_min_lr", "1.0", "scheduler_min_lr"),
        ("scheduler_type", "plateau", "cosine"),
    ],
)
def test_invalid_transformer_hyperparameters_fail_early(
    tmp_path, key, value, message
):
    override = tmp_path / "invalid.ini"
    override.write_text(
        f"[TRANSFORMER]\n{key} = {value}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=message):
        TransformerConfig(_CONFIG, extra_config_files=[str(override)])


def test_patch_size_cannot_exceed_parent_sequence(tmp_path):
    override = tmp_path / "patch.ini"
    override.write_text(
        "[TRANSFORMER]\npatch_size = 999999\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="sequence length"):
        load_configs(transformer_override_file=str(override))


def test_d_model_one_rejected_even_with_one_attention_head(tmp_path):
    override = tmp_path / "zero_width_head.ini"
    override.write_text(
        "[TRANSFORMER]\nd_model = 1\nn_heads = 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="d_model must be >= 2"):
        TransformerConfig(_CONFIG, extra_config_files=[str(override)])


def test_transformer_paths_are_absolute_cwd_independent_and_side_effect_free(
        tmp_path, monkeypatch):
    cfg = TransformerConfig(_CONFIG)
    expected_root = _ROOT
    assert cfg.get_dataset_path() == os.path.join(
        expected_root, "artifacts", "datasets", "training_dataset.npz"
    )
    assert cfg.get_model_save_path() == os.path.join(
        expected_root, "artifacts", "models", "transformer",
        "best_transformer_model.pth",
    )
    assert cfg.get_y_scaler_path() == os.path.join(
        expected_root, "artifacts", "models", "transformer",
        "transformer_y_scaler.pkl",
    )

    custom_dir = tmp_path / "not-created-by-getter"
    override = tmp_path / "paths.ini"
    override.write_text(
        f"[PATHS]\nmodel_artifacts_dir = {custom_dir}\n",
        encoding="utf-8",
    )
    custom = TransformerConfig(
        _CONFIG, extra_config_files=[str(override)]
    )
    before = custom.get_model_save_path()
    assert before == str(custom_dir / "best_transformer_model.pth")
    assert not custom_dir.exists()

    other_cwd = tmp_path / "cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    assert custom.get_model_save_path() == before
    assert not custom_dir.exists()


def test_missing_transformer_artifact_dir_falls_back_to_repository_root(
    tmp_path,
):
    config_path = tmp_path / "transformer.ini"
    config_path.write_text(
        "[TRANSFORMER]\n"
        "model_save_path = model.pth\n"
        "y_scaler_file = scaler.pkl\n",
        encoding="utf-8",
    )
    cfg = TransformerConfig(config_path, validate=False)

    expected = ARTIFACTS_DIR / "models" / "transformer"
    assert cfg.get_artifact_dir() == os.fspath(expected)
    assert cfg.get_model_save_path() == os.fspath(expected / "model.pth")
