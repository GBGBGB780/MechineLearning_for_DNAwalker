# coding=utf-8
"""Tests for shared ``Config`` and model-specific ``CNNConfig``.

Covers the additions from the optimization pass:
  - Config(validate=True) defaults: legitimate config loads cleanly.
  - Config(validate=False): skips validation (tests/sandboxes can pass partial configs).
  - _validate(): missing required keys / out-of-range learning_rate / batch_size /
    num_epochs / test_split_ratio raise ValueError before the object is usable.
  - get_weak_signal_floor / get_bin_edges: read the new [DATA_GENERATION] keys
    with sane defaults, and parse comma-separated edges.
  - get_model_save_path / get_x_scaler_file / get_y_scaler_file: return
    repository-anchored paths under artifacts/models/cnn.
"""

import os
import textwrap

import pytest

from dnawalker.cnn.config import CNNConfig
from dnawalker.cnn.model import InverseCNN
from dnawalker.config import Config
from dnawalker.paths import ARTIFACTS_DIR, RESULTS_DIR

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CANONICAL_CONFIG = os.path.join(_REPO_ROOT, "configs", "common.ini")


def _write_ini(tmp_path, body):
    """Write a temporary INI file and return its absolute path."""
    path = tmp_path / "test.ini"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Happy path — canonical shared + CNN layers load cleanly.
# ---------------------------------------------------------------------------
def test_canonical_config_loads_with_validation():
    cfg = CNNConfig()  # validate=True by default
    assert cfg.get_learning_rate() == pytest.approx(0.0005)
    assert cfg.get_batch_size() == 256
    assert cfg.get_num_epochs() == 2000
    assert cfg.get_output_size() == 7


def test_canonical_cnn_parameter_count():
    cfg = CNNConfig()
    model = InverseCNN(
        input_size=cfg.get_input_size(),
        output_size=cfg.get_output_size(),
        config=cfg,
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == 4_381_319
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_validate_false_skips_validation(tmp_path):
    """A partial config that would fail validation loads with validate=False."""
    path = _write_ini(tmp_path, """
        [TRAINING]
        sim_duration_minutes = 130
        num_curves = 3
    """)
    # validate=True would raise (many keys missing); validate=False permits it.
    cfg = Config(path, validate=False)
    assert cfg.config.getint('TRAINING', 'sim_duration_minutes') == 130


# ---------------------------------------------------------------------------
# _validate() — missing required keys
# ---------------------------------------------------------------------------
def test_validate_missing_required_keys_raises(tmp_path):
    path = _write_ini(tmp_path, """
        [TRAINING]
        sim_duration_minutes = 130
    """)
    with pytest.raises(ValueError, match="配置缺少必需项|missing required keys"):
        Config(path)


# ---------------------------------------------------------------------------
# _validate() — out-of-range numeric values
# ---------------------------------------------------------------------------
def _full_config_with(tmp_path, **overrides):
    """Build a minimal-but-complete config, with some keys overridable."""
    defaults = {
        "sim_duration_minutes": "130",
        "num_curves": "3",
        "output_size": "7",
        "learning_rate": "0.0005",
        "batch_size": "256",
        "num_epochs": "2000",
        "random_seed": "42",
        "test_split_ratio": "0.1",
        "val_split_ratio": "0.1",
    }
    defaults.update(overrides)
    body = "[TRAINING]\n" + "\n".join(f"{k} = {v}" for k, v in defaults.items())
    return _write_ini(tmp_path, body)


def test_validate_learning_rate_out_of_range(tmp_path):
    path = _full_config_with(tmp_path, learning_rate="2.0")  # >= 1
    with pytest.raises(ValueError, match="learning_rate"):
        CNNConfig(extra_config_files=[path])


def test_validate_batch_size_non_positive(tmp_path):
    path = _full_config_with(tmp_path, batch_size="0")
    with pytest.raises(ValueError, match="batch_size"):
        CNNConfig(extra_config_files=[path])


def test_validate_num_epochs_non_positive(tmp_path):
    path = _full_config_with(tmp_path, num_epochs="-5")
    with pytest.raises(ValueError, match="num_epochs"):
        CNNConfig(extra_config_files=[path])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sim_duration_minutes", "0"),
        ("num_curves", "0"),
        ("output_size", "0"),
    ],
)
def test_validate_core_dimensions_must_be_positive(tmp_path, field, value):
    path = _full_config_with(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        Config(path)


def test_validate_requires_three_fluorescence_channels(tmp_path):
    path = _full_config_with(tmp_path, num_curves="2")
    with pytest.raises(ValueError, match="exactly 3|必须为 3"):
        Config(path)


@pytest.mark.parametrize(
    ("field", "value"),
    [("random_seed", "-1"), ("random_seed", str(2 ** 32)),
     ("split_seed", "-1"), ("split_seed", str(2 ** 32))],
)
def test_validate_rejects_out_of_range_seeds(tmp_path, field, value):
    path = _full_config_with(tmp_path, **{field: value})
    with pytest.raises(ValueError, match=field):
        Config(path)


def test_validate_split_ratio_out_of_range(tmp_path):
    path = _full_config_with(tmp_path, test_split_ratio="1.5")
    with pytest.raises(ValueError, match="split"):
        Config(path)


@pytest.mark.parametrize(
    ("split_manifest_file", "train_subset_size"),
    [
        ("split.npz", ""),
        ("", "8000"),
        ("split.npz", "0"),
        ("split.npz", "-1"),
    ],
)
def test_validate_rejects_incomplete_or_invalid_explicit_split(
    tmp_path, split_manifest_file, train_subset_size
):
    path = _full_config_with(
        tmp_path,
        split_manifest_file=split_manifest_file,
        train_subset_size=train_subset_size,
    )
    with pytest.raises(ValueError, match="train_subset_size|同时配置"):
        Config(path)


def test_explicit_split_manifest_path_is_config_relative(tmp_path):
    path = _full_config_with(
        tmp_path,
        split_manifest_file="manifests/split.npz",
        train_subset_size="8000",
    )
    config = Config(path)

    assert config.get_split_manifest_file() == os.path.join(
        str(tmp_path), "manifests", "split.npz"
    )
    assert config.get_train_subset_size() == 8000


def test_validate_output_size_matches_trainable_parameters(tmp_path):
    path = _full_config_with(tmp_path, output_size="2")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[PHYSICAL_PARAMETERS]\n"
            "only_parameter =\n"
            "\n[TRAINING_PARAMETER_RANGES]\n"
            "only_parameter = 0, 1\n"
        )
    with pytest.raises(ValueError, match="output_size"):
        Config(path)


@pytest.mark.parametrize(
    "bad_range",
    ["not-a-range", "1", "2, 1", "0, inf", "0, 1, 2"],
)
def test_validate_rejects_malformed_parameter_ranges(tmp_path, bad_range):
    path = _full_config_with(tmp_path, output_size="1")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[PHYSICAL_PARAMETERS]\n"
            "parameter_a =\n"
            "\n[TRAINING_PARAMETER_RANGES]\n"
            f"parameter_a = {bad_range}\n"
        )
    with pytest.raises(ValueError, match="range|范围"):
        Config(path)


def test_validate_cross_checks_simulation_duration(tmp_path):
    path = _full_config_with(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n[NANOROBOT_MODELING]\nsim_total_time = 120\n")
    with pytest.raises(ValueError, match="sim_total_time"):
        Config(path)


def test_validate_rejects_invalid_prediction_settings(tmp_path):
    path = _full_config_with(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[PREDICTION]\n"
            "sg_window = 10\n"
            "sg_polyorder = 3\n"
            "ensemble_size = 1\n"
            "ensemble_noise_std = 0\n"
        )
    with pytest.raises(ValueError, match="Savitzky|SG"):
        Config(path)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("scheduler_factor", "1.0", "scheduler_factor"),
        ("scheduler_patience", "-1", "scheduler_patience"),
        ("scheduler_min_lr", "nan", "scheduler_min_lr"),
        ("loss_weight_mode", "typo", "loss_weight_mode"),
        ("early_stopping_patience", "-1", "early_stopping"),
    ],
)
def test_validate_rejects_invalid_optional_training_settings(
    tmp_path, key, value, message
):
    path = _full_config_with(tmp_path, **{key: value})
    with pytest.raises(ValueError, match=message):
        CNNConfig(extra_config_files=[path])


def test_validate_rejects_invalid_model_architecture(tmp_path):
    path = _full_config_with(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[MODEL_ARCHITECTURE]\n"
            "conv1_out_channels = 0\nconv1_kernel_size = 3\n"
            "conv1_stride = 1\nconv1_padding = 1\n"
            "conv2_out_channels = 2\nconv2_kernel_size = 3\n"
            "conv2_stride = 1\nconv2_padding = 1\n"
            "conv3_out_channels = 2\nconv3_kernel_size = 3\n"
            "conv3_stride = 1\nconv3_padding = 1\n"
            "conv4_out_channels = 2\nconv4_kernel_size = 3\n"
            "conv4_stride = 1\nconv4_padding = 1\n"
            "fc1_out_features = 2\n"
            "dropout_conv = 0.1\ndropout_fc = 0.1\n"
        )
    with pytest.raises(ValueError, match="conv1"):
        CNNConfig(extra_config_files=[path])


def test_validate_rejects_collapsed_cnn_sequence_geometry(tmp_path):
    path = _full_config_with(tmp_path, sim_duration_minutes="1")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[MODEL_ARCHITECTURE]\n"
            "conv1_out_channels = 2\nconv1_kernel_size = 1000\n"
            "conv1_stride = 1\nconv1_padding = 0\n"
            "conv2_out_channels = 2\nconv2_kernel_size = 3\n"
            "conv2_stride = 1\nconv2_padding = 1\n"
            "conv3_out_channels = 2\nconv3_kernel_size = 3\n"
            "conv3_stride = 1\nconv3_padding = 1\n"
            "conv4_out_channels = 2\nconv4_kernel_size = 3\n"
            "conv4_stride = 1\nconv4_padding = 1\n"
            "fc1_out_features = 2\n"
            "dropout_conv = 0.1\ndropout_fc = 0.1\n"
            "\n[NANOROBOT_MODELING]\nsim_total_time = 1\n"
        )
    with pytest.raises(ValueError, match="collapses sequence"):
        CNNConfig(extra_config_files=[path])


def test_validate_rejects_nan_replacement_inside_safe_range(tmp_path):
    path = _full_config_with(tmp_path)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(
            "\n[DATA_PROCESSING]\n"
            "safe_threshold = 100\n"
            "nan_replacement_value = 0\n"
            "log_epsilon = 1e-9\n"
        )
    with pytest.raises(ValueError, match="nan_replacement"):
        Config(path)


# ---------------------------------------------------------------------------
# get_weak_signal_floor / get_bin_edges — new gendata-driven getters.
# ---------------------------------------------------------------------------
def test_weak_signal_floor_default_when_missing(tmp_path):
    """If [DATA_GENERATION].weak_signal_floor is absent, fallback = 0.01."""
    path = _full_config_with(tmp_path)  # no DATA_GENERATION section
    cfg = Config(path)
    assert cfg.get_weak_signal_floor() == pytest.approx(0.01)


def test_weak_signal_floor_reads_configured(tmp_path):
    path = _full_config_with(tmp_path)
    # Append a [DATA_GENERATION] section.
    with open(path, "a") as fh:
        fh.write("\n[DATA_GENERATION]\nweak_signal_floor = 0.05\n")
    cfg = Config(path)
    assert cfg.get_weak_signal_floor() == pytest.approx(0.05)


def test_bin_edges_default_when_missing(tmp_path):
    path = _full_config_with(tmp_path)
    cfg = Config(path)
    assert cfg.get_bin_edges() == [0.01, 0.05, 0.1, 0.2, 0.4, 2.0]


def test_bin_edges_parses_comma_separated(tmp_path):
    path = _full_config_with(tmp_path)
    with open(path, "a") as fh:
        fh.write("\n[DATA_GENERATION]\nbin_edges = 0.0, 0.1, 0.5, 1.0\n")
    cfg = Config(path)
    assert cfg.get_bin_edges() == [0.0, 0.1, 0.5, 1.0]


def test_canonical_bin_edges_match_historical_hardcode():
    """Regression guard: common.ini edges match the historical generator."""
    cfg = Config(_CANONICAL_CONFIG)
    assert cfg.get_bin_edges() == [0.01, 0.05, 0.1, 0.2, 0.4, 2.0]
    assert cfg.get_weak_signal_floor() == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Path getters — repository-owned absolute locations, independent of cwd.
# ---------------------------------------------------------------------------
def test_path_getters_use_repository_boundaries(tmp_path, monkeypatch):
    cfg = CNNConfig()
    expected_artifacts = os.path.join(_REPO_ROOT, "artifacts")
    expected_cnn = os.path.join(expected_artifacts, "models", "cnn")

    assert cfg.get_output_path() == os.path.join(_REPO_ROOT, "results")
    assert cfg.get_artifacts_path() == expected_artifacts
    assert cfg.get_dataset_dir() == os.path.join(expected_artifacts, "datasets")
    assert cfg.get_dataset_file() == os.path.join(
        expected_artifacts, "datasets", "training_dataset.npz"
    )
    assert cfg.get_cnn_artifacts_path() == expected_cnn
    assert cfg.get_model_save_path() == os.path.join(
        expected_cnn, "best_mlp_model.pth"
    )
    assert cfg.get_x_scaler_file() == os.path.join(expected_cnn, "x_scaler.pkl")
    assert cfg.get_y_scaler_file() == os.path.join(expected_cnn, "y_scaler.pkl")
    assert cfg.get_experimental_data_path() == os.path.join(
        _REPO_ROOT, "data", "experimental", "Fig3a_fitting.xlsx"
    )

    monkeypatch.chdir(tmp_path)
    from_other_cwd = CNNConfig()
    assert from_other_cwd.get_dataset_file() == cfg.get_dataset_file()
    assert from_other_cwd.get_model_save_path() == cfg.get_model_save_path()
    assert from_other_cwd.get_experimental_data_path() == (
        cfg.get_experimental_data_path()
    )


def test_missing_shared_path_options_fall_back_to_repository_roots(tmp_path):
    path = _full_config_with(tmp_path)
    cfg = Config(path)

    assert cfg.get_output_path() == os.fspath(RESULTS_DIR)
    assert cfg.get_artifacts_path() == os.fspath(ARTIFACTS_DIR)
    assert cfg.get_dataset_dir() == os.fspath(ARTIFACTS_DIR / "datasets")


def test_split_seed_is_independent_from_model_seed(tmp_path):
    """A model-seed override must not change train/val/test membership."""
    base = _full_config_with(tmp_path)
    with open(base, "a", encoding="utf-8") as fh:
        fh.write("\nsplit_seed = 17\n")

    override = tmp_path / "seed_override.ini"
    override.write_text("[TRAINING]\nrandom_seed = 99\n", encoding="utf-8")

    cfg = Config(base, extra_config_files=[str(override)])
    assert cfg.get_random_seed() == 99
    assert cfg.get_split_seed() == 17


def test_split_seed_falls_back_for_legacy_configs(tmp_path):
    """Old configs retain their historical split until split_seed is added."""
    path = _full_config_with(tmp_path, random_seed="23")
    cfg = Config(path)
    assert cfg.get_split_seed() == 23
