"""Regression tests for fixed train/validation/test membership."""

import numpy as np
from sklearn.model_selection import train_test_split as sklearn_train_test_split

from dnawalker.physics import pysim
from dnawalker.cnn import data as cnn_data
from dnawalker.transformer import data as transformer_data


class _ParentConfig:
    def __init__(self, scaler_path):
        self._scaler_path = str(scaler_path)

    def get_trainable_param_names(self):
        return list(pysim.PARAM_NAMES)

    def get_num_curves(self):
        return 3

    def get_seq_length(self):
        return 4

    def get_log_epsilon(self):
        return 1e-9

    def get_log_transform_params(self):
        return []

    def get_amplitude_filter_enabled(self):
        return False

    def get_safe_threshold(self):
        return 1e20

    def get_nan_replacement_value(self):
        return -1e30

    def get_test_split_ratio(self):
        return 0.2

    def get_val_split_ratio(self):
        return 0.25

    def get_split_seed(self):
        return 42

    def get_random_seed(self):
        raise AssertionError("training loaders must not use the model seed")

    def get_y_scaler_file(self):
        return self._scaler_path


class _TransformerConfig:
    def __init__(self, scaler_path):
        self._scaler_path = str(scaler_path)

    def get_y_scaler_path(self):
        return self._scaler_path


def _synthetic_dataset():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(20, 3, 4)).astype(np.float32)
    y = rng.normal(size=(20, len(pysim.PARAM_NAMES))).astype(np.float32)
    return x, y


def _record_split_seeds(monkeypatch, module):
    seeds = []

    def recording_split(*args, **kwargs):
        seeds.append(kwargs.get("random_state"))
        return sklearn_train_test_split(*args, **kwargs)

    monkeypatch.setattr(module, "train_test_split", recording_split)
    return seeds


def _replace_dataset_loader(monkeypatch, module):
    x, y = _synthetic_dataset()

    def fake_loader(*_args, **_kwargs):
        return x.copy(), y.copy(), list(pysim.PARAM_NAMES)

    monkeypatch.setattr(module, "load_npz_dataset", fake_loader)


def test_cnn_training_loader_uses_only_fixed_split_seed(monkeypatch, tmp_path):
    _replace_dataset_loader(monkeypatch, cnn_data)
    seeds = _record_split_seeds(monkeypatch, cnn_data)
    config = _ParentConfig(tmp_path / "cnn_scaler.pkl")

    train, val, test, _ = cnn_data.load_and_preprocess_data(
        "synthetic.npz",
        batch_size=4,
        config=config,
    )

    assert seeds == [42, 42]
    assert [len(loader.dataset) for loader in (train, val, test)] == [12, 4, 4]


def test_cnn_training_loader_honors_resolved_scaler_override(
        monkeypatch, tmp_path):
    _replace_dataset_loader(monkeypatch, cnn_data)
    configured_path = tmp_path / "configured" / "scaler.pkl"
    override_path = tmp_path / "resolved" / "scaler.pkl"
    config = _ParentConfig(configured_path)

    cnn_data.load_and_preprocess_data(
        "synthetic.npz",
        batch_size=4,
        config=config,
        y_scaler_file=override_path,
    )

    assert override_path.is_file()
    assert not configured_path.exists()


def test_transformer_training_loader_uses_only_fixed_split_seed(
    monkeypatch, tmp_path
):
    _replace_dataset_loader(monkeypatch, transformer_data)
    seeds = _record_split_seeds(monkeypatch, transformer_data)
    parent = _ParentConfig(tmp_path / "unused.pkl")
    transformer = _TransformerConfig(tmp_path / "transformer_scaler.pkl")

    train, val, test, _ = transformer_data.load_and_preprocess_data_3d(
        "synthetic.npz",
        batch_size=4,
        parent_config=parent,
        transformer_config=transformer,
    )

    assert seeds == [42, 42]
    assert [len(loader.dataset) for loader in (train, val, test)] == [12, 4, 4]
