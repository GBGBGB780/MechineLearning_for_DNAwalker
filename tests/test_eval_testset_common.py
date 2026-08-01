import numpy as np
import pytest

from dnawalker.shared import evaluation as eval_testset_common
from dnawalker.physics import pysim
from dnawalker.shared.artifacts import sha256_file


class _IdentityScaler:
    @staticmethod
    def transform(values):
        return np.asarray(values)


class _Config:
    names = [
        "k0",
        "e_b",
        "drt_s",
        "e_b_azo_cis",
        "k_mig",
        "drt_z",
        "e_b_azo_trans",
    ]

    def get_trainable_param_names(self):
        return list(self.names)

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
    def get_amplitude_thresholds():
        return [1.0, 1.0, 1.0]

    @staticmethod
    def get_safe_threshold():
        return 1e20

    @staticmethod
    def get_nan_replacement_value():
        return -1e30


class _Predictor:
    y_scaler = _IdentityScaler()

    def __init__(self, prediction, names=None):
        self.prediction = prediction
        self.names = list(names or _Config.names)

    def predict(self, _curves):
        return self.prediction.copy()

    def get_param_names(self):
        return list(self.names)


def test_run_evaluation_maps_reordered_columns_by_name(tmp_path, monkeypatch):
    config = _Config()
    values_by_name = {
        "E_b": -1.1,
        "E_b_azo_trans": -0.9,
        "E_b_azo_cis": -0.2,
        "k_mig": 0.07,
        "k0": 4e-6,
        "drt_z": 0.4,
        "drt_s": 0.08,
    }
    values_by_lower_name = {
        name.lower(): value for name, value in values_by_name.items()
    }
    prediction_names = list(reversed(config.names))
    prediction = np.asarray(
        [[values_by_lower_name[name] for name in prediction_names]],
        dtype=np.float64,
    )
    labels = np.asarray(
        [[values_by_lower_name[name] for name in config.names]],
        dtype=np.float64,
    )
    curves = np.zeros((1, 3, 4), dtype=np.float64)
    captured = {}

    monkeypatch.setattr(
        eval_testset_common,
        "get_test_split",
        lambda _config: (curves.copy(), labels.copy()),
    )

    def fake_simulation(params):
        captured.update(params)
        return curves[0].copy(), 1.0

    monkeypatch.setattr(pysim, "run_simulation", fake_simulation)

    summary = eval_testset_common.run_evaluation(
        predictor=_Predictor(prediction, names=prediction_names),
        config=config,
        results_dir=str(tmp_path),
        model_label="test",
        tag="reordered",
    )

    assert summary["n_valid"] == 1
    assert captured == values_by_name


def test_run_evaluation_rejects_wrong_prediction_shape(tmp_path, monkeypatch):
    config = _Config()
    curves = np.zeros((1, 3, 4), dtype=np.float64)
    labels = np.zeros((1, len(config.names)), dtype=np.float64)
    monkeypatch.setattr(
        eval_testset_common,
        "get_test_split",
        lambda _config: (curves, labels),
    )

    predictor = _Predictor(np.zeros((1, len(config.names) - 1)))
    with pytest.raises(ValueError, match="shape"):
        eval_testset_common.run_evaluation(
            predictor=predictor,
            config=config,
            results_dir=str(tmp_path),
            model_label="test",
            tag="bad-shape",
        )


def test_run_evaluation_rejects_nonfinite_prediction(tmp_path, monkeypatch):
    config = _Config()
    curves = np.zeros((1, 3, 4), dtype=np.float64)
    labels = np.zeros((1, len(config.names)), dtype=np.float64)
    monkeypatch.setattr(
        eval_testset_common,
        "get_test_split",
        lambda _config: (curves, labels),
    )
    prediction = np.zeros((1, len(config.names)))
    prediction[0, 0] = np.nan

    with pytest.raises(ValueError, match="values"):
        eval_testset_common.run_evaluation(
            predictor=_Predictor(prediction),
            config=config,
            results_dir=str(tmp_path),
            model_label="test",
            tag="nonfinite",
        )


def test_run_evaluation_counts_nonfinite_dt_as_invalid(
        tmp_path, monkeypatch):
    config = _Config()
    curves = np.zeros((1, 3, 4), dtype=np.float64)
    labels = np.zeros((1, len(config.names)), dtype=np.float64)
    prediction = np.zeros((1, len(config.names)))
    monkeypatch.setattr(
        eval_testset_common,
        "get_test_split",
        lambda _config: (curves, labels),
    )
    monkeypatch.setattr(
        pysim,
        "run_simulation",
        lambda _params: (curves[0], float("nan")),
    )

    summary = eval_testset_common.run_evaluation(
        predictor=_Predictor(prediction),
        config=config,
        results_dir=str(tmp_path),
        model_label="test",
        tag="nan-dt",
    )

    assert summary["n_valid"] == 0
    assert summary["n_invalid"] == 1


def test_clean_like_training_uses_transformed_float32_label_mask():
    class LogConfig(_Config):
        @staticmethod
        def get_log_transform_params():
            return ["K0"]

    config = LogConfig()
    curves = np.zeros((3, 3, 4), dtype=np.float32)
    labels = np.ones((3, len(config.names)), dtype=np.float64)
    k0_index = config.names.index("k0")
    labels[:, k0_index] = [1e-6, -1e-9, 1e-6]
    curves[2, 0, 0] = np.inf

    clean_curves, clean_labels = eval_testset_common.clean_like_training(
        curves, labels, config
    )

    assert clean_curves.shape[0] == 1
    np.testing.assert_array_equal(clean_labels[0], labels[0])


def test_evaluation_provenance_validates_checkpoint_artifact_hashes(
        tmp_path):
    dataset = tmp_path / "dataset.npz"
    checkpoint = tmp_path / "model.pth"
    scaler = tmp_path / "scaler.pkl"
    dataset.write_bytes(b"dataset")
    checkpoint.write_bytes(b"checkpoint")
    scaler.write_bytes(b"scaler")

    class Config(_Config):
        @staticmethod
        def get_split_seed():
            return 42

        @staticmethod
        def get_dataset_file():
            return str(dataset)

    predictor = _Predictor(np.zeros((1, len(_Config.names))))
    predictor.model_path = str(checkpoint)
    predictor.y_scaler_path = str(scaler)
    predictor.checkpoint_dataset_sha256 = sha256_file(dataset)
    predictor.checkpoint_y_scaler_sha256 = sha256_file(scaler)

    provenance = eval_testset_common.evaluation_provenance(
        predictor, Config()
    )

    assert provenance["dataset_sha256"] == sha256_file(dataset)
    assert provenance["checkpoint_sha256"] == sha256_file(checkpoint)
    assert provenance["y_scaler_sha256"] == sha256_file(scaler)
    assert (
        provenance["checkpoint_dataset_sha256"]
        == provenance["dataset_sha256"]
    )
    assert (
        provenance["checkpoint_y_scaler_sha256"]
        == provenance["y_scaler_sha256"]
    )


def test_evaluation_provenance_rejects_dataset_hash_mismatch(tmp_path):
    dataset = tmp_path / "dataset.npz"
    dataset.write_bytes(b"dataset")

    class Config(_Config):
        @staticmethod
        def get_dataset_file():
            return str(dataset)

    predictor = _Predictor(np.zeros((1, len(_Config.names))))
    predictor.checkpoint_dataset_sha256 = "0" * 64

    with pytest.raises(ValueError, match="dataset SHA-256 mismatch"):
        eval_testset_common.evaluation_provenance(predictor, Config())
