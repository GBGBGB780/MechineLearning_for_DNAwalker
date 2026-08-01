import numpy as np
import pytest
import torch
from types import SimpleNamespace

from dnawalker.physics import pysim
from dnawalker.cnn import inference as cnn_inference
from dnawalker.cnn.inference import (
    NanorobotPredictor,
    _resolve_config_artifact,
)
from dnawalker.cnn import evaluate as cnn_eval_dual
from dnawalker.cnn import predict as cnn_predict_refine
from dnawalker.transformer import evaluate as transformer_eval_dual
from dnawalker.transformer import inference as transformer_inference
from dnawalker.transformer import predict as transformer_predict_refine
from dnawalker.transformer.inference import TransformerPredictor
from dnawalker.shared.artifacts import optional_checkpoint_seed
from dnawalker.shared.artifacts import sha256_file


class _Config:
    def __init__(self, log_params=None):
        self._log_params = [] if log_params is None else log_params

    def get_num_curves(self):
        return 3

    def get_seq_length(self):
        return 4

    def get_log_transform_params(self):
        return self._log_params

    def get_log_epsilon(self):
        return 1e-12


def test_cnn_config_artifacts_are_independent_of_caller_cwd(
        tmp_path, monkeypatch):
    relative = _resolve_config_artifact("results/model.pth")
    monkeypatch.chdir(tmp_path)
    from_other_cwd = _resolve_config_artifact("results/model.pth")
    assert relative == from_other_cwd
    assert relative.endswith("artifacts/models/cnn/model.pth")

    absolute = str(tmp_path / "model.pth")
    assert _resolve_config_artifact(absolute) == absolute


def test_checkpoint_seed_metadata_rejects_lossy_coercions():
    assert optional_checkpoint_seed(None, "model_seed") is None
    assert optional_checkpoint_seed(np.int64(42), "model_seed") == 42
    for invalid in (True, 42.5, "42", -1, 2 ** 32):
        with pytest.raises(ValueError, match="model_seed"):
            optional_checkpoint_seed(invalid, "model_seed")


class _IdentityScaler:
    def inverse_transform(self, values):
        return np.asarray(values)


class _Model:
    def __init__(self, columns=7, fill=0.0):
        self.columns = columns
        self.fill = fill

    def __call__(self, values):
        return torch.full(
            (values.shape[0], self.columns),
            self.fill,
            dtype=torch.float32,
            device=values.device,
        )


class _RecordingModel(_Model):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def __call__(self, values):
        self.batch_sizes.append(values.shape[0])
        return super().__call__(values)


class _LoadableModel:
    def to(self, _device):
        return self

    def load_state_dict(self, _state):
        return None

    def eval(self):
        return self


def _cnn_predictor(model=None, config=None):
    predictor = NanorobotPredictor.__new__(NanorobotPredictor)
    predictor.config = config or _Config()
    predictor.input_size = 12
    predictor.output_size = len(pysim.PARAM_NAMES)
    predictor.param_names = list(pysim.PARAM_NAMES)
    predictor.device = torch.device("cpu")
    predictor.model = model or _Model()
    predictor.y_scaler = _IdentityScaler()
    return predictor


def _transformer_predictor(model=None, config=None):
    predictor = TransformerPredictor.__new__(TransformerPredictor)
    predictor.parent_config = config or _Config()
    predictor.param_names = list(pysim.PARAM_NAMES)
    predictor.device = torch.device("cpu")
    predictor.model = model or _Model()
    predictor.y_scaler = _IdentityScaler()
    return predictor


@pytest.mark.parametrize(
    "curves",
    [
        np.ones(12),
        np.ones((2, 2, 3)),
        np.ones((1, 1, 12)),
        np.empty((0, 12)),
        np.full((3, 4), np.nan),
    ],
)
def test_cnn_rejects_invalid_inputs(curves):
    with pytest.raises(ValueError, match="shape|sample|NaN|Inf"):
        _cnn_predictor().predict(curves)


def test_cnn_accepts_curve_and_flattened_batches():
    predictor = _cnn_predictor()
    assert predictor.predict(np.ones((3, 4))).shape == (1, 7)
    assert predictor.predict(np.ones((2, 12))).shape == (2, 7)
    assert predictor.predict(np.ones((2, 3, 4))).shape == (2, 7)


def test_cnn_casts_to_float32_before_shared_normalization(monkeypatch):
    observed = {}

    def record_dtype(curves, **_kwargs):
        observed["dtype"] = curves.dtype
        return curves

    monkeypatch.setattr(
        cnn_inference, "normalize_per_sample", record_dtype
    )
    predictor = _cnn_predictor()
    predictor.predict(np.ones((3, 4), dtype=np.float64))

    assert observed["dtype"] == np.dtype(np.float32)


@pytest.mark.parametrize(
    "curves",
    [
        np.ones(12),
        np.ones((1, 2, 4)),
        np.empty((0, 3, 4)),
        np.empty((3, 0)),
        np.full((3, 4), np.inf),
    ],
)
def test_transformer_rejects_invalid_inputs(curves):
    with pytest.raises(ValueError, match="shape|channel|sample|length|NaN|Inf"):
        _transformer_predictor().predict(curves)


def test_transformer_accepts_single_and_batched_curves():
    predictor = _transformer_predictor()
    assert predictor.predict(np.ones((3, 4))).shape == (1, 7)
    assert predictor.predict(np.ones((2, 3, 4))).shape == (2, 7)


def test_transformer_predicts_in_bounded_batches():
    model = _RecordingModel()
    predictor = _transformer_predictor(model=model)
    predictor.inference_batch_size = 2

    prediction = predictor.predict(np.ones((5, 3, 4)))

    assert prediction.shape == (5, 7)
    assert model.batch_sizes == [2, 2, 1]


def test_cnn_rejects_scaler_hash_mismatch_before_pickle(
        tmp_path, monkeypatch):
    scaler_path = tmp_path / "scaler.pkl"
    model_path = tmp_path / "model.pth"
    scaler_path.write_bytes(b"not loaded")
    model_path.write_bytes(b"checkpoint placeholder")

    class Config:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def get_input_size():
            return 12

        @staticmethod
        def get_output_size():
            return len(pysim.PARAM_NAMES)

        @staticmethod
        def get_param_ranges():
            return {}

        @staticmethod
        def get_model_save_path():
            return str(model_path)

        @staticmethod
        def get_y_scaler_file():
            return str(scaler_path)

        @staticmethod
        def get_trainable_param_names():
            return list(pysim.PARAM_NAMES)

        @staticmethod
        def get_random_seed():
            return 42

        @staticmethod
        def get_split_seed():
            return 42

    checkpoint = {
        "model_state": {},
        "param_names": list(pysim.PARAM_NAMES),
        "model_seed": 42,
        "split_seed": 42,
        "y_scaler_sha256": "0" * 64,
    }
    pickle_called = False

    def fail_pickle(_stream):
        nonlocal pickle_called
        pickle_called = True
        raise AssertionError("pickle must not be loaded before hash validation")

    monkeypatch.setattr(cnn_inference, "CNNConfig", Config)
    monkeypatch.setattr(
        cnn_inference, "InverseCNN", lambda *_args: _LoadableModel()
    )
    monkeypatch.setattr(
        cnn_inference, "pick_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(cnn_inference.torch, "load", lambda *_a, **_k: checkpoint)
    monkeypatch.setattr(cnn_inference.pickle, "load", fail_pickle)

    with pytest.raises(ValueError, match="y_scaler SHA-256 mismatch"):
        NanorobotPredictor(model_path=str(model_path))
    assert not pickle_called


def test_transformer_rejects_scaler_hash_mismatch_before_pickle(
        tmp_path, monkeypatch):
    scaler_path = tmp_path / "scaler.pkl"
    model_path = tmp_path / "model.pth"
    scaler_path.write_bytes(b"not loaded")
    model_path.write_bytes(b"checkpoint placeholder")

    class ParentConfig:
        @staticmethod
        def get_output_size():
            return len(pysim.PARAM_NAMES)

        @staticmethod
        def get_trainable_param_names():
            return list(pysim.PARAM_NAMES)

        @staticmethod
        def get_random_seed():
            return 42

        @staticmethod
        def get_split_seed():
            return 42

    class TransformerConfig:
        @staticmethod
        def get_model_save_path():
            return str(model_path)

        @staticmethod
        def get_y_scaler_path():
            return str(scaler_path)

        @staticmethod
        def get_batch_size():
            return 2

    checkpoint = {
        "model_state": {},
        "param_names": list(pysim.PARAM_NAMES),
        "model_seed": 42,
        "split_seed": 42,
        "y_scaler_sha256": "0" * 64,
    }
    pickle_called = False

    def fail_pickle(_stream):
        nonlocal pickle_called
        pickle_called = True
        raise AssertionError("pickle must not be loaded before hash validation")

    monkeypatch.setattr(
        transformer_inference,
        "load_configs",
        lambda **_kwargs: (ParentConfig(), TransformerConfig()),
    )
    monkeypatch.setattr(
        transformer_inference,
        "build_transformer_model",
        lambda *_args: _LoadableModel(),
    )
    monkeypatch.setattr(
        transformer_inference, "pick_device", lambda: torch.device("cpu")
    )
    monkeypatch.setattr(
        transformer_inference.torch,
        "load",
        lambda *_args, **_kwargs: checkpoint,
    )
    monkeypatch.setattr(
        transformer_inference.pickle, "load", fail_pickle
    )

    with pytest.raises(ValueError, match="y_scaler SHA-256 mismatch"):
        TransformerPredictor(model_path=str(model_path))
    assert not pickle_called


@pytest.mark.parametrize(
    "factory",
    [_cnn_predictor, _transformer_predictor],
)
def test_predictors_reject_wrong_model_output_shape(factory):
    with pytest.raises(ValueError, match="output shape"):
        factory(_Model(columns=6)).predict(np.ones((3, 4)))


@pytest.mark.parametrize(
    "factory",
    [_cnn_predictor, _transformer_predictor],
)
def test_predictors_reject_nonfinite_model_output(factory):
    with pytest.raises(ValueError, match="NaN|Inf"):
        factory(_Model(fill=np.nan)).predict(np.ones((3, 4)))


@pytest.mark.parametrize(
    "factory",
    [_cnn_predictor, _transformer_predictor],
)
def test_predictors_match_log_parameter_names_case_insensitively(factory):
    config = _Config(log_params=["K0"])
    predictor = factory(_Model(fill=1.0), config=config)

    prediction = predictor.predict(np.ones((3, 4)))
    k0_index = predictor.param_names.index("k0")

    assert prediction[0, k0_index] == pytest.approx(
        10.0 - config.get_log_epsilon()
    )
    np.testing.assert_allclose(
        np.delete(prediction[0], k0_index),
        np.ones(len(pysim.PARAM_NAMES) - 1),
    )


@pytest.mark.parametrize("value", [-1, 1.5, True])
@pytest.mark.parametrize(
    "invoke",
    [
        lambda value: cnn_predict_refine.run(multistart=value),
        lambda value: transformer_predict_refine.run(multistart=value),
        lambda value: cnn_eval_dual.eval_one(
            None, None, None, None, 0, 0.0, 1, value, 0
        ),
        lambda value: transformer_eval_dual.eval_one(
            None, None, None, None, 0, 0.0, 1, value, 0
        ),
    ],
    ids=[
        "cnn-predict-refine",
        "transformer-predict-refine",
        "cnn-eval-dual",
        "transformer-eval-dual",
    ],
)
def test_refinement_entrypoints_reject_invalid_multistart_before_io(
        invoke, value):
    with pytest.raises(ValueError, match="multistart"):
        invoke(value)


@pytest.mark.parametrize(
    ("module", "model_name"),
    [
        (cnn_eval_dual, "cnn"),
        (transformer_eval_dual, "transformer"),
    ],
)
def test_dual_evaluation_metadata_binds_inputs_and_settings(
        tmp_path, monkeypatch, module, model_name):
    original = tmp_path / "original.xlsx"
    generalization = tmp_path / "generalization.xlsx"
    original.write_bytes(b"original")
    generalization.write_bytes(b"generalization")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(module, "DATASETS", {
        "original": str(original),
        "generalization": str(generalization),
    })
    args = SimpleNamespace(
        checkpoint_selection="minimum_validation_mse",
        ensemble=20,
        noise_std=0.005,
        no_refine=False,
        maxiter=500,
        multistart=8,
        seed=3,
    )

    metadata = module._build_metadata(args)

    assert metadata["schema_version"] == 2
    assert metadata["model"] == model_name
    assert metadata["checkpoint_selection"] == "minimum_validation_mse"
    assert metadata["evaluation_settings"] == {
        "ensemble": 20,
        "noise_std": 0.005,
        "refine_enabled": True,
        "refinement_method": "Powell",
        "maxiter": 500,
        "multistart": 8,
        "seed": 3,
    }
    assert metadata["experimental_inputs"]["original"] == {
        "path": "original.xlsx",
        "sha256": sha256_file(original),
    }
    assert metadata["experimental_inputs"]["generalization"] == {
        "path": "generalization.xlsx",
        "sha256": sha256_file(generalization),
    }
