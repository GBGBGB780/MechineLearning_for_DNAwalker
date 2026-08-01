# coding=utf-8
"""
Transformer inference module.

Handles config loading, checkpoint loading, preprocessing, and inference.
"""

import os
import pickle
from typing import List, Optional

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_THIS_DIR = os.path.join(_REPO_ROOT, "train_transformer")

from .config import load_configs
from .model import build_transformer_model
from dnawalker.shared.parameters import resolve_checkpoint_param_names
from dnawalker.data.preprocessing import normalize_per_sample
from dnawalker.shared.artifacts import (
    optional_checkpoint_seed,
    optional_positive_int,
    require_matching_sha256,
)
from dnawalker.shared.device import pick_device


def _format_checkpoint_metric(checkpoint):
    for key in ("val_mse", "selection_score"):
        if key not in checkpoint:
            continue
        value = checkpoint.get(key)
        try:
            return key, f"{float(value):.6f}"
        except (TypeError, ValueError):
            return key, str(value)
    return None, "N/A"


class TransformerPredictor:
    """
    Load a trained transformer model and perform inference.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        parent_config_override_file: Optional[str] = None,
        transformer_config_override_file: Optional[str] = None,
    ) -> None:
        print("--- Initializing Transformer predictor ---")

        self.parent_config, self.transformer_config = load_configs(
            parent_override_file=os.path.abspath(parent_config_override_file)
            if parent_config_override_file
            else None,
            transformer_override_file=os.path.abspath(transformer_config_override_file)
            if transformer_config_override_file
            else None,
        )

        self.model_path = model_path or self.transformer_config.get_model_save_path()
        self.y_scaler_path = self.transformer_config.get_y_scaler_path()
        self.inference_batch_size = self.transformer_config.get_batch_size()

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(
                f"Missing y_scaler file: {self.y_scaler_path}. Train the model first."
            )

        self.device = pick_device()
        print(f"  Device: {self.device}")

        self.model = build_transformer_model(
            self.parent_config, self.transformer_config
        ).to(self.device)
        self._load_weights()

        # _load_weights validates the checkpoint/scaler hash before pickle load.
        with open(self.y_scaler_path, "rb") as f:
            self.y_scaler = pickle.load(f)
        output_size = self.parent_config.get_output_size()
        scaler_features = getattr(
            self.y_scaler, "n_features_in_", output_size
        )
        if int(scaler_features) != output_size:
            raise ValueError(
                "Y scaler feature count does not match model output size: "
                f"{scaler_features} vs {output_size}"
            )
        print(f"  Loaded y_scaler: {self.y_scaler_path}")

        self.model.eval()
        print(f"  Loaded model weights from {self.model_path}")
        print("--- Transformer resources ready ---\n")

    def _load_weights(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=True
        )
        self.checkpoint_model_seed = None
        self.checkpoint_split_seed = None
        self.checkpoint_dataset_sha256 = None
        self.checkpoint_y_scaler_sha256 = None
        self.checkpoint_split_manifest_sha256 = None
        self.checkpoint_train_subset_size = None
        self.checkpoint_epoch = None
        self.checkpoint_val_mse = None
        self.checkpoint_param_names_present = (
            isinstance(checkpoint, dict) and "param_names" in checkpoint
        )
        self.param_names = resolve_checkpoint_param_names(
            checkpoint,
            self.parent_config.get_trainable_param_names(),
        )
        if isinstance(checkpoint, dict):
            model_seed = checkpoint.get("model_seed")
            split_seed = checkpoint.get("split_seed")
            self.checkpoint_dataset_sha256 = checkpoint.get(
                "dataset_sha256"
            )
            self.checkpoint_y_scaler_sha256 = checkpoint.get(
                "y_scaler_sha256"
            )
            self.checkpoint_split_manifest_sha256 = checkpoint.get(
                "split_manifest_sha256"
            )
            self.checkpoint_train_subset_size = optional_positive_int(
                checkpoint.get("train_subset_size"),
                "train_subset_size",
            )
            self.checkpoint_epoch = checkpoint.get("epoch")
            self.checkpoint_val_mse = checkpoint.get("val_mse")
            self.checkpoint_model_seed = optional_checkpoint_seed(
                model_seed, "model_seed"
            )
            self.checkpoint_split_seed = optional_checkpoint_seed(
                split_seed, "split_seed"
            )
            if (self.checkpoint_model_seed is not None
                    and self.checkpoint_model_seed
                    != self.parent_config.get_random_seed()):
                raise ValueError(
                    "Checkpoint model_seed does not match configuration: "
                    f"{self.checkpoint_model_seed} vs "
                    f"{self.parent_config.get_random_seed()}"
                )
            if (self.checkpoint_split_seed is not None
                    and self.checkpoint_split_seed
                    != self.parent_config.get_split_seed()):
                raise ValueError(
                    "Checkpoint split_seed does not match configuration: "
                    f"{self.checkpoint_split_seed} vs "
                    f"{self.parent_config.get_split_seed()}"
                )

        manifest_getter = getattr(
            self.parent_config, "get_split_manifest_file", lambda: None
        )
        size_getter = getattr(
            self.parent_config, "get_train_subset_size", lambda: None
        )
        configured_manifest = manifest_getter()
        configured_train_size = size_getter()
        if configured_manifest is not None:
            if (
                self.checkpoint_split_manifest_sha256 is None
                or self.checkpoint_train_subset_size is None
            ):
                raise ValueError(
                    "Checkpoint lacks required explicit-split provenance"
                )
            require_matching_sha256(
                configured_manifest,
                self.checkpoint_split_manifest_sha256,
                "split_manifest",
            )
            if self.checkpoint_train_subset_size != configured_train_size:
                raise ValueError(
                    "Checkpoint train_subset_size does not match configuration: "
                    f"{self.checkpoint_train_subset_size} vs "
                    f"{configured_train_size}"
                )
        elif (
            self.checkpoint_split_manifest_sha256 is not None
            or self.checkpoint_train_subset_size is not None
        ):
            raise ValueError(
                "Checkpoint requires an explicit split manifest configuration"
            )

        self.y_scaler_sha256 = require_matching_sha256(
            self.y_scaler_path,
            self.checkpoint_y_scaler_sha256,
            "y_scaler",
        )

        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
            metric_key, metric_value = _format_checkpoint_metric(checkpoint)
            epoch = checkpoint.get("epoch", "unknown")
            if metric_key == "selection_score":
                print(f"  (Loaded from epoch {epoch}, selection score: {metric_value})")
            elif metric_key == "val_mse":
                print(f"  (Loaded from epoch {epoch}, val MSE: {metric_value})")
            else:
                print(f"  (Loaded from epoch {epoch})")
            return

        self.model.load_state_dict(checkpoint)

    @staticmethod
    def _resize_to_expected_length(x_input: np.ndarray, expected_len: int) -> np.ndarray:
        """
        Resize each curve to the expected length using linear interpolation.
        """
        batch_size, channels, seq_len = x_input.shape
        if seq_len == expected_len:
            return x_input
        if seq_len <= 0:
            raise ValueError("Input sequence length must be greater than 0.")

        src_axis = np.linspace(0.0, 1.0, seq_len, dtype=np.float64)
        dst_axis = np.linspace(0.0, 1.0, expected_len, dtype=np.float64)
        resized = np.empty((batch_size, channels, expected_len), dtype=np.float32)

        for batch_idx in range(batch_size):
            for channel_idx in range(channels):
                curve = np.asarray(x_input[batch_idx, channel_idx], dtype=np.float64)
                finite_mask = np.isfinite(curve)

                if not finite_mask.any():
                    resized[batch_idx, channel_idx] = 0.0
                    continue

                if finite_mask.sum() == 1:
                    resized[batch_idx, channel_idx] = curve[finite_mask][0]
                    continue

                cleaned_curve = np.interp(
                    src_axis, src_axis[finite_mask], curve[finite_mask]
                )
                resized[batch_idx, channel_idx] = np.interp(
                    dst_axis, src_axis, cleaned_curve
                )

        return resized

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        """
        Predict physical parameters from one or more input curve samples.

        Accepts either `(3, T)` or `(B, 3, T)` and returns `(B, output_size)`.
        """
        try:
            x_raw = np.asarray(x_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("Input curves must be a numeric array") from exc
        if (not np.issubdtype(x_raw.dtype, np.number)
                or np.issubdtype(x_raw.dtype, np.complexfloating)):
            raise ValueError("Input curves must be a real numeric array")
        if x_raw.ndim == 2:
            x_input = x_raw[np.newaxis, ...]
        elif x_raw.ndim == 3:
            x_input = x_raw
        else:
            raise ValueError(
                f"Input X shape must be (3, T) or (B, 3, T), got {x_raw.shape}"
            )

        _, channels, seq_len = x_input.shape
        if x_input.shape[0] == 0:
            raise ValueError("Input batch must contain at least one sample")
        if seq_len <= 0:
            raise ValueError("Input sequence length must be greater than 0")
        if not np.all(np.isfinite(x_input)):
            raise ValueError("Input curves contain NaN or Inf")
        expected_channels = self.parent_config.get_num_curves()
        if channels != expected_channels:
            raise ValueError(
                f"Input channel count mismatch. Expected {expected_channels}, got {channels}"
            )

        expected_len = self.parent_config.get_seq_length()
        if seq_len != expected_len:
            print(
                f"Warning: input length {seq_len} does not match expected length "
                f"{expected_len}; resizing automatically."
            )
            x_input = self._resize_to_expected_length(x_input, expected_len)

        # Keep model forwards bounded by the training batch size. Besides
        # limiting device memory, this avoids incorrect MPS results observed
        # when an entire 1,000-sample evaluation set was submitted at once.
        batch_size = getattr(
            self, "inference_batch_size", x_input.shape[0]
        )
        if (isinstance(batch_size, bool)
                or not isinstance(batch_size, (int, np.integer))
                or batch_size <= 0):
            raise ValueError(
                "inference_batch_size must be a positive integer, got "
                f"{batch_size!r}"
            )

        output_size = len(self.param_names)
        scaled_predictions = []
        for start in range(0, x_input.shape[0], int(batch_size)):
            batch = np.asarray(
                x_input[start:start + int(batch_size)],
                dtype=np.float32,
            )
            # Per-sample joint-channel normalization, shared with training and
            # CNN inference via preprocessing.normalize_per_sample.
            x_scaled = normalize_per_sample(batch).astype(
                np.float32, copy=False
            )
            x_tensor = torch.from_numpy(
                np.ascontiguousarray(x_scaled)
            ).to(self.device)
            with torch.no_grad():
                batch_predictions = self.model(x_tensor).cpu().numpy()
            expected_batch_shape = (batch.shape[0], output_size)
            if batch_predictions.shape != expected_batch_shape:
                raise ValueError(
                    "Model output shape mismatch: "
                    f"expected {expected_batch_shape}, "
                    f"got {batch_predictions.shape}"
                )
            if not np.all(np.isfinite(batch_predictions)):
                raise ValueError("Model produced NaN or Inf predictions")
            scaled_predictions.append(batch_predictions)

        y_pred_scaled = np.concatenate(scaled_predictions, axis=0)
        expected_output_shape = (x_input.shape[0], output_size)

        y_pred_real = np.asarray(
            self.y_scaler.inverse_transform(y_pred_scaled)
        ).copy()
        if y_pred_real.shape != expected_output_shape:
            raise ValueError(
                "Y scaler output shape mismatch: "
                f"expected {expected_output_shape}, got {y_pred_real.shape}"
            )
        if not np.all(np.isfinite(y_pred_real)):
            raise ValueError("Y scaler produced NaN or Inf predictions")

        param_names = self.param_names
        log_transform_params = self.parent_config.get_log_transform_params()
        log_epsilon = self.parent_config.get_log_epsilon()
        param_index = {
            str(name).lower(): index
            for index, name in enumerate(param_names)
        }

        with np.errstate(over="ignore", invalid="ignore"):
            for param_name in log_transform_params:
                idx = param_index.get(str(param_name).lower())
                if idx is not None:
                    y_pred_real[:, idx] = (
                        np.power(10.0, y_pred_real[:, idx]) - log_epsilon
                    )
        if not np.all(np.isfinite(y_pred_real)):
            raise ValueError("Inverse log transform produced NaN or Inf predictions")

        return y_pred_real

    def get_param_names(self) -> List[str]:
        return list(self.param_names)
