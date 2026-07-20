# coding=utf-8
"""
Transformer inference module.

Handles config loading, checkpoint loading, preprocessing, and inference.
"""

import os
import pickle
import sys

import numpy as np
import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config_loader_transformer import load_configs
from model_transformer import build_transformer_model


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
        model_path=None,
        parent_config_override_file=None,
        transformer_config_override_file=None,
    ):
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

        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(
                f"Missing y_scaler file: {self.y_scaler_path}. Train the model first."
            )
        with open(self.y_scaler_path, "rb") as f:
            self.y_scaler = pickle.load(f)
        print(f"  Loaded y_scaler: {self.y_scaler_path}")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"  Device: {self.device}")

        self.model = build_transformer_model(
            self.parent_config, self.transformer_config
        ).to(self.device)
        self._load_weights()
        self.model.eval()
        print(f"  Loaded model weights from {self.model_path}")
        print("--- Transformer resources ready ---\n")

    def _load_weights(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Missing model file: {self.model_path}")

        checkpoint = torch.load(
            self.model_path, map_location=self.device, weights_only=False
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
    def _resize_to_expected_length(x_input, expected_len):
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

    def predict(self, x_raw):
        """
        Predict physical parameters from one or more input curve samples.

        Accepts either `(3, T)` or `(B, 3, T)` and returns `(B, output_size)`.
        """
        x_raw = np.asarray(x_raw, dtype=np.float32)
        if x_raw.ndim == 2:
            x_input = x_raw[np.newaxis, ...]
        elif x_raw.ndim == 3:
            x_input = x_raw
        else:
            raise ValueError(
                f"Input X shape must be (3, T) or (B, 3, T), got {x_raw.shape}"
            )

        _, channels, seq_len = x_input.shape
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

        sample_means = np.nanmean(x_input, axis=(1, 2), keepdims=True)
        sample_stds = np.nanstd(x_input, axis=(1, 2), keepdims=True) + 1e-8
        x_scaled = (x_input - sample_means) / sample_stds
        x_scaled = np.nan_to_num(x_scaled, nan=0.0).astype(np.float32)

        x_tensor = torch.tensor(x_scaled).to(self.device)
        with torch.no_grad():
            y_pred_scaled = self.model(x_tensor).cpu().numpy()

        y_pred_real = self.y_scaler.inverse_transform(y_pred_scaled)

        param_names = self.parent_config.get_trainable_param_names()
        log_transform_params = self.parent_config.get_log_transform_params()
        log_epsilon = self.parent_config.get_log_epsilon()

        for param_name in log_transform_params:
            if param_name in param_names:
                idx = param_names.index(param_name)
                y_pred_real[:, idx] = np.power(10.0, y_pred_real[:, idx]) - log_epsilon

        return y_pred_real

    def get_param_names(self):
        return self.parent_config.get_trainable_param_names()
