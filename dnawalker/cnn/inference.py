# coding=utf-8
"""
dnawalker.cnn.inference — CNN 推理模块，封装模型加载和预测逻辑
dnawalker.cnn.inference — CNN inference module (model loading + prediction).
"""

from typing import List, Optional

import torch
import numpy as np
import pickle
import os

from .model import InverseCNN
from .config import CNNConfig, resolve_artifact_path
from dnawalker.shared.parameters import resolve_checkpoint_param_names
from dnawalker.data.preprocessing import normalize_per_sample
from dnawalker.shared.artifacts import (
    optional_checkpoint_seed,
    optional_positive_int,
    require_matching_sha256,
)
from dnawalker.shared.device import pick_device


_resolve_config_artifact = resolve_artifact_path


class NanorobotPredictor:
    """
    CNN 预测器：加载训练好的模型并执行推理。
    CNN predictor: loads a trained model and performs inference.
    """

    def __init__(self, config_file: Optional[str] = None,
                 model_path: Optional[str] = None,
                 config_override_file: Optional[str] = None) -> None:
        """
        初始化预测器。/ Initialize predictor.

        Args:
            config_file: Shared INI path (default: ``configs/common.ini``).
            model_path: Optional checkpoint path (defaults to configuration).
        """
        print("--- Loading NanorobotPredictor Resources ---")

        primary_config = (
            os.path.abspath(config_file) if config_file is not None else None
        )
        extra_config_files = (
            [os.path.abspath(config_override_file)]
            if config_override_file
            else None
        )
        self.config = CNNConfig(
            primary_config, extra_config_files=extra_config_files
        )
        self.input_size = self.config.get_input_size()
        self.output_size = self.config.get_output_size()
        self.param_ranges = self.config.get_param_ranges()

        # 路径 / Paths
        self.model_path = (
            model_path
            if model_path
            else _resolve_config_artifact(self.config.get_model_save_path())
        )
        self.y_scaler_path = _resolve_config_artifact(
            self.config.get_y_scaler_file()
        )

        # Validate existence before loading either paired artifact.
        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"Y Scaler not found: {self.y_scaler_path}. Train first.")

        # 加载模型 / Load model
        self.device = pick_device()
        print(f"Device: {self.device}")
        self.model = InverseCNN(self.input_size, self.output_size, self.config).to(self.device)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}. Train first.")
        # weights_only=True: 仅反序列化张量权重，避免加载不受信 .pth 时的任意代码执行；
        # 本模型 checkpoint 只存 state_dict，故安全 (torch>=2.6 已将其设为默认)。
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
            self.config.get_trainable_param_names(),
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
                    != self.config.get_random_seed()):
                raise ValueError(
                    "Checkpoint model_seed does not match configuration: "
                    f"{self.checkpoint_model_seed} vs "
                    f"{self.config.get_random_seed()}"
                )
            if (self.checkpoint_split_seed is not None
                    and self.checkpoint_split_seed
                    != self.config.get_split_seed()):
                raise ValueError(
                    "Checkpoint split_seed does not match configuration: "
                    f"{self.checkpoint_split_seed} vs "
                    f"{self.config.get_split_seed()}"
                )

        manifest_getter = getattr(
            self.config, "get_split_manifest_file", lambda: None
        )
        size_getter = getattr(
            self.config, "get_train_subset_size", lambda: None
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

        # Check the checkpoint/scaler pairing before unpickling the scaler.
        self.y_scaler_sha256 = require_matching_sha256(
            self.y_scaler_path,
            self.checkpoint_y_scaler_sha256,
            "y_scaler",
        )
        with open(self.y_scaler_path, 'rb') as f:
            self.y_scaler = pickle.load(f)
        scaler_features = getattr(
            self.y_scaler, "n_features_in_", self.output_size
        )
        if int(scaler_features) != self.output_size:
            raise ValueError(
                "Y scaler feature count does not match model output size: "
                f"{scaler_features} vs {self.output_size}"
            )

        state_dict = (
            checkpoint["model_state"]
            if isinstance(checkpoint, dict) and "model_state" in checkpoint
            else checkpoint
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print("--- Resources Loaded Successfully ---\n")

    def predict(self, X_data: np.ndarray) -> np.ndarray:
        """
        执行推理。/ Run prediction.

        Args:
            X_data: numpy 数组 / numpy array, shapes:
                    (3, T) 单样本 / single sample → reshaped to (1, 3*T)
                    (N, 3, T) 批次 / batch → reshaped to (N, 3*T)
                    (N, Features) 已展平 / already flattened

        Returns:
            predicted_real_params: (N, output_size) 物理参数值 / physical parameter values
        """
        try:
            # Training normalizes the stored float32 curves. Cast before
            # normalization so inference follows the same arithmetic path.
            X_array = np.asarray(X_data, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Input curves must be a numeric array") from exc

        num_ch = self.config.get_num_curves()
        seq_len = self.input_size // num_ch
        curve_shape = (num_ch, seq_len)

        # 形状处理 / Shape handling
        if X_array.ndim == 2 and X_array.shape == curve_shape:
            X_flat = X_array.reshape(1, -1)
        elif X_array.ndim == 2 and X_array.shape[1] == self.input_size:
            X_flat = X_array
        elif X_array.ndim == 3 and X_array.shape[1:] == curve_shape:
            X_flat = X_array.reshape(X_array.shape[0], -1)
        else:
            raise ValueError(
                "Input X shape must be (C, T), (N, C, T), or "
                f"(N, C*T) with (C, T)={curve_shape}; got {X_array.shape}"
            )

        if X_flat.shape[0] == 0:
            raise ValueError("Input batch must contain at least one sample")
        if not np.all(np.isfinite(X_flat)):
            raise ValueError("Input curves contain NaN or Inf")

        # 单样本联合通道归一化 / Per-sample joint-channel normalization
        # 与训练侧 data_loader 共用 preprocessing.normalize_per_sample，分布一致。
        X_3d = normalize_per_sample(X_flat.reshape(-1, num_ch, seq_len))
        X_scaled = X_3d.reshape(X_flat.shape[0], -1)

        # 推理 / Inference
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()
        expected_output_shape = (X_flat.shape[0], self.output_size)
        if pred_scaled.shape != expected_output_shape:
            raise ValueError(
                "Model output shape mismatch: "
                f"expected {expected_output_shape}, got {pred_scaled.shape}"
            )
        if not np.all(np.isfinite(pred_scaled)):
            raise ValueError("Model produced NaN or Inf predictions")

        # 反归一化 / Inverse scaling
        pred_real = np.asarray(
            self.y_scaler.inverse_transform(pred_scaled)
        ).copy()
        if pred_real.shape != expected_output_shape:
            raise ValueError(
                "Y scaler output shape mismatch: "
                f"expected {expected_output_shape}, got {pred_real.shape}"
            )
        if not np.all(np.isfinite(pred_real)):
            raise ValueError("Y scaler produced NaN or Inf predictions")

        # 反对数变换 / Inverse log transform
        log_params = self.config.get_log_transform_params()
        log_eps = self.config.get_log_epsilon()
        param_names = self.param_names
        param_index = {
            str(name).lower(): index
            for index, name in enumerate(param_names)
        }
        with np.errstate(over="ignore", invalid="ignore"):
            for p in log_params:
                idx = param_index.get(str(p).lower())
                if idx is not None:
                    pred_real[:, idx] = np.power(10, pred_real[:, idx]) - log_eps
        if not np.all(np.isfinite(pred_real)):
            raise ValueError("Inverse log transform produced NaN or Inf predictions")

        return pred_real

    def get_param_names(self) -> List[str]:
        """返回待训练参数名称列表。/ Returns trainable parameter names."""
        return list(self.param_names)
