# coding=utf-8
"""
inference_cnn.py — CNN 推理模块，封装模型加载和预测逻辑
inference_cnn.py — CNN inference module encapsulating model loading and prediction logic
"""

import torch
import numpy as np
import pickle
import os
import sys

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from model_cnn import InverseCNN
from config_loader import Config


class NanorobotPredictor:
    """
    CNN 预测器：加载训练好的模型并执行推理。
    CNN predictor: loads a trained model and performs inference.
    """

    def __init__(self, config_file=None, model_path=None):
        """
        初始化预测器。/ Initialize predictor.

        Args:
            config_file: 配置文件路径 / config file path (default: parent configfile.ini)
            model_path:  模型路径（可选）/ model path (optional, reads from config if None)
        """
        print("--- Loading NanorobotPredictor Resources ---")

        if config_file is None:
            config_file = os.path.join(_PARENT_DIR, 'configfile.ini')

        self.config = Config(config_file)
        self.input_size = self.config.get_input_size()
        self.output_size = self.config.get_output_size()
        self.param_ranges = self.config.get_param_ranges()

        # 路径 / Paths
        self.model_path = model_path if model_path else self.config.get_model_save_path()
        self.y_scaler_path = self.config.get_y_scaler_file()

        # 加载 y_scaler / Load y_scaler
        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"Y Scaler not found: {self.y_scaler_path}. Train first.")
        with open(self.y_scaler_path, 'rb') as f:
            self.y_scaler = pickle.load(f)

        # 加载模型 / Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        self.model = InverseCNN(self.input_size, self.output_size, self.config).to(self.device)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}. Train first.")
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        print("--- Resources Loaded Successfully ---\n")

    def predict(self, X_data):
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
        # 形状处理 / Shape handling
        if X_data.ndim == 2 and X_data.shape[0] == 3:
            X_flat = X_data.reshape(1, -1)
        elif X_data.ndim == 3:
            X_flat = X_data.reshape(X_data.shape[0], -1)
        else:
            X_flat = X_data

        if X_flat.shape[1] != self.input_size:
            raise ValueError(f"维度不匹配 / Dimension mismatch: expected {self.input_size}, got {X_flat.shape[1]}")

        # 单样本联合通道归一化 / Per-sample joint-channel normalization
        num_ch = self.config.get_num_curves()
        seq_len = self.input_size // num_ch
        X_3d = X_flat.reshape(-1, num_ch, seq_len)
        means = np.nanmean(X_3d, axis=(1, 2), keepdims=True)
        stds = np.nanstd(X_3d, axis=(1, 2), keepdims=True) + 1e-8
        X_3d = (X_3d - means) / stds
        X_3d = np.where(np.isnan(X_3d), 0.0, X_3d)
        X_scaled = X_3d.reshape(X_flat.shape[0], -1)

        # 推理 / Inference
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).cpu().numpy()

        # 反归一化 / Inverse scaling
        pred_real = self.y_scaler.inverse_transform(pred_scaled)

        # 反对数变换 / Inverse log transform
        log_params = self.config.get_log_transform_params()
        log_eps = self.config.get_log_epsilon()
        param_names = self.config.get_trainable_param_names()
        for p in log_params:
            if p in param_names:
                idx = param_names.index(p)
                pred_real[:, idx] = np.power(10, pred_real[:, idx]) - log_eps

        return pred_real

    def get_param_names(self):
        """返回待训练参数名称列表。/ Returns trainable parameter names."""
        return self.config.get_trainable_param_names()
