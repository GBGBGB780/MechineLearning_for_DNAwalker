# coding=utf-8
"""
inference_transformer.py — Transformer 推理模块
inference_transformer.py — Transformer inference module

封装模型加载、预处理和推理逻辑。
Encapsulates model loading, preprocessing, and inference logic.
"""

import os
import sys
import torch
import numpy as np
import pickle

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config_loader_transformer import load_configs
from model_transformer import build_transformer

class TransformerPredictor:
    """
    Transformer 预测器：加载训练好的模型并执行推理。
    Transformer predictor: loads a trained model and performs inference.
    """
    def __init__(self, model_path=None):
        print("--- 正在初始化 Transformer 预测器 ---")
        
        # 1. 加载配置
        self.parent_config, self.transformer_config = load_configs()
        
        # 2. 路径设置
        self.model_path = model_path if model_path else self.transformer_config.get_model_save_path()
        self.y_scaler_path = self.transformer_config.get_y_scaler_path()
        
        # 3. 加载 y_scaler (MinMaxScaler)
        if not os.path.exists(self.y_scaler_path):
            raise FileNotFoundError(f"未找到 y_scaler 文件: {self.y_scaler_path}。请确保已完成训练。")
        with open(self.y_scaler_path, 'rb') as f:
            self.y_scaler = pickle.load(f)
        print(f"  已加载 y_scaler: {self.y_scaler_path}")

        # 4. 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  使用设备: {self.device}")
        
        self.model = build_transformer(self.parent_config, self.transformer_config).to(self.device)
        self._load_weights()
        self.model.eval()
        print(f"  已从 {self.model_path} 加载模型权重")
        print("--- Transformer 资源加载成功 ---\n")

    def _load_weights(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        # 支持保存了 epoch, model_state, val_mse 的字典格式，也支持直接保存 state_dict
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state'])
            print(f"  (加载自 Epoch {checkpoint.get('epoch', 'unknown')}, Val MSE: {checkpoint.get('val_mse', 'N/A'):.6f})")
        else:
            self.model.load_state_dict(checkpoint)

    def predict(self, X_raw):
        """
        对输入数据进行预测并反归一化。
        
        Args:
            X_raw: Numpy 数组，形状可以是:
                   - (3, T): 单个样本的 3 通道曲线 (FAM, TYE, CY5)
                   - (B, 3, T): 多个样本的批次数据
        
        Returns:
            predicted_real: 反归一化后的物理参数 (B, output_size)
        """
        # 1. 维度处理
        if X_raw.ndim == 2:
            # (3, T) -> (1, 3, T)
            X_input = X_raw[np.newaxis, ...]
        elif X_raw.ndim == 3:
            X_input = X_raw
        else:
            raise ValueError(f"输入 X 维度不匹配。期望 (B, 3, T) 或 (3, T)，得到 {X_raw.shape}")

        B, C, L = X_input.shape
        expected_len = self.parent_config.get_seq_length()
        if L != expected_len:
            print(f"警告: 输入长度 ({L}) 与期望长度 ({expected_len}) 不匹配，将进行插值补全/截断。")
            # 这里简单判断，实际应用中建议先在外部做 pd.read_excel -> interp1d 处理
            pass

        # 2. 单样本联合通道归一化 (Domain Invariant)
        # 计算每个样本 (3, T) 的均值和标推差
        sample_means = np.nanmean(X_input, axis=(1, 2), keepdims=True)  # (B, 1, 1)
        sample_stds  = np.nanstd(X_input,  axis=(1, 2), keepdims=True) + 1e-8
        X_scaled = (X_input - sample_means) / sample_stds
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        # 3. 推理
        X_tensor = torch.tensor(X_scaled).to(self.device)
        with torch.no_grad():
            Y_pred_scaled = self.model(X_tensor).cpu().numpy() # (B, output_size)

        # 4. 反归一化 y_scaler (MinMaxScaler [0.1, 0.9] -> log space / partial space)
        Y_pred_real = self.y_scaler.inverse_transform(Y_pred_scaled)

        # 5. 反对数变换 (10^x)
        param_names = self.parent_config.get_trainable_param_names()
        log_transform_params = self.parent_config.get_log_transform_params()
        log_epsilon = self.parent_config.get_log_epsilon()

        for p in log_transform_params:
            if p in param_names:
                idx = param_names.index(p)
                Y_pred_real[:, idx] = np.power(10, Y_pred_real[:, idx]) - log_epsilon

        return Y_pred_real

    def get_param_names(self):
        return self.parent_config.get_trainable_param_names()
