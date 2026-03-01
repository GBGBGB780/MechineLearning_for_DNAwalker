# coding=utf-8
import torch
import numpy as np
import pickle
import os
import sys


# Local imports
try:
    from model import InverseCNN
    from config_loader import Config
except ImportError as e:
    print(f"Error: Could not import necessary modules: {e}")
    sys.exit(1)

class NanorobotPredictor:
    """
    Encapsulates the logic for loading the model and running predictions.
    """
    
    def __init__(self, config_file='configfile.ini', model_path=None):
        """
        初始化预测器
        
        Args:
            config_file: 配置文件路径
            model_path: 模型路径（可选，如果None则从config读取）
        """
        print(f"--- Loading NanorobotPredictor Resources ---")
        
        # 1. Load Config
        try:
            self.config = Config(config_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_file}: {e}")

        self.input_size = self.config.get_input_size()
        self.output_size = self.config.get_output_size()
        self.param_ranges = self.config.get_param_ranges()
        
        # 2. Key Paths
        # Allow overriding model_path, otherwise use config
        self.model_path = model_path if model_path else self.config.get_model_save_path()
        self.x_scaler_path = self.config.get_x_scaler_file()
        self.y_scaler_path = self.config.get_y_scaler_file()
        
        # 3. Load Scalers
        # X 采用单样本联合通道归一化（运行时计算），不依赖全局 x_scaler
        self.y_scaler = self._load_pickle(self.y_scaler_path, "Y Scaler")
        
        # 4. Load Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = InverseCNN(self.input_size, self.output_size, self.config).to(self.device)
        self._load_model_weights()
        self.model.eval() # Set to evaluation mode
        
        print("--- Resources Loaded Successfully ---\n")

    def _load_pickle(self, path, name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} not found at {path}. Please run training first.")
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_model_weights(self):
        if not os.path.exists(self.model_path):
             raise FileNotFoundError(f"Model file not found at {self.model_path}. Please run training first.")
        
        try:
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        except Exception as e:
             raise RuntimeError(f"Failed to load model weights: {e}")

    def predict(self, X_data):
        """
        Run prediction on input data.
        
        Args:
            X_data: Numpy array. Can be:
                    - (3, T) single sample -> will be reshaped to (1, 3*T)
                    - (N, 3, T) batch -> will be reshaped to (N, 3*T)
                    - (N, Features) already flattened -> used as is
        
        Returns:
            predicted_real_params: Numpy array of shape (N, OutputSize) containing actual physical values.
        """
        # 1. Shape Handling & Flattening
        if X_data.ndim == 2 and X_data.shape[0] == 3: 
             # Assume single sample (3, T) -> reshape to (1, 3*T)
             X_flat = X_data.reshape(1, -1)
        elif X_data.ndim == 3:
             # Assume batch (N, 3, T) -> reshape to (N, 3*T)
             X_flat = X_data.reshape(X_data.shape[0], -1)
        else:
             # Assume already flattened or compatible (N, Features)
             X_flat = X_data
             
        # Verify flattened size
        expected_size = self.input_size # e.g. 23403
        if X_flat.shape[1] != expected_size:
            raise ValueError(f"Input data dimension mismatch. Expected {expected_size} features, got {X_flat.shape[1]}.")

        # 2. 单样本联合通道归一化 (Domain Invariance) — 与 utils.py 完全一致
        # 我们把 FAM, TYE, CY5 拼在一起计算这一个样本的总均值和总方差进行缩放。
        # 不使用全局均值，从而避免实验环境系统误差导致的整体验移对模型产生的误导。
        num_channels = self.config.get_num_curves()  # 3
        seq_len = self.input_size // num_channels     # 7801
        X_3d = X_flat.reshape(-1, num_channels, seq_len)
        
        # 沿着通道(axis=1)和时间点(axis=2)一起计算，保留各个曲线之间的绝对高度关系
        sample_means = np.nanmean(X_3d, axis=(1, 2), keepdims=True)  # Shape: (N, 1, 1)
        sample_stds  = np.nanstd(X_3d, axis=(1, 2), keepdims=True) + 1e-8 # Shape: (N, 1, 1)
        
        X_3d = (X_3d - sample_means) / sample_stds
        # 空白区域（NaN）填 0：z-score 空间的均值，不提供任何信息
        X_3d = np.where(np.isnan(X_3d), 0.0, X_3d)
        X_scaled = X_3d.reshape(X_flat.shape[0], -1)  # 展平回 (N, 23403)
        
        # 3. Convert to Tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 4. Model Inference
        with torch.no_grad():
            predicted_scaled = self.model(X_tensor)
            
        # 5. Inverse Scaling & Post-processing
        # Move to CPU and numpy
        predicted_scaled_np = predicted_scaled.cpu().numpy()
        
        # 5. 反归一化 (Scaler Inverse Transform) -> 得到物理参数值
        # 对于非 log 变换的参数，直接得到原始值（含符号）
        # 对于 log 变换的参数（如 k0），得到 log10 值，需要再做 10^x
        predicted_real = self.y_scaler.inverse_transform(predicted_scaled_np)
        
        # 6. 仅对 log 变换过的参数做逆变换 (10^x)
        log_transform_params = self.config.get_log_transform_params()
        log_epsilon = self.config.get_log_epsilon()
        param_names = self.config.get_trainable_param_names()
        
        for p in log_transform_params:
            if p in param_names:
                idx = param_names.index(p)
                predicted_real[:, idx] = np.power(10, predicted_real[:, idx]) - log_epsilon

        return predicted_real

    def get_param_names(self):
        """Returns the list of trainable parameter names from config."""
        return self.config.get_trainable_param_names()
