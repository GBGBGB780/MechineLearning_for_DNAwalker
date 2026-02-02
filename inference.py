# coding=utf-8
import torch
import numpy as np
import pickle
import os
import sys

# user_wants_negative_drt_s_hack = False # Placeholder
user_wants_negative_drt_s_hack = False

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
        self.x_scaler = self._load_pickle(self.x_scaler_path, "X Scaler")
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

        # 2. Scaling (Normalization)
        X_scaled = self.x_scaler.transform(X_flat)
        
        # 3. Convert to Tensor
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # 4. Model Inference
        with torch.no_grad():
            predicted_scaled = self.model(X_tensor)
            
        # 5. Inverse Scaling & Post-processing
        # Move to CPU and numpy
        predicted_scaled_np = predicted_scaled.cpu().numpy()
        
        # 5. 反归一化 (Scaler Inverse Transform) -> 得到 log10(|Y|)
        predicted_log_abs = self.y_scaler.inverse_transform(predicted_scaled_np)
        
        # 6. 转回实数域 (10^x)，得到绝对值 |Y|
        predicted_abs = np.power(10, predicted_log_abs)
        
        # 7. 恢复符号 (Sign Restoration)
        # 模型训练时使用了 abs()，丢失了符号信息。
        # 我们根据 config 中的参数范围 [min, max] 来恢复符号。
        # 规则: 
        #   - 如果 max <= 0, 则参数必为负 -> multiply by -1
        #   - 如果 min >= 0, 则参数必为正 -> keep positive
        #   - 如果 min < 0 < max, 这里存在二义性。
        #     目前的策略是: 如果 max <= 0 (严格负) 或者 参数名以 'E_' 开头且 max < 0.6 (能量通常为负), 则视为负。
        #     注: 更稳健的方法是仅依赖 ranges。但 E_b_azo_trans 范围是 [-2, 0.5]。
        #     根据用户反馈，E_b 等能量项应为负。
        
        predicted_final = np.zeros_like(predicted_abs)
        param_names = self.config.get_trainable_param_names()
        
        # 针对每个样本
        for i in range(predicted_abs.shape[0]):
            for j, name in enumerate(param_names):
                val_abs = predicted_abs[i, j]
                min_val, max_val = self.param_ranges.get(name, (-1e9, 1e9))
                
                # 判定符号
                if max_val <= 0:
                    # 范围全负，一定是负数
                    predicted_final[i, j] = -val_abs
                elif min_val >= 0:
                     # 范围全正，一定是正数
                    predicted_final[i, j] = val_abs
                else:
                    # 范围跨越 0 (Mixed range)
                    # 启发式规则:
                    # 1. 能量项 (E_...) 且范围主要在负半轴 -> 视为负
                    # 2. 其他情况默认正，或者看中点?
                    # 用户指出的 Reference: E_b_azo_trans (-1.35) 是负的。范围 [-2.0, 0.5].
                    # 注意: config keys 通常是小写，所以使用 lower() checks
                    if name.lower().startswith('e_'):
                        predicted_final[i, j] = -val_abs
                    elif name == 'drt_s' and max_val <= 1.0 and user_wants_negative_drt_s_hack:
                        # 特殊情况? 不，config中 drt_s 是 [0, 1]. 用户提供的 reference -0.05 可能是错的或者噪音。
                        # 我们严格遵守 Config 范围。如果不符合 range，我们在 verify 中可能会看到偏移。
                        # 这里我们只处理 ambiguous case.
                        # 对于 drt_s [0, 1]，它是 min >= 0 case，上面已经处理为正。
                        predicted_final[i, j] = val_abs
                    else:
                        predicted_final[i, j] = val_abs

        return predicted_final

    def get_param_names(self):
        """Returns the list of trainable parameter names from config."""
        return self.config.get_trainable_param_names()
