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
        Initialize the predictor by loading config, scalers, and the model.
        """
        print(f"--- Loading NanorobotPredictor Resources ---")
        
        # 1. Load Config
        try:
            self.config = Config(config_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load config from {config_file}: {e}")

        self.input_size = self.config.get_input_size()
        self.output_size = self.config.get_output_size()
        
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
        
        # Inverse transform: Scaled [0,1] -> Log10 space
        predicted_log = self.y_scaler.inverse_transform(predicted_scaled_np)
        
        # Anti-log: Log10 space -> Real Physical Space (10^x)
        predicted_real = np.power(10, predicted_log)
        
        return predicted_real

    def get_param_names(self):
        """Returns the list of trainable parameter names from config."""
        return self.config.get_trainable_param_names()
