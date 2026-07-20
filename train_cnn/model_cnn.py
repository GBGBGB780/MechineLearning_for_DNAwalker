# coding=utf-8
"""
model_cnn.py — 1D-CNN 逆向模型
model_cnn.py — 1D-CNN inverse model

模型列表 / Models:
  - InverseCNN:     荧光曲线 → 物理参数 (逆向推理)
                    Fluorescence curves → physical parameters (inverse inference)
"""

import torch
import torch.nn as nn


class _MPSCompatAdaptiveAvgPool1d(nn.Module):
    """AdaptiveAvgPool1d 的 MPS 兼容封装。

    PyTorch 截至 2.x 在 MPS 上不支持 input_size 不能整除 output_size 的
    自适应池化（见 pytorch#96056）。这里只在输入张量位于 MPS 时把
    本层挪到 CPU 计算，其他设备路径与原始算子完全等价。
    """

    def __init__(self, output_size: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type == "mps":
            return self.pool(x.cpu()).to(x.device)
        return self.pool(x)

# 上层目录路径设置 / Parent directory path setup
import os
import sys
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)


class InverseCNN(nn.Module):
    """
    1D-CNN 逆向模型：从荧光曲线反推物理参数。
    1D-CNN inverse model: infers physical parameters from fluorescence curves.

    架构 / Architecture:
        Conv1d×4 (特征提取) → AdaptiveAvgPool → FC×3 (回归) → Sigmoid
    特性 / Features:
        BatchNorm + Dropout 防过拟合 / prevents overfitting
        Safe Sigmoid Lock 输出 (0,1) / output bounded to (0,1)
    """

    def __init__(self, input_size, output_size, config):
        """
        初始化模型。/ Initialize model.

        Args:
            input_size:  输入维度 (num_curves × seq_length) / input dimension
            output_size: 输出维度 (待预测参数数量) / output dimension (number of parameters)
            config:      Config 配置对象 / Config object
        """
        super(InverseCNN, self).__init__()

        if config is None:
            raise ValueError("Config object is required for InverseCNN initialization.")

        # 从配置读取参数 / Read parameters from config
        self.num_curves = config.get_num_curves()
        self.seq_length = config.get_seq_length()
        conv1 = config.get_conv1_params()
        conv2 = config.get_conv2_params()
        conv3 = config.get_conv3_params()
        conv4 = config.get_conv4_params()
        fc1_features = config.get_fc1_out_features()
        dropout_conv = config.get_dropout_conv()
        dropout_fc = config.get_dropout_fc()

        # 卷积特征提取器 / Convolutional feature extractor
        self.features = nn.Sequential(
            # 第一层：识别基础波形 / Layer 1: detect basic waveforms
            nn.Conv1d(self.num_curves, conv1['out_channels'],
                      kernel_size=conv1['kernel_size'], stride=conv1['stride'], padding=conv1['padding']),
            nn.BatchNorm1d(conv1['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),

            # 第二层：识别复杂组合 / Layer 2: detect complex patterns
            nn.Conv1d(conv1['out_channels'], conv2['out_channels'],
                      kernel_size=conv2['kernel_size'], stride=conv2['stride'], padding=conv2['padding']),
            nn.BatchNorm1d(conv2['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),

            # 第三层：提取深层物理特征 / Layer 3: extract deep physical features
            nn.Conv1d(conv2['out_channels'], conv3['out_channels'],
                      kernel_size=conv3['kernel_size'], stride=conv3['stride'], padding=conv3['padding']),
            nn.BatchNorm1d(conv3['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),

            # 第四层：极致压缩 / Layer 4: aggressive compression
            nn.Conv1d(conv3['out_channels'], conv4['out_channels'],
                      kernel_size=conv4['kernel_size'], stride=conv4['stride'], padding=conv4['padding']),
            nn.BatchNorm1d(conv4['out_channels']),
            nn.ReLU(),
            _MPSCompatAdaptiveAvgPool1d(64),  # 保留均值信息，适合回归 / retains mean info, suits regression
        )

        # 回归预测层 / Regression head
        # Safe Sigmoid Lock: 输出限制在 (0,1)，目标映射到 [0.1, 0.9]
        # Safe Sigmoid Lock: output bounded to (0,1), targets mapped to [0.1, 0.9]
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv4['out_channels'] * 64, conv4['out_channels']),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(conv4['out_channels'], fc1_features),
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc1_features, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        前向传播 / Forward pass.

        Args:
            x: (B, num_curves*seq_length) 展平的输入 / flattened input
        Returns:
            (B, output_size) 预测参数 / predicted parameters
        """
        # 还原为 3D 形状 / Reshape to 3D for Conv1d
        x = x.view(-1, self.num_curves, self.seq_length)
        x = self.features(x)
        x = self.regressor(x)
        return x
