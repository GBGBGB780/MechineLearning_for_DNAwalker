# coding=utf-8
import torch
import torch.nn as nn

class InverseCNN(nn.Module):
    """
    1D-CNN 用于从荧光曲线反推物理参数。
    特性：BatchNorm + Dropout 防过拟合，加深的全连接回归头。
    """

    def __init__(self, input_size, output_size, config):
        """
        初始化模型
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            config: Config对象 (必须提供)
        """
        super(InverseCNN, self).__init__()
        
        if config is None:
            raise ValueError("Config object is required for InverseCNN initialization.")
            
        # 从配置读取参数
        self.num_curves = config.get_num_curves()
        self.seq_length = config.get_seq_length()
        conv1 = config.get_conv1_params()
        conv2 = config.get_conv2_params()
        conv3 = config.get_conv3_params()
        conv4 = config.get_conv4_params()
        fc1_features = config.get_fc1_out_features()
        
        # 从配置读取 Dropout 率
        dropout_conv = config.get_dropout_conv()
        dropout_fc = config.get_dropout_fc()
        
        # --- 卷积特征提取器 (Feature Extractor) ---
        self.features = nn.Sequential(
            # 第一层卷积: 识别基础形状 (斜率, 简单波形)
            nn.Conv1d(in_channels=self.num_curves, out_channels=conv1['out_channels'], 
                     kernel_size=conv1['kernel_size'], stride=conv1['stride'], padding=conv1['padding']),
            nn.BatchNorm1d(conv1['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),
            
            # 第二层卷积: 识别复杂组合
            nn.Conv1d(conv1['out_channels'], conv2['out_channels'], 
                     kernel_size=conv2['kernel_size'], stride=conv2['stride'], padding=conv2['padding']),
            nn.BatchNorm1d(conv2['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),
            
            # 第三层卷积: 提取深层物理规律
            nn.Conv1d(conv2['out_channels'], conv3['out_channels'], 
                     kernel_size=conv3['kernel_size'], stride=conv3['stride'], padding=conv3['padding']),
            nn.BatchNorm1d(conv3['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_conv),
            
            # 第四层卷积: 极致压缩
            nn.Conv1d(conv3['out_channels'], conv4['out_channels'], 
                     kernel_size=conv4['kernel_size'], stride=conv4['stride'], padding=conv4['padding']),
            nn.BatchNorm1d(conv4['out_channels']),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 全局平均池化，强制压缩成核心特征
        )
        
        # --- 回归预测层 (Regressor) ---
        # 加深: 256 → 256 → ReLU+Dropout → 128 → ReLU+Dropout → 7
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv4['out_channels'], conv4['out_channels']),  # 256→256
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(conv4['out_channels'], fc1_features),           # 256→128
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc1_features, output_size)                      # 128→7
        )

    def forward(self, x):
        # x 的形状目前是 (Batch, Flattened_Size) (被压平了)
        # 我们必须把它还原回 (Batch, num_curves, seq_length) 才能给 CNN 处理
        x = x.view(-1, self.num_curves, self.seq_length)
        
        # 1. 提取曲线特征
        x = self.features(x)
        
        # 2. 预测物理参数
        x = self.regressor(x)
        
        return x
