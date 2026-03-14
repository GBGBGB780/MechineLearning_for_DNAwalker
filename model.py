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
            nn.AdaptiveAvgPool1d(64) # AvgPool保留均值形状信息(64点)，比MaxPool更适合回归任务
        )
        
        # --- 回归预测层 (Regressor) ---
        # 256 → 256 → ReLU+Dropout → 128 → ReLU+Dropout → 7
        # Safe Sigmoid Lock: 输出限制在 (0,1)，同时目标被映射到 [0.1, 0.9]
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv4['out_channels'] * 64, conv4['out_channels']),  # 256*64→256 (对应AdaptiveAvgPool1d(64))
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(conv4['out_channels'], fc1_features),           # 256→128
            nn.ReLU(),
            nn.Dropout(dropout_fc),
            nn.Linear(fc1_features, output_size),                     # 128→7
            nn.Sigmoid()                                              # Safe Sigmoid Lock
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


class ForwardDecoder(nn.Module):
    """
    正向解码器 (Neural Surrogate Model / 数字孪生近似器)。
    学习从 7 个归一化物理参数 → 3 条荧光曲线 (3×seq_length) 的映射。
    本质上是 MATLAB 物理模拟器的可微分近似。
    
    架构: MLP 扩展 → 1D 转置卷积上采样
    - 输入: (Batch, 7)  [Sigmoid 归一化后的参数空间]
    - 输出: (Batch, 3*seq_length) [重构的荧光曲线，展平]
    """

    def __init__(self, input_size, config):
        """
        初始化 ForwardDecoder。
        
        Args:
            input_size: 参数维度 (通常为 7)
            config: Config 对象
        """
        super(ForwardDecoder, self).__init__()
        
        if config is None:
            raise ValueError("Config object is required for ForwardDecoder initialization.")
        
        self.num_curves = config.get_num_curves()    # 3
        self.seq_length = config.get_seq_length()     # 7801
        
        # 从配置读取 Decoder 结构参数
        # 使用与 Encoder 对称的通道数，但方向相反
        conv4 = config.get_conv4_params()  # Encoder 最深层: 256 通道
        conv3 = config.get_conv3_params()  # 128 通道
        conv2 = config.get_conv2_params()  # 64 通道
        conv1 = config.get_conv1_params()  # 32 通道
        
        dec_init_len = 64  # 对应 Encoder 的 AdaptiveAvgPool1d(64) 输出
        
        # --- MLP 扩展层: 7 → 256*64 ---
        self.expander = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, conv4['out_channels'] * dec_init_len),  # 7 → 256*64 = 16384
            nn.ReLU(),
        )
        
        # --- 转置卷积上采样层 (镜像 Encoder 的卷积+池化) ---
        # Encoder 路径: (3, 7801) → pool → conv1(32) → pool → conv2(64) → pool → conv3(128) → pool → conv4(256) → AdaptiveAvgPool(64)
        # Decoder 路径: (256, 64) → upsample → deconv4(128) → upsample → deconv3(64) → upsample → deconv2(32) → upsample → deconv1(3) → adjust
        self.decoder_convs = nn.Sequential(
            # 阶段 1: (256, 64) → (128, 128)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(conv4['out_channels'], conv3['out_channels'], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv3['out_channels']),
            nn.ReLU(),
            
            # 阶段 2: (128, 128) → (64, 256)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(conv3['out_channels'], conv2['out_channels'], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv2['out_channels']),
            nn.ReLU(),
            
            # 阶段 3: (64, 256) → (32, 512)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(conv2['out_channels'], conv1['out_channels'], kernel_size=5, padding=2),
            nn.BatchNorm1d(conv1['out_channels']),
            nn.ReLU(),
            
            # 阶段 4: (32, 512) → (16, 1024)
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.Conv1d(conv1['out_channels'], 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
        )
        
        # --- 最终调整层: 将通道数降为 3，并插值到精确的 seq_length ---
        self.final_conv = nn.Sequential(
            nn.Conv1d(16, self.num_curves, kernel_size=7, padding=3),
            # 不加激活函数，因为输出是归一化后的荧光值（可正可负）
        )

    def forward(self, params):
        """
        正向传播: 从参数重构荧光曲线。
        
        Args:
            params: (Batch, 7) 归一化后的参数
        
        Returns:
            x_recon: (Batch, 3*seq_length) 重构的荧光曲线（展平）
        """
        # 1. MLP 扩展: (B, 7) → (B, 256*64)
        x = self.expander(params)
        
        # 2. Reshape 为 3D: (B, 256*64) → (B, 256, 64)
        x = x.view(-1, 256, 64)
        
        # 3. 转置卷积上采样: (B, 256, 64) → (B, 16, 1024)
        x = self.decoder_convs(x)
        
        # 4. 最终卷积: (B, 16, 1024) → (B, 3, 1024)
        x = self.final_conv(x)
        
        # 5. 插值到精确的 seq_length: (B, 3, 1024) → (B, 3, 7801)
        x = nn.functional.interpolate(x, size=self.seq_length, mode='linear', align_corners=False)
        
        # 6. 展平: (B, 3, 7801) → (B, 23403)
        x = x.view(x.size(0), -1)
        
        return x
