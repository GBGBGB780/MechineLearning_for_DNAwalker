import torch
import torch.nn as nn

class InverseMLP(nn.Module):
    """
    注意：虽然类名还叫 InverseMLP 以保持兼容性，
    但内部已经升级为强大的 1D-CNN (一维卷积神经网络)。
    """

    def __init__(self, input_size, output_size):
        super(InverseMLP, self).__init__()
        
        # 我们知道输入是 3条曲线 * 7801 个时间点
        self.num_curves = 3
        self.seq_length = 7801 
        
        # --- 卷积特征提取器 (Feature Extractor) ---
        self.features = nn.Sequential(
            # 第一层卷积: 识别基础形状 (斜率, 简单波形)
            # 修改: 卷积核加大到 21 (padding=10)，通道数翻倍到 32
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=21, stride=2, padding=10),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度减半
            
            # 第二层卷积: 识别复杂组合
            # 修改: 通道数翻倍 32 -> 64
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度再减半
            
            # 第三层卷积: 提取深层物理规律
            # 修改: 通道数翻倍 64 -> 128
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度再减半
            
            # 第四层卷积: 极致压缩
            # 修改: 通道数翻倍 128 -> 256
            nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 全局平均池化，强制压缩成 256 个核心特征
        )
        
        # --- 回归预测层 (Regressor) ---
        self.regressor = nn.Sequential(
            nn.Flatten(),           # 压平 (Batch, 256, 1) -> (Batch, 256)
            # 修改: 承接层从 128->64 改为 256->128
            nn.Linear(256, 128),
            nn.ReLU(),
            # 修改: 注释掉 Dropout 以追求更高精度
            # nn.Dropout(0.1),
            nn.Linear(128, output_size) # 输出 7 个参数
        )

    def forward(self, x):
        # x 的形状目前是 (Batch, Flattened_Size) (被压平了)
        # 我们必须把它还原回 (Batch, 3, seq_length) 才能给 CNN 处理
        x = x.view(-1, self.num_curves, self.seq_length)
        
        # 1. 提取曲线特征
        x = self.features(x)
        
        # 2. 预测物理参数
        x = self.regressor(x)
        
        return x