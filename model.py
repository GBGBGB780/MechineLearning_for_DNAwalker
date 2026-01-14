import torch
import torch.nn as nn

class InverseMLP(nn.Module):
    """
    注意：虽然类名还叫 InverseMLP 以保持兼容性，
    但内部已经升级为强大的 1D-CNN (一维卷积神经网络)。
    """

    def __init__(self, input_size, output_size, config=None):
        """
        初始化模型
        
        Args:
            input_size: 输入维度
            output_size: 输出维度
            config: Config对象，如果为None则使用硬编码值（向后兼容）
        """
        super(InverseMLP, self).__init__()
        
        # 从配置读取或使用默认值
        if config is not None:
            self.num_curves = config.get_num_curves()
            self.seq_length = config.get_seq_length()
            conv1 = config.get_conv1_params()
            conv2 = config.get_conv2_params()
            conv3 = config.get_conv3_params()
            conv4 = config.get_conv4_params()
            fc1_features = config.get_fc1_out_features()
        else:
            # 默认硬编码值（向后兼容）
            self.num_curves = 3
            self.seq_length = 7801
            conv1 = {'out_channels': 32, 'kernel_size': 21, 'stride': 2, 'padding': 10}
            conv2 = {'out_channels': 64, 'kernel_size': 5, 'stride': 2, 'padding': 2}
            conv3 = {'out_channels': 128, 'kernel_size': 5, 'stride': 2, 'padding': 2}
            conv4 = {'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1}
            fc1_features = 128
        
        # --- 卷积特征提取器 (Feature Extractor) ---
        self.features = nn.Sequential(
            # 第一层卷积: 识别基础形状 (斜率, 简单波形)
            nn.Conv1d(in_channels=self.num_curves, out_channels=conv1['out_channels'], 
                     kernel_size=conv1['kernel_size'], stride=conv1['stride'], padding=conv1['padding']),
            nn.BatchNorm1d(conv1['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度减半
            
            # 第二层卷积: 识别复杂组合
            nn.Conv1d(conv1['out_channels'], conv2['out_channels'], 
                     kernel_size=conv2['kernel_size'], stride=conv2['stride'], padding=conv2['padding']),
            nn.BatchNorm1d(conv2['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度再减半
            
            # 第三层卷积: 提取深层物理规律
            nn.Conv1d(conv2['out_channels'], conv3['out_channels'], 
                     kernel_size=conv3['kernel_size'], stride=conv3['stride'], padding=conv3['padding']),
            nn.BatchNorm1d(conv3['out_channels']),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), # 长度再减半
            
            # 第四层卷积: 极致压缩
            nn.Conv1d(conv3['out_channels'], conv4['out_channels'], 
                     kernel_size=conv4['kernel_size'], stride=conv4['stride'], padding=conv4['padding']),
            nn.BatchNorm1d(conv4['out_channels']),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # 全局平均池化，强制压缩成核心特征
        )
        
        # --- 回归预测层 (Regressor) ---
        self.regressor = nn.Sequential(
            nn.Flatten(),           # 压平 (Batch, conv4_channels, 1) -> (Batch, conv4_channels)
            nn.Linear(conv4['out_channels'], fc1_features),
            nn.ReLU(),
            nn.Linear(fc1_features, output_size) # 输出物理参数
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