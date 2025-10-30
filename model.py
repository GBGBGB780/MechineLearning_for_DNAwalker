import torch
import torch.nn as nn


class InverseMLP(nn.Module):
    """
    一个简单的MLP(多层感知机)来学习 曲线 -> 参数 的逆向映射
    """

    def __init__(self, input_size, output_size):
        super(InverseMLP, self).__init__()

        # 定义网络结构
        self.model = nn.Sequential(
            # 输入层: 300 (3*100) 个特征
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # 添加 Dropout 防止过拟合

            # 隐藏层 1
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),

            # 隐藏层 2
            nn.Linear(512, 256),
            nn.ReLU(),

            # 输出层: 7 个物理参数
            nn.Linear(256, output_size),

            # 我们预测的是 [0, 1] 范围内的归一化值，Sigmoid 是一个好选择
            # 但在实践中，直接线性输出（下面这行）通常更稳健
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)