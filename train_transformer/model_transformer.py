# coding=utf-8
"""
model_transformer.py  —  DNA Walker Transformer 模型

架构：PatchTST-Inspired + Cross-Channel Attention
  输入: (B, C, L)  C=3 通道 (FAM/TYE/CY5), L=7801 时间点
  ↓ Patch Embedding → (B, C, P, d_model)
  ↓ Temporal Self-Attention (per channel) → (B, C, P, d_model)
  ↓ Cross-Channel Attention (per patch) → (B, C, P, d_model)
  ↓ Global Average Pool → (B, d_model)
  ↓ Regression Head → (B, output_size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# 1. Patch Embedding
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    将每条时间序列分割成重叠 Patch，并通过线性投影映射到 d_model 维。

    实现：使用 Conv1d(in=1, out=d_model, kernel=patch_size, stride=stride)
    等效于 "切 Patch + 线性投影"，并行处理所有通道。

    输入:  (B, C, L)
    输出:  (B, C, num_patches, d_model)
    """

    def __init__(self, patch_size: int, stride: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        # per-channel linear projection（共享权重）
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride
        )
        # 投影后做 LayerNorm
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def num_patches(seq_len: int, patch_size: int, stride: int) -> int:
        return (seq_len - patch_size) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, L)
        """
        B, C, L = x.shape
        # 将通道维度合并到 batch，逐通道做 Conv1d
        x = x.reshape(B * C, 1, L)          # (B*C, 1, L)
        x = self.proj(x)                     # (B*C, d_model, num_patches)
        x = x.permute(0, 2, 1)              # (B*C, num_patches, d_model)
        x = self.norm(x)
        P = x.shape[1]
        x = x.reshape(B, C, P, self.d_model) # (B, C, P, d_model)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 2. 可学习位置编码
# ─────────────────────────────────────────────────────────────────────────────

class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码，为每个 Patch 位置添加唯一的可学习偏置向量。
    比固定正弦编码更灵活，适合本任务的非标准序列长度。
    """

    def __init__(self, max_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_patches, d_model)
        self.dropout = nn.Dropout(dropout)
        # 预计算位置索引（0, 1, ..., max_patches-1）
        self.register_buffer('pos_ids', torch.arange(max_patches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, P, d_model) 或 (B*C, P, d_model)
        """
        P = x.shape[-2]
        pe = self.pe(self.pos_ids[:P])  # (P, d_model)
        return self.dropout(x + pe)


# ─────────────────────────────────────────────────────────────────────────────
# 3. 时间维度 Transformer Block（每个通道独立）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTransformerBlock(nn.Module):
    """
    单层 Transformer Block（Pre-LayerNorm 变体，训练更稳定）。
    作用于时间（Patch）维度，每个通道独立计算自注意力。

    输入/输出: (B*C, P, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须整除 n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.norm1 = nn.LayerNorm(d_model)
        # QKV 合并投影（一次矩阵乘法，效率更高）
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN Self-Attention（Flash Attention via F.scaled_dot_product_attention）
        residual = x
        x = self.norm1(x)
        BT, S, D = x.shape
        # QKV 分拆 → (BT, n_heads, S, head_dim)
        qkv = self.qkv_proj(x).reshape(BT, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)          # 各 (BT, S, n_heads, head_dim)
        q = q.transpose(1, 2)                 # (BT, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Flash Attention（训练时启用 dropout，推理时为 0）
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(BT, S, D)  # (BT, S, D)
        x = self.out_proj(attn_out) + residual
        # Pre-LN FFN
        residual = x
        x = self.ffn(self.norm2(x))
        return x + residual


# ─────────────────────────────────────────────────────────────────────────────
# 4. 通道间注意力（Cross-Channel Attention）
# ─────────────────────────────────────────────────────────────────────────────

class CrossChannelAttentionBlock(nn.Module):
    """
    在每个 Patch 位置，对三条曲线（通道维度）做 Multi-Head Attention。
    让模型显式学习 FAM ↔ TYE ↔ CY5 之间的相互关系。

    输入/输出:  (B, C, P, d_model)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须整除 n_heads"
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.dropout  = dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, P, D = x.shape
        # Rearrange: (B*P, C, D) — 对每个 Patch position 做通道间注意力
        x = x.permute(0, 2, 1, 3).reshape(B * P, C, D)

        residual = x
        x = self.norm1(x)
        BP, S, _ = x.shape   # S = C = 3
        qkv = self.qkv_proj(x).reshape(BP, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)   # (BP, n_heads, S, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        dropout_p = self.dropout if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        attn_out = attn_out.transpose(1, 2).reshape(BP, S, D)
        x = self.out_proj(attn_out) + residual

        residual = x
        x = self.ffn(self.norm2(x))
        x = x + residual

        # Restore: (B, C, P, D)
        x = x.reshape(B, P, C, D).permute(0, 2, 1, 3)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. 主模型 DNAWalkerTransformer
# ─────────────────────────────────────────────────────────────────────────────

class DNAWalkerTransformer(nn.Module):
    """
    PatchTST-Inspired Transformer，用于 DNA Walker 反问题：
    三条荧光曲线 (FAM, TYE, CY5)  →  7 个物理参数

    整体流程：
      Patch Embedding → Positional Encoding
      → N × TemporalTransformerBlock (per channel)
      → M × CrossChannelAttentionBlock
      → Global Average Pool (时间 + 通道)
      → Regression Head → Sigmoid (Safe Sigmoid Lock)
    """

    def __init__(
        self,
        seq_len: int,
        num_channels: int,
        output_size: int,
        patch_size: int,
        stride: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        cross_channel_layers: int,
        dropout: float,
        dropout_head: float,
    ):
        super().__init__()

        # 计算 Patch 数量
        num_patches = PatchEmbedding.num_patches(seq_len, patch_size, stride)
        self.num_patches  = num_patches
        self.num_channels = num_channels
        self.d_model      = d_model

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(patch_size, stride, d_model)

        # --- 可学习位置编码 ---
        self.pos_enc = LearnablePositionalEncoding(num_patches, d_model, dropout)

        # --- 时间维度 Transformer（通道独立）---
        self.temporal_blocks = nn.ModuleList([
            TemporalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # --- 通道间注意力 ---
        self.cross_channel_blocks = nn.ModuleList([
            CrossChannelAttentionBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(cross_channel_layers)
        ])

        # --- 回归头 ---
        # 输入维度 = d_model（全局均值池化后）
        head_hidden = d_model // 2
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, output_size),
            nn.Sigmoid()   # Safe Sigmoid Lock: 输出 (0,1)，配合 y 缩放到 [0.1,0.9]
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """Xavier 均匀初始化 Linear 层，使训练初期梯度更稳定"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, L)  — 已归一化的三通道时间序列
        Returns:
            out: (B, output_size)  — 预测的物理参数（已过 Sigmoid）
        """
        B, C, L = x.shape

        # 1. Patch Embedding → (B, C, P, d_model)
        x = self.patch_embed(x)

        # 2. 可学习位置编码（广播到每个通道）
        B, C, P, D = x.shape
        # 展平通道到 batch 维，适配 LearnablePositionalEncoding
        x = x.reshape(B * C, P, D)
        x = self.pos_enc(x)
        x = x.reshape(B, C, P, D)

        # 3. 通道独立时间自注意力
        x = x.reshape(B * C, P, D)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(B, C, P, D)

        # 4. 通道间注意力
        for block in self.cross_channel_blocks:
            x = block(x)

        # 5. 全局均值池化（Patch 维 + Channel 维）
        # (B, C, P, D) → mean over P → (B, C, D) → mean over C → (B, D)
        x = x.mean(dim=2)   # (B, C, D)
        x = x.mean(dim=1)   # (B, D)

        # 6. 回归头
        return self.head(x)


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_transformer(parent_config, transformer_config) -> DNAWalkerTransformer:
    """
    根据配置对象构建模型实例，供 train_transformer.py 调用。
    """
    seq_len      = parent_config.get_seq_length()
    num_channels = parent_config.get_num_curves()
    output_size  = parent_config.get_output_size()

    model = DNAWalkerTransformer(
        seq_len              = seq_len,
        num_channels         = num_channels,
        output_size          = output_size,
        patch_size           = transformer_config.get_patch_size(),
        stride               = transformer_config.get_stride(),
        d_model              = transformer_config.get_d_model(),
        n_heads              = transformer_config.get_n_heads(),
        n_layers             = transformer_config.get_n_layers(),
        d_ff                 = transformer_config.get_d_ff(),
        cross_channel_layers = transformer_config.get_cross_channel_layers(),
        dropout              = transformer_config.get_dropout(),
        dropout_head         = transformer_config.get_dropout_head(),
    )
    return model


if __name__ == '__main__':
    # 快速验证：随机输入 → 检查输出形状
    B, C, L = 4, 3, 7801
    x = torch.randn(B, C, L)
    model = DNAWalkerTransformer(
        seq_len=L, num_channels=C, output_size=7,
        patch_size=50, stride=25,
        d_model=256, n_heads=8, n_layers=4, d_ff=512,
        cross_channel_layers=2, dropout=0.15, dropout_head=0.3
    )
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}  (期望: [{B}, 7])")
    print(f"模型参数量: {n_params:,}")
    assert out.shape == (B, 7), "输出形状错误！"
    assert out.min() >= 0 and out.max() <= 1, "Sigmoid 输出范围错误！"
    print("✅ 模型架构验证通过")
