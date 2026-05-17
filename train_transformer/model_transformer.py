# coding=utf-8
"""
model_transformer.py — PatchTST + MHA Transformer 逆向模型定义
model_transformer.py — PatchTST + MHA Transformer inverse model definition

架构 / Architecture:
    PatchEmbedding → LearnablePositionalEncoding
    → TemporalTransformerBlock × n_layers (时间维度自注意力 / temporal self-attention)
    → CrossChannelAttentionBlock × cross_channel_layers (跨通道注意力 / cross-channel attention)
    → GlobalMeanPool → RegressionHead (Sigmoid)

输入 / Input:  (B, 3, 7801) — 三通道荧光曲线 / 3-channel fluorescence curves
输出 / Output: (B, 7)       — 归一化物理参数 / normalized physical parameters
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    将每条通道的序列切分为固定大小的 Patch，并映射到 d_model 维度。
    Splits each channel's sequence into fixed-size patches and projects to d_model.
    """

    def __init__(self, patch_size: int, stride: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def num_patches(seq_len: int, patch_size: int, stride: int) -> int:
        return (seq_len - patch_size) // stride + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, seq_len = x.shape
        x = x.reshape(batch_size * num_channels, 1, seq_len)
        x = self.proj(x).permute(0, 2, 1)
        x = self.norm(x)
        num_patches = x.shape[1]
        return x.reshape(batch_size, num_channels, num_patches, self.d_model)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码：为每个 Patch 添加位置信息。
    Learnable positional encoding: adds position information to each patch.
    """

    def __init__(self, max_patches: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_patches, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_ids", torch.arange(max_patches))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        num_patches = x.shape[-2]
        pe = self.pe(self.pos_ids[:num_patches])
        return self.dropout(x + pe)


class TemporalTransformerBlock(nn.Module):
    """
    时间维度 Transformer 块：在同一通道内做自注意力，捕捉曲线时序依赖。
    Temporal Transformer block: self-attention within a single channel to capture temporal dependencies.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual

        residual = x
        x = self.ffn(self.norm2(x))
        return x + residual


class CrossChannelAttentionBlock(nn.Module):
    """
    跨通道注意力块：在同一时间位置上对不同通道（FAM, TYE, CY5）做注意力交互。
    Cross-channel attention block: attention across channels (FAM, TYE, CY5) at the same temporal position.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_patches, d_model = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * num_patches, num_channels, d_model)

        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + residual

        residual = x
        x = self.ffn(self.norm2(x))
        x = x + residual

        return x.reshape(batch_size, num_patches, num_channels, d_model).permute(0, 2, 1, 3)


class DNAWalkerTransformer(nn.Module):
    """
    DNA Walker Transformer 逆向模型：从 3 通道荧光曲线预测 7 个物理参数。
    DNA Walker Transformer inverse model: predicts 7 physical parameters from 3-channel fluorescence curves.

    架构 / Architecture:
        PatchEmbedding → Positional Encoding
        → Temporal Self-Attention × n_layers (单通道内时序建模)
        → Cross-Channel Attention × cross_channel_layers (FAM/TYE/CY5 跨通道交互)
        → Global Mean Pool → MLP Head (Sigmoid)
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
        num_patches = PatchEmbedding.num_patches(seq_len, patch_size, stride)
        self.num_patches = num_patches
        self.patch_embed = PatchEmbedding(patch_size, stride, d_model)
        self.pos_enc = LearnablePositionalEncoding(num_patches, d_model, dropout)
        self.temporal_blocks = nn.ModuleList(
            [TemporalTransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.cross_channel_blocks = nn.ModuleList(
            [CrossChannelAttentionBlock(d_model, n_heads, d_ff, dropout) for _ in range(cross_channel_layers)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(d_model // 2, output_size),
            nn.Sigmoid(),
        )
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, _ = x.shape
        x = self.patch_embed(x)
        _, _, num_patches, d_model = x.shape
        x = x.reshape(batch_size * num_channels, num_patches, d_model)
        x = self.pos_enc(x)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(batch_size, num_channels, num_patches, d_model)
        for block in self.cross_channel_blocks:
            x = block(x)
        x = x.mean(dim=2).mean(dim=1)
        return self.head(x)


def build_transformer_model(parent_config, transformer_config):
    """
    根据配置构建 Transformer 模型。/ Build a Transformer model from configs.

    Args:
        parent_config:      根目录 Config 对象 / root Config object
        transformer_config: TransformerConfig 对象 / TransformerConfig object
    Returns:
        DNAWalkerTransformer 实例 / DNAWalkerTransformer instance
    """
    return DNAWalkerTransformer(
        seq_len=parent_config.get_seq_length(),
        num_channels=parent_config.get_num_curves(),
        output_size=parent_config.get_output_size(),
        patch_size=transformer_config.get_patch_size(),
        stride=transformer_config.get_stride(),
        d_model=transformer_config.get_d_model(),
        n_heads=transformer_config.get_n_heads(),
        n_layers=transformer_config.get_n_layers(),
        d_ff=transformer_config.get_d_ff(),
        cross_channel_layers=transformer_config.get_cross_channel_layers(),
        dropout=transformer_config.get_dropout(),
        dropout_head=transformer_config.get_dropout_head(),
    )
