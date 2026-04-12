# coding=utf-8
"""
model_transformer.py  —  DNA Walker Transformer 模型 (TokenMixer 版)

架构：PatchTST-Inspired + TokenMixer (替换 MHA)
  输入: (B, C, L)  C=3 通道 (FAM/TYE/CY5), L=7801 时间点
  ↓ Patch Embedding → (B, C, P, d_model)
  ↓ Temporal TokenMixer Block (per channel) → (B, C, P, d_model)
  ↓ Cross-Channel TokenMixer Block (per patch) → (B, C, P, d_model)
  ↓ Global Average Pool → (B, d_model)
  ↓ Regression Head → (B, output_size)

TokenMixer 核心思想：
  仅通过维度重排 + 转置实现 token 间信息混合，复杂度 O(B*T*D)。
  要求 num_heads == num_tokens (H=T)，每个新 token 都混入了其他 token 的信息。
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
# 3. TokenMixer — 无参数 token 信息混合
# ─────────────────────────────────────────────────────────────────────────────

class TokenMixer(nn.Module):
    """
    通过维度重排 + 转置实现 token 间信息混合，复杂度 O(B*T*D)。
    要求 num_heads == num_tokens (H=T)，
    每个新 token = 自己的块 + 其他所有 token 的块（重排后拼接）。

    无可学习参数，仅做 reshape + permute。
    """

    def __init__(self, num_tokens: int, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        self.num_heads = num_tokens  # H = T
        assert d_model % num_tokens == 0, \
            f"d_model({d_model}) 必须能整除 num_tokens({num_tokens})"
        self.d_head = d_model // num_tokens
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D)
        """
        B, T, D = x.shape
        # [B, T, D] → [B, T, H, Dh]  把每个 token 切分为 H 个小块
        x = x.reshape(B, T, self.num_heads, self.d_head)
        # [B, T, H, Dh] → [B, H, T, Dh]  把所有 token 的第 i 块放一起
        x = x.permute(0, 2, 1, 3)
        # [B, H, T, Dh] → [B, T, D]  每个新 token = 来自不同原始 token 的块
        x = x.reshape(B, T, D)
        x = self.dropout(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. 时间维度 TokenMixer Block（每个通道独立）
# ─────────────────────────────────────────────────────────────────────────────

class TemporalTokenMixerBlock(nn.Module):
    """
    单层 TokenMixer Block（Pre-LayerNorm 变体）。
    用 TokenMixer 替代 MHA，保留 FFN 作为可学习部分。
    作用于时间（Patch）维度，每个通道独立计算。

    输入/输出: (B*C, P, d_model)
    """

    def __init__(self, num_tokens: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixer = TokenMixer(num_tokens, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-LN TokenMixer (替代 Self-Attention)
        residual = x
        x = self.token_mixer(self.norm1(x)) + residual
        # Pre-LN FFN
        residual = x
        x = self.ffn(self.norm2(x)) + residual
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. 通道间 TokenMixer Block（Cross-Channel TokenMixer）
# ─────────────────────────────────────────────────────────────────────────────

class CrossChannelTokenMixerBlock(nn.Module):
    """
    在每个 Patch 位置，对三条曲线（通道维度）做 TokenMixer 混合。
    让模型显式学习 FAM ↔ TYE ↔ CY5 之间的相互关系。
    保留 FFN 作为可学习部分。

    输入/输出:  (B, C, P, d_model)
    """

    def __init__(self, num_channels: int, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.token_mixer = TokenMixer(num_channels, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, P, D = x.shape
        # Rearrange: (B*P, C, D) — 对每个 Patch position 做通道间 TokenMixer
        x = x.permute(0, 2, 1, 3).reshape(B * P, C, D)

        # Pre-LN TokenMixer (替代 Cross-Channel Attention)
        residual = x
        x = self.token_mixer(self.norm1(x)) + residual
        # Pre-LN FFN
        residual = x
        x = self.ffn(self.norm2(x)) + residual

        # Restore: (B, C, P, D)
        x = x.reshape(B, P, C, D).permute(0, 2, 1, 3)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 6. 主模型 DNAWalkerTransformer
# ─────────────────────────────────────────────────────────────────────────────

class DNAWalkerTransformer(nn.Module):
    """
    PatchTST-Inspired Transformer (TokenMixer 版)，用于 DNA Walker 反问题：
    三条荧光曲线 (FAM, TYE, CY5)  →  7 个物理参数

    整体流程：
      Patch Embedding → Positional Encoding
      → N × TemporalTokenMixerBlock (per channel)
      → M × CrossChannelTokenMixerBlock
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
        n_heads: int,           # 保留参数接口兼容，但 TokenMixer 不使用
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

        # 维度约束检查 (TokenMixer 要求 d_model 能被 num_tokens 整除)
        assert d_model % num_patches == 0, \
            f"d_model({d_model}) 必须能整除 num_patches({num_patches})，" \
            f"当前 patch_size={patch_size}, stride={stride}"
        assert d_model % num_channels == 0, \
            f"d_model({d_model}) 必须能整除 num_channels({num_channels})"

        # --- Patch Embedding ---
        self.patch_embed = PatchEmbedding(patch_size, stride, d_model)

        # --- 可学习位置编码 ---
        self.pos_enc = LearnablePositionalEncoding(num_patches, d_model, dropout)

        # --- 时间维度 TokenMixer（通道独立）---
        self.temporal_blocks = nn.ModuleList([
            TemporalTokenMixerBlock(num_patches, d_model, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # --- 通道间 TokenMixer ---
        self.cross_channel_blocks = nn.ModuleList([
            CrossChannelTokenMixerBlock(num_channels, d_model, d_ff, dropout)
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

        # 3. 通道独立时间 TokenMixer
        x = x.reshape(B * C, P, D)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(B, C, P, D)

        # 4. 通道间 TokenMixer
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
    D_MODEL = 576
    # 参数需满足: d_model % num_patches == 0 且 d_model % C == 0
    # patch_size=161, stride=80 → num_patches=96, d_model=576 → 576/96=6, 576/3=192
    model = DNAWalkerTransformer(
        seq_len=L, num_channels=C, output_size=7,
        patch_size=161, stride=80,
        d_model=D_MODEL, n_heads=8, n_layers=6, d_ff=1152,
        cross_channel_layers=3, dropout=0.15, dropout_head=0.3
    )
    x = torch.randn(B, C, L)
    out = model(x)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {out.shape}  (期望: [{B}, 7])")
    print(f"Patch 数量: {model.num_patches}")
    print(f"Temporal d_head: {D_MODEL // model.num_patches}")
    print(f"Cross-Channel d_head: {D_MODEL // C}")
    print(f"模型参数量: {n_params:,}")
    assert out.shape == (B, 7), "输出形状错误！"
    assert out.min() >= 0 and out.max() <= 1, "Sigmoid 输出范围错误！"
    print("✅ 模型架构验证通过 (TokenMixer Scaled-Up 版)")
