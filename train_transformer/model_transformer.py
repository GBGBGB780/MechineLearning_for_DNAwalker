# coding=utf-8
"""
PatchTST-style Transformer model for the DNA Walker inverse problem.

Input curves are split into temporal patches per fluorescence channel, processed
with self-attention along time, then refined with cross-channel attention.
"""

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Split each channel into patches and project every patch to d_model."""

    def __init__(self, patch_size, stride, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.d_model = d_model
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=stride,
        )
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def num_patches(seq_len, patch_size, stride):
        if seq_len < patch_size:
            raise ValueError(
                f"seq_len ({seq_len}) must be >= patch_size ({patch_size})"
            )
        return (seq_len - patch_size) // stride + 1

    def forward(self, x):
        batch_size, channels, _ = x.shape
        x = x.reshape(batch_size * channels, 1, -1)
        x = self.proj(x).transpose(1, 2)
        x = self.norm(x)
        return x.reshape(batch_size, channels, x.shape[1], self.d_model)


class LearnablePositionalEncoding(nn.Module):
    """Learned positional embedding for temporal patch tokens."""

    def __init__(self, max_patches, d_model, dropout):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 1, max_patches, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        return self.dropout(x + self.pos_embedding[:, :, : x.shape[2], :])


class TransformerEncoderBlock(nn.Module):
    """Pre-LN self-attention block for temporal patch mixing."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
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

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        return x + self.ffn(self.norm2(x))


class CrossChannelAttentionBlock(nn.Module):
    """Apply attention across FAM/TYE/CY5 at each patch position."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
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

    def forward(self, x):
        batch_size, channels, patches, d_model = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size * patches, channels, d_model)
        residual = x
        x_norm = self.norm1(x)
        x = residual + self.attn(x_norm, x_norm, x_norm, need_weights=False)[0]
        x = x + self.ffn(self.norm2(x))
        return x.reshape(batch_size, patches, channels, d_model).permute(0, 2, 1, 3)


class DNAWalkerTransformer(nn.Module):
    """PatchTST + cross-channel MHA regressor for seven physical parameters."""

    def __init__(
        self,
        seq_len,
        num_channels,
        output_size,
        patch_size,
        stride,
        d_model,
        n_heads,
        n_layers,
        d_ff,
        cross_channel_layers,
        dropout,
        dropout_head,
    ):
        super().__init__()
        self.num_patches = PatchEmbedding.num_patches(seq_len, patch_size, stride)
        self.num_channels = num_channels
        self.d_model = d_model

        self.patch_embed = PatchEmbedding(patch_size, stride, d_model)
        self.pos_enc = LearnablePositionalEncoding(self.num_patches, d_model, dropout)
        self.temporal_blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.cross_channel_blocks = nn.ModuleList(
            [
                CrossChannelAttentionBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(cross_channel_layers)
            ]
        )

        head_hidden = max(d_model // 2, output_size)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout_head),
            nn.Linear(head_hidden, output_size),
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

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = self.pos_enc(x)

        _, channels, patches, d_model = x.shape
        x = x.reshape(batch_size * channels, patches, d_model)
        for block in self.temporal_blocks:
            x = block(x)
        x = x.reshape(batch_size, channels, patches, d_model)

        for block in self.cross_channel_blocks:
            x = block(x)

        x = x.mean(dim=(1, 2))
        return self.head(x)


def build_transformer_model(parent_config, transformer_config):
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


def build_transformer(parent_config, transformer_config):
    return build_transformer_model(parent_config, transformer_config)


if __name__ == "__main__":
    model = DNAWalkerTransformer(
        seq_len=7801,
        num_channels=3,
        output_size=7,
        patch_size=100,
        stride=100,
        d_model=256,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        cross_channel_layers=2,
        dropout=0.15,
        dropout_head=0.3,
    )
    x = torch.randn(2, 3, 7801)
    y = model(x)
    print(f"output: {tuple(y.shape)}")
    print(f"patches: {model.num_patches}")
