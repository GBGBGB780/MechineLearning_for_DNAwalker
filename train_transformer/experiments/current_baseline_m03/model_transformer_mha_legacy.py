import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
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


class DNAWalkerTransformerLegacyMHA(nn.Module):
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


def build_transformer(parent_config, transformer_config):
    return DNAWalkerTransformerLegacyMHA(
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
