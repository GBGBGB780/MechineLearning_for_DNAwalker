# coding=utf-8
"""Default local transformer implementation now tracks the legacy MHA m03 baseline."""

try:
    from model_transformer_mha_legacy import (  # type: ignore
        DNAWalkerTransformerLegacyMHA,
        build_transformer,
    )
except ImportError:  # pragma: no cover - package-style import fallback
    from train_transformer.model_transformer_mha_legacy import (  # type: ignore
        DNAWalkerTransformerLegacyMHA,
        build_transformer,
    )

__all__ = ["DNAWalkerTransformerLegacyMHA", "build_transformer"]
