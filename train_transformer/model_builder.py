# coding=utf-8
"""Build the default local transformer model.

The repo-local default has been switched from the TokenMixer branch to the
legacy MHA m03 baseline.
"""

try:
    from model_transformer_mha_legacy import build_transformer as build_legacy_mha_transformer
except ImportError:  # pragma: no cover - package-style import fallback
    from train_transformer.model_transformer_mha_legacy import build_transformer as build_legacy_mha_transformer


def build_transformer_model(parent_config, transformer_config):
    return build_legacy_mha_transformer(parent_config, transformer_config)
