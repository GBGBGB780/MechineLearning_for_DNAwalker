# coding=utf-8
"""Transformer configuration layered over ``configs/common.ini``."""

import configparser
import math
import os

from dnawalker.paths import (
    ARTIFACTS_DIR,
    DEFAULT_CONFIG,
    DEFAULT_TRANSFORMER_CONFIG,
)

from dnawalker.config import Config as ParentConfig


def _normalize_config_files(config_file, extra_config_files=None):
    """Return an ordered list of config files with existence checks."""
    config_files = [config_file]
    if extra_config_files:
        if isinstance(extra_config_files, (str, os.PathLike)):
            config_files.append(extra_config_files)
        else:
            config_files.extend(extra_config_files)

    normalized = []
    for path in config_files:
        if path is None:
            continue
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"配置文件不存在 / Config not found: {path}")
        normalized.append(path)
    return normalized


class TransformerConfig:
    """Load and validate Transformer-specific settings."""

    def __init__(self, config_file=None, extra_config_files=None, validate=True):
        if config_file is None:
            config_file = os.fspath(DEFAULT_TRANSFORMER_CONFIG)
        config_files = _normalize_config_files(config_file, extra_config_files)
        self._cfg = configparser.ConfigParser()
        self._cfg.read(config_files, encoding='utf-8')
        self.config_file = config_files[0]
        self.config_files = config_files
        self._config_dir = os.path.dirname(os.path.abspath(self.config_file))
        if validate:
            self._validate()

    def _validate(self):
        """Reject architecture/training values that would fail or misbehave."""
        required = (
            "patch_size", "stride", "d_model", "n_heads", "n_layers", "d_ff",
            "dropout", "learning_rate", "weight_decay", "batch_size",
            "num_epochs", "model_save_path", "y_scaler_file",
        )
        missing = [
            f"[TRANSFORMER].{name}"
            for name in required
            if not self._cfg.has_option("TRANSFORMER", name)
        ]
        if not self._cfg.has_option("PATHS", "dataset_file"):
            missing.append("[PATHS].dataset_file")
        if missing:
            raise ValueError(
                f"Transformer 配置缺少必需项 / missing required keys: {missing}"
            )

        positive_ints = {
            "patch_size": self.get_patch_size(),
            "stride": self.get_stride(),
            "d_model": self.get_d_model(),
            "n_heads": self.get_n_heads(),
            "n_layers": self.get_n_layers(),
            "d_ff": self.get_d_ff(),
            "batch_size": self.get_batch_size(),
            "num_epochs": self.get_num_epochs(),
        }
        nonnegative_ints = {
            "cross_channel_layers": self.get_cross_channel_layers(),
            "early_stopping_patience": self.get_early_stopping_patience(),
        }
        bad_positive = {k: v for k, v in positive_ints.items() if v <= 0}
        bad_nonnegative = {k: v for k, v in nonnegative_ints.items() if v < 0}
        if bad_positive or bad_nonnegative:
            raise ValueError(
                "Transformer integer hyperparameters out of range: "
                f"{bad_positive or bad_nonnegative}"
            )
        if self.get_d_model() < 2:
            raise ValueError(
                "d_model must be >= 2 so the regression head has a non-zero "
                f"hidden width, got {self.get_d_model()}"
            )
        if self.get_d_model() % self.get_n_heads() != 0:
            raise ValueError(
                "d_model 必须能被 n_heads 整除 / must be divisible: "
                f"{self.get_d_model()} % {self.get_n_heads()} != 0"
            )

        for name, value in (
            ("dropout", self.get_dropout()),
            ("dropout_head", self.get_dropout_head()),
        ):
            if not math.isfinite(value) or not (0.0 <= value < 1.0):
                raise ValueError(f"{name} must be in [0, 1), got {value}")

        learning_rate = self.get_learning_rate()
        weight_decay = self.get_weight_decay()
        warmup_ratio = self.get_warmup_ratio()
        min_lr = self.get_scheduler_min_lr()
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be finite and > 0, got {learning_rate}"
            )
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(
                f"weight_decay must be finite and >= 0, got {weight_decay}"
            )
        if not math.isfinite(warmup_ratio) or not (0.0 <= warmup_ratio < 1.0):
            raise ValueError(
                f"warmup_ratio must be in [0, 1), got {warmup_ratio}"
            )
        if not math.isfinite(min_lr) or not (0.0 <= min_lr <= learning_rate):
            raise ValueError(
                "scheduler_min_lr must be between 0 and learning_rate: "
                f"min_lr={min_lr}, learning_rate={learning_rate}"
            )
        if self.get_scheduler_type() != "cosine":
            raise ValueError(
                "Only scheduler_type=cosine is implemented, got "
                f"{self.get_scheduler_type()!r}"
            )

    def validate_sequence_length(self, seq_len):
        """Validate patch geometry against the parent sequence length."""
        if self.get_patch_size() > seq_len:
            raise ValueError(
                "patch_size cannot exceed the input sequence length: "
                f"patch_size={self.get_patch_size()}, seq_len={seq_len}"
            )

    def _resolve_from_config_dir(self, path: str) -> str:
        """先按配置文件所在目录解析，再标准化。/ Resolve a path against the config dir."""
        if path.startswith('./'):
            path = path[2:]
        if not os.path.isabs(path):
            path = os.path.join(self._config_dir, path)
        return os.path.normpath(path)

    @staticmethod
    def _to_cwd_relative(path: str) -> str:
        """将路径转换为相对当前工作目录。/ Convert a path to be relative to cwd."""
        return os.path.relpath(os.path.normpath(path), start=os.getcwd())

    # ===== TRANSFORMER 模型参数 / Model Parameters =====

    def get_patch_size(self) -> int:
        """Patch 大小 / Patch size."""
        return self._cfg.getint('TRANSFORMER', 'patch_size')

    def get_stride(self) -> int:
        """Patch 滑动步长 / Patch sliding stride."""
        return self._cfg.getint('TRANSFORMER', 'stride')

    def get_d_model(self) -> int:
        """嵌入维度 / Embedding dimension."""
        return self._cfg.getint('TRANSFORMER', 'd_model')

    def get_n_heads(self) -> int:
        """多头注意力头数 / Number of attention heads."""
        return self._cfg.getint('TRANSFORMER', 'n_heads')

    def get_n_layers(self) -> int:
        """时间维度 Transformer 层数 / Temporal Transformer block count."""
        return self._cfg.getint('TRANSFORMER', 'n_layers')

    def get_d_ff(self) -> int:
        """FFN 中间层维度 / FFN hidden dimension."""
        return self._cfg.getint('TRANSFORMER', 'd_ff')

    def get_cross_channel_layers(self) -> int:
        """跨通道注意力层数 / Cross-channel attention block count."""
        return self._cfg.getint('TRANSFORMER', 'cross_channel_layers', fallback=2)

    def get_dropout(self) -> float:
        """Attention + FFN Dropout 率 / Attention + FFN Dropout rate."""
        return self._cfg.getfloat('TRANSFORMER', 'dropout')

    def get_dropout_head(self) -> float:
        """回归头 Dropout / Regression head Dropout rate."""
        return self._cfg.getfloat('TRANSFORMER', 'dropout_head', fallback=0.3)

    # ===== 优化器参数 / Optimizer Parameters =====

    def get_learning_rate(self) -> float:
        """AdamW 学习率 / AdamW learning rate."""
        return self._cfg.getfloat('TRANSFORMER', 'learning_rate')

    def get_weight_decay(self) -> float:
        """权重衰减 / Weight decay (L2 regularization)."""
        return self._cfg.getfloat('TRANSFORMER', 'weight_decay')

    def get_warmup_ratio(self) -> float:
        """Warmup 比例 / Warmup ratio (fraction of total epochs)."""
        return self._cfg.getfloat('TRANSFORMER', 'warmup_ratio', fallback=0.1)

    def get_scheduler_type(self) -> str:
        """调度器类型 / Scheduler type (cosine)."""
        return self._cfg.get('TRANSFORMER', 'scheduler_type', fallback='cosine').strip().lower()

    def get_scheduler_min_lr(self) -> float:
        """最小学习率 / Minimum learning rate."""
        return self._cfg.getfloat('TRANSFORMER', 'scheduler_min_lr', fallback=1e-6)

    # ===== 训练超参数 / Training Hyperparameters =====

    def get_batch_size(self) -> int:
        """批次大小 / Batch size."""
        return self._cfg.getint('TRANSFORMER', 'batch_size')

    def get_num_epochs(self) -> int:
        """训练轮数 / Number of training epochs."""
        return self._cfg.getint('TRANSFORMER', 'num_epochs')

    def get_early_stopping_patience(self) -> int:
        """Early Stopping 耐心值 / Early stopping patience."""
        return self._cfg.getint('TRANSFORMER', 'early_stopping_patience', fallback=150)

    # ===== 文件路径 / File Paths =====

    def get_dataset_path(self) -> str:
        """Get the configured dataset path, independent of caller cwd."""
        rel = self._cfg.get('PATHS', 'dataset_file')
        return self._resolve_from_config_dir(rel)

    def get_artifact_dir(self) -> str:
        """Get the Transformer checkpoint/scaler directory."""
        rel = self._cfg.get(
            'PATHS', 'model_artifacts_dir',
            fallback=os.fspath(ARTIFACTS_DIR / 'models' / 'transformer'),
        )
        return self._resolve_from_config_dir(rel)

    def _resolve_artifact(self, path: str) -> str:
        """Resolve an artifact name and redirect legacy ``results/`` values."""
        path = os.fspath(path)
        if os.path.isabs(path):
            return os.path.normpath(path)
        normalized = os.path.normpath(path)
        parts = normalized.split(os.sep)
        if parts and parts[0] == 'results':
            normalized = os.path.join(*parts[1:]) if len(parts) > 1 else ''
        return os.path.normpath(os.path.join(self.get_artifact_dir(), normalized))

    def get_model_save_path(self) -> str:
        """Get the absolute Transformer checkpoint path without side effects."""
        return self._resolve_artifact(
            self._cfg.get('TRANSFORMER', 'model_save_path')
        )

    def get_y_scaler_path(self) -> str:
        """Get the absolute Transformer Y-scaler path without side effects."""
        return self._resolve_artifact(
            self._cfg.get('TRANSFORMER', 'y_scaler_file')
        )



def load_configs(parent_config_file=None, transformer_config_file=None,
                 parent_override_file=None, transformer_override_file=None):
    """Load the shared and Transformer-specific configuration layers."""
    parent_config_file = parent_config_file or os.fspath(DEFAULT_CONFIG)
    transformer_config_file = (
        transformer_config_file or os.fspath(DEFAULT_TRANSFORMER_CONFIG)
    )

    parent_extras = [parent_override_file] if parent_override_file else None
    transformer_extras = [transformer_override_file] if transformer_override_file else None

    parent_config = ParentConfig(
        config_file=parent_config_file,
        extra_config_files=parent_extras
    )
    transformer_config = TransformerConfig(
        config_file=transformer_config_file,
        extra_config_files=transformer_extras
    )
    transformer_config.validate_sequence_length(parent_config.get_seq_length())
    return parent_config, transformer_config


if __name__ == '__main__':
    pc, tc = load_configs()
    print("=== Parent Config / 父配置 ===")
    print(f"  input_size:  {pc.get_input_size()}")
    print(f"  output_size: {pc.get_output_size()}")
    print(f"  params:      {pc.get_trainable_param_names()}")
    print("=== Transformer Config ===")
    print(f"  patch_size:  {tc.get_patch_size()}")
    print(f"  d_model:     {tc.get_d_model()}")
    print(f"  n_layers:    {tc.get_n_layers()}")
    print(f"  dataset:     {tc.get_dataset_path()}")
    print(f"  model_save:  {tc.get_model_save_path()}")
