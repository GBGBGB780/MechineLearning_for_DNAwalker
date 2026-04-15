# coding=utf-8
"""
config_loader_transformer.py — Transformer 专用配置加载器
config_loader_transformer.py — Transformer-specific configuration loader

设计 / Design:
  - TransformerConfig: 读取 config_transformer.ini 中的 Transformer 特定参数
                       Reads Transformer-specific params from config_transformer.ini
  - 通用参数复用上层 Config 类 / Shared params reuse parent Config class
"""

import sys
import os
import configparser

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import Config as ParentConfig  # noqa: E402


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
    """
    Transformer 专用配置加载器。/ Transformer-specific config loader.
    读取 config_transformer.ini 中的模型结构和训练超参数。
    Reads model architecture and training hyperparameters from config_transformer.ini.
    """

    def __init__(self, config_file=None, extra_config_files=None):
        if config_file is None:
            config_file = os.path.join(_THIS_DIR, 'config_transformer.ini')
        config_files = _normalize_config_files(config_file, extra_config_files)
        self._cfg = configparser.ConfigParser()
        self._cfg.read(config_files, encoding='utf-8')
        self.config_file = config_files[0]
        self.config_files = config_files
        self._config_dir = os.path.dirname(os.path.abspath(self.config_file))

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
        """注意力头数（TokenMixer 模式下保留兼容）/ Attention heads (kept for compatibility)."""
        return self._cfg.getint('TRANSFORMER', 'n_heads')

    def get_n_layers(self) -> int:
        """时间维度 TokenMixer 层数 / Temporal TokenMixer block count."""
        return self._cfg.getint('TRANSFORMER', 'n_layers')

    def get_d_ff(self) -> int:
        """FFN 中间层维度 / FFN hidden dimension."""
        return self._cfg.getint('TRANSFORMER', 'd_ff')

    def get_cross_channel_layers(self) -> int:
        """通道间 TokenMixer 层数 / Cross-channel TokenMixer block count."""
        return self._cfg.getint('TRANSFORMER', 'cross_channel_layers', fallback=2)

    def get_dropout(self) -> float:
        """TokenMixer + FFN Dropout / TokenMixer + FFN Dropout rate."""
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
        """获取相对当前工作目录的数据集路径 / Get cwd-relative dataset path."""
        rel = self._cfg.get('PATHS', 'dataset_file')
        return self._to_cwd_relative(self._resolve_from_config_dir(rel))

    def get_model_save_path(self) -> str:
        """获取相对当前工作目录的模型保存路径 / Get cwd-relative model save path."""
        rel = self._cfg.get('TRANSFORMER', 'model_save_path')
        path = self._to_cwd_relative(self._resolve_from_config_dir(rel))
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        return path

    def get_y_scaler_path(self) -> str:
        """获取相对当前工作目录的 y_scaler 路径 / Get cwd-relative y_scaler path."""
        rel = self._cfg.get('TRANSFORMER', 'y_scaler_file')
        path = self._to_cwd_relative(self._resolve_from_config_dir(rel))
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        return path

    # ===== AUTOENCODER 参数 / Autoencoder Parameters =====

    def get_decoder_num_epochs(self) -> int:
        """Decoder 训练轮数 / Decoder training epochs."""
        return self._cfg.getint('AUTOENCODER', 'decoder_num_epochs', fallback=500)

    def get_decoder_lr(self) -> float:
        """Decoder 学习率 / Decoder learning rate."""
        return self._cfg.getfloat('AUTOENCODER', 'decoder_lr', fallback=0.001)

    def get_decoder_patience(self) -> int:
        """Decoder Early Stopping 耐心值 / Decoder early stopping patience."""
        return self._cfg.getint('AUTOENCODER', 'decoder_patience', fallback=100)

    def get_alpha_recon(self) -> float:
        """重构 Loss 权重 α / Reconstruction loss weight α."""
        return self._cfg.getfloat('AUTOENCODER', 'alpha_recon', fallback=0.7)

    def get_beta_param(self) -> float:
        """参数 Loss 权重 β / Parameter loss weight β."""
        return self._cfg.getfloat('AUTOENCODER', 'beta_param', fallback=0.3)


def load_configs(parent_config_file=None, transformer_config_file=None,
                 parent_override_file=None, transformer_override_file=None):
    """
    加载完整配置，返回 (parent_config, transformer_config)。
    Load full configuration, returns (parent_config, transformer_config).

    parent_config:      上层 configfile.ini / parent Config object
    transformer_config: 本目录 config_transformer.ini / TransformerConfig object
    """
    if parent_config_file is None:
        parent_config_file = os.path.join(_PARENT_DIR, 'configfile.ini')
    if transformer_config_file is None:
        transformer_config_file = os.path.join(_THIS_DIR, 'config_transformer.ini')

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
