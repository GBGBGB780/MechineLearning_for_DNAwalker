# coding=utf-8
"""
config_loader_transformer.py  —  Transformer 专用配置加载器

设计：
  - TransformerConfig  读取 train_transformer/config_transformer.ini
  - 复用上层目录的 Config 类 (config_loader.py) 读取 configfile.ini 中的通用参数
"""

import sys
import os
import configparser

# 将上层目录加入路径，以导入 config_loader.Config
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import Config as ParentConfig  # noqa: E402


class TransformerConfig:
    """读取 train_transformer/config_transformer.ini 的配置加载器"""

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = os.path.join(_THIS_DIR, 'config_transformer.ini')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        self._cfg = configparser.ConfigParser()
        self._cfg.read(config_file, encoding='utf-8')
        self._this_dir = _THIS_DIR

    # ===== TRANSFORMER 节 =====

    def get_patch_size(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'patch_size')

    def get_stride(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'stride')

    def get_d_model(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'd_model')

    def get_n_heads(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'n_heads')

    def get_n_layers(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'n_layers')

    def get_d_ff(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'd_ff')

    def get_cross_channel_layers(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'cross_channel_layers', fallback=2)

    def get_dropout(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'dropout')

    def get_dropout_head(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'dropout_head', fallback=0.3)

    def get_learning_rate(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'learning_rate')

    def get_weight_decay(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'weight_decay')

    def get_warmup_ratio(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'warmup_ratio', fallback=0.1)

    def get_scheduler_type(self) -> str:
        return self._cfg.get('TRANSFORMER', 'scheduler_type', fallback='cosine').strip().lower()

    def get_scheduler_min_lr(self) -> float:
        return self._cfg.getfloat('TRANSFORMER', 'scheduler_min_lr', fallback=1e-6)

    def get_batch_size(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'batch_size')

    def get_num_epochs(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'num_epochs')

    def get_early_stopping_patience(self) -> int:
        return self._cfg.getint('TRANSFORMER', 'early_stopping_patience', fallback=150)

    # ===== PATHS 节 =====

    def get_dataset_path(self) -> str:
        """获取数据集路径（相对 train_transformer/ 目录的路径，转为绝对路径）"""
        rel = self._cfg.get('PATHS', 'dataset_file')
        return os.path.normpath(os.path.join(self._this_dir, rel))

    def get_model_save_path(self) -> str:
        rel = self._cfg.get('TRANSFORMER', 'model_save_path')
        path = os.path.normpath(os.path.join(self._this_dir, rel))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path

    def get_y_scaler_path(self) -> str:
        rel = self._cfg.get('TRANSFORMER', 'y_scaler_file')
        path = os.path.normpath(os.path.join(self._this_dir, rel))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


def load_configs():
    """
    加载完整配置，返回 (parent_config, transformer_config)

    parent_config      : 上层 configfile.ini 的 Config 对象
    transformer_config : 本目录 config_transformer.ini 的 TransformerConfig 对象
    """
    parent_ini = os.path.join(_PARENT_DIR, 'configfile.ini')
    parent_config = ParentConfig(config_file=parent_ini)
    transformer_config = TransformerConfig()
    return parent_config, transformer_config


if __name__ == '__main__':
    pc, tc = load_configs()
    print("=== 父配置 ===")
    print(f"  input_size   : {pc.get_input_size()}")
    print(f"  output_size  : {pc.get_output_size()}")
    print(f"  param_names  : {pc.get_trainable_param_names()}")
    print("=== Transformer 配置 ===")
    print(f"  patch_size   : {tc.get_patch_size()}")
    print(f"  stride       : {tc.get_stride()}")
    print(f"  d_model      : {tc.get_d_model()}")
    print(f"  n_heads      : {tc.get_n_heads()}")
    print(f"  n_layers     : {tc.get_n_layers()}")
    print(f"  d_ff         : {tc.get_d_ff()}")
    print(f"  dataset_path : {tc.get_dataset_path()}")
    print(f"  model_save   : {tc.get_model_save_path()}")
