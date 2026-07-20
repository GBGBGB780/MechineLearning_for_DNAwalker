# coding=utf-8
"""
Transformer-specific configuration loader.

Shared physics/data settings come from the root configfile.ini, while model and
optimizer settings come from train_transformer/config_transformer.ini.
"""

import configparser
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from config_loader import Config as ParentConfig  # noqa: E402


def _normalize_config_files(config_file, extra_config_files=None):
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
    """Read Transformer architecture, training, and path settings."""

    def __init__(self, config_file=None, extra_config_files=None):
        if config_file is None:
            config_file = os.path.join(_THIS_DIR, "config_transformer.ini")
        config_files = _normalize_config_files(config_file, extra_config_files)
        self._cfg = configparser.ConfigParser()
        self._cfg.read(config_files, encoding="utf-8")
        self.config_file = config_files[0]
        self.config_files = config_files
        self._config_dir = os.path.dirname(os.path.abspath(self.config_file))

    def _resolve_from_config_dir(self, path):
        if path.startswith("./"):
            path = path[2:]
        if not os.path.isabs(path):
            path = os.path.join(self._config_dir, path)
        return os.path.normpath(path)

    @staticmethod
    def _to_cwd_relative(path):
        return os.path.relpath(os.path.normpath(path), start=os.getcwd())

    def get_patch_size(self):
        return self._cfg.getint("TRANSFORMER", "patch_size")

    def get_stride(self):
        return self._cfg.getint("TRANSFORMER", "stride")

    def get_d_model(self):
        return self._cfg.getint("TRANSFORMER", "d_model")

    def get_n_heads(self):
        return self._cfg.getint("TRANSFORMER", "n_heads")

    def get_n_layers(self):
        return self._cfg.getint("TRANSFORMER", "n_layers")

    def get_d_ff(self):
        return self._cfg.getint("TRANSFORMER", "d_ff")

    def get_cross_channel_layers(self):
        return self._cfg.getint("TRANSFORMER", "cross_channel_layers", fallback=2)

    def get_dropout(self):
        return self._cfg.getfloat("TRANSFORMER", "dropout")

    def get_dropout_head(self):
        return self._cfg.getfloat("TRANSFORMER", "dropout_head", fallback=0.3)

    def get_learning_rate(self):
        return self._cfg.getfloat("TRANSFORMER", "learning_rate")

    def get_weight_decay(self):
        return self._cfg.getfloat("TRANSFORMER", "weight_decay")

    def get_warmup_ratio(self):
        return self._cfg.getfloat("TRANSFORMER", "warmup_ratio", fallback=0.1)

    def get_scheduler_type(self):
        return self._cfg.get("TRANSFORMER", "scheduler_type", fallback="cosine").strip().lower()

    def get_scheduler_min_lr(self):
        return self._cfg.getfloat("TRANSFORMER", "scheduler_min_lr", fallback=1e-6)

    def get_batch_size(self):
        return self._cfg.getint("TRANSFORMER", "batch_size")

    def get_num_epochs(self):
        return self._cfg.getint("TRANSFORMER", "num_epochs")

    def get_early_stopping_patience(self):
        return self._cfg.getint("TRANSFORMER", "early_stopping_patience", fallback=150)

    def get_dataset_path(self):
        rel = self._cfg.get("PATHS", "dataset_file")
        return self._to_cwd_relative(self._resolve_from_config_dir(rel))

    def get_model_save_path(self):
        rel = self._cfg.get("TRANSFORMER", "model_save_path")
        path = self._to_cwd_relative(self._resolve_from_config_dir(rel))
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        return path

    def get_y_scaler_path(self):
        rel = self._cfg.get("TRANSFORMER", "y_scaler_file")
        path = self._to_cwd_relative(self._resolve_from_config_dir(rel))
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        return path


def load_configs(
    parent_config_file=None,
    transformer_config_file=None,
    parent_override_file=None,
    transformer_override_file=None,
):
    if parent_config_file is None:
        parent_config_file = os.path.join(_PARENT_DIR, "configfile.ini")
    if transformer_config_file is None:
        transformer_config_file = os.path.join(_THIS_DIR, "config_transformer.ini")

    parent_config = ParentConfig(
        config_file=parent_config_file,
        extra_config_files=[parent_override_file] if parent_override_file else None,
    )
    transformer_config = TransformerConfig(
        config_file=transformer_config_file,
        extra_config_files=[transformer_override_file] if transformer_override_file else None,
    )
    return parent_config, transformer_config


if __name__ == "__main__":
    pc, tc = load_configs()
    print("=== Parent Config ===")
    print(f"  input_size:  {pc.get_input_size()}")
    print(f"  output_size: {pc.get_output_size()}")
    print("=== Transformer Config ===")
    print(f"  patch_size:  {tc.get_patch_size()}")
    print(f"  d_model:     {tc.get_d_model()}")
    print(f"  n_layers:    {tc.get_n_layers()}")
    print(f"  dataset:     {tc.get_dataset_path()}")
