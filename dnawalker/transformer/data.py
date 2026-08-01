# coding=utf-8
"""
dnawalker.transformer.data — Transformer 专用数据加载模块
dnawalker.transformer.data — Transformer-specific data loading module

与 CNN 数据适配器的区别 / Difference from dnawalker.cnn.data:
  - X 保持 3D 形状 (N, 3, 7801)，不展平 / X stays 3D for Patch Embedding
  - 其余预处理一致 / All other preprocessing is identical
"""

import os
import pickle
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dnawalker.data.preprocessing import (
    normalize_per_sample,
    prepare_labels_and_sample_mask,
)
from dnawalker.data.splits import configured_explicit_split
from dnawalker.shared.parameters import load_npz_dataset


def load_and_preprocess_data_3d(npz_filename, batch_size, parent_config, transformer_config):
    """
    加载、预处理数据，返回 3D DataLoaders。
    Load and preprocess data, returning 3D DataLoaders (N, C, T).

    Args:
        npz_filename   (str):     数据集 .npz 路径
        batch_size     (int):     批次大小
        parent_config  (Config):  configs/common.ini 对应的 Config 对象
        transformer_config (TransformerConfig): Transformer 配置对象

    Returns:
        train_loader, val_loader, test_loader, param_names
    """
    print("--- 1. 加载和预处理数据 (Transformer 3D 模式) ---")

    # ---------- 读取参数名 ----------
    param_names = parent_config.get_trainable_param_names()
    print(f"从配置文件读取到 {len(param_names)} 个待训练参数: {param_names}")

    # ---------- 加载 .npz ----------
    try:
        # Metadata-aware shared loader keeps CNN and Transformer label columns
        # identical even when a config or NPZ stores parameters in a new order.
        X_data, Y_data, param_names = load_npz_dataset(
            npz_filename,
            param_names,
            allow_legacy_canonical=True,
            x_dtype=np.float32,
            y_dtype=np.float32,
            expected_x_shape=(
                parent_config.get_num_curves(),
                parent_config.get_seq_length(),
            ),
        )
        print(f"成功加载 {npz_filename}")
        print(f"原始 X 形状: {X_data.shape},  原始 Y 形状: {Y_data.shape}")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        # FileNotFoundError/OSError: 路径错；KeyError: 缺 X/Y；ValueError: npz 损坏
        print(f"错误：无法加载 {npz_filename}。\n错误信息: {e}")
        return None, None, None, None

    log_transform_params = parent_config.get_log_transform_params()
    for p in log_transform_params:
        print(f"  参数 '{p}' 已做 log10 变换")

    amplitude_thresholds = (
        parent_config.get_amplitude_thresholds()
        if parent_config.get_amplitude_filter_enabled()
        else None
    )
    Y_transformed, amplitude_mask, final_mask = (
        prepare_labels_and_sample_mask(
            X_data,
            Y_data,
            param_names,
            log_transform_params=log_transform_params,
            log_epsilon=parent_config.get_log_epsilon(),
            amplitude_thresholds=amplitude_thresholds,
            safe_threshold=parent_config.get_safe_threshold(),
            nan_replacement=parent_config.get_nan_replacement_value(),
        )
    )
    finite_y = Y_transformed[np.isfinite(Y_transformed)]
    if finite_y.size:
        print(
            f"Y 变换后范围: [{finite_y.min():.4f}, {finite_y.max():.4f}]"
        )

    num_samples = X_data.shape[0]
    num_amp_filtered = int(num_samples - amplitude_mask.sum())
    if amplitude_thresholds is not None:
        print(
            f"振幅过滤后剩余: {int(amplitude_mask.sum())} 个样本 "
            f"(移除 {num_amp_filtered})"
        )

    num_bad = int(amplitude_mask.sum() - final_mask.sum())
    if num_bad > 0:
        print(f"警告：发现并移除 {num_bad} 个含 Inf/NaN/极端值的坏样本")
    num_clean = int(final_mask.sum())
    print(f"清洗后样本数: {num_clean}")

    if num_clean == 0:
        print("错误：数据集中没有有效样本！")
        return None, None, None, None

    explicit_split = configured_explicit_split(
        parent_config,
        npz_filename,
        n_samples=num_samples,
        valid_mask=final_mask,
    )
    if explicit_split is not None:
        X_train = normalize_per_sample(
            X_data[explicit_split.train], copy=False
        )
        X_val = normalize_per_sample(
            X_data[explicit_split.val], copy=False
        )
        X_test = normalize_per_sample(
            X_data[explicit_split.test], copy=False
        )
        Y_train_raw = Y_transformed[explicit_split.train]
        Y_val_raw = Y_transformed[explicit_split.val]
        Y_test_raw = Y_transformed[explicit_split.test]
        print(
            "使用显式分层切分 / Explicit stratified split: "
            f"{explicit_split.manifest_path}"
        )
        del X_data, Y_data, Y_transformed, amplitude_mask, final_mask
    else:
        X_clean = X_data[final_mask]
        Y_clean = Y_transformed[final_mask]
        del X_data, Y_data, Y_transformed, amplitude_mask, final_mask

        # 共享逐样本归一化，保持 3D 形状。
        X_scaled = normalize_per_sample(X_clean, copy=False)
        test_ratio = parent_config.get_test_split_ratio()
        val_ratio = parent_config.get_val_split_ratio()
        split_seed = parent_config.get_split_seed()

        X_tv, X_test, Y_tv_raw, Y_test_raw = train_test_split(
            X_scaled, Y_clean, test_size=test_ratio, random_state=split_seed
        )
        del X_scaled, X_clean, Y_clean

        X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(
            X_tv, Y_tv_raw, test_size=val_ratio, random_state=split_seed
        )
        del X_tv, Y_tv_raw

    print(
        "X 单样本联合通道归一化完成 "
        "(共享 normalize_per_sample, 保持 3D 形状)"
    )
    gc.collect()

    # 仅在训练集上拟合 y_scaler，避免标签泄漏 / Fit y_scaler on train split only
    y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    Y_train = y_scaler.fit_transform(Y_train_raw).astype(np.float32)
    Y_val = y_scaler.transform(Y_val_raw).astype(np.float32)
    Y_test = y_scaler.transform(Y_test_raw).astype(np.float32)

    # 保存 y_scaler
    y_scaler_path = transformer_config.get_y_scaler_path()
    os.makedirs(os.path.dirname(y_scaler_path), exist_ok=True)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"y_scaler 已保存至: {y_scaler_path}")

    print(f"训练集: {X_train.shape[0]}  验证集: {X_val.shape[0]}  测试集: {X_test.shape[0]}")

    # ---------- 转为 Tensor → DataLoader ----------
    # pin_memory 仅在 CUDA 下有意义；MPS 不支持且会触发告警。
    use_pin = torch.cuda.is_available()

    def make_loader(X, Y, shuffle=False):
        ds = TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(Y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=use_pin)

    train_loader = make_loader(X_train, Y_train, shuffle=True)
    val_loader   = make_loader(X_val,   Y_val)
    test_loader  = make_loader(X_test,  Y_test)

    print("DataLoaders 创建完毕（X 形状: batch × 3 × seq_len）")
    print("----------------------------\n")
    return train_loader, val_loader, test_loader, param_names
