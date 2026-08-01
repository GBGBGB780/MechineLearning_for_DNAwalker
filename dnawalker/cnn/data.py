# coding=utf-8
"""
dnawalker.cnn.data — CNN 数据加载与预处理模块
dnawalker.cnn.data — CNN data loading and preprocessing module

处理流程 / Processing pipeline:
    加载 .npz → 对数变换 (Y) → 振幅过滤 → NaN/Inf 清洗 → 单样本联合通道归一化 (X)
    → MinMaxScaler [0.1,0.9] (Y) → 数据集拆分 → DataLoader
    Load .npz → log transform (Y) → amplitude filter → NaN/Inf cleanup → per-sample
    joint-channel normalization (X) → MinMaxScaler [0.1,0.9] (Y) → split → DataLoader
"""

from typing import List, Optional, Tuple

import numpy as np
import pickle
import torch
import os
import gc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dnawalker.data.preprocessing import (
    normalize_per_sample,
    prepare_labels_and_sample_mask,
)
from dnawalker.data.splits import configured_explicit_split
from dnawalker.shared.parameters import load_npz_dataset


def load_and_preprocess_data(
    npz_filename: str,
    batch_size: int = 64,
    config=None,
    *,
    y_scaler_file=None,
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader], Optional[List[str]]]:
    """
    加载、预处理、拆分数据，并创建 PyTorch DataLoaders。
    Load, preprocess, split data, and create PyTorch DataLoaders.

    Args:
        npz_filename: 数据集文件路径 / dataset file path
        batch_size:   批次大小 / batch size
        config:       Config 配置对象（必须提供）/ Config object (required)
        y_scaler_file: Optional resolved scaler destination. Trainers pass an
            absolute package-owned path so output is independent of caller cwd.

    Returns:
        (train_loader, val_loader, test_loader, param_names) 或失败时返回四个 None
        (train_loader, val_loader, test_loader, param_names) or four Nones on failure
    """
    print("--- 1. 加载和预处理数据 / Loading and preprocessing data ---")

    if config is None:
        raise ValueError("Config object is required for data loading.")

    # 读取参数名称 / Read parameter names
    param_names = config.get_trainable_param_names()
    print(f"待训练参数 / Trainable params ({len(param_names)}): {param_names}")

    # 加载 NPZ / Load NPZ
    try:
        X_data, Y_data, param_names = load_npz_dataset(
            npz_filename,
            param_names,
            # Explicit compatibility mode for pre-metadata datasets generated
            # by the historical MATLAB generator (canonical pysim order).
            allow_legacy_canonical=True,
            x_dtype=np.float32,
            y_dtype=np.float32,
            expected_x_shape=(config.get_num_curves(), config.get_seq_length()),
        )
        print(f"成功加载 / Loaded: {npz_filename} (float32)")
        print(f"原始 X: {X_data.shape}, Y: {Y_data.shape}")
    except (FileNotFoundError, KeyError, ValueError, OSError) as e:
        # FileNotFoundError/OSError: 路径错；KeyError: 缺 X/Y；ValueError: npz 损坏
        print(f"错误 / Error: 无法加载 / Cannot load {npz_filename}: {e}")
        return None, None, None, None

    log_transform_params = config.get_log_transform_params()
    for p in log_transform_params:
        print(f"  '{p}' → log10 变换 / transformed")

    amplitude_thresholds = (
        config.get_amplitude_thresholds()
        if config.get_amplitude_filter_enabled()
        else None
    )
    Y_transformed, amplitude_mask, final_mask = (
        prepare_labels_and_sample_mask(
            X_data,
            Y_data,
            param_names,
            log_transform_params=log_transform_params,
            log_epsilon=config.get_log_epsilon(),
            amplitude_thresholds=amplitude_thresholds,
            safe_threshold=config.get_safe_threshold(),
            nan_replacement=config.get_nan_replacement_value(),
        )
    )

    num_samples_original = X_data.shape[0]
    num_amp_filtered = int(num_samples_original - amplitude_mask.sum())
    if amplitude_thresholds is not None:
        print(
            "振幅过滤后 / After filter: "
            f"{int(amplitude_mask.sum())} 样本/samples "
            f"(移除/removed {num_amp_filtered})"
        )

    num_good = int(final_mask.sum())
    num_bad = int(amplitude_mask.sum()) - num_good
    if num_bad > 0:
        print(f"警告 / Warning: 移除 {num_bad} 个坏样本 / removed {num_bad} bad samples")

    if num_good == 0:
        print("错误 / Error: 无有效数据 / No valid data!")
        return None, None, None, None

    num_channels = config.get_num_curves()
    seq_len = config.get_seq_length()
    flat_width = num_channels * seq_len
    explicit_split = configured_explicit_split(
        config,
        npz_filename,
        n_samples=num_samples_original,
        valid_mask=final_mask,
    )

    if explicit_split is not None:
        def prepare_curves(indices):
            selected = X_data[indices]
            normalized = normalize_per_sample(selected, copy=False)
            return normalized.reshape(indices.size, flat_width)

        X_train = prepare_curves(explicit_split.train)
        X_val = prepare_curves(explicit_split.val)
        X_test = prepare_curves(explicit_split.test)
        Y_train_raw = Y_transformed[explicit_split.train]
        Y_val_raw = Y_transformed[explicit_split.val]
        Y_test_raw = Y_transformed[explicit_split.test]
        print(
            "使用显式分层切分 / Explicit stratified split: "
            f"{explicit_split.manifest_path}"
        )
        del X_data, Y_data, Y_transformed, amplitude_mask, final_mask
    else:
        X_clean = X_data[final_mask].reshape(num_good, flat_width)
        Y_clean = Y_transformed[final_mask]
        del X_data, Y_data, Y_transformed, amplitude_mask, final_mask

        # 单样本联合通道归一化 / Per-sample joint-channel normalization
        X_3d = normalize_per_sample(
            X_clean.reshape(-1, num_channels, seq_len),
            copy=False,
        )
        X_scaled = X_3d.reshape(X_clean.shape[0], -1)

        test_ratio = config.get_test_split_ratio()
        val_ratio = config.get_val_split_ratio()
        split_seed = config.get_split_seed()
        X_tv, X_test, Y_tv_raw, Y_test_raw = train_test_split(
            X_scaled, Y_clean, test_size=test_ratio, random_state=split_seed
        )
        del X_scaled, X_3d, X_clean, Y_clean

        X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(
            X_tv, Y_tv_raw, test_size=val_ratio, random_state=split_seed
        )
        del X_tv, Y_tv_raw

    print("X 归一化完成 / X normalization done (Domain Invariant)")
    gc.collect()

    # 仅在训练集上拟合 y_scaler，避免标签泄漏 / Fit y_scaler on train split only
    y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    Y_train = y_scaler.fit_transform(Y_train_raw)
    Y_val = y_scaler.transform(Y_val_raw)
    Y_test = y_scaler.transform(Y_test_raw)

    # 保存 y_scaler / Save y_scaler
    if y_scaler_file is None:
        y_scaler_file = config.get_y_scaler_file()
    y_scaler_file = os.fspath(y_scaler_file)
    scaler_dir = os.path.dirname(y_scaler_file)
    if scaler_dir and not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir, exist_ok=True)
    with open(y_scaler_file, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"Y Scaler 已保存 / saved: {y_scaler_file}")

    # 转 float32 / Convert to float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    Y_train = Y_train.astype(np.float32)
    Y_val = Y_val.astype(np.float32)
    Y_test = Y_test.astype(np.float32)
    print(f"训练/Train: {X_train.shape[0]}, 验证/Val: {X_val.shape[0]}, 测试/Test: {X_test.shape[0]}")

    del Y_train_raw, Y_val_raw, Y_test_raw
    gc.collect()

    # 创建 DataLoaders / Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train)),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val)),
                            batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test)),
                             batch_size=batch_size)

    print("DataLoaders 创建完毕 / DataLoaders created\n")
    return train_loader, val_loader, test_loader, param_names
