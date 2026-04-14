# coding=utf-8
"""
data_loader.py — CNN 数据加载与预处理模块
data_loader.py — CNN data loading and preprocessing module

处理流程 / Processing pipeline:
    加载 .npz → 对数变换 (Y) → 振幅过滤 → NaN/Inf 清洗 → 单样本联合通道归一化 (X)
    → MinMaxScaler [0.1,0.9] (Y) → 数据集拆分 → DataLoader
    Load .npz → log transform (Y) → amplitude filter → NaN/Inf cleanup → per-sample
    joint-channel normalization (X) → MinMaxScaler [0.1,0.9] (Y) → split → DataLoader
"""

import numpy as np
import pickle
import torch
import os
import gc
import sys
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 上层目录路径设置 / Parent directory path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)


def load_and_preprocess_data(npz_filename, batch_size=64, config=None):
    """
    加载、预处理、拆分数据，并创建 PyTorch DataLoaders。
    Load, preprocess, split data, and create PyTorch DataLoaders.

    Args:
        npz_filename: 数据集文件路径 / dataset file path
        batch_size:   批次大小 / batch size
        config:       Config 配置对象（必须提供）/ Config object (required)

    Returns:
        (train_loader, val_loader, test_loader, param_names) 或失败时返回四个 None
        (train_loader, val_loader, test_loader, param_names) or four Nones on failure
    """
    print(f"--- 1. 加载和预处理数据 / Loading and preprocessing data ---")

    if config is None:
        raise ValueError("Config object is required for data loading.")

    # 读取参数名称 / Read parameter names
    param_names = config.get_trainable_param_names()
    print(f"待训练参数 / Trainable params ({len(param_names)}): {param_names}")

    # 加载 NPZ / Load NPZ
    try:
        with np.load(npz_filename) as dataset:
            X_data = dataset['X'].astype(np.float32)
            Y_data = dataset['Y'].astype(np.float32)
        print(f"成功加载 / Loaded: {npz_filename} (float32)")
        print(f"原始 X: {X_data.shape}, Y: {Y_data.shape}")
    except Exception as e:
        print(f"错误 / Error: 无法加载 / Cannot load {npz_filename}: {e}")
        return None, None, None, None

    # 选择性对数变换 (Y) / Selective log transform (Y)
    log_epsilon = config.get_log_epsilon()
    log_transform_params = config.get_log_transform_params()
    for p in log_transform_params:
        if p in param_names:
            idx = param_names.index(p)
            Y_data[:, idx] = np.log10(Y_data[:, idx] + log_epsilon)
            print(f"  '{p}' (列/col {idx}) → log10 变换 / transformed")

    # 振幅过滤 / Amplitude filtering
    num_samples_original = X_data.shape[0]
    if config.get_amplitude_filter_enabled():
        amp_thresholds = config.get_amplitude_thresholds()
        channel_names = ['FAM', 'TYE', 'CY5']
        amp_mask = np.ones(num_samples_original, dtype=bool)
        for c in range(min(X_data.shape[1], len(amp_thresholds))):
            amplitudes = X_data[:, c, :].max(axis=1) - X_data[:, c, :].min(axis=1)
            ch_mask = amplitudes <= amp_thresholds[c]
            num_filtered = np.sum(~ch_mask & amp_mask)
            amp_mask &= ch_mask
            print(f"  振幅过滤 / Amplitude filter {channel_names[c]}: "
                  f"阈值/threshold={amp_thresholds[c]:.2f}, 过滤/filtered {num_filtered}")
        X_data, Y_data = X_data[amp_mask], Y_data[amp_mask]
        print(f"振幅过滤后 / After filter: {X_data.shape[0]} 样本/samples")
        num_samples_original = X_data.shape[0]

    # 展平 X / Flatten X
    X_flat = X_data.reshape(num_samples_original, -1)

    # NaN/Inf 清洗 / NaN/Inf cleanup
    safe_threshold = config.get_safe_threshold()
    nan_replacement = config.get_nan_replacement_value()
    np.nan_to_num(X_flat, nan=nan_replacement, posinf=nan_replacement, neginf=nan_replacement, copy=False)
    np.nan_to_num(Y_data, nan=nan_replacement, posinf=nan_replacement, neginf=nan_replacement, copy=False)

    limit = safe_threshold
    mask_x = (np.max(X_flat, axis=1) < limit) & (np.min(X_flat, axis=1) > -limit)
    mask_y = (np.max(Y_data, axis=1) < limit) & (np.min(Y_data, axis=1) > -limit)
    final_mask = mask_x & mask_y
    num_good = np.sum(final_mask)
    num_bad = num_samples_original - num_good
    if num_bad > 0:
        print(f"警告 / Warning: 移除 {num_bad} 个坏样本 / removed {num_bad} bad samples")

    X_clean = X_flat[final_mask].copy()
    Y_clean = Y_data[final_mask].copy()
    del X_data, X_flat, Y_data, mask_x, mask_y, final_mask
    gc.collect()

    if num_good == 0:
        print("错误 / Error: 无有效数据 / No valid data!")
        return None, None, None, None

    # 单样本联合通道归一化 / Per-sample joint-channel normalization (Domain Invariant)
    num_channels = config.get_num_curves()
    seq_len = config.get_seq_length()
    X_3d = X_clean.reshape(-1, num_channels, seq_len)
    sample_means = np.nanmean(X_3d, axis=(1, 2), keepdims=True)
    sample_stds = np.nanstd(X_3d, axis=(1, 2), keepdims=True) + 1e-8
    X_3d = (X_3d - sample_means) / sample_stds
    X_scaled = X_3d.reshape(X_clean.shape[0], -1)
    np.nan_to_num(X_scaled, nan=0.0, copy=False)
    print("X 归一化完成 / X normalization done (Domain Invariant)")

    # 拆分数据集 / Split dataset
    test_ratio = config.get_test_split_ratio()
    val_ratio = config.get_val_split_ratio()
    seed = config.get_random_seed()
    X_tv, X_test, Y_tv_raw, Y_test_raw = train_test_split(
        X_scaled, Y_clean, test_size=test_ratio, random_state=seed
    )
    X_train, X_val, Y_train_raw, Y_val_raw = train_test_split(
        X_tv, Y_tv_raw, test_size=val_ratio, random_state=seed
    )

    # 仅在训练集上拟合 y_scaler，避免标签泄漏 / Fit y_scaler on train split only
    y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    Y_train = y_scaler.fit_transform(Y_train_raw)
    Y_val = y_scaler.transform(Y_val_raw)
    Y_test = y_scaler.transform(Y_test_raw)

    # 保存 y_scaler / Save y_scaler
    y_scaler_file = config.get_y_scaler_file()
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

    del X_scaled, X_tv, Y_clean, Y_tv_raw, Y_train_raw, Y_val_raw, Y_test_raw
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
