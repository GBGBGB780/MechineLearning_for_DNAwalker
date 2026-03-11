# coding=utf-8
"""
dataset.py  —  DNA Walker Transformer 数据加载模块

与 CNN 的 utils.py 主要区别:
  - X 保持 3D 形状 (N, 3, 7801)，不展平，直接喂给 Transformer 的 Patch Embedding
  - 其余预处理（振幅过滤、单样本联合通道归一化、y_scaler）完全复用同一套逻辑
"""

import sys
import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 将上层目录加入 Python 模块搜索路径，以复用 config_loader.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)


def load_and_preprocess_data_3d(npz_filename, batch_size, parent_config, transformer_config):
    """
    加载、预处理数据，返回保持 3D 形状 (N, 3, seq_len) 的 DataLoaders。

    Args:
        npz_filename   (str):     数据集 .npz 路径
        batch_size     (int):     批次大小
        parent_config  (Config):  上层 configfile.ini 对应的 Config 对象
        transformer_config (TransformerConfig): 本文件夹的配置对象

    Returns:
        train_loader, val_loader, test_loader, param_names
    """
    print("--- 1. 加载和预处理数据 (Transformer 3D 模式) ---")

    # ---------- 读取参数名 ----------
    param_names = parent_config.get_trainable_param_names()
    print(f"从配置文件读取到 {len(param_names)} 个待训练参数: {param_names}")

    # ---------- 加载 .npz ----------
    try:
        dataset = np.load(npz_filename)
        X_data = dataset['X']   # (N, 3, 7801) 或 (N, num_curves, seq_len)
        Y_data = dataset['Y']   # (N, 7)
        print(f"成功加载 {npz_filename}")
        print(f"原始 X 形状: {X_data.shape},  原始 Y 形状: {Y_data.shape}")
    except Exception as e:
        print(f"错误：无法加载 {npz_filename}。\n错误信息: {e}")
        return None, None, None, None

    # ---------- 对数变换 (Y) ----------
    log_epsilon = parent_config.get_log_epsilon()
    log_transform_params = parent_config.get_log_transform_params()
    for p in log_transform_params:
        if p in param_names:
            idx = param_names.index(p)
            Y_data[:, idx] = np.log10(Y_data[:, idx] + log_epsilon)
            print(f"  参数 '{p}' (列 {idx}) 已做 log10 变换")
    print(f"Y 变换后范围: [{Y_data.min():.4f}, {Y_data.max():.4f}]")

    # ---------- 振幅过滤 (X) ----------
    num_samples = X_data.shape[0]
    if parent_config.get_amplitude_filter_enabled():
        amp_thresholds = parent_config.get_amplitude_thresholds()
        channel_names = ['FAM', 'TYE', 'CY5']
        amp_mask = np.ones(num_samples, dtype=bool)
        for c in range(min(X_data.shape[1], len(amp_thresholds))):
            amplitudes = X_data[:, c, :].max(axis=1) - X_data[:, c, :].min(axis=1)
            ch_mask = amplitudes <= amp_thresholds[c]
            num_filtered = np.sum(~ch_mask & amp_mask)
            amp_mask &= ch_mask
            print(f"  振幅过滤 {channel_names[c]}: 阈值={amp_thresholds[c]:.4f}, "
                  f"本通道过滤 {num_filtered} 个样本")
        num_amp_filtered = num_samples - np.sum(amp_mask)
        X_data = X_data[amp_mask]
        Y_data = Y_data[amp_mask]
        print(f"振幅过滤后剩余: {X_data.shape[0]} 个样本 (移除 {num_amp_filtered})")
        num_samples = X_data.shape[0]

    # ---------- 过滤 NaN / Inf / 极端值 ----------
    safe_threshold = parent_config.get_safe_threshold()
    nan_replacement = parent_config.get_nan_replacement_value()

    X_flat_check = X_data.reshape(num_samples, -1)
    X_nn = np.nan_to_num(X_flat_check, nan=nan_replacement)
    Y_nn = np.nan_to_num(Y_data, nan=nan_replacement)

    mask_x = (np.abs(X_nn) < safe_threshold).all(axis=1)
    mask_y = (np.abs(Y_nn) < safe_threshold).all(axis=1)
    final_mask = mask_x & mask_y

    num_bad = num_samples - np.sum(final_mask)
    if num_bad > 0:
        print(f"警告：发现并移除 {num_bad} 个含 Inf/NaN/极端值的坏样本")
    X_data = X_data[final_mask]   # shape: (N_clean, 3, 7801)
    Y_data = Y_data[final_mask]
    num_samples = X_data.shape[0]
    print(f"清洗后: X={X_data.shape}, Y={Y_data.shape}")

    if num_samples == 0:
        print("错误：数据集中没有有效样本！")
        return None, None, None, None

    # ---------- 单样本联合通道归一化 (X) ----------
    # 对每个样本，将三条曲线合并计算均值和标准差，整体归一化
    # 等效于 utils.py 中的 Domain Invariant 归一化，但保持 3D 形状
    sample_means = np.nanmean(X_data, axis=(1, 2), keepdims=True)  # (N, 1, 1)
    sample_stds  = np.nanstd(X_data,  axis=(1, 2), keepdims=True) + 1e-8
    X_scaled = (X_data - sample_means) / sample_stds            # (N, 3, 7801)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)
    print("X 单样本联合通道归一化完成 (保持 3D 形状)")

    # ---------- MinMaxScaler 缩放 Y → [0.1, 0.9] (Safe Sigmoid Lock) ----------
    y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    Y_scaled = y_scaler.fit_transform(Y_data).astype(np.float32)

    # 保存 y_scaler
    y_scaler_path = transformer_config.get_y_scaler_path()
    os.makedirs(os.path.dirname(y_scaler_path), exist_ok=True)
    with open(y_scaler_path, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"y_scaler 已保存至: {y_scaler_path}")

    # ---------- 数据集拆分 ----------
    test_ratio  = parent_config.get_test_split_ratio()
    val_ratio   = parent_config.get_val_split_ratio()
    random_seed = parent_config.get_random_seed()

    X_tv, X_test, Y_tv, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=test_ratio, random_state=random_seed)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_tv, Y_tv, test_size=val_ratio, random_state=random_seed)

    print(f"训练集: {X_train.shape[0]}  验证集: {X_val.shape[0]}  测试集: {X_test.shape[0]}")

    # ---------- 转为 Tensor → DataLoader ----------
    def make_loader(X, Y, shuffle=False):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(Y, dtype=torch.float32))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)

    train_loader = make_loader(X_train, Y_train, shuffle=True)
    val_loader   = make_loader(X_val,   Y_val)
    test_loader  = make_loader(X_test,  Y_test)

    print("DataLoaders 创建完毕（X 形状: batch × 3 × seq_len）")
    print("----------------------------\n")
    return train_loader, val_loader, test_loader, param_names
