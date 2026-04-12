# coding=utf-8
"""
check_npz.py — 检查 NPZ 数据集内容的诊断工具
check_npz.py — Diagnostic tool for inspecting NPZ dataset contents

用法 / Usage:
    python check_npz.py <npz_path> [sample_idx]
    python check_npz.py ../results/training_dataset.npz 100
"""

import sys
import numpy as np


def check_npz(npz_path, sample_idx=0):
    """
    加载并检查 NPZ 数据集的结构和数据质量。
    Load and inspect the structure and data quality of an NPZ dataset.

    Args:
        npz_path:   NPZ 文件路径 / path to the NPZ file
        sample_idx: 要检查的样本索引 / sample index to inspect
    """
    try:
        data = np.load(npz_path)
    except FileNotFoundError:
        print(f"错误 / Error: 未找到 '{npz_path}'")
        return

    print(f"NPZ 文件内容 / NPZ contents: {data.files}")

    X = data['X']
    Y = data['Y']

    # 参数名称 / Parameter names
    param_names = ['E_b', 'E_b_azo_trans', 'E_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']
    curve_labels = ['sim_fam', 'sim_tye', 'sim_cy5']

    print(f"\nX (曲线/curves) shape: {X.shape}")
    print(f"Y (参数/params) shape: {Y.shape}")

    # 校正索引范围 / Clamp index to valid range
    if sample_idx >= X.shape[0]:
        print(f"警告 / Warning: sample_idx {sample_idx} 超出范围，重置为 0 / out of range, reset to 0")
        sample_idx = 0

    sample_X = X[sample_idx]
    sample_Y = Y[sample_idx]

    # 显示 X 的前 10 个时间点 / Show first 10 time points
    num_curves = sample_X.shape[0] if sample_X.ndim == 2 else 1
    print(f"\n--- 样本 {sample_idx} 的前 10 个时间点 / First 10 time points of sample {sample_idx} ---")
    for c in range(min(num_curves, len(curve_labels))):
        pts = sample_X[c, :10] if sample_X.ndim == 2 else sample_X[:10]
        print(f"  {curve_labels[c]:<15}: {pts}")

    # 显示 Y 参数值 / Show Y parameter values
    print(f"\n--- 样本 {sample_idx} 的参数值 / Parameter values of sample {sample_idx} ---")
    for i, val in enumerate(sample_Y):
        label = param_names[i] if i < len(param_names) else f"Param_{i}"
        print(f"  {label:<15}: {val:.6e}")

    # 数据质量检查 / Data quality check
    nan_x = np.isnan(X).any(axis=tuple(range(1, X.ndim))).sum()
    inf_x = np.isinf(X).any(axis=tuple(range(1, X.ndim))).sum()
    nan_y = np.isnan(Y).sum()
    inf_y = np.isinf(Y).sum()

    print(f"\n--- 数据质量 / Data quality ---")
    print(f"X: NaN 样本数={nan_x}, Inf 样本数={inf_x}")
    print(f"Y: NaN 个数={nan_y}, Inf 个数={inf_y}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "../results/training_dataset.npz"
    idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    check_npz(path, idx)
