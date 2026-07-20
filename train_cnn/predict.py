# coding=utf-8
"""
predict.py — CNN 实验数据预测脚本
predict.py — CNN experimental data prediction script

完整流程 / Full pipeline:
    1. 加载实验 Excel → 插值 + SG 平滑        / Load Excel → interpolation + SG smoothing
    2. DL 模型预测（Test-Time Ensemble）       / DL prediction (Test-Time Ensemble)
    3. 反归一化 → 物理参数                     / Inverse scaling → physical parameters
    4. 写入 matlab_input_params.txt            / Write MATLAB input file

用法 / Usage:
    cd train_cnn/
    python predict.py
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from inference_cnn import NanorobotPredictor


def _to_cwd_relative(path):
    """将脚本内文件路径转换为相对当前工作目录。"""
    return os.path.relpath(os.path.normpath(path), start=os.getcwd())


MATLAB_INPUT_FILE = _to_cwd_relative(os.path.join(_THIS_DIR, "matlab_input_params.txt"))


def load_real_experimental_data(config, data_path):
    """
    加载、插值并平滑真实实验数据，使其与训练数据格式一致。
    Load, interpolate, and smooth real experimental data to match training data format.

    Args:
        config:    Config 对象 / Config object
        data_path: 实验数据 Excel 路径 / experimental data Excel path

    Returns:
        (3, T) numpy 数组或 None / numpy array or None on failure
    """
    print(f"--- 1. 加载实验数据 / Loading experimental data: {data_path} ---")

    sim_total_time = config.get_sim_total_time()
    num_time_points = config.get_num_time_points()
    standard_time_axis = np.linspace(0, sim_total_time, num_time_points)

    # 读取 Excel / Read Excel
    try:
        from exp_data_io import load_experimental_curves
        exp_time, exp_fam, exp_tye, exp_cy5 = load_experimental_curves(data_path)
    except Exception as e:
        print(f"错误 / Error: {e}")
        return None

    # 线性插值 / Linear interpolation (load_experimental_curves 已剔除 NaN 行)
    funcs = [interp1d(exp_time, d, kind='linear', bounds_error=False, fill_value=(d[0], d[-1]))
             for d in [exp_fam, exp_tye, exp_cy5]]
    curves = [f(standard_time_axis) for f in funcs]

    # SG 平滑 / Savitzky-Golay smoothing
    sg_window = config.get_sg_window()
    sg_poly = config.get_sg_polyorder()
    try:
        curves = [savgol_filter(c, sg_window, sg_poly) for c in curves]
        print(f"SG 平滑完成 / SG smoothing done (window={sg_window}, poly={sg_poly})")
    except Exception as e:
        print(f"警告 / Warning: SG 平滑失败 / SG smoothing failed: {e}")

    return np.stack(curves, axis=0)


def write_matlab_input(params_dict):
    """
    写入 MATLAB 输入文件。/ Write MATLAB input file.
    """
    print(f"\n写入 / Writing: {MATLAB_INPUT_FILE}")
    try:
        with open(MATLAB_INPUT_FILE, 'w') as f:
            for name, value in params_dict.items():
                f.write(f"{name}={repr(float(value))}\n")
            f.write("END_OF_PARAMS=1\n")
        print("写入完毕 / Write complete")
    except IOError as e:
        print(f"Error: {e}")


def predict_parameters(config_override=None):
    """
    完整预测流程。/ Full prediction pipeline.
    """
    # 初始化 / Initialize
    try:
        predictor = NanorobotPredictor(config_override_file=config_override)
    except Exception as e:
        print(f"初始化失败 / Init failed: {e}")
        return

    data_path = predictor.config.get_experimental_data_path()

    # 加载数据 / Load data
    X_raw = load_real_experimental_data(predictor.config, data_path)
    if X_raw is None:
        return

    # DL 预测 (Test-Time Ensemble) / DL prediction
    print("\n--- 2. DL 模型预测 / DL prediction (Test-Time Ensemble) ---")
    num_ch = predictor.config.get_num_curves()
    seq_len = predictor.input_size // num_ch

    X_flat = X_raw.reshape(1, -1).astype(np.float32)
    X_3d = X_flat.reshape(1, num_ch, seq_len)
    mean = np.nanmean(X_3d, axis=(1, 2), keepdims=True)
    std = np.nanstd(X_3d, axis=(1, 2), keepdims=True) + 1e-8
    X_3d = np.where(np.isnan((X_3d - mean) / std), 0.0, (X_3d - mean) / std)
    X_scaled = X_3d.reshape(1, -1)

    N_ENS = predictor.config.get_ensemble_size()
    NOISE_STD = predictor.config.get_ensemble_noise_std()
    np.random.seed(42)

    preds = []
    with torch.no_grad():
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(predictor.device)
        preds.append(predictor.model(X_t).cpu().numpy()[0])
        for _ in range(N_ENS - 1):
            noise = np.random.normal(0, NOISE_STD, X_scaled.shape).astype(np.float32)
            X_noisy = torch.tensor(X_scaled + noise, dtype=torch.float32).to(predictor.device)
            preds.append(predictor.model(X_noisy).cpu().numpy()[0])

    preds = np.array(preds)
    pred_scaled = np.median(preds, axis=0)
    print(f"  集成预测 / Ensemble (N={N_ENS}): {pred_scaled}")
    print(f"  集成标准差 / Ensemble std: {np.round(np.std(preds, axis=0), 5)}")

    # 反归一化 / Inverse scaling
    print("\n--- 3. 反归一化 / Inverse scaling ---")
    pred_real = predictor.y_scaler.inverse_transform(pred_scaled.reshape(1, -1))
    log_params = predictor.config.get_log_transform_params()
    log_eps = predictor.config.get_log_epsilon()
    param_names = predictor.get_param_names()
    for p in log_params:
        if p in param_names:
            idx = param_names.index(p)
            pred_real[0, idx] = np.power(10, pred_real[0, idx]) - log_eps

    # 输出 / Output
    params_dict = {n: v for n, v in zip(param_names, pred_real[0])}
    write_matlab_input(params_dict)

    print("\n--- 4. 最终预测参数 / Final predicted parameters ---")
    print("=" * 35)
    for n, v in zip(param_names, pred_real[0]):
        print(f"{n:<20}: {v:<15.6e}")
    print("=" * 35)
    print(f"\n参数已保存至 / Saved to '{MATLAB_INPUT_FILE}'")
    print("运行 verify.m 验证 / Run verify.m to verify")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN prediction on experimental data.")
    parser.add_argument(
        "--config",
        help="Optional override INI layered on top of the repo root configfile.ini."
    )
    args = parser.parse_args()
    predict_parameters(config_override=args.config)
