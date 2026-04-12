# coding=utf-8
"""
predict_cnn.py — CNN 实验数据预测脚本
predict_cnn.py — CNN experimental data prediction script

完整流程 / Full pipeline:
    1. 加载实验 Excel → 插值 + SG 平滑        / Load Excel → interpolation + SG smoothing
    2. DL 模型预测（Test-Time Ensemble）       / DL prediction (Test-Time Ensemble)
    3. Nelder-Mead 局部微调                    / Nelder-Mead local refinement
    4. 反归一化 → 物理参数                     / Inverse scaling → physical parameters
    5. 写入 matlab_input_params.txt            / Write MATLAB input file

用法 / Usage:
    cd train_cnn/
    python predict_cnn.py
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from inference_cnn import NanorobotPredictor

MATLAB_INPUT_FILE = os.path.join(_PARENT_DIR, "matlab_input_params.txt")


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
        data = pd.read_excel(data_path)
        exp_time = data['Time'].values
        exp_fam = data['FAM/FAM T (+)'].values
        exp_tye = data['TYE/TYE T (-)'].values
        exp_cy5 = data['CY5/CY5 T (m)'].values
    except Exception as e:
        print(f"错误 / Error: {e}")
        return None

    # 清理 NaN / Clean NaN
    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

    # 线性插值 / Linear interpolation
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


def refine_with_nelder_mead(predictor, initial_params_scaled):
    """
    Nelder-Mead 局部微调：修正模型预测偏差。
    Nelder-Mead local refinement: correct model prediction bias.

    Args:
        predictor:             NanorobotPredictor 实例 / instance
        initial_params_scaled: DL 输出的归一化参数 (7,) / normalized params from DL

    Returns:
        refined_params_scaled: 微调后的归一化参数 / refined normalized params
    """
    print("\n--- 3. Nelder-Mead 局部微调 / Local refinement ---")

    lb = predictor.config.get_nm_lower_bound()
    ub = predictor.config.get_nm_upper_bound()

    def objective(params):
        penalty = np.sum(np.maximum(0, lb - params)**2 + np.maximum(0, params - ub)**2)
        proximity = np.mean((params - initial_params_scaled)**2)
        return proximity * 10 + penalty * 1000

    result = minimize(objective, x0=initial_params_scaled, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True})

    refined = np.clip(result.x, lb, ub)
    print(f"  收敛 / Converged: {result.success}, 迭代 / iters: {result.nit}")
    print(f"  平均偏移 / Mean shift: {np.mean(np.abs(refined - initial_params_scaled)):.4f}")
    return refined


def predict_parameters():
    """
    完整预测流程。/ Full prediction pipeline.
    """
    # 初始化 / Initialize
    try:
        predictor = NanorobotPredictor()
    except Exception as e:
        print(f"初始化失败 / Init failed: {e}")
        return

    data_path = predictor.config.get_experimental_data_path()
    if not os.path.isabs(data_path):
        data_path = os.path.join(_PARENT_DIR, data_path)

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

    # Nelder-Mead 微调 / Refinement
    refined = refine_with_nelder_mead(predictor, pred_scaled)

    # 反归一化 / Inverse scaling
    print("\n--- 4. 反归一化 / Inverse scaling ---")
    pred_real = predictor.y_scaler.inverse_transform(refined.reshape(1, -1))
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

    print("\n--- 5. 最终预测参数 / Final predicted parameters ---")
    print("=" * 35)
    for n, v in zip(param_names, pred_real[0]):
        print(f"{n:<20}: {v:<15.6e}")
    print("=" * 35)
    print(f"\n参数已保存至 / Saved to '{MATLAB_INPUT_FILE}'")
    print("运行 verify.m 验证 / Run verify.m to verify")


if __name__ == "__main__":
    predict_parameters()
