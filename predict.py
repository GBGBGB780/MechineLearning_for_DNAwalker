# coding=utf-8
import torch
import numpy as np
import pandas as pd
import pickle
import sys
from scipy.interpolate import interp1d

# 从我们的本地文件中导入
try:
    from inference import NanorobotPredictor
except ImportError as e:
    print(f"错误: 找不到必要的模块: {e}")
    print("请确保 'inference.py' 与此脚本位于同一文件夹中。")
    sys.exit(1)


def load_real_experimental_data(config, data_path):
    """
    加载、清理并插值真实的实验数据，使其与训练数据格式完全一致。
    """
    print(f"--- 1. 正在加载并预处理真实实验数据: {data_path} ---")

    # --- a. 从 config 中获取模拟设置 ---
    try:
        sim_total_time = config.get_sim_total_time()
        num_time_points = config.get_num_time_points()
        p_unbind_track = config.get_p_unbind_track()
    except Exception as e:
        print(f"错误: 无法从配置文件中读取关键设置: {e}")
        return None

    # --- b. 创建标准时间轴 (必须与训练时 一致) ---
    standard_time_axis = np.linspace(0, sim_total_time, num_time_points)

    # --- c. 加载真实的CSV数据 ---
    try:
        data = pd.read_excel(data_path)
        exp_time = data['Time'].values
        exp_fam = data['FAM/FAM T (+)'].values
        exp_tye = data['TYE/TYE T (-)'].values
        exp_cy5 = data['CY5/CY5 T (m)'].values
    except Exception as e:
        print(f"错误: 无法从 {data_path} 读取数据列。")
        print(f"请确保文件存在且包含 'Time', 'FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)' 列。")
        print(f"错误信息: {e}")
        return None

    # --- d. 清理并调整数据 ---
    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

    # --- e. 映射到标准时间轴（保留空白区域为 NaN）---
    # 对于实验数据存在的区域，用线性插值取近似值；
    # 对于空白区域（距离最近数据点超过阈值），直接置 NaN，
    # 后续在 inference.py 中归一化时空白处填 0（不提供任何信息给模型）。
    GAP_THRESHOLD = 5.0  # 分钟。超过此距离视为空白（可根据实验数据间隔调整）

    try:
        interp_fam_func = interp1d(exp_time, exp_fam, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_tye_func = interp1d(exp_time, exp_tye, kind='linear', bounds_error=False, fill_value=np.nan)
        interp_cy5_func = interp1d(exp_time, exp_cy5, kind='linear', bounds_error=False, fill_value=np.nan)

        curve_fam = interp_fam_func(standard_time_axis)
        curve_tye = interp_tye_func(standard_time_axis)
        curve_cy5 = interp_cy5_func(standard_time_axis)
    except ValueError as e:
        print(f"错误: 无法映射数据。错误信息: {e}")
        return None

    # 检测空白区域：对每个标准时间点，找最近的实验时间点距离
    # 若距离超过 GAP_THRESHOLD，视为空白，置 NaN
    nearest_dist = np.min(np.abs(standard_time_axis[:, None] - exp_time[None, :]), axis=1)
    gap_mask = nearest_dist > GAP_THRESHOLD
    n_gaps = gap_mask.sum()
    if n_gaps > 0:
        print(f"  检测到 {n_gaps} 个时间点在空白区域（距最近数据点 > {GAP_THRESHOLD} min），已置 NaN")
        curve_fam[gap_mask] = np.nan
        curve_tye[gap_mask] = np.nan
        curve_cy5[gap_mask] = np.nan

    # --- g. 组合并返回 ---
    X_sample_raw = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)  # (3, T)
    valid_pct = 100 * (1 - n_gaps / len(standard_time_axis))
    print(f"实验数据加载完成: (3, {num_time_points}), 有效时间点占比 {valid_pct:.1f}%")
    return X_sample_raw



def predict_parameters():
    """
    加载模型和数据，执行并打印最终的参数预测。
    """

    # --- 1. 初始化预测器 ---
    try:
        predictor = NanorobotPredictor()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 获取配置中的数据路径
    DATA_FILE = predictor.config.get_experimental_data_path()

    # --- 2. 加载并处理真实的输入数据 ---
    # 注意: load_real_experimental_data 需要 config 对象
    X_sample_raw = load_real_experimental_data(predictor.config, DATA_FILE)

    if X_sample_raw is None:
        print("无法处理输入数据，预测中止。")
        return

    # --- 3. 执行预测 ---
    print("\n--- 2. 正在执行模型预测 ---")
    try:
        predicted_real_params = predictor.predict(X_sample_raw)
    except Exception as e:
        print(f"预测出错: {e}")
        return

    # --- 4. 打印最终结果 ---
    print("\n--- 3. 预测的物理参数 ---")
    print("=" * 30)

    param_names = predictor.get_param_names()
    for i in range(len(param_names)):
        name = param_names[i]
        pred_val = predicted_real_params[0, i]
        print(f"{name:<15}: {pred_val:<15.6e}")

    print("=" * 30)
    print("预测完成。")


if __name__ == "__main__":
    predict_parameters()
