# coding=utf-8
"""
predict_transformer.py  —  DNA Walker Transformer 实验数据预测脚本

与主目录的 predict.py 逻辑一致，但使用 Transformer 模型进行推理。
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

# ── 路径设置：确保可以从 train_transformer/ 目录运行 ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from inference_transformer import TransformerPredictor

# --- 全局常量 ---
MATLAB_INPUT_FILE = "matlab_input_params_transformer.txt"


def load_real_experimental_data(config, data_path):
    """
    加载并预处理真实实验数据（插值 + SG平滑）。
    逻辑直接复用自根目录 predict.py。
    """
    print(f"--- 1. 正在加载并预处理真实实验数据: {data_path} ---")

    # 获取配置
    sim_total_time = config.get_sim_total_time()
    num_time_points = config.get_num_time_points()
    standard_time_axis = np.linspace(0, sim_total_time, num_time_points)

    # 读取 Excel
    try:
        data = pd.read_excel(data_path)
        exp_time = data['Time'].values
        exp_fam = data['FAM/FAM T (+)'].values
        exp_tye = data['TYE/TYE T (-)'].values
        exp_cy5 = data['CY5/CY5 T (m)'].values
    except Exception as e:
        print(f"错误: 无法读取数据列。请确保 Excel 包含 'Time', 'FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)' 列。")
        print(f"错误信息: {e}")
        return None

    # 清理并插值
    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

    interp_fam_func = interp1d(exp_time, exp_fam, kind='linear', bounds_error=False, fill_value=(exp_fam[0], exp_fam[-1]))
    interp_tye_func = interp1d(exp_time, exp_tye, kind='linear', bounds_error=False, fill_value=(exp_tye[0], exp_tye[-1]))
    interp_cy5_func = interp1d(exp_time, exp_cy5, kind='linear', bounds_error=False, fill_value=(exp_cy5[0], exp_cy5[-1]))

    curve_fam = interp_fam_func(standard_time_axis)
    curve_tye = interp_tye_func(standard_time_axis)
    curve_cy5 = interp_cy5_func(standard_time_axis)

    # SG 平滑 (window=61, polyorder=3)
    try:
        curve_fam = savgol_filter(curve_fam, 61, 3)
        curve_tye = savgol_filter(curve_tye, 61, 3)
        curve_cy5 = savgol_filter(curve_cy5, 61, 3)
        print("Savitzky-Golay 平滑完成。")
    except:
        print("警告: SG 平滑失败，使用原始插值。")

    return np.stack([curve_fam, curve_tye, curve_cy5], axis=0)  # (3, 7801)


def write_matlab_input(params_dict):
    """写入 MATLAB 输入文件供验证"""
    print(f"\n正在写入 MATLAB 输入文件: {MATLAB_INPUT_FILE}")
    try:
        with open(MATLAB_INPUT_FILE, 'w') as f:
            for name, value in params_dict.items():
                f.write(f"{name}={repr(float(value))}\n")
            f.write("END_OF_PARAMS=1\n")
        print("写入成功。")
    except Exception as e:
        print(f"写入失败: {e}")


def main():
    # 1. 初始化预测器
    try:
        predictor = TransformerPredictor()
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        return

    # 2. 读取实验数据路径
    data_file = predictor.parent_config.get_experimental_data_path()
    if not os.path.isabs(data_file):
        data_file = os.path.join(_PARENT_DIR, data_file)

    # 3. 加载并处理数据
    X_sample_raw = load_real_experimental_data(predictor.parent_config, data_file)
    if X_sample_raw is None:
        return

    # 4. 执行推理 (内含归一化和反对数变换)
    print("\n--- 2. 正在执行 Transformer 模型推理 ---")
    results = predictor.predict(X_sample_raw)
    predicted_real = results[0] # 取第 0 个样本

    # 5. 打印并保存结果
    param_names = predictor.get_param_names()
    params_dict = {name: val for name, val in zip(param_names, predicted_real)}

    print("\n--- 3. 预测完成 (Transformer Results) ---")
    print("=" * 45)
    for name, val in params_dict.items():
        print(f"  {name:<20}: {val:<15.6e}")
    print("=" * 45)

    write_matlab_input(params_dict)
    print(f"\n结果已保存至 {MATLAB_INPUT_FILE}")
    print("请在 MATLAB 中使用 verify.m (需修改读取路径) 或将参数复制到 verify.m 进行物理验证。")


if __name__ == '__main__':
    main()
