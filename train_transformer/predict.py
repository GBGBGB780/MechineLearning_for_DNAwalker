# coding=utf-8
"""
predict.py — Transformer 实验数据预测脚本
predict.py — Transformer experimental data prediction script

与 CNN 的 predict.py 逻辑一致，但使用 Transformer 模型推理。
Same logic as CNN predict.py, but uses Transformer model for inference.
"""

import argparse
import os
import sys
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

from inference_transformer import TransformerPredictor


def _to_cwd_relative(path):
    """将脚本内文件路径转换为相对当前工作目录。"""
    return os.path.relpath(os.path.normpath(path), start=os.getcwd())


# --- 全局常量 ---
MATLAB_INPUT_FILE = _to_cwd_relative(os.path.join(_THIS_DIR, "matlab_input_params.txt"))


def load_real_experimental_data(config, data_path):
    """
    加载并预处理真实实验数据（插值 + SG 平滑）。
    Load and preprocess real experimental data (interpolation + SG smoothing).
    """
    print(f"--- 1. 正在加载并预处理真实实验数据: {data_path} ---")

    # 获取配置
    sim_total_time = config.get_sim_total_time()
    num_time_points = config.get_num_time_points()
    standard_time_axis = np.linspace(0, sim_total_time, num_time_points)

    # 读取 Excel
    try:
        from exp_data_io import load_experimental_curves
        exp_time, exp_fam, exp_tye, exp_cy5 = load_experimental_curves(data_path)
    except Exception as e:
        print(f"错误: 无法读取实验数据列: {e}")
        return None

    # 清理并插值 (load_experimental_curves 已剔除 NaN 行)
    interp_fam_func = interp1d(exp_time, exp_fam, kind='linear', bounds_error=False, fill_value=(exp_fam[0], exp_fam[-1]))
    interp_tye_func = interp1d(exp_time, exp_tye, kind='linear', bounds_error=False, fill_value=(exp_tye[0], exp_tye[-1]))
    interp_cy5_func = interp1d(exp_time, exp_cy5, kind='linear', bounds_error=False, fill_value=(exp_cy5[0], exp_cy5[-1]))

    curve_fam = interp_fam_func(standard_time_axis)
    curve_tye = interp_tye_func(standard_time_axis)
    curve_cy5 = interp_cy5_func(standard_time_axis)

    # SG 平滑 (from config) / SG smoothing (from config)
    sg_window = config.get_sg_window()
    sg_poly = config.get_sg_polyorder()
    try:
        curve_fam = savgol_filter(curve_fam, sg_window, sg_poly)
        curve_tye = savgol_filter(curve_tye, sg_window, sg_poly)
        curve_cy5 = savgol_filter(curve_cy5, sg_window, sg_poly)
        print(f"SG 平滑完成 / SG smoothing done (window={sg_window}, poly={sg_poly})")
    except Exception as e:
        print(f"警告 / Warning: SG 平滑失败 / SG smoothing failed: {e}")

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


def main(parent_config_override=None, transformer_config_override=None):
    # 1. 初始化预测器
    try:
        predictor = TransformerPredictor(
            parent_config_override_file=parent_config_override,
            transformer_config_override_file=transformer_config_override
        )
    except Exception as e:
        print(f"初始化预测器失败: {e}")
        return

    # 2. 读取实验数据路径
    data_file = predictor.parent_config.get_experimental_data_path()

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
    parser = argparse.ArgumentParser(description='Run Transformer prediction on experimental data.')
    parser.add_argument(
        '--config',
        help='Optional override INI layered on top of the repo root configfile.ini'
    )
    parser.add_argument(
        '--transformer-config',
        help='Optional override INI layered on top of train_transformer/config_transformer.ini'
    )
    args = parser.parse_args()
    main(
        parent_config_override=args.config,
        transformer_config_override=args.transformer_config
    )
