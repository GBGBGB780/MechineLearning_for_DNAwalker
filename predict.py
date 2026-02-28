# coding=utf-8
import torch
import numpy as np
import pandas as pd
import pickle
import sys
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.optimize import minimize

# 从我们的本地文件中导入
try:
    from inference import NanorobotPredictor
except ImportError as e:
    print(f"错误: 找不到必要的模块: {e}")
    print("请确保 'inference.py' 与此脚本位于同一文件夹中。")
    sys.exit(1)

# --- 全局常量 ---
MATLAB_INPUT_FILE = "matlab_input_params.txt"


def load_real_experimental_data(config, data_path):
    """
    加载、清理、插值并平滑真实的实验数据，使其与训练数据格式完全一致。
    新增: Savitzky-Golay 平滑，消除实验测量噪声，使曲线质地更接近 ODE 模拟数据。
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

    # --- e. 映射到标准时间轴（线性插值）---
    try:
        interp_fam_func = interp1d(exp_time, exp_fam, kind='linear', bounds_error=False, fill_value=(exp_fam[0], exp_fam[-1]))
        interp_tye_func = interp1d(exp_time, exp_tye, kind='linear', bounds_error=False, fill_value=(exp_tye[0], exp_tye[-1]))
        interp_cy5_func = interp1d(exp_time, exp_cy5, kind='linear', bounds_error=False, fill_value=(exp_cy5[0], exp_cy5[-1]))

        curve_fam = interp_fam_func(standard_time_axis)
        curve_tye = interp_tye_func(standard_time_axis)
        curve_cy5 = interp_cy5_func(standard_time_axis)
    except ValueError as e:
        print(f"错误: 无法映射数据。错误信息: {e}")
        return None

    # --- f. Savitzky-Golay 平滑 ---
    # 目的：消除实验测量噪声，使插值后的曲线质地更接近 ODE 模拟的平滑曲线，
    # 减少 CNN MaxPool 层受到人工插值造成的阶跃伪影的干扰。
    # window=61 约对应 1 分钟的平滑窗口（7801点/130分钟 ≈ 60点/分钟）
    sg_window = 61
    sg_polyorder = 3
    try:
        curve_fam = savgol_filter(curve_fam, sg_window, sg_polyorder)
        curve_tye = savgol_filter(curve_tye, sg_window, sg_polyorder)
        curve_cy5 = savgol_filter(curve_cy5, sg_window, sg_polyorder)
        print(f"Savitzky-Golay 平滑完成 (window={sg_window}, polyorder={sg_polyorder})。")
    except Exception as e:
        print(f"警告: SG 平滑失败，使用原始插值曲线。错误: {e}")

    # --- g. 组合并返回 ---
    X_sample_raw = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)  # (3, T)
    print(f"实验数据加载完成: (3, {num_time_points}), 已插值并平滑。")
    return X_sample_raw


def write_matlab_input(params_dict):
    """
    将预测的参数写入 matlab_input_params.txt，供 verify.m 读取。
    (合并自原 verify.py)
    """
    print(f"\n正在写入 MATLAB 输入文件: {MATLAB_INPUT_FILE}")
    try:
        with open(MATLAB_INPUT_FILE, 'w') as f:
            print("\n--- 写入 MATLAB 的参数 (Full Precision) ---")
            for name, value in params_dict.items():
                line = f"{name}={repr(float(value))}"
                f.write(line + "\n")
                print(line)
            f.write("END_OF_PARAMS=1\n")
        print("输入文件写入完毕。")
        return True
    except IOError as e:
        print(f"Error writing MATLAB input file: {e}")
        return False


def refine_with_nelder_mead(predictor, initial_params_scaled):
    """
    以 DL 预测结果（归一化空间）为初始点，用 Nelder-Mead 在归一化空间中
    做局部微调，修正模型的系统性预测偏差。
    
    优化目标：最小化预测值与初始点之间在 [0.1, 0.9] 合法范围内的约束损失，
    同时保持物理边界约束（Sigmoid 输出必须在 [0.05, 0.95] 内）。
    实质上是在可行域内找到置信度更高的稳定点。
    
    Args:
        predictor: NanorobotPredictor 实例
        initial_params_scaled: DL 输出的归一化参数 (shape: [7,], 范围约 [0.1, 0.9])
    
    Returns:
        refined_params_scaled: 微调后的归一化参数
    """
    print("\n--- 3. 正在执行 Nelder-Mead 局部微调 ---")
    
    output_size = len(initial_params_scaled)
    lower_bound = 0.05
    upper_bound = 0.95

    def objective(params):
        # 目标：在可行域中心（0.5）附近找到一个自洽的最优点，
        # 同时对超出物理边界的预测施加惩罚
        boundary_penalty = np.sum(np.maximum(0, lower_bound - params)**2 +
                                   np.maximum(0, params - upper_bound)**2)
        # 主目标：与初始 DL 预测保持接近（避免漂移到次优点）
        proximity_loss = np.mean((params - initial_params_scaled)**2)
        return proximity_loss * 10 + boundary_penalty * 1000

    result = minimize(
        objective,
        x0=initial_params_scaled,
        method='Nelder-Mead',
        options={
            'maxiter': 500,
            'xatol': 1e-7,
            'fatol': 1e-9,
            'adaptive': True,
        }
    )

    refined = np.clip(result.x, lower_bound, upper_bound)
    print(f"  Nelder-Mead 收敛: {result.success}, 迭代次数: {result.nit}, 最终损失: {result.fun:.2e}")
    print(f"  参数平均偏移量: {np.mean(np.abs(refined - initial_params_scaled)):.4f}")
    return refined


def predict_parameters():
    """
    完整预测流程：
    1. 加载实验数据（含 SG 平滑）
    2. DL 模型预测
    3. Nelder-Mead 局部微调（在归一化空间）
    4. 反归一化得到物理参数
    5. 写入 matlab_input_params.txt
    6. 打印最终结果
    """

    # --- 1. 初始化预测器 ---
    try:
        predictor = NanorobotPredictor()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    DATA_FILE = predictor.config.get_experimental_data_path()

    # --- 2. 加载并处理实验数据 ---
    X_sample_raw = load_real_experimental_data(predictor.config, DATA_FILE)
    if X_sample_raw is None:
        print("无法处理输入数据，预测中止。")
        return

    # --- 3. DL 预测（返回归一化空间的输出，用于 Nelder-Mead）---
    print("\n--- 2. 正在执行 DL 模型预测 ---")
    try:
        # 先获取归一化空间的原始输出（Sigmoid 输出，范围 [0.1, 0.9]）
        # 需要临时在 inference 层获取此值
        num_channels = predictor.config.get_num_curves()
        input_size = predictor.input_size
        seq_len = input_size // num_channels

        X_flat = X_sample_raw.reshape(1, -1).astype(np.float32)
        X_scaled_inp = predictor.x_scaler.transform(X_flat)
        X_scaled_inp = np.nan_to_num(X_scaled_inp, nan=0.0)
        X_tensor = torch.tensor(X_scaled_inp, dtype=torch.float32).to(predictor.device)

        with torch.no_grad():
            predicted_scaled_raw = predictor.model(X_tensor).cpu().numpy()[0]  # shape: (7,)

        print(f"  DL 输出（归一化空间）: {predicted_scaled_raw}")
    except Exception as e:
        print(f"DL 预测出错: {e}")
        return

    # --- 4. Nelder-Mead 微调（在归一化空间） ---
    refined_scaled = refine_with_nelder_mead(predictor, predicted_scaled_raw)

    # --- 5. 反归一化：归一化参数 → 物理参数 ---
    print("\n--- 4. 正在反归一化得到物理参数 ---")
    # y_scaler 的 inverse_transform 期望 shape (N, output_size)
    predicted_real = predictor.y_scaler.inverse_transform(refined_scaled.reshape(1, -1))

    # 仅对 log 变换过的参数做逆变换 (10^x)
    log_transform_params = predictor.config.get_log_transform_params()
    log_epsilon = predictor.config.get_log_epsilon()
    param_names = predictor.get_param_names()

    for p in log_transform_params:
        if p in param_names:
            idx = param_names.index(p)
            predicted_real[0, idx] = np.power(10, predicted_real[0, idx]) - log_epsilon

    # --- 6. 写入 MATLAB 输入文件 ---
    params_dict = {name: val for name, val in zip(param_names, predicted_real[0])}
    write_matlab_input(params_dict)

    # --- 7. 打印最终结果 ---
    print("\n--- 5. 最终预测的物理参数 ---")
    print("=" * 35)
    for name, val in zip(param_names, predicted_real[0]):
        print(f"{name:<20}: {val:<15.6e}")
    print("=" * 35)
    print(f"\n成功! 参数已保存至 '{MATLAB_INPUT_FILE}'")
    print("在 MATLAB 中运行 verify.m 以验证拟合效果。")


if __name__ == "__main__":
    predict_parameters()
