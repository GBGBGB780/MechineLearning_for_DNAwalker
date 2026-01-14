# coding=gb2312
import torch
import numpy as np
import pandas as pd
import pickle
import sys
from scipy.interpolate import interp1d

# 从我们的本地文件中导入
try:
    from model import InverseMLP
    from config_loader import Config
except ImportError as e:
    print(f"错误: 找不到必要的模块: {e}")
    print("请确保 'model.py' 和 'config_loader.py' 与此脚本位于同一文件夹中。")
    sys.exit(1)


def load_real_experimental_data(config, data_path):
    """
    加载、清理并插值真实的实验数据，使其与训练数据格式完全一致。
    """
    print(f"--- 1. 正在加载并预处理真实实验数据: {data_path} ---")

    # --- a. 从 config 中获取模拟设置 ---
    try:
        sim_total_time = float(config["NANOROBOT_MODELING"]["sim_total_time"])
        num_time_points = int(config['DATA_GENERATION']['num_time_points'])
        p_unbind_track = float(config['PHYSICAL_PARAMETERS']['p_unbind_track'])
    except KeyError as e:
        print(f"错误: 无法从 {CONFIG_FILE} 中读取关键设置: {e}")
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
    # (这部分逻辑也借鉴自 nanorobot_solver.py)
    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

    # !!! 关键步骤 !!!
    # 我们的训练数据 在计算 sim_cy5 时加上了 p_unbind_track
    # 为了使真实数据匹配，我们必须从真实数据中 *减去* 这个基线值
    exp_cy5_adjusted = exp_cy5 - p_unbind_track

    # --- e. 插值 (Interpolate) ---
    # (这部分逻辑借鉴自 generate_dataset.py 和 nanorobot_solver.py)
    try:
        interp_fam_func = interp1d(exp_time, exp_fam, kind='linear', fill_value='extrapolate')
        interp_tye_func = interp1d(exp_time, exp_tye, kind='linear', fill_value='extrapolate')
        interp_cy5_func = interp1d(exp_time, exp_cy5_adjusted, kind='linear', fill_value='extrapolate')

        curve_fam = interp_fam_func(standard_time_axis)
        curve_tye = interp_tye_func(standard_time_axis)
        curve_cy5 = interp_cy5_func(standard_time_axis)
    except ValueError as e:
        print(f"错误: 无法插值数据。这可能是因为 {data_path} 中的时间点不足。")
        print(f"错误信息: {e}")
        return None

    # --- f. 组合并返回 ---
    # axis=0 使形状为 (3, T)，与 gendata.m 和 model.py 一致
    X_sample_raw = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)

    print(f"真实实验数据已成功加载并转换为 (3, {num_time_points}) 格式。")
    return X_sample_raw


def predict_parameters():
    """
    加载模型和数据，执行并打印最终的参数预测。
    """

    # --- 1. 加载所有必要的工具 ---
    print(f"--- 正在加载工具 ---")
    try:
        # a. 加载 Config
        config = Config()
        
        # 从配置获取文件路径和参数
        INPUT_SIZE = config.get_input_size()
        OUTPUT_SIZE = config.get_output_size()
        MODEL_FILE = config.get_model_save_path()
        X_SCALER_FILE = config.get_x_scaler_file()
        Y_SCALER_FILE = config.get_y_scaler_file()
        DATA_FILE = config.get_experimental_data_path()

        # b. 加载 Scalers
        with open(X_SCALER_FILE, 'rb') as f:
            x_scaler = pickle.load(f)
        with open(Y_SCALER_FILE, 'rb') as f:
            y_scaler = pickle.load(f)

        # c. 加载参数名称
        param_names = config.get_trainable_param_names()

        # d. 加载模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = InverseMLP(INPUT_SIZE, OUTPUT_SIZE, config).to(device)
        
        # [修改 1] 增加 map_location 以防止在不同设备(CPU/GPU)间切换时报错
        model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
        model.eval()  # ** 必须 ** 设为评估模式

        print(f"成功加载: Config, Scalers, Model, ParamNames")
        print(f"预测设备: {device}")

    except Exception as e:
        print(f"*** 致命错误: 无法加载必要的工具文件 ***")
        print(f"请确保所有配置文件和模型文件都在此文件夹中。")
        print(f"错误信息: {e}")
        return

    # --- 2. 加载并处理真实的输入数据 ---
    X_sample_raw = load_real_experimental_data(config, DATA_FILE)

    if X_sample_raw is None:
        print("无法处理输入数据，预测中止。")
        return

    # --- 3. 预处理 (与训练时 完全一致) ---
    print("\n--- 2. 正在准备模型输入 ---")
    # a. 扁平化: (3, 7801) -> (1, 23403)
    X_sample_flat = X_sample_raw.reshape(1, -1)

    # b. 归一化 (使用 x_scaler)
    X_sample_scaled = x_scaler.transform(X_sample_flat)

    # c. 转换为 Tensor
    X_sample_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32).to(device)
    print(f"输入向量 (1, {X_sample_flat.shape[1]}) 已准备就绪。")

    # --- 4. 执行预测 ---
    print("\n--- 3. 正在执行模型预测 ---")
    with torch.no_grad():  # 预测时不需要计算梯度
        predicted_scaled_params = model(X_sample_tensor)  # (1, 7)

    # --- 5. 逆向转换 (最关键的一步) ---
    # [修改 2] 逻辑更新：先反归一化得到Log值，再反对数变换(10^x)得到物理值
    predicted_log_params = y_scaler.inverse_transform(
        predicted_scaled_params.cpu().numpy()
    )
    predicted_real_params = np.power(10, predicted_log_params)

    # --- 6. 打印最终结果 ---
    print("\n--- 4. 预测的物理参数 ---")
    print("=" * 30)

    for i in range(len(param_names)):
        name = param_names[i]
        pred_val = predicted_real_params[0, i]
        # [修改 3] 使用 .6e 科学计数法，方便查看跨度很大的物理量
        print(f"{name:<15}: {pred_val:<15.6e}")

    print("=" * 30)
    print("预测完成。")


if __name__ == "__main__":
    predict_parameters()
