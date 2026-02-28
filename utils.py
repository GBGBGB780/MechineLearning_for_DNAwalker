# coding=utf-8
import numpy as np
import pickle
import torch
import os
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_and_preprocess_data(npz_filename, batch_size=64, config=None):
    """
    加载、预处理、拆分数据，并创建PyTorch DataLoaders。
    
    Args:
        npz_filename: 数据集文件路径
        batch_size: 批次大小
        config: Config对象 (必须提供)
    """
    print(f"--- 1. 加载和预处理数据 ---")
    
    if config is None:
        raise ValueError("Config object is required for data loading.")

    # 从配置读取参数名称
    param_names = config.get_trainable_param_names()
    print(f"从配置文件读取到 {len(param_names)} 个待训练参数: {param_names}")
    # --- 加载数据 ---
    try:
        dataset = np.load(npz_filename)
        X_data = dataset['X']
        Y_data = dataset['Y']
        print(f"成功加载 {npz_filename}。")
        print(f"原始 X 形状: {X_data.shape}, 原始 Y 形状: {Y_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载 {npz_filename}。请确保文件存在且未损坏。")
        print(f"错误信息: {e}")
        return None, None, None, None


    # 从config读取log_epsilon和需要log变换的参数
    log_epsilon = config.get_log_epsilon()
    log_transform_params = config.get_log_transform_params()
    
    # 选择性对数变换：仅对跨数量级的参数 (如 k0) 做 log10
    # 其余参数保持原值，避免 abs() 丢失符号信息
    for p in log_transform_params:
        if p in param_names:
            idx = param_names.index(p)
            Y_data[:, idx] = np.log10(Y_data[:, idx] + log_epsilon)
            print(f"  参数 '{p}' (列 {idx}) 已做 log10 变换")
    
    print(f"Y 数据变换完成 (log变换参数: {log_transform_params})")
    print(f"  Y 变换后的范围: [{Y_data.min():.4f}, {Y_data.max():.4f}]")
    # --- 预处理 X (输入) ---
    # 1. 扁平化: 将 (N, 3, 7801) 变为 (N, 23403)
    num_samples_original = X_data.shape[0]
    
    # --- 曲线振幅过滤 (在扁平化之前，用 3D 数据) ---
    if config.get_amplitude_filter_enabled():
        amp_thresholds = config.get_amplitude_thresholds()  # [FAM, TYE, CY5]
        channel_names = ['FAM', 'TYE', 'CY5']
        
        amp_mask = np.ones(num_samples_original, dtype=bool)
        for c in range(min(X_data.shape[1], len(amp_thresholds))):
            amplitudes = X_data[:, c, :].max(axis=1) - X_data[:, c, :].min(axis=1)
            ch_mask = amplitudes <= amp_thresholds[c]
            num_filtered = np.sum(~ch_mask & amp_mask)  # 本通道新过滤的数量
            amp_mask &= ch_mask
            print(f"  振幅过滤 {channel_names[c]}: 阈值={amp_thresholds[c]:.2f}, "
                  f"本通道过滤 {num_filtered} 个样本")
        
        num_amp_filtered = num_samples_original - np.sum(amp_mask)
        X_data = X_data[amp_mask]
        Y_data = Y_data[amp_mask]
        print(f"振幅过滤: 移除 {num_amp_filtered} 个狂暴样本 "
              f"({100*num_amp_filtered/num_samples_original:.1f}%), "
              f"剩余 {X_data.shape[0]} 个")
        num_samples_original = X_data.shape[0]
    
    X_flat = X_data.reshape(num_samples_original, -1)  # 形状变为 [N, 23403]
    print(f"X 压平后的形状: {X_flat.shape}")

    # --- [已修正] 使用更严格的过滤条件 ---
    print("正在查找并跳过包含 Inf/NaN 或极端值的“坏”样本...")

    # 从config读取阈值参数
    safe_threshold = config.get_safe_threshold()
    nan_replacement = config.get_nan_replacement_value()
    
    # 我们设置一个非常大但计算安全的阈值
    # np.abs(X_flat) < safe_threshold 会自动处理 Inf (返回 False)
    # 但我们仍然需要 np.nan_to_num 来处理 NaN，以防万一

    # (更简单的方法) 先替换 NaN，再检查阈值
    X_flat_no_nan = np.nan_to_num(X_flat, nan=nan_replacement)
    Y_data_no_nan = np.nan_to_num(Y_data, nan=nan_replacement)

    # 检查 X 和 Y 中的所有值是否都在安全范围内
    mask_x_good = (np.abs(X_flat_no_nan) < safe_threshold).all(axis=1)
    mask_y_good = (np.abs(Y_data_no_nan) < safe_threshold).all(axis=1)

    # 我们只保留 X 和 Y *都* 完好的行
    final_mask = mask_x_good & mask_y_good

    num_good = np.sum(final_mask)
    num_bad = num_samples_original - num_good

    if num_bad > 0:
        print(f"警告: 发现并跳过了 {num_bad} 个“坏”样本 (包含 Inf, NaN 或极端值)。")

    # 应用掩码
    X_clean = X_flat[final_mask]
    Y_clean = Y_data[final_mask]

    print(f"清理后的 X 形状: {X_clean.shape}, 清理后的 Y 形状: {Y_clean.shape}")

    if num_good == 0:
        print("错误: 数据集中没有剩余的有效数据！")
        return None, None, None, None
    # --- [修正结束] ---

    # --- 预处理 X (输入): 单样本联合通道归一化 (Domain Invariance) ---
    # 先还原为 (N, 3, 7801)
    # 我们不再使用全局归一化，因为实验环境数据的基准线和缩放可能会有偏差（Domain Shift）。
    # 但我们也不能像最开始那样对每条曲线独立进行归一化，这会抹除 FAM、TYE、CY5 之间的相对高度差异（这决定了物理参数）。
    # 解决方案：对“每个样本”计算它 三条曲线组合在一起 的总均值和总方差，然后整体缩放。
    # 这样既不受全局环境基线飘移的影响，又能完美保留三条曲线互相之间的相对高度和距离。
    num_channels = X_data.shape[1]   # 3
    X_clean_3d = X_clean.reshape(-1, num_channels, X_data.shape[2])  # (N, 3, 7801)
    
    # 沿着通道(axis=1)和时间点(axis=2)一起计算
    sample_means = np.nanmean(X_clean_3d, axis=(1, 2), keepdims=True)  # Shape: (N, 1, 1)
    sample_stds  = np.nanstd(X_clean_3d, axis=(1, 2), keepdims=True) + 1e-8 # Shape: (N, 1, 1)
    
    # 执行归一化
    X_clean_3d = (X_clean_3d - sample_means) / sample_stds
        
    X_scaled = X_clean_3d.reshape(X_clean.shape[0], -1)  # 展平回 (N, 23403)
    print("X 单样本联合通道归一化完成 (Domain Invariant)。")
    
    # 填充所有的潜在的 NaN 以防止训练报错 (安全机制)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    # --- 预处理 Y (标签) ---
    # Safe Sigmoid Lock策略: 缩放到 [0.1, 0.9] 避免 Sigmoid 两端的梯度死区
    y_scaler = MinMaxScaler(feature_range=(0.1, 0.9))
    Y_scaled = y_scaler.fit_transform(Y_clean)

    # --- 保存 scaler ---
    x_scaler_file = config.get_x_scaler_file()
    y_scaler_file = config.get_y_scaler_file()
    scaler_dir = os.path.dirname(y_scaler_file)
    if scaler_dir and not os.path.exists(scaler_dir):
        os.makedirs(scaler_dir, exist_ok=True)
    with open(x_scaler_file, 'wb') as f:
        pickle.dump(None, f)   # 采用单样本联合归一化，不再依赖全局 X Scaler
    with open(y_scaler_file, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"Scalers 已保存: {x_scaler_file} 和 {y_scaler_file}")

    # --- 数据格式转换以节省内存 ---
    # 将 float64 转为 float32：内存占用减半，防止 train_test_split 时报 OOM 错误
    # PyTorch 默认使用的也是 float32，不影响精度
    X_scaled = X_scaled.astype(np.float32)
    Y_scaled = Y_scaled.astype(np.float32)

    # --- 拆分数据集 ---
    # 从config读取拆分比例和随机种子
    test_ratio = config.get_test_split_ratio()
    val_ratio = config.get_val_split_ratio()
    random_seed = config.get_random_seed()
    
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=test_ratio, random_state=random_seed
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=val_ratio, random_state=random_seed
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"验证集大小: {X_val.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    # --- 转换为 PyTorch Tensors ---
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    Y_val_t = torch.tensor(Y_val, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    Y_test_t = torch.tensor(Y_test, dtype=torch.float32)

    # --- 创建 DataLoaders ---
    train_dataset = TensorDataset(X_train_t, Y_train_t)
    val_dataset = TensorDataset(X_val_t, Y_val_t)
    test_dataset = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("PyTorch DataLoaders 创建完毕。")
    print("----------------------------\n")

    return train_loader, val_loader, test_loader, param_names
