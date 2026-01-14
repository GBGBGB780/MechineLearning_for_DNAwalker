import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_preprocess_data(npz_filename, batch_size=64, config=None):
    """
    加载、预处理、拆分数据，并创建PyTorch DataLoaders。
    
    Args:
        npz_filename: 数据集文件路径
        batch_size: 批次大小
        config: Config对象，如果为None则使用硬编码值（向后兼容）
    """
    print(f"--- 1. 加载和预处理数据 ---")
    
    # 如果提供了config，从配置读取参数名称；否则使用硬编码
    if config is not None:
        param_names = config.get_trainable_param_names()
        print(f"从配置文件读取到 {len(param_names)} 个待训练参数: {param_names}")
    else:
        param_names = [
            'E_b',
            'E_b_azo_trans',
            'E_b_azo_cis',
            'k_mig',
            'k0',
            'drt_z',
            'drt_s'
        ]
        print("警告: 未提供config，使用硬编码的参数名称")
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


    # 从config读取log_epsilon，或使用默认值
    log_epsilon = config.get_log_epsilon() if config is not None else 1e-9
    # 加一个极小值防止 log(0) 报错
    Y_data = np.log10(np.abs(Y_data) + log_epsilon)
    # --- 预处理 X (输入) ---
    # 1. 扁平化: 将 (N, 3, 7801) 变为 (N, 23403)
    num_samples_original = X_data.shape[0]
    X_flat = X_data.reshape(num_samples_original, -1)  # 形状变为 [N, 23403]
    print(f"X 压平后的形状: {X_flat.shape}")

    # --- [已修正] 使用更严格的过滤条件 ---
    print("正在查找并跳过包含 Inf/NaN 或极端值的“坏”样本...")

    # 从config读取阈值参数，或使用默认值
    safe_threshold = config.get_safe_threshold() if config is not None else 1e20
    nan_replacement = config.get_nan_replacement_value() if config is not None else -1e30
    
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

    # 2. 归一化 (StandardScaler: 均值为0, 方差为1)
    x_scaler = StandardScaler()
    # [已修正] 在“干净”的数据上进行拟合
    X_scaled = x_scaler.fit_transform(X_clean)

    # --- 预处理 Y (标签) ---
    # 1. 归一化 (MinMaxScaler: 范围为 [0, 1])
    y_scaler = MinMaxScaler()
    # [已修正] 在“干净”的数据上进行拟合
    Y_scaled = y_scaler.fit_transform(Y_clean)

    # --- 保存 Scaler (至关重要!) ---
    # 从config读取scaler文件路径，或使用默认值
    x_scaler_file = config.get_x_scaler_file() if config is not None else 'x_scaler.pkl'
    y_scaler_file = config.get_y_scaler_file() if config is not None else 'y_scaler.pkl'
    
    with open(x_scaler_file, 'wb') as f:
        pickle.dump(x_scaler, f)
    with open(y_scaler_file, 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"X 和 Y 的归一化 'scalers' 已保存: {x_scaler_file}, {y_scaler_file}")

    # --- 拆分数据集 ---
    # 从config读取拆分比例和随机种子，或使用默认值
    test_ratio = config.get_test_split_ratio() if config is not None else 0.1
    val_ratio = config.get_val_split_ratio() if config is not None else 0.1111
    random_seed = config.get_random_seed() if config is not None else 42
    
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
