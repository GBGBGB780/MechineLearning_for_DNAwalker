import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def load_and_preprocess_data(npz_filename, batch_size=64):
    """
    加载、预处理、拆分数据，并创建PyTorch DataLoaders。
    """
    print(f"--- 1. 加载和预处理数据 ---")

    # --- 加载数据 ---
    try:
        dataset = np.load(npz_filename)
        X_data = dataset['X']
        Y_data = dataset['Y']
        param_names = dataset['parameter_names']
        print(f"成功加载 {npz_filename}。")
        print(f"原始 X 形状: {X_data.shape}, 原始 Y 形状: {Y_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载 {npz_filename}。请确保文件存在且未损坏。")
        print(f"错误信息: {e}")
        return None, None, None, None

    # --- 预处理 X (输入) ---
    # 1. 扁平化: 将 (N, 3, 100) 变为 (N, 300)
    num_samples = X_data.shape[0]
    X_flat = X_data.reshape(num_samples, -1)  # 形状变为 [N, 300]

    # 2. 归一化 (StandardScaler: 均值为0, 方差为1)
    x_scaler = StandardScaler()
    X_scaled = x_scaler.fit_transform(X_flat)

    # --- 预处理 Y (标签) ---
    # 1. 归一化 (MinMaxScaler: 范围为 [0, 1])
    # 这一步至关重要，因为您的7个参数尺度差异巨大 (例如 k0 vs E_b)
    y_scaler = MinMaxScaler()
    Y_scaled = y_scaler.fit_transform(Y_data)

    # --- 保存 Scaler (至关重要!) ---
    # 我们在 "预测" 阶段需要它们来“逆向”翻译结果
    with open('x_scaler.pkl', 'wb') as f:
        pickle.dump(x_scaler, f)
    with open('y_scaler.pkl', 'wb') as f:
        pickle.dump(y_scaler, f)
    print("X 和 Y 的归一化 'scalers' 已保存到 .pkl 文件。")

    # --- 拆分数据集 ---
    # 我们分为 训练集 (80%), 验证集 (10%), 测试集 (10%)
    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X_scaled, Y_scaled, test_size=0.1, random_state=42
    )
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.1111, random_state=42  # 0.1111 * 0.9 = 0.1
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