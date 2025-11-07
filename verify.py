import torch
import numpy as np
import pickle
import sys

# 从我们的本地文件中导入
from model import InverseMLP


def predict_from_new_data():
    print(f"--- 1. 加载训练好的模型和 Scalers ---")

    # --- 加载 Scalers ---
    try:
        with open('x_scaler.pkl', 'rb') as f:
            x_scaler = pickle.load(f)
        with open('y_scaler.pkl', 'rb') as f:
            y_scaler = pickle.load(f)
    except FileNotFoundError:
        print("错误: 找不到 'x_scaler.pkl' 或 'y_scaler.pkl'。")
        print("请先成功运行 train_mlp.py 来生成这些文件。")
        return

    # --- 加载模型 ---
    INPUT_SIZE = 300
    OUTPUT_SIZE = 7
    MODEL_SAVE_PATH = 'best_mlp_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InverseMLP(INPUT_SIZE, OUTPUT_SIZE).to(device)

    try:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
        model.eval()  # 设为评估模式
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{MODEL_SAVE_PATH}'。")
        print("请先成功运行 train_mlp.py。")
        return

    print(f"成功加载模型和 Scalers。")

    # --- 2. 加载用于预测的数据 ---
    # *****************************************************************
    # * 在真实场景中:
    # * 您会在这里加载您的 'Fig3a_fitting.xlsx' 文件。
    # * 然后，您需要用和 generate_dataset.py 中完全相同的方法：
    # * 1. 提取 3 条曲线。
    # * 2. 将它们插值 (interpolate) 到 100 个标准时间点。
    # * 3. 形状变为 (3, 100)
    # *****************************************************************

    # --- 为了本示例，我们从 'training_dataset.npz' 中加载测试集数据 ---
    try:
        dataset = np.load('training_dataset.npz')
        # (我们必须复现 train_mlp.py 中的数据拆分逻辑来找到测试集)
        # (为了简单起见，我们直接加载整个 X 并选择一个样本)
        X_data_raw = dataset['X']
        Y_data_raw = dataset['Y']
        param_names = dataset['parameter_names']

        # 选取一个样本 (例如，第 1000 个)
        X_sample_raw = X_data_raw[1000]  # 形状 (3, 100)
        Y_sample_real = Y_data_raw[1000]  # 形状 (7,)

    except Exception as e:
        print(f"错误: 无法加载 {dataset['__file__']}。{e}")
        return

    print(f"\n--- 2. 准备预测样本 (样本 #1000) ---")

    # --- 3. 预处理样本 (必须使用和训练时完全相同的 Scaler!) ---
    X_sample_flat = X_sample_raw.reshape(1, -1)  # (1, 300)
    X_sample_scaled = x_scaler.transform(X_sample_flat)  # 归一化
    X_sample_tensor = torch.tensor(X_sample_scaled, dtype=torch.float32).to(device)

    # --- 4. 进行预测 ---
    print("... 正在预测 ...")
    with torch.no_grad():
        predicted_scaled_params = model(X_sample_tensor)  # (1, 7)

    # --- 5. 逆向转换 (最重要!) ---
    # 将模型输出的 [0, 1] 范围的值，“解压缩”回真实的物理单位
    predicted_real_params = y_scaler.inverse_transform(predicted_scaled_params.cpu().numpy())

    print("\n--- 3. 预测结果对比 ---")
    print(f"{'参数':<15} | {'真实值 (Y)':<15} | {'预测值 (Y_pred)':<15}")
    print("-" * 49)

    for i in range(len(param_names)):
        name = param_names[i]
        real_val = Y_sample_real[i]
        pred_val = predicted_real_params[0, i]
        print(f"{name:<15} | {real_val:<15.6f} | {pred_val:<15.6f}")


if __name__ == "__main__":
    predict_from_new_data()