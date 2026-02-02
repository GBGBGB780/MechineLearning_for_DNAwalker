# coding=utf-8
import torch
import numpy as np
import pickle
import sys

# 从我们的本地文件中导入
try:
    from inference import NanorobotPredictor
except ImportError as e:
    print(f"错误: 找不到必要的模块: {e}")
    sys.exit(1)


def predict_from_new_data():
    print(f"--- 1. 初始化预测器 ---")
    try:
        predictor = NanorobotPredictor()
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # --- 2. 加载用于预测的数据 ---
    # *****************************************************************
    # * 在真实场景中:
    # * 您会在这里加载您的 'Fig3a_fitting.xlsx' 文件。
    # * 然后，您需要用和 generate_dataset.py 中完全相同的方法：
    # * 1. 提取 3 条曲线。
    # * 2. 将它们插值 (interpolate) 到 7801 个标准时间点。
    # * 3. 形状变为 (3, 7801)
    # *****************************************************************

    # --- 为了本示例，我们从 'training_dataset.npz' 中加载测试集数据 ---
    param_names = predictor.get_param_names()

    try:
        # 获取数据集路径 (可以使用 config 中的或者硬编码，这里假设在同目录下)
        # predictor.config.get_dataset_file() 也可以
        dataset_file = predictor.config.get_dataset_file()
             
        dataset = np.load(dataset_file)
        # (我们必须复现 train_mlp.py 中的数据拆分逻辑来找到测试集)
        # (为了简单起见，我们直接加载整个 X 并选择一个样本)
        X_data_raw = dataset['X']
        Y_data_raw = dataset['Y']

        # 选取一个样本 (例如，第 7 个)
        X_sample_raw = X_data_raw[7]  # 形状 (3, 7801)
        Y_sample_real = Y_data_raw[7]  # 形状 (7,)

    except Exception as e:
        print(f"错误: 无法加载数据: {e}")
        return

    print(f"\n--- 2. 准备预测样本 (样本 #7) ---")

    # --- 3. 进行预测 ---
    print("... 正在预测 ...")
    try:
        predicted_real_params = predictor.predict(X_sample_raw)
    except Exception as e:
        print(f"预测出错: {e}")
        return

    print("\n--- 3. 预测结果对比 ---")
    print(f"{'参数':<15} | {'真实值 (Y)':<15} | {'预测值 (Y_pred)':<15}")
    print("-" * 49)

    for i in range(len(param_names)):
        name = param_names[i]
        real_val = Y_sample_real[i]
        pred_val = predicted_real_params[0, i]
        # 使用科学计数法显示
        print(f"{name:<15} | {real_val:<15.6e} | {pred_val:<15.6e}")


if __name__ == "__main__":
    predict_from_new_data()