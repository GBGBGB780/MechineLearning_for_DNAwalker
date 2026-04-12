# coding=utf-8
"""
train_mlp.py — 1D-CNN (MLP) 模型训练脚本
train_mlp.py — 1D-CNN (MLP) model training script

使用自适应加权 MSE Loss 训练 InverseCNN 逆向模型。
Trains the InverseCNN inverse model with adaptive weighted MSE loss.

用法 / Usage:
    cd train_cnn/
    python train_mlp.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data_loader import load_and_preprocess_data
from model_cnn import InverseCNN
from config_loader import Config


def train():
    """
    完整训练流程：加载配置 → 加载数据 → 训练 → 测试评估。
    Full training pipeline: load config → load data → train → test evaluation.
    """
    # --- 0. 加载配置 / Load config ---
    print("--- 0. 加载配置文件 / Loading config ---")
    config = Config(config_file=os.path.join(_PARENT_DIR, 'configfile.ini'))

    INPUT_SIZE = config.get_input_size()
    OUTPUT_SIZE = config.get_output_size()
    LEARNING_RATE = config.get_learning_rate()
    BATCH_SIZE = config.get_batch_size()
    NUM_EPOCHS = config.get_num_epochs()
    DATASET_FILE = config.get_dataset_file()
    MODEL_SAVE_PATH = config.get_model_save_path()
    EARLY_STOPPING_PATIENCE = config.get_early_stopping_patience()

    # 确保输出目录存在 / Ensure output directory exists
    output_dir = os.path.dirname(MODEL_SAVE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"  输入维度/Input: {INPUT_SIZE}, 输出维度/Output: {OUTPUT_SIZE}")
    print(f"  学习率/LR: {LEARNING_RATE}, 批次/Batch: {BATCH_SIZE}, 轮数/Epochs: {NUM_EPOCHS}")
    print(f"  Early Stopping: {EARLY_STOPPING_PATIENCE}\n")

    # --- 1. 加载数据 / Load data ---
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data(
        DATASET_FILE, BATCH_SIZE, config
    )
    if train_loader is None:
        return

    # --- 2. 初始化模型 / Initialize model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 2. 开始训练 / Start training ---")
    print(f"设备 / Device: {device}")

    model = InverseCNN(INPUT_SIZE, OUTPUT_SIZE, config).to(device)

    # 自适应加权 MSE / Adaptive weighted MSE
    param_names = config.get_trainable_param_names()
    param_weights = torch.ones(OUTPUT_SIZE, dtype=torch.float32, device=device)

    def weighted_mse_loss(pred, target):
        return torch.mean(param_weights * (pred - target) ** 2)

    criterion = weighted_mse_loss
    print("损失函数 / Loss: 自适应加权 MSE / Adaptive Weighted MSE")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.get_scheduler_mode(),
        factor=config.get_scheduler_factor(),
        patience=config.get_scheduler_patience(),
        min_lr=config.get_scheduler_min_lr()
    )

    # --- 3. 训练循环 / Training loop ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    mse_fn = nn.MSELoss(reduction='none')

    for epoch in range(NUM_EPOCHS):
        # 训练阶段 / Training phase
        model.train()
        total_train_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证阶段 / Validation phase
        model.eval()
        val_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
        val_sample_count = 0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val, Y_val = X_val.to(device), Y_val.to(device)
                Y_pred_val = model(X_val)
                batch_mse = mse_fn(Y_pred_val, Y_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_val.shape[0]

        avg_val_mse_per_param = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = avg_val_mse_per_param.mean()
        scheduler.step(avg_val_mse)

        # 自适应损失加权 / Adaptive loss weighting
        raw_w = np.sqrt(avg_val_mse_per_param + 1e-6)
        norm_w = raw_w / raw_w.mean()
        param_weights = 0.9 * param_weights + 0.1 * torch.tensor(norm_w, dtype=torch.float32, device=device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train: {avg_train_loss:.6f} | "
              f"Val MSE: {avg_val_mse:.6f} | LR: {current_lr:.8f}")

        # 每 100 轮打印详情 / Print details every 100 epochs
        if (epoch + 1) % 100 == 0:
            mse_str = ", ".join([f"{n}={v:.6f}" for n, v in zip(param_names, avg_val_mse_per_param)])
            print(f"  [Per-Param MSE] {mse_str}")

        # Early Stopping
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 最佳模型已保存 / Best model saved (Val MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early Stopping: {EARLY_STOPPING_PATIENCE} 轮未改善 / epochs without improvement ***")
                break

    print("--- 训练完成 / Training complete ---")

    # --- 4. 测试评估 / Test evaluation ---
    print(f"\n--- 3. 测试集评估 / Test evaluation ---")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    test_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
    test_sample_count = 0
    with torch.no_grad():
        for X_test, Y_test in test_loader:
            X_test, Y_test = X_test.to(device), Y_test.to(device)
            batch_mse = mse_fn(model(X_test), Y_test)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count += X_test.shape[0]

    avg_test_mse = (test_mse_per_param / test_sample_count).cpu().numpy()
    print(f"\n{'='*60}")
    print(f"  逐参数 MSE / Per-Parameter MSE:")
    for name, mse_val in zip(param_names, avg_test_mse):
        print(f"    {name:20s}: {mse_val:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
