# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# 从我们的本地文件中导入
from utils import load_and_preprocess_data
from model import InverseCNN
from config_loader import Config


def train():
    # --- 1. 加载配置 ---
    print("--- 0. 加载配置文件 ---")
    config = Config()
    
    # 从配置文件读取所有超参数
    INPUT_SIZE = config.get_input_size()
    OUTPUT_SIZE = config.get_output_size()
    LEARNING_RATE = config.get_learning_rate()
    BATCH_SIZE = config.get_batch_size()
    NUM_EPOCHS = config.get_num_epochs()
    DATASET_FILE = config.get_dataset_file()
    MODEL_SAVE_PATH = config.get_model_save_path()
    EARLY_STOPPING_PATIENCE = config.get_early_stopping_patience()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(MODEL_SAVE_PATH)
    if not os.path.exists(output_dir):
        print(f"创建输出目录: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"配置已加载:")
    print(f"  - 输入维度: {INPUT_SIZE}")
    print(f"  - 输出维度: {OUTPUT_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
    print(f"  - Early Stopping 耐心值: {EARLY_STOPPING_PATIENCE}")
    print()
    
    # --- 2. 加载数据 ---
    # 注意：utils.py 中的函数 会自动帮我们处理好一切
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data(
        DATASET_FILE, BATCH_SIZE, config
    )

    if train_loader is None:
        return  # 数据加载失败

    # --- 3. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 2. 开始训练 ---")
    print(f"使用设备: {device}")

    # 传递config给模型，让模型从配置文件读取结构参数
    model = InverseCNN(INPUT_SIZE, OUTPUT_SIZE, config).to(device)

    # 损失函数: 标准 MSE
    criterion = nn.MSELoss()
    print("使用标准 MSE Loss")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 从配置读取学习率调度器参数
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=config.get_scheduler_mode(),
        factor=config.get_scheduler_factor(),
        patience=config.get_scheduler_patience(),
        min_lr=config.get_scheduler_min_lr()
    )

    # --- 4. 训练循环 ---
    best_val_loss = float('inf')
    epochs_no_improve = 0  # Early Stopping 计数器
    mse_fn = nn.MSELoss(reduction='none')  # 用于计算逐参数 MSE

    for epoch in range(NUM_EPOCHS):
        # --- 训练阶段 ---
        model.train()  # 将模型设为训练模式
        total_train_loss = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # 1. 前向传播
            Y_pred = model(X_batch)

            # 2. 计算损失
            loss = criterion(Y_pred, Y_batch)

            # 3. 反向传播和优化
            optimizer.zero_grad()  # 清空梯度
            loss.backward()        # 计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 防止梯度爆炸
            optimizer.step()       # 更新权重

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()  # 将模型设为评估模式
        total_val_loss = 0
        val_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
        val_sample_count = 0
        with torch.no_grad():  # 在验证时不需要计算梯度
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)

                Y_pred_val = model(X_batch_val)
                val_loss = criterion(Y_pred_val, Y_batch_val)
                total_val_loss += val_loss.item()
                
                # 累计逐参数 MSE
                batch_mse = mse_fn(Y_pred_val, Y_batch_val)  # (batch, output_size)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_batch_val.shape[0]

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_mse_per_param = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = avg_val_mse_per_param.mean()  # 纯 MSE 用于早停和调度
        scheduler.step(avg_val_mse)
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 打印时带上当前的学习率
        print(f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val MSE: {avg_val_mse:.6f} | LR: {current_lr:.8f}")
        
        # 每100轮打印逐参数 MSE
        if (epoch + 1) % 100 == 0:
            mse_str = ", ".join([f"{n}={v:.6f}" for n, v in zip(param_names, avg_val_mse_per_param)])
            print(f"  [Per-Param MSE] {mse_str}")

        # --- 保存最佳模型 + Early Stopping (基于纯 MSE) ---
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 新的最佳模型已保存到 {MODEL_SAVE_PATH} (Val MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early Stopping: 验证损失连续 {EARLY_STOPPING_PATIENCE} 轮未改善，提前停止训练 ***")
                print(f"*** 最佳验证损失: {best_val_loss:.6f} (在 Epoch {epoch + 1 - EARLY_STOPPING_PATIENCE}) ***")
                break

    print("--- 训练完成 ---")

    # --- 5. 最终测试 ---
    print(f"\n--- 3. 在测试集上评估最佳模型 ---")
    # 加载回表现最好的那个模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    total_test_loss = 0
    test_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
    test_sample_count = 0
    with torch.no_grad():
        for X_batch_test, Y_batch_test in test_loader:
            X_batch_test, Y_batch_test = X_batch_test.to(device), Y_batch_test.to(device)

            Y_pred_test = model(X_batch_test)
            test_loss = criterion(Y_pred_test, Y_batch_test)
            total_test_loss += test_loss.item()
            
            batch_mse = mse_fn(Y_pred_test, Y_batch_test)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count += X_batch_test.shape[0]

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_mse_per_param = (test_mse_per_param / test_sample_count).cpu().numpy()
    
    print(f"\n{'='*60}")
    print(f"  Test Loss (total): {avg_test_loss:.6f}")
    print(f"  Per-Parameter MSE (scaled space):")
    for name, mse_val in zip(param_names, avg_test_mse_per_param):
        print(f"    {name:20s}: {mse_val:.6f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()

