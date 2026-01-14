import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 从我们的本地文件中导入
from utils import load_and_preprocess_data
from model import InverseMLP
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
    
    print(f"配置已加载:")
    print(f"  - 输入维度: {INPUT_SIZE}")
    print(f"  - 输出维度: {OUTPUT_SIZE}")
    print(f"  - 学习率: {LEARNING_RATE}")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - 训练轮数: {NUM_EPOCHS}")
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
    model = InverseMLP(INPUT_SIZE, OUTPUT_SIZE, config).to(device)

    # 损失函数: 均方误差(MSE)，因为这是回归问题
    criterion = nn.MSELoss()

    # 优化器: Adam 是一个稳健的好选择
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
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # --- 验证阶段 ---
        model.eval()  # 将模型设为评估模式
        total_val_loss = 0
        with torch.no_grad():  # 在验证时不需要计算梯度
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)

                Y_pred_val = model(X_batch_val)
                val_loss = criterion(Y_pred_val, Y_batch_val)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 打印时带上当前的学习率
        print(f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.8f}")

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> 新的最佳模型已保存到 {MODEL_SAVE_PATH} (Val Loss: {avg_val_loss:.6f})")

    print("--- 训练完成 ---")

    # --- 5. 最终测试 ---
    print(f"\n--- 3. 在测试集上评估最佳模型 ---")
    # 加载回表现最好的那个模型
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()

    total_test_loss = 0
    with torch.no_grad():
        for X_batch_test, Y_batch_test in test_loader:
            X_batch_test, Y_batch_test = X_batch_test.to(device), Y_batch_test.to(device)

            Y_pred_test = model(X_batch_test)
            test_loss = criterion(Y_pred_test, Y_batch_test)
            total_test_loss += test_loss.item()

    avg_test_loss = total_test_loss / len(test_loader)
    print(f"最终测试集上的平均损失 (MSE): {avg_test_loss:.6f}")
    print("----------------------------------\n")


if __name__ == "__main__":
    train()
