import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 从我们的本地文件中导入
from utils import load_and_preprocess_data
from model import InverseMLP

# --- 1. 设置超参数 ---
INPUT_SIZE = 300  # 3 条曲线 * 100 个时间点
OUTPUT_SIZE = 7  # 7 个物理参数
LEARNING_RATE = 0.001
BATCH_SIZE = 64
NUM_EPOCHS = 100  # 您可以先从50-100开始，看损失曲线
DATASET_FILE = 'training_dataset.npz'
MODEL_SAVE_PATH = 'best_mlp_model.pth'


def train():
    # --- 2. 加载数据 ---
    # 注意：utils.py 中的函数 会自动帮我们处理好一切
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data(
        DATASET_FILE, BATCH_SIZE
    )

    if train_loader is None:
        return  # 数据加载失败

    # --- 3. 初始化模型、损失函数和优化器 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 2. 开始训练 ---")
    print(f"使用设备: {device}")

    model = InverseMLP(INPUT_SIZE, OUTPUT_SIZE).to(device)

    # 损失函数: 均方误差(MSE)，因为这是回归问题
    criterion = nn.MSELoss()

    # 优化器: Adam 是一个稳健的好选择
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

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

        print(f"Epoch {epoch + 1:03d}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

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