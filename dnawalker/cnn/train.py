# coding=utf-8
"""
dnawalker.cnn.train — 1D-CNN (MLP) 模型训练脚本
dnawalker.cnn.train — 1D-CNN (MLP) model training script

使用自适应加权 MSE Loss 训练 InverseCNN 逆向模型。
Trains the InverseCNN inverse model with adaptive weighted MSE loss.

用法 / Usage:
    python -m dnawalker.cnn.train
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import load_and_preprocess_data
from .model import InverseCNN
from .config import CNNConfig, resolve_artifact_path
from dnawalker.shared.artifacts import sha256_file
from dnawalker.shared.seeding import seed_everything
from dnawalker.shared.device import pick_device


def _resolve_cli_path(path):
    """Resolve a CLI-provided path against the current working directory."""
    if path is None:
        return None
    return os.path.abspath(path)


def train(config_override=None):
    """
    完整训练流程：加载配置 → 加载数据 → 训练 → 测试评估。
    Full training pipeline: load config → load data → train → test evaluation.
    """
    # --- 0. 加载配置 / Load config ---
    print("--- 0. 加载配置文件 / Loading config ---")
    override_files = [_resolve_cli_path(config_override)] if config_override else None
    config = CNNConfig(extra_config_files=override_files)

    INPUT_SIZE = config.get_input_size()
    OUTPUT_SIZE = config.get_output_size()
    LEARNING_RATE = config.get_learning_rate()
    BATCH_SIZE = config.get_batch_size()
    NUM_EPOCHS = config.get_num_epochs()
    DATASET_FILE = config.get_dataset_file()
    MODEL_SAVE_PATH = resolve_artifact_path(config.get_model_save_path())
    Y_SCALER_PATH = resolve_artifact_path(config.get_y_scaler_file())
    EARLY_STOPPING_PATIENCE = config.get_early_stopping_patience()

    # 确保输出目录存在 / Ensure output directory exists
    output_dir = os.path.dirname(MODEL_SAVE_PATH)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"  输入维度/Input: {INPUT_SIZE}, 输出维度/Output: {OUTPUT_SIZE}")
    print(f"  学习率/LR: {LEARNING_RATE}, 批次/Batch: {BATCH_SIZE}, 轮数/Epochs: {NUM_EPOCHS}")
    print(f"  Early Stopping: {EARLY_STOPPING_PATIENCE}\n")

    # 全局种子：必须在 train_test_split / DataLoader shuffle / 权重初始化 / Dropout
    # 之前设定，使 [TRAINING] random_seed 控制全部主要随机源 (而非仅数据划分)。
    seed = config.get_random_seed()
    seed_everything(seed)
    print(f"  随机种子 / Random seed: {seed} (固定随机源 / seeded run)")

    # --- 1. 加载数据 / Load data ---
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data(
        DATASET_FILE,
        BATCH_SIZE,
        config,
        y_scaler_file=Y_SCALER_PATH,
    )
    if train_loader is None:
        raise RuntimeError(f"Training data could not be loaded: {DATASET_FILE}")
    dataset_sha256 = sha256_file(DATASET_FILE)
    y_scaler_sha256 = sha256_file(Y_SCALER_PATH)
    split_metadata = {}
    split_manifest = config.get_split_manifest_file()
    if split_manifest is not None:
        split_metadata = {
            "split_manifest_sha256": sha256_file(split_manifest),
            "train_subset_size": int(config.get_train_subset_size()),
        }

    # --- 2. 初始化模型 / Initialize model ---
    # 复用共享的设备选择 (CUDA>MPS>CPU)，与推理适配器 (dnawalker.cnn.inference) 口径一致。
    device = pick_device()
    print("--- 2. 开始训练 / Start training ---")
    print(f"设备 / Device: {device}")

    model = InverseCNN(INPUT_SIZE, OUTPUT_SIZE, config).to(device)

    # 损失函数：由 [TRAINING] loss_weight_mode 控制 / Loss selected by config.
    #   'adaptive' (默认): 逐参数权重按验证集 MSE EMA 自适应 (历史行为)。
    #   'none'          : 标准等权 MSE (param_weights 恒为 1，下方自适应更新被跳过)。
    param_names = config.get_trainable_param_names()
    loss_weight_mode = config.get_loss_weight_mode()
    # 大小写无关比较：'none'/'None'/'NONE' 都关闭自适应加权 (其余值均视为 adaptive)。
    adaptive_weighting = (loss_weight_mode.lower() != 'none')
    param_weights = torch.ones(OUTPUT_SIZE, dtype=torch.float32, device=device)

    def weighted_mse_loss(pred, target):
        return torch.mean(param_weights * (pred - target) ** 2)

    criterion = weighted_mse_loss
    if adaptive_weighting:
        print(f"损失函数 / Loss: 自适应加权 MSE / Adaptive Weighted MSE "
              f"(loss_weight_mode={loss_weight_mode})")
    else:
        print("损失函数 / Loss: 标准等权 MSE / Unweighted MSE (loss_weight_mode=none)")

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
    checkpoint_saved = False
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

        # 自适应损失加权 / Adaptive loss weighting (仅 loss_weight_mode != none 时启用)
        if adaptive_weighting:
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
            torch.save({
                'epoch': int(epoch + 1),
                'model_state': model.state_dict(),
                'val_mse': float(best_val_loss),
                'param_names': list(param_names),
                'model_seed': int(config.get_random_seed()),
                'split_seed': int(config.get_split_seed()),
                'dataset_sha256': dataset_sha256,
                'y_scaler_sha256': y_scaler_sha256,
                **split_metadata,
            }, MODEL_SAVE_PATH)
            checkpoint_saved = True
            print(f"  -> 最佳模型已保存 / Best model saved (Val MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if EARLY_STOPPING_PATIENCE > 0 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n*** Early Stopping: {EARLY_STOPPING_PATIENCE} 轮未改善 / epochs without improvement ***")
                break

    print("--- 训练完成 / Training complete ---")
    if not checkpoint_saved:
        raise RuntimeError(
            "Training produced no finite validation checkpoint; refusing to "
            f"reuse a stale model at {MODEL_SAVE_PATH}"
        )

    # --- 4. 测试评估 / Test evaluation ---
    print("\n--- 3. 测试集评估 / Test evaluation ---")
    # map_location + weights_only: 跨设备可移植, 且仅反序列化权重 (安全)。
    checkpoint = torch.load(
        MODEL_SAVE_PATH, map_location=device, weights_only=True
    )
    state_dict = (
        checkpoint["model_state"]
        if isinstance(checkpoint, dict) and "model_state" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict)
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
    print("  逐参数 MSE / Per-Parameter MSE:")
    for name, mse_val in zip(param_names, avg_test_mse):
        print(f"    {name:20s}: {mse_val:.6f}")
    print(f"{'='*60}\n")


def main(argv=None):
    """Run the CNN training CLI."""
    parser = argparse.ArgumentParser(description="Train the CNN inverse model.")
    parser.add_argument(
        "--config",
        help="Optional override layered after common.ini and cnn.ini.",
    )
    args = parser.parse_args(argv)
    train(config_override=args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
