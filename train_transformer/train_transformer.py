# coding=utf-8
"""
train_transformer.py — Transformer 主训练脚本
train_transformer.py — Transformer main training script

用法 / Usage (from train_transformer/ directory):
    python train_transformer.py           # 正式训练 / Full training
    python train_transformer.py --smoke   # 烟雾测试 / Smoke test (1 epoch)
"""

import argparse
import math
import os
import sys
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

# ── 路径设置：允许从任意工作目录运行 ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config_loader_transformer import load_configs
from dataset import load_and_preprocess_data_3d
from model_builder import build_transformer_model


# ─────────────────────────────────────────────────────────────────────────────
# 学习率调度：Cosine Annealing with Linear Warmup
# ─────────────────────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int,
                                    min_lr_ratio: float = 0.0):
    """
    前 warmup_steps 步线性升温，之后余弦退火到 min_lr_ratio × peak_lr。
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# 主训练函数
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_cli_path(path):
    """Resolve a CLI-provided path against the current working directory."""
    if path is None:
        return None
    return os.path.abspath(path)


def train(smoke_test: bool = False, parent_config_override=None, transformer_config_override=None):
    # ── 0. 加载配置 ──────────────────────────────────────────────────────────
    print("=" * 65)
    print("  DNA Walker Transformer Trainer")
    print("=" * 65)
    print("\n--- 0. 加载配置 ---")
    parent_config, transformer_config = load_configs(
        parent_override_file=_resolve_cli_path(parent_config_override),
        transformer_override_file=_resolve_cli_path(transformer_config_override)
    )

    OUTPUT_SIZE  = parent_config.get_output_size()
    BATCH_SIZE   = transformer_config.get_batch_size()
    NUM_EPOCHS   = 2 if smoke_test else transformer_config.get_num_epochs()
    LR           = transformer_config.get_learning_rate()
    WEIGHT_DECAY = transformer_config.get_weight_decay()
    WARMUP_RATIO = transformer_config.get_warmup_ratio()
    MIN_LR       = transformer_config.get_scheduler_min_lr()
    PATIENCE     = transformer_config.get_early_stopping_patience()
    MODEL_SAVE   = transformer_config.get_model_save_path()
    DATASET_PATH = transformer_config.get_dataset_path()

    print(f"  数据集路径    : {DATASET_PATH}")
    print(f"  模型保存路径  : {MODEL_SAVE}")
    print(f"  batch_size    : {BATCH_SIZE}")
    print(f"  num_epochs    : {NUM_EPOCHS}{'  (smoke test)' if smoke_test else ''}")
    print(f"  learning_rate : {LR}")
    print(f"  weight_decay  : {WEIGHT_DECAY}")

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data_3d(
        npz_filename=DATASET_PATH,
        batch_size=BATCH_SIZE,
        parent_config=parent_config,
        transformer_config=transformer_config
    )
    if train_loader is None:
        print("数据加载失败，退出。")
        return

    # ── 2. 初始化模型 ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- 2. 初始化模型 ---")
    print(f"  运行设备: {device}")

    model = build_transformer_model(parent_config, transformer_config).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数量: {n_params:,}")
    print(f"  Patch 数量: {model.num_patches}")

    # ── 3. 损失函数 / 优化器 / 调度器 ────────────────────────────────────────
    # 自适应加权 MSE (Adaptive Loss Weighting)
    # 初始等权重，随后根据验证集各参数的相对 MSE 自适应调整，
    # 哪个参数预测得最差（MSE最大），谁的权重就越高，自动平衡！
    initial_weights = [1.0] * OUTPUT_SIZE
    param_weights = torch.tensor(initial_weights, dtype=torch.float32, device=device)

    def weighted_mse_loss(pred, target):
        return torch.mean(param_weights * (pred - target) ** 2)

    criterion = weighted_mse_loss
    mse_fn    = nn.MSELoss(reduction='none')   # 用于逐参数统计
    print("  损失函数: 自适应加权 MSE Loss (Adaptive Loss Weighting)")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    total_steps   = NUM_EPOCHS * len(train_loader)
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    min_lr_ratio  = MIN_LR / LR if LR > 0 else 0.0

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps, min_lr_ratio
    )

    print(f"  优化器: AdamW  |  调度器: Cosine Warmup ({warmup_steps} steps warmup)")
    print()

    # ── 4. 训练循环 ──────────────────────────────────────────────────────────
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--- 3. 训练中 (即将开始训练，当前模型总参与训练参数量: {n_params:,}) ---")
    best_val_mse    = float('inf')
    epochs_no_impr  = 0
    train_start     = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()

        # ---- 训练阶段 ----
        model.train()
        total_train_loss = 0.0

        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            Y_pred = model(X_batch)
            loss   = criterion(Y_pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- 验证阶段 ----
        model.eval()
        total_val_loss   = 0.0
        val_mse_per_param = torch.zeros(OUTPUT_SIZE, device=device)
        val_sample_count  = 0

        with torch.no_grad():
            for X_v, Y_v in val_loader:
                X_v = X_v.to(device, non_blocking=True)
                Y_v = Y_v.to(device, non_blocking=True)

                Y_pred_v = model(X_v)
                total_val_loss += criterion(Y_pred_v, Y_v).item()

                batch_mse = mse_fn(Y_pred_v, Y_v)   # (batch, output_size)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count  += X_v.shape[0]

        avg_val_mse_per_p   = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse         = float(avg_val_mse_per_p.mean())
        current_lr          = optimizer.param_groups[0]['lr']
        epoch_time          = time.time() - epoch_start

        print(f"Epoch {epoch:04d}/{NUM_EPOCHS} | "
              f"Train: {avg_train_loss:.6f} | "
              f"Val MSE: {avg_val_mse:.6f} | "
              f"LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # --- 自适应损失加权 (Adaptive Loss Weighting) ---
        # 根据本轮验证集各参数的 MSE，动态调整下一轮训练的权重
        raw_weights = np.sqrt(avg_val_mse_per_p + 1e-6)
        normalized_weights = raw_weights / raw_weights.mean()
        new_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
        param_weights = 0.9 * param_weights + 0.1 * new_weights_tensor

        # 每 50 epoch 打印逐参数 MSE 和 当前权重
        if epoch % 50 == 0 or epoch == NUM_EPOCHS:
            mse_str = ", ".join(
                [f"{n}={v:.5f}" for n, v in zip(param_names, avg_val_mse_per_p)])
            weight_str = ", ".join(
                [f"{n}={v:.3f}" for n, v in zip(param_names, param_weights.cpu().numpy())])
            print(f"  [Per-Param MSE] {mse_str}")
            print(f"  [Current Weights] {weight_str}")

        # ---- 保存最佳模型 + Early Stopping ----
        if avg_val_mse < best_val_mse:
            best_val_mse   = avg_val_mse
            epochs_no_impr = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'val_mse'    : best_val_mse,
                'param_names': param_names,
            }, MODEL_SAVE)
            print(f"  -> 最佳模型已保存 (Val MSE: {best_val_mse:.6f})")
        else:
            epochs_no_impr += 1
            if PATIENCE > 0 and epochs_no_impr >= PATIENCE:
                print(f"\n*** Early Stopping: 连续 {PATIENCE} 轮无改善，停止训练 ***")
                print(f"*** 最佳 Val MSE: {best_val_mse:.6f} ***")
                break

    total_time = time.time() - train_start
    print(f"\n训练完成，耗时 {total_time/60:.1f} 分钟")

    # ── 5. 测试集评估 ────────────────────────────────────────────────────────
    print(f"\n--- 4. 测试集评估（加载最佳模型）---")
    if not os.path.exists(MODEL_SAVE):
        print("警告：未找到已保存的最佳模型文件，跳过测试评估。")
        return
    checkpoint = torch.load(MODEL_SAVE, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    print(f"  (最佳模型来自 Epoch {checkpoint['epoch']}, Val MSE={checkpoint['val_mse']:.6f})")
    model.eval()

    total_test_loss   = 0.0
    test_mse_per_param = torch.zeros(OUTPUT_SIZE, device=device)
    test_sample_count  = 0

    with torch.no_grad():
        for X_t, Y_t in test_loader:
            X_t = X_t.to(device, non_blocking=True)
            Y_t = Y_t.to(device, non_blocking=True)

            Y_pred_t = model(X_t)
            total_test_loss   += criterion(Y_pred_t, Y_t).item()

            batch_mse = mse_fn(Y_pred_t, Y_t)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count  += X_t.shape[0]

    avg_test_loss     = total_test_loss / len(test_loader)
    avg_test_mse_per  = (test_mse_per_param / test_sample_count).cpu().numpy()

    print(f"\n{'='*65}")
    print(f"  测试集 Loss (total MSE):  {avg_test_loss:.6f}")
    print(f"  逐参数 MSE (scaled space):")
    for name, mse_val in zip(param_names, avg_test_mse_per):
        bar = "█" * int(mse_val * 3000)
        print(f"    {name:20s}: {mse_val:.6f}  {bar}")
    print(f"{'='*65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNA Walker Transformer Trainer')
    parser.add_argument(
        '--config',
        help='Optional override INI layered on top of the repo root configfile.ini'
    )
    parser.add_argument(
        '--transformer-config',
        help='Optional override INI layered on top of train_transformer/config_transformer.ini'
    )
    parser.add_argument('--smoke', action='store_true',
                        help='仅运行 2 个 epoch 的烟雾测试，验证流程正常')
    args = parser.parse_args()
    train(
        smoke_test=args.smoke,
        parent_config_override=args.config,
        transformer_config_override=args.transformer_config
    )
