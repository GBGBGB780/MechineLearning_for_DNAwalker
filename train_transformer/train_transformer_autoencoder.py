# coding=utf-8
"""
train_transformer_autoencoder.py  —  Transformer 三阶段 Encoder-Decoder 训练脚本
=============================================================================
Phase 1: 独立训练 ForwardDecoder (数字孪生近似器)
         输入: 参数 Y(7) → 输出: 曲线 X(3×7801)
         Loss: MSE(X_pred, X_true)

Phase 2: 冻结 Decoder，训练 Transformer Encoder (DNAWalkerTransformer)
         X_3d → Encoder → Ŷ → [Frozen Decoder] → X̂
         Loss: α × Recon_MSE(X, X̂) + β × Adaptive_Param_MSE(Y, Ŷ)

Phase 3 (可选): 联合微调 Encoder + Decoder
         用更小的学习率端到端训练

★ HPC 断点续训 (Checkpoint/Resume):
  - 每 N 个 epoch 自动保存完整训练状态 (checkpoint)
  - 监控 wall-clock 时间，接近 HPC 时限前自动保存并安全退出
  - 支持 --resume 重新提交作业后从断点恢复
  - 所有训练状态 (模型权重/优化器/调度器/自适应权重/随机种子) 完整保存

用法:
    python train_transformer_autoencoder.py                     # 从头开始
    python train_transformer_autoencoder.py --resume            # 从上次断点恢复
    python train_transformer_autoencoder.py --max-hours 23      # 设置最大运行时间
    python train_transformer_autoencoder.py --resume --phase 2  # 强制跳到 Phase 2
    python train_transformer_autoencoder.py --smoke             # 烟雾测试
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

# ── 路径设置：允许从任意工作目录运行 / Path setup: allow running from any directory ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
_CNN_DIR = os.path.join(_PARENT_DIR, 'train_cnn')
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
if _CNN_DIR not in sys.path:
    sys.path.insert(0, _CNN_DIR)

from config_loader_transformer import load_configs
from config_loader import Config as ParentConfig
from dataset import load_and_preprocess_data_3d
from model_transformer import build_transformer
from model_cnn import ForwardDecoder  # CNN Decoder 位于 train_cnn/ / CNN Decoder in train_cnn/


# ─────────────────────────────────────────────────────────────────────────────
# 常量与默认参数 / Constants and default parameters
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT_FILENAME = 'results/transformer_autoencoder_checkpoint.pth'
DECODER_SAVE_FILENAME = 'results/best_transformer_decoder.pth'
ENCODER_SAVE_FILENAME = 'results/best_transformer_model.pth'

CHECKPOINT_EVERY_EPOCHS = 20   # 每 N epoch 强制保存一次 checkpoint
DEFAULT_MAX_HOURS = 23.0       # 默认最大运行小时数 (HPC 24h 限制，留 1h 余量)

# Phase 1 默认参数 / Phase 1 default parameters
P1_NUM_EPOCHS = 500
P1_LR = 0.001
P1_PATIENCE = 100

# Phase 2 默认参数 / Phase 2 default parameters
P2_ALPHA = 0.7          # 重构 Loss 权重
P2_BETA = 0.3           # 参数 Loss 权重

# Phase 3 默认参数 / Phase 3 default parameters
P3_ALPHA = 0.7
P3_BETA = 0.3
P3_NUM_EPOCHS = 200
P3_PATIENCE = 50


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数 / Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio=0.0):
    """前 warmup_steps 步线性升温，之后余弦退火。"""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def time_str(seconds):
    """将秒数格式化为 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def should_stop_for_time(start_time, max_hours):
    """检查是否应该因为接近时间限制而停止"""
    elapsed = time.time() - start_time
    max_seconds = max_hours * 3600
    remaining = max_seconds - elapsed
    # 留 15 分钟余量用于保存 checkpoint
    if remaining < 900:
        return True, elapsed, remaining
    return False, elapsed, remaining


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint 管理 / Checkpoint management
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, phase, epoch, decoder, encoder, optimizer, scheduler,
                    best_val_loss, epochs_no_improve, param_weights,
                    total_wall_time, extra=None):
    """
    保存完整的训练状态，支持任意阶段的断点恢复。
    """
    checkpoint = {
        'phase': phase,
        'epoch': epoch,
        'decoder_state': decoder.state_dict(),
        'encoder_state': encoder.state_dict() if encoder is not None else None,
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve,
        'param_weights': param_weights.cpu() if param_weights is not None else None,
        'total_wall_time': total_wall_time,
    }
    if extra:
        checkpoint.update(extra)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    # 先写临时文件再重命名，确保原子性（防止写到一半断电）
    tmp_path = path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, path)
    print(f"  💾 Checkpoint 已保存: phase={phase}, epoch={epoch}, "
          f"wall_time={time_str(total_wall_time)}")


def load_checkpoint(path, decoder, encoder, device):
    """
    加载 checkpoint 并恢复状态。
    返回 checkpoint dict（调用者负责恢复 optimizer/scheduler/weights）。
    """
    print(f"\n📂 加载 checkpoint: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)

    decoder.load_state_dict(ckpt['decoder_state'])
    if ckpt['encoder_state'] is not None and encoder is not None:
        encoder.load_state_dict(ckpt['encoder_state'])

    print(f"  恢复自: Phase {ckpt['phase']}, Epoch {ckpt['epoch']}, "
          f"Best Loss: {ckpt['best_val_loss']:.6f}, "
          f"已运行: {time_str(ckpt['total_wall_time'])}")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: 训练 ForwardDecoder / Phase 1: Train ForwardDecoder
# ─────────────────────────────────────────────────────────────────────────────

def train_phase1_decoder(decoder, train_loader, val_loader, device,
                         num_epochs, lr, patience, save_path,
                         start_epoch=0, best_val_loss=float('inf'),
                         epochs_no_improve=0,
                         optimizer_state=None, scheduler_state=None,
                         global_start_time=None, max_hours=DEFAULT_MAX_HOURS,
                         prev_wall_time=0.0, checkpoint_path=None,
                         encoder=None):
    """
    Phase 1: 独立训练 ForwardDecoder
    DataLoader 中 (X_3d, Y): X_3d=(B,3,7801), Y=(B,7)
    Decoder: Y(B,7) → X_flat(B,23403)
    """
    print("=" * 70)
    print("  PHASE 1: 训练 ForwardDecoder (数字孪生近似器)")
    print("=" * 70)
    if start_epoch > 0:
        print(f"  ⏩ 从 Epoch {start_epoch + 1} 恢复训练")
    print(f"  参数量: {count_parameters(decoder):,}")
    print(f"  学习率: {lr}, 最大轮数: {num_epochs}, Early Stopping 耐心: {patience}")
    print()

    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6
    )
    # 恢复 optimizer/scheduler 状态
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)

    criterion = nn.MSELoss()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"  [即将开始 Phase 1 训练] 当前 Decoder 参与训练参数量: {count_parameters(decoder):,}")
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 检查时间限制 ----
        if global_start_time is not None:
            should_stop, elapsed, remaining = should_stop_for_time(global_start_time, max_hours)
            if should_stop:
                wall_time = prev_wall_time + (time.time() - global_start_time)
                save_checkpoint(
                    checkpoint_path, phase=1, epoch=epoch,
                    decoder=decoder, encoder=encoder,
                    optimizer=optimizer, scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    epochs_no_improve=epochs_no_improve,
                    param_weights=None,
                    total_wall_time=wall_time,
                )
                print(f"\n⏰ 接近 HPC 时间限制 ({max_hours}h)，已安全保存退出。")
                print(f"   已完成: Phase 1, Epoch {epoch}/{num_epochs}")
                print(f"   请重新提交作业，使用 --resume 恢复训练。")
                return best_val_loss, True  # True = 因时间停止

        # ---- 训练阶段 ----
        decoder.train()
        total_train_loss = 0

        for X_3d_batch, Y_batch in train_loader:
            X_3d_batch = X_3d_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            # Decoder: Y(B,7) → X_flat(B,23403)
            X_recon_flat = decoder(Y_batch)

            # 展平真实 X 做对比
            X_true_flat = X_3d_batch.view(X_3d_batch.size(0), -1)  # (B, 23403)
            loss = criterion(X_recon_flat, X_true_flat)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- 验证阶段 ----
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_3d_val, Y_val in val_loader:
                X_3d_val = X_3d_val.to(device, non_blocking=True)
                Y_val = Y_val.to(device, non_blocking=True)
                X_recon_val = decoder(Y_val)
                X_true_val = X_3d_val.view(X_3d_val.size(0), -1)
                total_val_loss += criterion(X_recon_val, X_true_val).item()

        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        epoch_time = time.time() - epoch_start

        print(f"[Phase1] Epoch {epoch + 1:04d}/{num_epochs} | "
              f"Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # ---- Early Stopping ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(decoder.state_dict(), save_path)
            print(f"  -> 新的最佳 Decoder 已保存 (Val Loss: {avg_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n*** Phase 1 Early Stopping: 连续 {patience} 轮未改善 ***")
                print(f"*** 最佳 Decoder Val Loss: {best_val_loss:.6f} ***")
                break

        # ---- 定期保存 checkpoint ----
        if checkpoint_path and (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            wall_time = prev_wall_time + (time.time() - global_start_time) if global_start_time else 0
            save_checkpoint(
                checkpoint_path, phase=1, epoch=epoch + 1,
                decoder=decoder, encoder=encoder,
                optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                epochs_no_improve=epochs_no_improve,
                param_weights=None,
                total_wall_time=wall_time,
            )

    # 加载最佳权重
    if os.path.exists(save_path):
        decoder.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
    print(f"\nPhase 1 完成! 最佳 Decoder Val MSE: {best_val_loss:.6f}\n")
    return best_val_loss, False  # False = 正常结束


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: 冻结 Decoder，训练 Transformer Encoder / Phase 2: Freeze Decoder, Train Transformer Encoder
# ─────────────────────────────────────────────────────────────────────────────

def train_phase2_encoder(encoder, decoder, train_loader, val_loader,
                         parent_config, transformer_config, device,
                         alpha, beta, num_epochs, patience,
                         encoder_save_path,
                         start_epoch=0, best_val_loss=float('inf'),
                         epochs_no_improve=0, param_weights_init=None,
                         optimizer_state=None, scheduler_state=None,
                         global_start_time=None, max_hours=DEFAULT_MAX_HOURS,
                         prev_wall_time=0.0, checkpoint_path=None,
                         scheduler_step_offset=0):
    """
    Phase 2: 冻结 Decoder，训练 Transformer Encoder
    X_3d(B,3,7801) → Transformer → Ŷ(B,7) → [Frozen Decoder] → X̂_flat(B,23403)
    Loss = α × Recon_MSE(X, X̂) + β × Adaptive_Param_MSE(Y, Ŷ)
    """
    print("=" * 70)
    print("  PHASE 2: 冻结 Decoder，训练 Transformer Encoder")
    print("=" * 70)
    if start_epoch > 0:
        print(f"  ⏩ 从 Epoch {start_epoch + 1} 恢复训练")
    print(f"  Loss 配比: α(重构)={alpha}, β(参数)={beta}")
    print(f"  Encoder 参数量: {count_parameters(encoder):,}")
    print()

    # 冻结 Decoder
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    print("  Decoder 已冻结 (不参与梯度更新)")

    OUTPUT_SIZE = parent_config.get_output_size()
    param_names = parent_config.get_trainable_param_names()
    NUM_CHANNELS = parent_config.get_num_curves()
    SEQ_LEN = parent_config.get_seq_length()

    LR = transformer_config.get_learning_rate()
    WEIGHT_DECAY = transformer_config.get_weight_decay()
    WARMUP_RATIO = transformer_config.get_warmup_ratio()
    MIN_LR = transformer_config.get_scheduler_min_lr()

    # 自适应加权 MSE
    if param_weights_init is not None:
        param_weights = param_weights_init.to(device)
    else:
        param_weights = torch.ones(OUTPUT_SIZE, dtype=torch.float32, device=device)

    recon_criterion = nn.MSELoss()
    mse_fn = nn.MSELoss(reduction='none')

    # 优化器: AdamW (Transformer 标配)
    optimizer = optim.AdamW(
        encoder.parameters(),
        lr=LR, weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999), eps=1e-8
    )

    # Cosine Warmup 调度器
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    min_lr_ratio = MIN_LR / LR if LR > 0 else 0.0
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)

    # 恢复状态
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    # 如果从断点恢复，需要快进 scheduler 到正确的 step
    if start_epoch > 0 and scheduler_state is None:
        for _ in range(start_epoch * len(train_loader)):
            scheduler.step()

    print(f"  优化器: AdamW | 调度器: Cosine Warmup ({warmup_steps} steps)")
    print()

    best_val_loss = best_val_loss
    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)

    print(f"  [即将开始 Phase 2 训练] 当前 Transformer Encoder 参与训练参数量: {count_parameters(encoder):,}")
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 检查时间限制 ----
        if global_start_time is not None:
            should_stop, elapsed, remaining = should_stop_for_time(global_start_time, max_hours)
            if should_stop:
                wall_time = prev_wall_time + (time.time() - global_start_time)
                save_checkpoint(
                    checkpoint_path, phase=2, epoch=epoch,
                    decoder=decoder, encoder=encoder,
                    optimizer=optimizer, scheduler=scheduler,
                    best_val_loss=best_val_loss,
                    epochs_no_improve=epochs_no_improve,
                    param_weights=param_weights,
                    total_wall_time=wall_time,
                )
                print(f"\n⏰ 接近 HPC 时间限制 ({max_hours}h)，已安全保存退出。")
                print(f"   已完成: Phase 2, Epoch {epoch}/{num_epochs}")
                print(f"   请重新提交作业，使用 --resume 恢复训练。")
                return best_val_loss, True

        # ---- 训练阶段 ----
        encoder.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_param_loss = 0

        for X_3d_batch, Y_batch in train_loader:
            X_3d_batch = X_3d_batch.to(device, non_blocking=True)  # (B, 3, 7801)
            Y_batch = Y_batch.to(device, non_blocking=True)         # (B, 7)

            # 1. Encoder: X_3d → Ŷ
            Y_pred = encoder(X_3d_batch)                            # (B, 7)

            # 2. Decoder (冻结): Ŷ → X̂_flat
            X_recon_flat = decoder(Y_pred)                          # (B, 23403)

            # 3. 展平真实 X 用于重构 Loss
            X_true_flat = X_3d_batch.view(X_3d_batch.size(0), -1)  # (B, 23403)

            # 4. 计算混合 Loss
            loss_recon = recon_criterion(X_recon_flat, X_true_flat)
            loss_param = torch.mean(param_weights * (Y_pred - Y_batch) ** 2)
            loss = alpha * loss_recon + beta * loss_param

            # 5. 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_param_loss += loss_param.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_param_loss = total_param_loss / len(train_loader)

        # ---- 验证阶段 ----
        encoder.eval()
        total_val_loss = 0
        total_val_recon = 0
        total_val_param = 0
        val_mse_per_param = torch.zeros(OUTPUT_SIZE, device=device)
        val_sample_count = 0

        with torch.no_grad():
            for X_3d_val, Y_val in val_loader:
                X_3d_val = X_3d_val.to(device, non_blocking=True)
                Y_val = Y_val.to(device, non_blocking=True)

                Y_pred_val = encoder(X_3d_val)
                X_recon_val = decoder(Y_pred_val)
                X_true_val = X_3d_val.view(X_3d_val.size(0), -1)

                val_loss_recon = recon_criterion(X_recon_val, X_true_val)
                val_loss_param = torch.mean(param_weights * (Y_pred_val - Y_val) ** 2)
                val_loss = alpha * val_loss_recon + beta * val_loss_param

                total_val_loss += val_loss.item()
                total_val_recon += val_loss_recon.item()
                total_val_param += val_loss_param.item()

                batch_mse = mse_fn(Y_pred_val, Y_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_3d_val.shape[0]

        avg_val_mse_per_p = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = float(avg_val_mse_per_p.mean())
        avg_val_recon = total_val_recon / len(val_loader)
        avg_val_param = total_val_param / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print(f"[Phase2] Epoch {epoch + 1:04d}/{num_epochs} | "
              f"Total: {avg_train_loss:.6f} | Recon: {avg_recon_loss:.6f} | "
              f"Param: {avg_param_loss:.6f} | Val MSE: {avg_val_mse:.6f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        # ---- 自适应损失加权 ----
        raw_weights = np.sqrt(avg_val_mse_per_p + 1e-6)
        normalized_weights = raw_weights / raw_weights.mean()
        new_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
        param_weights = 0.9 * param_weights + 0.1 * new_weights_tensor

        # 每 50 轮打印详情
        if (epoch + 1) % 50 == 0 or epoch + 1 == num_epochs:
            mse_str = ", ".join([f"{n}={v:.6f}" for n, v in zip(param_names, avg_val_mse_per_p)])
            weight_str = ", ".join([f"{n}={v:.3f}" for n, v in zip(param_names, param_weights.cpu().numpy())])
            print(f"  [Per-Param MSE]     {mse_str}")
            print(f"  [Adaptive Weights]  {weight_str}")
            print(f"  [Val Recon MSE]     {avg_val_recon:.6f}")

        # ---- Early Stopping (基于纯参数 MSE) ----
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state': encoder.state_dict(),
                'val_mse': best_val_loss,
                'param_names': param_names,
            }, encoder_save_path)
            print(f"  -> 新的最佳 Encoder 已保存 (Val Param MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n*** Phase 2 Early Stopping: 连续 {patience} 轮未改善 ***")
                print(f"*** 最佳 Val Param MSE: {best_val_loss:.6f} ***")
                break

        # ---- 定期保存 checkpoint ----
        if checkpoint_path and (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            wall_time = prev_wall_time + (time.time() - global_start_time) if global_start_time else 0
            save_checkpoint(
                checkpoint_path, phase=2, epoch=epoch + 1,
                decoder=decoder, encoder=encoder,
                optimizer=optimizer, scheduler=scheduler,
                best_val_loss=best_val_loss,
                epochs_no_improve=epochs_no_improve,
                param_weights=param_weights,
                total_wall_time=wall_time,
            )

    # 加载最佳 Encoder
    if os.path.exists(encoder_save_path):
        ckpt_best = torch.load(encoder_save_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt_best['model_state'])
    print(f"\nPhase 2 完成! 最佳 Encoder Val Param MSE: {best_val_loss:.6f}\n")
    return best_val_loss, False


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 (可选): 联合微调 / Phase 3 (Optional): Joint fine-tuning
# ─────────────────────────────────────────────────────────────────────────────

def train_phase3_joint(encoder, decoder, train_loader, val_loader,
                       parent_config, transformer_config, device,
                       alpha, beta, num_epochs, patience,
                       encoder_save_path, decoder_save_path,
                       start_epoch=0, best_val_loss=float('inf'),
                       epochs_no_improve=0,
                       optimizer_state=None, scheduler_state=None,
                       global_start_time=None, max_hours=DEFAULT_MAX_HOURS,
                       prev_wall_time=0.0, checkpoint_path=None):
    """
    Phase 3 (可选): 联合微调 Encoder + Decoder
    """
    print("=" * 70)
    print("  PHASE 3: 联合微调 Encoder + Decoder")
    print("=" * 70)
    if start_epoch > 0:
        print(f"  ⏩ 从 Epoch {start_epoch + 1} 恢复训练")

    # 解冻 Decoder
    for param in decoder.parameters():
        param.requires_grad = True
    print("  Decoder 已解冻")

    OUTPUT_SIZE = parent_config.get_output_size()
    param_names = parent_config.get_trainable_param_names()

    # 用更小的学习率
    joint_lr = transformer_config.get_learning_rate() * 0.1
    print(f"  联合学习率: {joint_lr}")

    optimizer = optim.AdamW([
        {'params': encoder.parameters(), 'lr': joint_lr},
        {'params': decoder.parameters(), 'lr': joint_lr * 0.5},
    ], weight_decay=transformer_config.get_weight_decay())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    if scheduler_state is not None:
        sched.load_state_dict(scheduler_state)

    recon_criterion = nn.MSELoss()
    mse_fn = nn.MSELoss(reduction='none')

    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)

    total_trainable = count_parameters(encoder) + count_parameters(decoder)
    print(f"  [即将开始 Phase 3 训练] 当前联合微调参与训练参数量: {total_trainable:,}")
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()

        # ---- 检查时间限制 ----
        if global_start_time is not None:
            should_stop, _, _ = should_stop_for_time(global_start_time, max_hours)
            if should_stop:
                wall_time = prev_wall_time + (time.time() - global_start_time)
                save_checkpoint(
                    checkpoint_path, phase=3, epoch=epoch,
                    decoder=decoder, encoder=encoder,
                    optimizer=optimizer, scheduler=sched,
                    best_val_loss=best_val_loss,
                    epochs_no_improve=epochs_no_improve,
                    param_weights=None,
                    total_wall_time=wall_time,
                )
                print(f"\n⏰ 接近 HPC 时间限制 ({max_hours}h)，已安全保存退出。")
                print(f"   已完成: Phase 3, Epoch {epoch}/{num_epochs}")
                return best_val_loss, True

        # ---- 训练阶段 ----
        encoder.train()
        decoder.train()
        total_train_loss = 0

        for X_3d_batch, Y_batch in train_loader:
            X_3d_batch = X_3d_batch.to(device, non_blocking=True)
            Y_batch = Y_batch.to(device, non_blocking=True)

            Y_pred = encoder(X_3d_batch)
            X_recon_flat = decoder(Y_pred)
            X_true_flat = X_3d_batch.view(X_3d_batch.size(0), -1)

            loss_recon = recon_criterion(X_recon_flat, X_true_flat)
            loss_param = nn.functional.mse_loss(Y_pred, Y_batch)
            loss = alpha * loss_recon + beta * loss_param

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ---- 验证阶段 ----
        encoder.eval()
        decoder.eval()
        val_mse_per_param = torch.zeros(OUTPUT_SIZE, device=device)
        val_sample_count = 0
        total_val_recon = 0

        with torch.no_grad():
            for X_3d_val, Y_val in val_loader:
                X_3d_val = X_3d_val.to(device, non_blocking=True)
                Y_val = Y_val.to(device, non_blocking=True)
                Y_pred_val = encoder(X_3d_val)
                X_recon_val = decoder(Y_pred_val)
                X_true_val = X_3d_val.view(X_3d_val.size(0), -1)

                total_val_recon += recon_criterion(X_recon_val, X_true_val).item()
                batch_mse = mse_fn(Y_pred_val, Y_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_3d_val.shape[0]

        avg_val_mse_per_p = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = float(avg_val_mse_per_p.mean())
        avg_val_recon = total_val_recon / len(val_loader)

        sched.step(avg_val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - epoch_start

        print(f"[Phase3] Epoch {epoch + 1:04d}/{num_epochs} | "
              f"Train: {avg_train_loss:.6f} | Val Param MSE: {avg_val_mse:.6f} | "
              f"Val Recon: {avg_val_recon:.6f} | LR: {current_lr:.2e} | "
              f"Time: {epoch_time:.1f}s")

        # ---- Early Stopping ----
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state': encoder.state_dict(),
                'val_mse': best_val_loss,
                'param_names': param_names,
            }, encoder_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            print(f"  -> 联合模型已保存 (Val Param MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"\n*** Phase 3 Early Stopping: 连续 {patience} 轮未改善 ***")
                break

        # ---- 定期保存 checkpoint ----
        if checkpoint_path and (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0:
            wall_time = prev_wall_time + (time.time() - global_start_time) if global_start_time else 0
            save_checkpoint(
                checkpoint_path, phase=3, epoch=epoch + 1,
                decoder=decoder, encoder=encoder,
                optimizer=optimizer, scheduler=sched,
                best_val_loss=best_val_loss,
                epochs_no_improve=epochs_no_improve,
                param_weights=None,
                total_wall_time=wall_time,
            )

    if os.path.exists(encoder_save_path):
        ckpt_best = torch.load(encoder_save_path, map_location=device, weights_only=False)
        encoder.load_state_dict(ckpt_best['model_state'])
    if os.path.exists(decoder_save_path):
        decoder.load_state_dict(torch.load(decoder_save_path, map_location=device, weights_only=False))
    print(f"\nPhase 3 完成! 最佳 Val Param MSE: {best_val_loss:.6f}\n")
    return best_val_loss, False


# ─────────────────────────────────────────────────────────────────────────────
# 测试集评估
# ─────────────────────────────────────────────────────────────────────────────

def final_test(encoder, decoder, test_loader, parent_config, device, param_names):
    """在测试集上评估最终模型"""
    print("=" * 70)
    print("  最终测试集评估")
    print("=" * 70)

    OUTPUT_SIZE = parent_config.get_output_size()

    encoder.eval()
    decoder.eval()

    mse_fn = nn.MSELoss(reduction='none')
    recon_criterion = nn.MSELoss()

    test_mse_per_param = torch.zeros(OUTPUT_SIZE, device=device)
    test_sample_count = 0
    total_test_recon = 0

    with torch.no_grad():
        for X_3d_test, Y_test in test_loader:
            X_3d_test = X_3d_test.to(device, non_blocking=True)
            Y_test = Y_test.to(device, non_blocking=True)

            Y_pred_test = encoder(X_3d_test)
            X_recon_test = decoder(Y_pred_test)
            X_true_test = X_3d_test.view(X_3d_test.size(0), -1)

            batch_mse = mse_fn(Y_pred_test, Y_test)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count += X_3d_test.shape[0]
            total_test_recon += recon_criterion(X_recon_test, X_true_test).item()

    avg_test_mse_per_p = (test_mse_per_param / test_sample_count).cpu().numpy()
    avg_test_recon = total_test_recon / len(test_loader)

    print(f"\n{'=' * 65}")
    print(f"  Test Reconstruction MSE: {avg_test_recon:.6f}")
    print(f"  Per-Parameter MSE (scaled space):")
    for name, mse_val in zip(param_names, avg_test_mse_per_p):
        bar = "█" * int(mse_val * 3000)
        print(f"    {name:20s}: {mse_val:.6f}  {bar}")
    print(f"  Average Param MSE: {avg_test_mse_per_p.mean():.6f}")
    print(f"{'=' * 65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# 主函数
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='DNA Walker Transformer 三阶段 Encoder-Decoder 训练 (支持断点续训)'
    )
    parser.add_argument('--resume', action='store_true',
                        help='从上次 checkpoint 恢复训练')
    parser.add_argument('--phase', type=int, default=0,
                        help='强制从指定阶段开始 (1/2/3)，0=自动')
    parser.add_argument('--max-hours', type=float, default=DEFAULT_MAX_HOURS,
                        help=f'最大运行小时数 (默认 {DEFAULT_MAX_HOURS}h)')
    parser.add_argument('--smoke', action='store_true',
                        help='烟雾测试: 每阶段仅 2 epoch')
    parser.add_argument('--enable-phase3', action='store_true',
                        help='启用 Phase 3 联合微调 (默认只跑 Phase 1+2)')
    parser.add_argument('--checkpoint-every', type=int, default=CHECKPOINT_EVERY_EPOCHS,
                        help=f'每 N epoch 保存一次 checkpoint (默认 {CHECKPOINT_EVERY_EPOCHS})')
    parser.add_argument('--pretrained-encoder', type=str, default=None,
                        help='预训练 Encoder 权重路径 (由 train_transformer.py 训练，'
                             '例如 results/best_transformer_model.pth)')
    args = parser.parse_args()

    global CHECKPOINT_EVERY_EPOCHS
    CHECKPOINT_EVERY_EPOCHS = args.checkpoint_every

    global_start_time = time.time()

    # ============================================================
    # 0. 加载配置
    # ============================================================
    print("=" * 70)
    print("  Transformer 三阶段 Encoder-Decoder 训练系统")
    print("  (支持 HPC 断点续训)")
    print("=" * 70)
    print(f"  最大运行时间: {args.max_hours}h")
    print(f"  Checkpoint 频率: 每 {CHECKPOINT_EVERY_EPOCHS} epoch")
    print()

    parent_config, transformer_config = load_configs()

    OUTPUT_SIZE = parent_config.get_output_size()
    BATCH_SIZE = transformer_config.get_batch_size()
    DATASET_PATH = transformer_config.get_dataset_path()
    param_names = parent_config.get_trainable_param_names()

    # 路径设置
    checkpoint_path = os.path.join(_THIS_DIR, CHECKPOINT_FILENAME)
    decoder_save_path = os.path.join(_THIS_DIR, DECODER_SAVE_FILENAME)
    encoder_save_path = os.path.join(_THIS_DIR, ENCODER_SAVE_FILENAME)

    print(f"  Encoder 保存路径:    {encoder_save_path}")
    print(f"  Decoder 保存路径:    {decoder_save_path}")
    print(f"  Checkpoint 路径:     {checkpoint_path}")
    print()

    # ============================================================
    # 1. 加载数据
    # ============================================================
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data_3d(
        npz_filename=DATASET_PATH,
        batch_size=BATCH_SIZE,
        parent_config=parent_config,
        transformer_config=transformer_config,
    )
    if train_loader is None:
        print("数据加载失败，退出。")
        return

    # ============================================================
    # 2. 初始化模型
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # Transformer Encoder (DNAWalkerTransformer)
    encoder = build_transformer(parent_config, transformer_config).to(device)
    print(f"Transformer Encoder 参数量: {count_parameters(encoder):,}")

    # ForwardDecoder (复用上层 model.py 中的架构)
    parent_config_for_decoder = ParentConfig(
        config_file=os.path.join(_PARENT_DIR, 'configfile.ini')
    )
    decoder = ForwardDecoder(OUTPUT_SIZE, parent_config_for_decoder).to(device)
    print(f"ForwardDecoder 参数量: {count_parameters(decoder):,}")
    print()

    # ============================================================
    # 3. 确定起始阶段 (Resume or Fresh)
    # ============================================================
    start_phase = args.phase if args.phase > 0 else 1
    resume_epoch = 0
    resume_best_loss = float('inf')
    resume_epochs_no_improve = 0
    resume_optimizer_state = None
    resume_scheduler_state = None
    resume_param_weights = None
    prev_wall_time = 0.0

    if args.resume and os.path.exists(checkpoint_path):
        ckpt = load_checkpoint(checkpoint_path, decoder, encoder, device)
        start_phase = ckpt['phase']
        resume_epoch = ckpt['epoch']
        resume_best_loss = ckpt['best_val_loss']
        resume_epochs_no_improve = ckpt.get('epochs_no_improve', 0)
        resume_optimizer_state = ckpt.get('optimizer_state')
        resume_scheduler_state = ckpt.get('scheduler_state')
        resume_param_weights = ckpt.get('param_weights')
        prev_wall_time = ckpt.get('total_wall_time', 0.0)
        print(f"  累计已训练时间: {time_str(prev_wall_time)}")
    elif args.resume:
        print("⚠️  未找到 checkpoint 文件，将从头开始训练。")

    # 烟雾测试覆盖
    if args.smoke:
        P1_EPOCHS_EFF = 2
        P2_EPOCHS_EFF = 2
        P3_EPOCHS_EFF = 2
        P1_PATIENCE_EFF = 999
        P2_PATIENCE_EFF = 999
        P3_PATIENCE_EFF = 999
    else:
        P1_EPOCHS_EFF = P1_NUM_EPOCHS
        P2_EPOCHS_EFF = transformer_config.get_num_epochs()
        P3_EPOCHS_EFF = P3_NUM_EPOCHS
        P1_PATIENCE_EFF = P1_PATIENCE
        P2_PATIENCE_EFF = transformer_config.get_early_stopping_patience()
        P3_PATIENCE_EFF = P3_PATIENCE

    # ============================================================
    # Phase 1: 训练 ForwardDecoder
    # ============================================================
    time_stopped = False

    if start_phase <= 1:
        p1_start_epoch = resume_epoch if start_phase == 1 else 0
        p1_best = resume_best_loss if start_phase == 1 else float('inf')
        p1_no_improve = resume_epochs_no_improve if start_phase == 1 else 0
        p1_opt_state = resume_optimizer_state if start_phase == 1 else None
        p1_sch_state = resume_scheduler_state if start_phase == 1 else None

        p1_loss, time_stopped = train_phase1_decoder(
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=P1_EPOCHS_EFF,
            lr=P1_LR,
            patience=P1_PATIENCE_EFF,
            save_path=decoder_save_path,
            start_epoch=p1_start_epoch,
            best_val_loss=p1_best,
            epochs_no_improve=p1_no_improve,
            optimizer_state=p1_opt_state,
            scheduler_state=p1_sch_state,
            global_start_time=global_start_time,
            max_hours=args.max_hours,
            prev_wall_time=prev_wall_time,
            checkpoint_path=checkpoint_path,
            encoder=encoder,
        )
        if time_stopped:
            return

        # Phase 1 完成后，写一个"Phase 1 已完成"的 checkpoint
        # 这样下次 --resume 会直接进入 Phase 2
        wall_time = prev_wall_time + (time.time() - global_start_time)
        save_checkpoint(
            checkpoint_path, phase=2, epoch=0,
            decoder=decoder, encoder=encoder,
            optimizer=None, scheduler=None,
            best_val_loss=float('inf'),
            epochs_no_improve=0,
            param_weights=None,
            total_wall_time=wall_time,
        )
        prev_wall_time = wall_time
        # 重置 resume 状态
        resume_epoch = 0
        resume_best_loss = float('inf')
        resume_epochs_no_improve = 0
        resume_optimizer_state = None
        resume_scheduler_state = None
        resume_param_weights = None

    # ============================================================
    # Phase 2: 冻结 Decoder，训练 Transformer Encoder
    # ============================================================
    if start_phase <= 2 and not time_stopped:
        # 如果跳过 Phase 1，确保加载已训练好的 Decoder
        if start_phase == 2 and resume_epoch == 0 and os.path.exists(decoder_save_path):
            decoder.load_state_dict(
                torch.load(decoder_save_path, map_location=device, weights_only=False)
            )
            print(f"  已加载 Phase 1 训练好的 Decoder: {decoder_save_path}")

        # ★ 加载预训练 Encoder 权重（若 Phase 2 已在训练中恢复则跳过）
        if args.pretrained_encoder and not (start_phase == 2 and resume_epoch > 0):
            if os.path.exists(args.pretrained_encoder):
                ckpt_enc = torch.load(
                    args.pretrained_encoder, map_location=device, weights_only=False
                )
                if isinstance(ckpt_enc, dict) and 'model_state' in ckpt_enc:
                    encoder.load_state_dict(ckpt_enc['model_state'])
                    src_epoch = ckpt_enc.get('epoch', '?')
                    src_mse = ckpt_enc.get('val_mse', '?')
                    if isinstance(src_mse, float):
                        src_mse = f'{src_mse:.6f}'
                    print(f"  ✅ 已加载预训练 Encoder "
                          f"(来自 Epoch {src_epoch}, Val MSE: {src_mse})")
                else:
                    encoder.load_state_dict(ckpt_enc)
                    print(f"  ✅ 已加载预训练 Encoder: {args.pretrained_encoder}")
            else:
                print(f"  ⚠️ 预训练 Encoder 文件不存在: "
                      f"{args.pretrained_encoder}，将使用随机初始化")

        p2_start_epoch = resume_epoch if start_phase == 2 else 0
        p2_best = resume_best_loss if start_phase == 2 else float('inf')
        p2_no_improve = resume_epochs_no_improve if start_phase == 2 else 0
        p2_opt_state = resume_optimizer_state if start_phase == 2 else None
        p2_sch_state = resume_scheduler_state if start_phase == 2 else None
        p2_param_w = resume_param_weights if start_phase == 2 else None

        p2_loss, time_stopped = train_phase2_encoder(
            encoder=encoder,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            parent_config=parent_config,
            transformer_config=transformer_config,
            device=device,
            alpha=P2_ALPHA,
            beta=P2_BETA,
            num_epochs=P2_EPOCHS_EFF,
            patience=P2_PATIENCE_EFF,
            encoder_save_path=encoder_save_path,
            start_epoch=p2_start_epoch,
            best_val_loss=p2_best,
            epochs_no_improve=p2_no_improve,
            param_weights_init=p2_param_w,
            optimizer_state=p2_opt_state,
            scheduler_state=p2_sch_state,
            global_start_time=global_start_time,
            max_hours=args.max_hours,
            prev_wall_time=prev_wall_time,
            checkpoint_path=checkpoint_path,
        )
        if time_stopped:
            return

        # Phase 2 完成后保存过渡 checkpoint
        wall_time = prev_wall_time + (time.time() - global_start_time)
        save_checkpoint(
            checkpoint_path, phase=3, epoch=0,
            decoder=decoder, encoder=encoder,
            optimizer=None, scheduler=None,
            best_val_loss=float('inf'),
            epochs_no_improve=0,
            param_weights=None,
            total_wall_time=wall_time,
        )
        prev_wall_time = wall_time
        resume_epoch = 0
        resume_best_loss = float('inf')
        resume_epochs_no_improve = 0
        resume_optimizer_state = None
        resume_scheduler_state = None

    # ============================================================
    # Phase 3 (可选): 联合微调
    # ============================================================
    if args.enable_phase3 and start_phase <= 3 and not time_stopped:
        p3_start_epoch = resume_epoch if start_phase == 3 else 0
        p3_best = resume_best_loss if start_phase == 3 else float('inf')
        p3_no_improve = resume_epochs_no_improve if start_phase == 3 else 0
        p3_opt_state = resume_optimizer_state if start_phase == 3 else None
        p3_sch_state = resume_scheduler_state if start_phase == 3 else None

        p3_loss, time_stopped = train_phase3_joint(
            encoder=encoder,
            decoder=decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            parent_config=parent_config,
            transformer_config=transformer_config,
            device=device,
            alpha=P3_ALPHA,
            beta=P3_BETA,
            num_epochs=P3_EPOCHS_EFF,
            patience=P3_PATIENCE_EFF,
            encoder_save_path=encoder_save_path,
            decoder_save_path=decoder_save_path,
            start_epoch=p3_start_epoch,
            best_val_loss=p3_best,
            epochs_no_improve=p3_no_improve,
            optimizer_state=p3_opt_state,
            scheduler_state=p3_sch_state,
            global_start_time=global_start_time,
            max_hours=args.max_hours,
            prev_wall_time=prev_wall_time,
            checkpoint_path=checkpoint_path,
        )
        if time_stopped:
            return

    # ============================================================
    # 最终测试
    # ============================================================
    if not time_stopped:
        if os.path.exists(encoder_save_path):
            ckpt_best = torch.load(encoder_save_path, map_location=device, weights_only=False)
            encoder.load_state_dict(ckpt_best['model_state'])
        final_test(encoder, decoder, test_loader, parent_config, device, param_names)

        total_time = prev_wall_time + (time.time() - global_start_time)
        print(f"🎉 全部训练完成! 累计训练时间: {time_str(total_time)}")
        print(f"   (本次运行: {time_str(time.time() - global_start_time)})")

        # 清理 checkpoint (训练已完成)
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            print(f"   Checkpoint 已清理: {checkpoint_path}")


if __name__ == '__main__':
    main()
