# coding=utf-8
"""
train_autoencoder.py — CNN 三阶段 Encoder-Decoder 训练脚本
train_autoencoder.py — CNN three-phase Encoder-Decoder training script

Phase 1: 独立训练 ForwardDecoder (数字孪生近似器)
         Train ForwardDecoder independently (digital twin surrogate)
Phase 2: 冻结 Decoder，训练 Encoder (InverseCNN)
         Freeze Decoder, train Encoder (InverseCNN)
Phase 3: (可选) 联合微调 / (Optional) Joint fine-tuning

用法 / Usage:
    cd train_cnn/
    python train_autoencoder.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import time

# 路径设置 / Path setup
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from data_loader import load_and_preprocess_data
from model_cnn import InverseCNN, ForwardDecoder
from config_loader import Config


def count_parameters(model):
    """统计可训练参数量。/ Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_phase1_decoder(decoder, train_loader, val_loader, config, device,
                         num_epochs, lr, patience, save_path):
    """
    Phase 1: 独立训练 ForwardDecoder。/ Train ForwardDecoder independently.
    Decoder 输入 Y(参数) → 输出 X(曲线)。/ Decoder: Y(params) → X(curves).
    """
    print("=" * 70)
    print("  PHASE 1: 训练 ForwardDecoder / Train ForwardDecoder")
    print("=" * 70)
    print(f"  参数量 / Params: {count_parameters(decoder):,}, LR: {lr}, Patience: {patience}\n")

    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        # 训练 / Train
        decoder.train()
        total_loss = 0
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            loss = criterion(decoder(Y_batch), X_batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # 验证 / Validate
        decoder.eval()
        val_loss = 0
        with torch.no_grad():
            for X_v, Y_v in val_loader:
                X_v, Y_v = X_v.to(device), Y_v.to(device)
                val_loss += criterion(decoder(Y_v), X_v).item()
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        print(f"[P1] Epoch {epoch+1:03d}/{num_epochs} | Train: {total_loss/len(train_loader):.6f} | "
              f"Val: {avg_val:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(decoder.state_dict(), save_path)
            print(f"  -> 最佳 Decoder 已保存 / Best Decoder saved (Val: {avg_val:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Phase 1 Early Stopping ({patience} epochs) ***")
                break

    decoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nPhase 1 完成 / Done! Best Val MSE: {best_val_loss:.6f}\n")
    return best_val_loss


def train_phase2_encoder(encoder, decoder, train_loader, val_loader, config, device,
                         alpha, beta, num_epochs, patience, save_path):
    """
    Phase 2: 冻结 Decoder，训练 Encoder。/ Freeze Decoder, train Encoder.
    Loss = α × Recon_MSE + β × Adaptive_Param_MSE
    """
    print("=" * 70)
    print("  PHASE 2: 冻结 Decoder，训练 Encoder / Freeze Decoder, Train Encoder")
    print("=" * 70)
    print(f"  α(重构/recon)={alpha}, β(参数/param)={beta}")
    print(f"  Encoder 参数量 / Params: {count_parameters(encoder):,}\n")

    # 冻结 Decoder / Freeze Decoder
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad = False

    OUTPUT_SIZE = config.get_output_size()
    param_names = config.get_trainable_param_names()
    LR = config.get_learning_rate()

    optimizer = optim.Adam(encoder.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config.get_scheduler_mode(), factor=config.get_scheduler_factor(),
        patience=config.get_scheduler_patience(), min_lr=config.get_scheduler_min_lr())

    param_weights = torch.ones(OUTPUT_SIZE, dtype=torch.float32, device=device)
    recon_crit = nn.MSELoss()
    mse_fn = nn.MSELoss(reduction='none')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        encoder.train()
        t_loss, t_recon, t_param = 0, 0, 0
        for X_b, Y_b in train_loader:
            X_b, Y_b = X_b.to(device), Y_b.to(device)
            Y_pred = encoder(X_b)
            X_recon = decoder(Y_pred)
            l_recon = recon_crit(X_recon, X_b)
            l_param = torch.mean(param_weights * (Y_pred - Y_b) ** 2)
            loss = alpha * l_recon + beta * l_param
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += loss.item()
            t_recon += l_recon.item()
            t_param += l_param.item()

        # 验证 / Validate
        encoder.eval()
        val_mse_pp = torch.zeros(OUTPUT_SIZE, device=device)
        val_cnt = 0
        with torch.no_grad():
            for X_v, Y_v in val_loader:
                X_v, Y_v = X_v.to(device), Y_v.to(device)
                Y_pv = encoder(X_v)
                val_mse_pp += mse_fn(Y_pv, Y_v).sum(dim=0)
                val_cnt += X_v.shape[0]

        avg_mse_pp = (val_mse_pp / val_cnt).cpu().numpy()
        avg_mse = avg_mse_pp.mean()
        scheduler.step(avg_mse)

        # 自适应加权 / Adaptive weighting
        raw_w = np.sqrt(avg_mse_pp + 1e-6)
        param_weights = 0.9 * param_weights + 0.1 * torch.tensor(raw_w / raw_w.mean(), dtype=torch.float32, device=device)

        print(f"[P2] Epoch {epoch+1:03d}/{num_epochs} | Total: {t_loss/len(train_loader):.6f} | "
              f"Val MSE: {avg_mse:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if (epoch + 1) % 100 == 0:
            print(f"  [Per-Param MSE] {', '.join(f'{n}={v:.6f}' for n, v in zip(param_names, avg_mse_pp))}")

        if avg_mse < best_val_loss:
            best_val_loss = avg_mse
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), save_path)
            print(f"  -> 最佳 Encoder 已保存 / Best Encoder saved (Val MSE: {avg_mse:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Phase 2 Early Stopping ({patience} epochs) ***")
                break

    encoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nPhase 2 完成 / Done! Best Val Param MSE: {best_val_loss:.6f}\n")
    return best_val_loss


def final_test(encoder, decoder, test_loader, config, device, param_names):
    """
    测试集最终评估。/ Final test set evaluation.
    """
    print("=" * 70)
    print("  最终测试集评估 / Final Test Evaluation")
    print("=" * 70)

    OUTPUT_SIZE = config.get_output_size()
    encoder.eval()
    decoder.eval()
    mse_fn = nn.MSELoss(reduction='none')
    recon_crit = nn.MSELoss()
    test_mse_pp = torch.zeros(OUTPUT_SIZE, device=device)
    test_cnt = 0
    test_recon = 0

    with torch.no_grad():
        for X_t, Y_t in test_loader:
            X_t, Y_t = X_t.to(device), Y_t.to(device)
            Y_pt = encoder(X_t)
            test_mse_pp += mse_fn(Y_pt, Y_t).sum(dim=0)
            test_cnt += X_t.shape[0]
            test_recon += recon_crit(decoder(Y_pt), X_t).item()

    avg_mse_pp = (test_mse_pp / test_cnt).cpu().numpy()
    print(f"\n{'='*60}")
    print(f"  Reconstruction MSE: {test_recon/len(test_loader):.6f}")
    for name, val in zip(param_names, avg_mse_pp):
        print(f"    {name:20s}: {val:.6f}")
    print(f"  Average Param MSE: {avg_mse_pp.mean():.6f}")
    print(f"{'='*60}\n")


def main():
    """主函数 / Main function."""
    start_time = time.time()

    print("=" * 70)
    print("  三阶段 Encoder-Decoder 训练 / Three-Phase Encoder-Decoder Training")
    print("=" * 70 + "\n")

    config = Config(config_file=os.path.join(_PARENT_DIR, 'configfile.ini'))

    INPUT_SIZE = config.get_input_size()
    OUTPUT_SIZE = config.get_output_size()
    BATCH_SIZE = config.get_batch_size()
    DATASET_FILE = config.get_dataset_file()
    MODEL_SAVE_PATH = config.get_model_save_path()
    PATIENCE = config.get_early_stopping_patience()

    output_dir = os.path.dirname(MODEL_SAVE_PATH) if os.path.dirname(MODEL_SAVE_PATH) else '.'
    DECODER_SAVE_PATH = os.path.join(output_dir, 'best_decoder.pth')
    os.makedirs(output_dir, exist_ok=True)

    # 从配置读取 Autoencoder 参数 / Read autoencoder params from config
    dec_epochs = config.get_decoder_num_epochs()
    dec_lr = config.get_decoder_lr()
    dec_patience = config.get_decoder_patience()
    alpha = config.get_alpha_recon()
    beta = config.get_beta_param()

    print(f"  Encoder: ({INPUT_SIZE}) → ({OUTPUT_SIZE}), Decoder: ({OUTPUT_SIZE}) → ({INPUT_SIZE})")
    print(f"  Batch: {BATCH_SIZE}, α={alpha}, β={beta}\n")

    # 加载数据 / Load data
    train_loader, val_loader, test_loader, _ = load_and_preprocess_data(DATASET_FILE, BATCH_SIZE, config)
    if train_loader is None:
        return

    param_names = config.get_trainable_param_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备 / Device: {device}")

    encoder = InverseCNN(INPUT_SIZE, OUTPUT_SIZE, config).to(device)
    decoder = ForwardDecoder(OUTPUT_SIZE, config).to(device)
    print(f"Encoder: {count_parameters(encoder):,} params, Decoder: {count_parameters(decoder):,} params\n")

    # Phase 1 / Phase 1
    train_phase1_decoder(decoder, train_loader, val_loader, config, device,
                         num_epochs=dec_epochs, lr=dec_lr, patience=dec_patience, save_path=DECODER_SAVE_PATH)

    # Phase 2 / Phase 2
    train_phase2_encoder(encoder, decoder, train_loader, val_loader, config, device,
                         alpha=alpha, beta=beta, num_epochs=config.get_num_epochs(),
                         patience=PATIENCE, save_path=MODEL_SAVE_PATH)

    # 测试 / Test
    encoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    final_test(encoder, decoder, test_loader, config, device, param_names)

    print(f"总耗时 / Total time: {(time.time()-start_time)/60:.1f} min")


if __name__ == "__main__":
    main()
