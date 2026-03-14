# coding=utf-8
"""
三阶段 Encoder-Decoder 训练脚本
================================
Phase 1: 独立训练 ForwardDecoder (数字孪生近似器)
         输入: 参数 Y(7) → 输出: 曲线 X(3×7801)
         Loss: MSE(X_pred, X_true)

Phase 2: 冻结 Decoder，训练 Encoder (InverseCNN)
         X → Encoder → Ŷ → [Frozen Decoder] → X̂
         Loss: α × Recon_MSE(X, X̂) + β × Adaptive_Param_MSE(Y, Ŷ)

Phase 3 (可选): 联合微调 Encoder + Decoder
         用更小的学习率端到端训练，持续监控防止捷径解
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time

# 从本地文件导入
from utils import load_and_preprocess_data
from model import InverseCNN, ForwardDecoder
from config_loader import Config


def count_parameters(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_phase1_decoder(decoder, train_loader, val_loader, config, device,
                         num_epochs=500, lr=0.001, patience=100, save_path='results/best_decoder.pth'):
    """
    Phase 1: 独立训练 ForwardDecoder
    用已有的 (参数, 曲线) 配对数据训练 Decoder 学习正向映射。
    
    注意: DataLoader 中 X=曲线(展平), Y=参数(归一化后)
    所以这里 Decoder 的输入是 Y_batch, 目标是 X_batch
    """
    print("=" * 70)
    print("  PHASE 1: 训练 ForwardDecoder (数字孪生近似器)")
    print("=" * 70)
    print(f"  Decoder 参数量: {count_parameters(decoder):,}")
    print(f"  学习率: {lr}, 最大轮数: {num_epochs}, Early Stopping 耐心: {patience}")
    print()

    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=30, min_lr=1e-6
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        decoder.train()
        total_train_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            # Decoder: 参数(Y) → 曲线(X)
            X_reconstructed = decoder(Y_batch)
            loss = criterion(X_reconstructed, X_batch)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                X_recon_val = decoder(Y_batch_val)
                val_loss = criterion(X_recon_val, X_batch_val)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        print(f"[Phase1] Epoch {epoch + 1:03d}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.8f}")
        
        # --- Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(decoder.state_dict(), save_path)
            print(f"  -> 新的最佳 Decoder 已保存 (Val Loss: {avg_val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Phase 1 Early Stopping: 连续 {patience} 轮未改善 ***")
                print(f"*** 最佳 Decoder Val Loss: {best_val_loss:.6f} ***")
                break
    
    # 加载最佳权重
    decoder.load_state_dict(torch.load(save_path, map_location=device))
    print(f"\nPhase 1 完成! 最佳 Decoder Val MSE: {best_val_loss:.6f}")
    print()
    
    return best_val_loss


def validate_decoder_gradients(decoder, val_loader, config, device, param_names):
    """
    验证 Decoder 的梯度方向是否符合物理直觉 (Jacobian 符号检查)。
    对验证集中的几个样本，微小扰动每个参数，检查输出变化方向的一致性。
    """
    print("--- 验证 Decoder 梯度方向 (Jacobian 符号检查) ---")
    decoder.eval()
    
    # 从验证集取一个 batch
    X_sample, Y_sample = next(iter(val_loader))
    Y_sample = Y_sample[:5].to(device)  # 取 5 个样本
    
    perturbation = 0.02  # 2% 扰动
    
    for param_idx, param_name in enumerate(param_names):
        # 基准输出
        with torch.no_grad():
            base_output = decoder(Y_sample)
        
        # 正扰动
        Y_plus = Y_sample.clone()
        Y_plus[:, param_idx] += perturbation
        with torch.no_grad():
            plus_output = decoder(Y_plus)
        
        # 负扰动  
        Y_minus = Y_sample.clone()
        Y_minus[:, param_idx] -= perturbation
        with torch.no_grad():
            minus_output = decoder(Y_minus)
        
        # 计算扰动引起的曲线变化的 L2 范数
        delta_plus = (plus_output - base_output).norm(dim=1).mean().item()
        delta_minus = (minus_output - base_output).norm(dim=1).mean().item()
        
        # 检查方向是否"对称" (即正负扰动引起的变化应该大致相等)
        ratio = delta_plus / (delta_minus + 1e-10)
        status = "✅ 正常" if 0.3 < ratio < 3.0 else "⚠️ 需关注"
        
        print(f"  {param_name:20s}: Δ+(L2)={delta_plus:.4f}, Δ-(L2)={delta_minus:.4f}, "
              f"比值={ratio:.2f} {status}")
    
    print()


def train_phase2_encoder(encoder, decoder, train_loader, val_loader, config, device,
                         alpha=0.7, beta=0.3, num_epochs=2000, patience=200,
                         encoder_save_path='results/best_mlp_model.pth'):
    """
    Phase 2: 冻结 Decoder，训练 Encoder
    Loss = α × Recon_MSE(X, X̂) + β × Adaptive_Param_MSE(Y, Ŷ)
    """
    print("=" * 70)
    print("  PHASE 2: 冻结 Decoder，训练 Encoder (InverseCNN)")
    print("=" * 70)
    print(f"  Loss 配比: α(重构)={alpha}, β(参数)={beta}")
    print(f"  Encoder 参数量: {count_parameters(encoder):,}")
    print()
    
    # 冻结 Decoder
    decoder.eval()
    for param in decoder.parameters():
        param.requires_grad = False
    print("  Decoder 已冻结 (不参与梯度更新)")
    
    OUTPUT_SIZE = config.get_output_size()
    LEARNING_RATE = config.get_learning_rate()
    param_names = config.get_trainable_param_names()
    
    optimizer = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.get_scheduler_mode(),
        factor=config.get_scheduler_factor(),
        patience=config.get_scheduler_patience(),
        min_lr=config.get_scheduler_min_lr()
    )
    
    # 自适应加权 MSE (仅用于参数空间 Loss)
    param_weights = torch.ones(OUTPUT_SIZE, dtype=torch.float32, device=device)
    
    recon_criterion = nn.MSELoss()
    mse_fn = nn.MSELoss(reduction='none')
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    os.makedirs(os.path.dirname(encoder_save_path), exist_ok=True)

    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        encoder.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_param_loss = 0

        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            # 1. Encoder 前向: X → Ŷ
            Y_pred = encoder(X_batch)

            # 2. Decoder 前向 (冻结): Ŷ → X̂
            X_reconstructed = decoder(Y_pred)

            # 3. 计算混合 Loss
            # 3a. 重构 Loss: MSE(X, X̂)
            loss_recon = recon_criterion(X_reconstructed, X_batch)
            
            # 3b. 参数 Loss: Adaptive Weighted MSE(Y, Ŷ)
            loss_param = torch.mean(param_weights * (Y_pred - Y_batch) ** 2)
            
            # 3c. 总 Loss
            loss = alpha * loss_recon + beta * loss_param

            # 4. 反向传播 (梯度只流过 Encoder)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            total_recon_loss += loss_recon.item()
            total_param_loss += loss_param.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_recon_loss = total_recon_loss / len(train_loader)
        avg_param_loss = total_param_loss / len(train_loader)

        # --- 验证阶段 ---
        encoder.eval()
        total_val_loss = 0
        total_val_recon = 0
        total_val_param = 0
        val_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
        val_sample_count = 0
        
        with torch.no_grad():
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)

                Y_pred_val = encoder(X_batch_val)
                X_recon_val = decoder(Y_pred_val)
                
                val_loss_recon = recon_criterion(X_recon_val, X_batch_val)
                val_loss_param = torch.mean(param_weights * (Y_pred_val - Y_batch_val) ** 2)
                val_loss = alpha * val_loss_recon + beta * val_loss_param
                
                total_val_loss += val_loss.item()
                total_val_recon += val_loss_recon.item()
                total_val_param += val_loss_param.item()
                
                # 累计逐参数 MSE (纯 MSE，用于自适应权重和早停)
                batch_mse = mse_fn(Y_pred_val, Y_batch_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_batch_val.shape[0]

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_recon = total_val_recon / len(val_loader)
        avg_val_param = total_val_param / len(val_loader)
        avg_val_mse_per_param = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = avg_val_mse_per_param.mean()
        
        scheduler.step(avg_val_mse)
        
        # --- 自适应损失加权 (Adaptive Loss Weighting) ---
        raw_weights = np.sqrt(avg_val_mse_per_param + 1e-6)
        normalized_weights = raw_weights / raw_weights.mean()
        new_weights_tensor = torch.tensor(normalized_weights, dtype=torch.float32).to(device)
        param_weights = 0.9 * param_weights + 0.1 * new_weights_tensor
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Phase2] Epoch {epoch + 1:03d}/{num_epochs} | "
              f"Total: {avg_train_loss:.6f} | Recon: {avg_recon_loss:.6f} | Param: {avg_param_loss:.6f} | "
              f"Val MSE: {avg_val_mse:.6f} | LR: {current_lr:.8f}")
        
        # 每 100 轮打印详情
        if (epoch + 1) % 100 == 0:
            mse_str = ", ".join([f"{n}={v:.6f}" for n, v in zip(param_names, avg_val_mse_per_param)])
            weight_str = ", ".join([f"{n}={v:.3f}" for n, v in zip(param_names, param_weights.cpu().numpy())])
            print(f"  [Per-Param MSE]     {mse_str}")
            print(f"  [Adaptive Weights]  {weight_str}")
            print(f"  [Val Recon MSE]     {avg_val_recon:.6f}")
            print(f"  [Val Param MSE]     {avg_val_param:.6f}")

        # --- Early Stopping (基于纯参数 MSE) ---
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), encoder_save_path)
            print(f"  -> 新的最佳 Encoder 已保存 (Val Param MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Phase 2 Early Stopping: 连续 {patience} 轮未改善 ***")
                print(f"*** 最佳 Val Param MSE: {best_val_loss:.6f} ***")
                break

    # 加载最佳权重
    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    print(f"\nPhase 2 完成! 最佳 Encoder Val Param MSE: {best_val_loss:.6f}")
    print()
    
    return best_val_loss


def train_phase3_joint(encoder, decoder, train_loader, val_loader, config, device,
                       alpha=0.7, beta=0.3, num_epochs=200, patience=50,
                       encoder_save_path='results/best_mlp_model.pth',
                       decoder_save_path='results/best_decoder.pth'):
    """
    Phase 3 (可选): 联合微调 Encoder + Decoder
    解冻 Decoder，用更小的学习率端到端训练。
    """
    print("=" * 70)
    print("  PHASE 3 (可选): 联合微调 Encoder + Decoder")
    print("=" * 70)
    
    # 解冻 Decoder
    for param in decoder.parameters():
        param.requires_grad = True
    print("  Decoder 已解冻")
    
    OUTPUT_SIZE = config.get_output_size()
    param_names = config.get_trainable_param_names()
    
    # 用更小的学习率
    joint_lr = config.get_learning_rate() * 0.1
    print(f"  联合学习率: {joint_lr} (原始的 1/10)")
    
    # Encoder 和 Decoder 用不同学习率
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': joint_lr},
        {'params': decoder.parameters(), 'lr': joint_lr * 0.5},  # Decoder 更保守
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-7
    )
    
    recon_criterion = nn.MSELoss()
    mse_fn = nn.MSELoss(reduction='none')
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        encoder.train()
        decoder.train()
        total_train_loss = 0
        
        for X_batch, Y_batch in train_loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            Y_pred = encoder(X_batch)
            X_reconstructed = decoder(Y_pred)
            
            loss_recon = recon_criterion(X_reconstructed, X_batch)
            loss_param = nn.functional.mse_loss(Y_pred, Y_batch)
            loss = alpha * loss_recon + beta * loss_param
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- 验证阶段 ---
        encoder.eval()
        decoder.eval()
        val_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
        val_sample_count = 0
        total_val_recon = 0
        
        with torch.no_grad():
            for X_batch_val, Y_batch_val in val_loader:
                X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                Y_pred_val = encoder(X_batch_val)
                X_recon_val = decoder(Y_pred_val)
                
                total_val_recon += recon_criterion(X_recon_val, X_batch_val).item()
                batch_mse = mse_fn(Y_pred_val, Y_batch_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += X_batch_val.shape[0]
        
        avg_val_mse_per_param = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = avg_val_mse_per_param.mean()
        avg_val_recon = total_val_recon / len(val_loader)
        
        scheduler.step(avg_val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Phase3] Epoch {epoch + 1:03d}/{num_epochs} | "
              f"Train: {avg_train_loss:.6f} | Val Param MSE: {avg_val_mse:.6f} | "
              f"Val Recon: {avg_val_recon:.6f} | LR: {current_lr:.8f}")
        
        # --- Early Stopping ---
        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), encoder_save_path)
            torch.save(decoder.state_dict(), decoder_save_path)
            print(f"  -> 联合模型已保存 (Val Param MSE: {avg_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n*** Phase 3 Early Stopping: 连续 {patience} 轮未改善 ***")
                break
    
    encoder.load_state_dict(torch.load(encoder_save_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_save_path, map_location=device))
    print(f"\nPhase 3 完成! 最佳 Val Param MSE: {best_val_loss:.6f}")
    print()
    
    return best_val_loss


def final_test(encoder, decoder, test_loader, config, device, param_names):
    """在测试集上评估最终模型"""
    print("=" * 70)
    print("  最终测试集评估")
    print("=" * 70)
    
    OUTPUT_SIZE = config.get_output_size()
    
    encoder.eval()
    decoder.eval()
    
    mse_fn = nn.MSELoss(reduction='none')
    recon_criterion = nn.MSELoss()
    
    test_mse_per_param = torch.zeros(OUTPUT_SIZE).to(device)
    test_sample_count = 0
    total_test_recon = 0
    
    with torch.no_grad():
        for X_batch_test, Y_batch_test in test_loader:
            X_batch_test, Y_batch_test = X_batch_test.to(device), Y_batch_test.to(device)
            
            Y_pred_test = encoder(X_batch_test)
            X_recon_test = decoder(Y_pred_test)
            
            batch_mse = mse_fn(Y_pred_test, Y_batch_test)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count += X_batch_test.shape[0]
            total_test_recon += recon_criterion(X_recon_test, X_batch_test).item()
    
    avg_test_mse_per_param = (test_mse_per_param / test_sample_count).cpu().numpy()
    avg_test_recon = total_test_recon / len(test_loader)
    
    print(f"\n{'=' * 60}")
    print(f"  Test Reconstruction MSE: {avg_test_recon:.6f}")
    print(f"  Per-Parameter MSE (scaled space):")
    for name, mse_val in zip(param_names, avg_test_mse_per_param):
        print(f"    {name:20s}: {mse_val:.6f}")
    print(f"  Average Param MSE: {avg_test_mse_per_param.mean():.6f}")
    print(f"{'=' * 60}\n")


def main():
    start_time = time.time()
    
    # ============================================================
    # 0. 加载配置
    # ============================================================
    print("=" * 70)
    print("  三阶段 Encoder-Decoder 训练系统")
    print("=" * 70)
    print()
    
    config = Config()
    
    INPUT_SIZE = config.get_input_size()
    OUTPUT_SIZE = config.get_output_size()
    BATCH_SIZE = config.get_batch_size()
    DATASET_FILE = config.get_dataset_file()
    MODEL_SAVE_PATH = config.get_model_save_path()
    EARLY_STOPPING_PATIENCE = config.get_early_stopping_patience()
    
    # Decoder 保存路径
    output_dir = os.path.dirname(MODEL_SAVE_PATH) if os.path.dirname(MODEL_SAVE_PATH) else '.'
    DECODER_SAVE_PATH = os.path.join(output_dir, 'best_decoder.pth')
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"配置已加载:")
    print(f"  - Encoder: 曲线({INPUT_SIZE}) → 参数({OUTPUT_SIZE})")
    print(f"  - Decoder: 参数({OUTPUT_SIZE}) → 曲线({INPUT_SIZE})")
    print(f"  - 批次大小: {BATCH_SIZE}")
    print(f"  - Encoder 保存路径: {MODEL_SAVE_PATH}")
    print(f"  - Decoder 保存路径: {DECODER_SAVE_PATH}")
    print()
    
    # ============================================================
    # 1. 加载数据
    # ============================================================
    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data(
        DATASET_FILE, BATCH_SIZE, config
    )
    
    if train_loader is None:
        return
    
    param_names = config.get_trainable_param_names()
    
    # ============================================================
    # 2. 初始化模型
    # ============================================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    encoder = InverseCNN(INPUT_SIZE, OUTPUT_SIZE, config).to(device)
    decoder = ForwardDecoder(OUTPUT_SIZE, config).to(device)
    
    print(f"Encoder (InverseCNN) 参数量: {count_parameters(encoder):,}")
    print(f"Decoder (ForwardDecoder) 参数量: {count_parameters(decoder):,}")
    print()
    
    # ============================================================
    # Phase 1: 训练 Decoder
    # ============================================================
    phase1_loss = train_phase1_decoder(
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        num_epochs=500,
        lr=0.001,
        patience=100,
        save_path=DECODER_SAVE_PATH,
    )
    
    # 验证 Decoder 梯度方向
    validate_decoder_gradients(decoder, val_loader, config, device, param_names)
    
    # ============================================================
    # Phase 2: 冻结 Decoder，训练 Encoder
    # ============================================================
    phase2_loss = train_phase2_encoder(
        encoder=encoder,
        decoder=decoder,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        alpha=0.7,   # 重构 Loss 权重
        beta=0.3,    # 参数 Loss 权重
        num_epochs=config.get_num_epochs(),
        patience=EARLY_STOPPING_PATIENCE,
        encoder_save_path=MODEL_SAVE_PATH,
    )
    
    # ============================================================
    # Phase 3 (可选): 联合微调
    # ============================================================
    # 取消下面的注释以启用 Phase 3
    # 注意：Phase 3 有捷径解风险，建议先用 Phase 1+2 验证效果
    
    # phase3_loss = train_phase3_joint(
    #     encoder=encoder,
    #     decoder=decoder,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     config=config,
    #     device=device,
    #     alpha=0.7,
    #     beta=0.3,
    #     num_epochs=200,
    #     patience=50,
    #     encoder_save_path=MODEL_SAVE_PATH,
    #     decoder_save_path=DECODER_SAVE_PATH,
    # )
    
    # ============================================================
    # 最终测试
    # ============================================================
    encoder.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    final_test(encoder, decoder, test_loader, config, device, param_names)
    
    elapsed = time.time() - start_time
    print(f"总训练耗时: {elapsed / 60:.1f} 分钟")


if __name__ == "__main__":
    main()
