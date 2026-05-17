import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
TRAIN_TRANSFORMER_DIR = os.path.join(PROJECT_ROOT, "train_transformer")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if TRAIN_TRANSFORMER_DIR not in sys.path:
    sys.path.insert(0, TRAIN_TRANSFORMER_DIR)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from config_loader import Config as ParentConfig
from config_loader_transformer import TransformerConfig
from dataset import load_and_preprocess_data_3d
from model_transformer_mha_legacy import build_transformer


def build_configs(parent_config_file, transformer_config_file):
    parent = ParentConfig(config_file=parent_config_file)
    transformer = TransformerConfig(config_file=transformer_config_file)
    return parent, transformer


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    parent_config, transformer_config = build_configs(args.config_base, args.transformer_config)

    output_size = parent_config.get_output_size()
    batch_size = transformer_config.get_batch_size()
    num_epochs = transformer_config.get_num_epochs()
    learning_rate = transformer_config.get_learning_rate()
    weight_decay = transformer_config.get_weight_decay()
    warmup_ratio = transformer_config.get_warmup_ratio()
    min_lr = transformer_config.get_scheduler_min_lr()
    patience = transformer_config.get_early_stopping_patience()
    model_save = transformer_config.get_model_save_path()
    dataset_path = transformer_config.get_dataset_path()

    train_loader, val_loader, test_loader, param_names = load_and_preprocess_data_3d(
        npz_filename=dataset_path,
        batch_size=batch_size,
        parent_config=parent_config,
        transformer_config=transformer_config,
    )
    if train_loader is None:
        raise RuntimeError("Failed to load dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(parent_config, transformer_config).to(device)
    if args.init_model:
        checkpoint = torch.load(args.init_model, map_location=device, weights_only=False)
        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Initialized from checkpoint: {args.init_model}")

    mse_fn = nn.MSELoss(reduction="none")
    adaptive_weights = torch.ones(output_size, dtype=torch.float32, device=device)

    if args.adaptive_weight:
        def criterion(pred, target):
            return torch.mean(adaptive_weights * (pred - target) ** 2)
    else:
        criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(total_steps * warmup_ratio)
    min_lr_ratio = min_lr / learning_rate if learning_rate > 0 else 0.0
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, min_lr_ratio)

    best_val_mse = float("inf")
    epochs_no_improve = 0
    train_start = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        val_mse_per_param = torch.zeros(output_size, device=device)
        val_sample_count = 0

        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device, non_blocking=True)
                y_val = y_val.to(device, non_blocking=True)
                y_pred_val = model(x_val)
                total_val_loss += criterion(y_pred_val, y_val).item()
                batch_mse = mse_fn(y_pred_val, y_val)
                val_mse_per_param += batch_mse.sum(dim=0)
                val_sample_count += x_val.shape[0]

        avg_val_mse_per_param = (val_mse_per_param / val_sample_count).cpu().numpy()
        avg_val_mse = float(avg_val_mse_per_param.mean())
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:04d}/{num_epochs} | Train: {avg_train_loss:.6f} | "
            f"Val MSE: {avg_val_mse:.6f} | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"
        )

        if args.adaptive_weight:
            raw_weights = np.sqrt(avg_val_mse_per_param + 1e-6)
            normalized = raw_weights / raw_weights.mean()
            new_weights = torch.tensor(normalized, dtype=torch.float32, device=device)
            adaptive_weights = 0.9 * adaptive_weights + 0.1 * new_weights

        if epoch % 50 == 0 or epoch == num_epochs:
            mse_str = ", ".join([f"{name}={value:.5f}" for name, value in zip(param_names, avg_val_mse_per_param)])
            print(f"  [Per-Param MSE] {mse_str}")
            if args.adaptive_weight:
                weight_str = ", ".join([f"{name}={value:.3f}" for name, value in zip(param_names, adaptive_weights.cpu().numpy())])
                print(f"  [Adaptive Weights] {weight_str}")

        if avg_val_mse < best_val_mse:
            best_val_mse = avg_val_mse
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_mse": best_val_mse,
                    "param_names": param_names,
                    "adaptive_weight": bool(args.adaptive_weight),
                },
                model_save,
            )
            print(f"  -> Saved best model (Val MSE: {best_val_mse:.6f})")
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

    total_time = time.time() - train_start
    print(f"Training finished in {total_time / 60.0:.1f} minutes.")

    checkpoint = torch.load(model_save, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    criterion_test = nn.MSELoss()
    total_test_loss = 0.0
    test_mse_per_param = torch.zeros(output_size, device=device)
    test_sample_count = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device, non_blocking=True)
            y_test = y_test.to(device, non_blocking=True)
            y_pred_test = model(x_test)
            total_test_loss += criterion_test(y_pred_test, y_test).item()
            batch_mse = mse_fn(y_pred_test, y_test)
            test_mse_per_param += batch_mse.sum(dim=0)
            test_sample_count += x_test.shape[0]

    avg_test_loss = total_test_loss / len(test_loader)
    avg_test_mse = (test_mse_per_param / test_sample_count).cpu().numpy()
    print(f"Test loss: {avg_test_loss:.6f}")
    for name, value in zip(param_names, avg_test_mse):
        print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the reconstructed legacy MHA transformer.")
    parser.add_argument("--config-base", default=os.path.join(PROJECT_ROOT, "configfile.ini"))
    parser.add_argument("--transformer-config", required=True)
    parser.add_argument("--adaptive-weight", action="store_true")
    parser.add_argument("--init-model", default=None)
    args = parser.parse_args()
    train(args)
