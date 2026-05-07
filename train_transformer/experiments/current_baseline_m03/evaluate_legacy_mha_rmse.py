import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


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
from model_transformer_mha_legacy import build_transformer


def build_parent_config(base_config, override_config=None):
    extras = [override_config] if override_config else None
    return ParentConfig(config_file=base_config, extra_config_files=extras)


def load_real_experimental_data(config, data_path):
    data = pd.read_excel(data_path)
    data = data.iloc[:, :4].copy()
    exp_time = pd.to_numeric(data.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
    exp_fam = pd.to_numeric(data.iloc[:, 1], errors="coerce").to_numpy(dtype=float)
    exp_tye = pd.to_numeric(data.iloc[:, 2], errors="coerce").to_numpy(dtype=float)
    exp_cy5 = pd.to_numeric(data.iloc[:, 3], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(exp_time) & np.isfinite(exp_fam) & np.isfinite(exp_tye) & np.isfinite(exp_cy5)
    exp_time = exp_time[mask]
    exp_fam = exp_fam[mask]
    exp_tye = exp_tye[mask]
    exp_cy5 = exp_cy5[mask]

    standard_time_axis = np.linspace(0.0, config.get_sim_total_time(), config.get_num_time_points())
    curve_fam = interp1d(exp_time, exp_fam, kind="linear", bounds_error=False, fill_value=(exp_fam[0], exp_fam[-1]))(standard_time_axis)
    curve_tye = interp1d(exp_time, exp_tye, kind="linear", bounds_error=False, fill_value=(exp_tye[0], exp_tye[-1]))(standard_time_axis)
    curve_cy5 = interp1d(exp_time, exp_cy5, kind="linear", bounds_error=False, fill_value=(exp_cy5[0], exp_cy5[-1]))(standard_time_axis)
    try:
        curve_fam = savgol_filter(curve_fam, config.get_sg_window(), config.get_sg_polyorder())
        curve_tye = savgol_filter(curve_tye, config.get_sg_window(), config.get_sg_polyorder())
        curve_cy5 = savgol_filter(curve_cy5, config.get_sg_window(), config.get_sg_polyorder())
    except Exception:
        pass
    return np.stack([curve_fam, curve_tye, curve_cy5], axis=0), (exp_time, exp_fam, exp_tye, exp_cy5)


def resize_to_expected_length(x_input, expected_len):
    batch_size, channels, seq_len = x_input.shape
    if seq_len == expected_len:
        return x_input
    src_axis = np.linspace(0.0, 1.0, seq_len, dtype=np.float64)
    dst_axis = np.linspace(0.0, 1.0, expected_len, dtype=np.float64)
    resized = np.empty((batch_size, channels, expected_len), dtype=np.float32)
    for batch_idx in range(batch_size):
        for channel_idx in range(channels):
            curve = np.asarray(x_input[batch_idx, channel_idx], dtype=np.float64)
            finite_mask = np.isfinite(curve)
            if not finite_mask.any():
                resized[batch_idx, channel_idx] = 0.0
                continue
            if finite_mask.sum() == 1:
                resized[batch_idx, channel_idx] = curve[finite_mask][0]
                continue
            cleaned_curve = np.interp(src_axis, src_axis[finite_mask], curve[finite_mask])
            resized[batch_idx, channel_idx] = np.interp(dst_axis, src_axis, cleaned_curve)
    return resized


def predict_params(model, y_scaler, parent_config, device, x_raw):
    x_input = np.asarray(x_raw, dtype=np.float32)
    if x_input.ndim == 2:
        x_input = x_input[np.newaxis, ...]
    expected_len = parent_config.get_seq_length()
    if x_input.shape[2] != expected_len:
        x_input = resize_to_expected_length(x_input, expected_len)
    sample_means = np.nanmean(x_input, axis=(1, 2), keepdims=True)
    sample_stds = np.nanstd(x_input, axis=(1, 2), keepdims=True) + 1e-8
    x_scaled = (x_input - sample_means) / sample_stds
    x_scaled = np.nan_to_num(x_scaled, nan=0.0).astype(np.float32)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)
    with torch.no_grad():
        y_pred_scaled = model(x_tensor).cpu().numpy()
    y_pred_real = y_scaler.inverse_transform(y_pred_scaled)
    param_names = parent_config.get_trainable_param_names()
    for param_name in parent_config.get_log_transform_params():
        if param_name in param_names:
            idx = param_names.index(param_name)
            y_pred_real[:, idx] = np.power(10.0, y_pred_real[:, idx]) - parent_config.get_log_epsilon()
    return y_pred_real[0]


def write_matlab_input(params_dict, path):
    with open(path, "w", encoding="utf-8") as f:
        for name, value in params_dict.items():
            f.write(f"{name}={repr(float(value))}\n")
        f.write("END_OF_PARAMS=1\n")


def compute_rmse(result_signal_path, exp_data):
    exp_time, exp_fam, exp_tye, exp_cy5 = exp_data
    sim_arr = np.loadtxt(result_signal_path)
    sim_time = sim_arr[:, 0]
    sim_fam = sim_arr[:, 1]
    sim_tye = sim_arr[:, 2]
    sim_cy5 = sim_arr[:, 3]
    fam_interp = interp1d(exp_time, exp_fam, kind="linear", bounds_error=False, fill_value="extrapolate")(sim_time)
    tye_interp = interp1d(exp_time, exp_tye, kind="linear", bounds_error=False, fill_value="extrapolate")(sim_time)
    cy5_interp = interp1d(exp_time, exp_cy5, kind="linear", bounds_error=False, fill_value="extrapolate")(sim_time)
    fam_rmse = float(np.sqrt(np.mean((sim_fam - fam_interp) ** 2)))
    tye_rmse = float(np.sqrt(np.mean((sim_tye - tye_interp) ** 2)))
    cy5_rmse = float(np.sqrt(np.mean((sim_cy5 - cy5_interp) ** 2)))
    avg_rmse = float(np.mean([fam_rmse, tye_rmse, cy5_rmse]))
    return fam_rmse, tye_rmse, cy5_rmse, avg_rmse


def run_single_evaluation(model, y_scaler, device, output_dir, tag_suffix, override_config):
    eval_parent = build_parent_config(args.config_base, override_config)
    data_file = eval_parent.get_experimental_data_path()
    x_raw, exp_data = load_real_experimental_data(eval_parent, data_file)
    predicted = predict_params(model, y_scaler, eval_parent, device, x_raw)
    params_dict = {name: value for name, value in zip(eval_parent.get_trainable_param_names(), predicted)}

    matlab_input_path = os.path.join(TRAIN_TRANSFORMER_DIR, "matlab_input_params.txt")
    result_signal_path = os.path.join(TRAIN_TRANSFORMER_DIR, "result_signal.txt")
    if os.path.exists(result_signal_path):
        os.remove(result_signal_path)
    write_matlab_input(params_dict, matlab_input_path)

    matlab_log = os.path.join(output_dir, f"matlab_eval.{tag_suffix}.log")
    with open(matlab_log, "w", encoding="utf-8") as logf:
        proc = subprocess.run(["matlab", "-batch", "verify"], cwd=TRAIN_TRANSFORMER_DIR, stdout=logf, stderr=subprocess.STDOUT, check=False)

    if not os.path.exists(result_signal_path):
        raise FileNotFoundError(f"verify.m did not generate result_signal.txt (exit={proc.returncode})")

    shutil.copy2(result_signal_path, os.path.join(output_dir, f"result_signal.{tag_suffix}.txt"))
    shutil.copy2(matlab_input_path, os.path.join(output_dir, f"matlab_input_params.{tag_suffix}.txt"))
    fam_rmse, tye_rmse, cy5_rmse, avg_rmse = compute_rmse(result_signal_path, exp_data)
    summary = {
        "tag": tag_suffix,
        "model_path": args.model,
        "transformer_config": args.transformer_config,
        "experimental_data": data_file,
        "fam_rmse": fam_rmse,
        "tye_rmse": tye_rmse,
        "cy5_rmse": cy5_rmse,
        "avg_rmse": avg_rmse,
        "matlab_exit_code": proc.returncode,
    }
    with open(os.path.join(output_dir, f"rmse_summary.{tag_suffix}.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def main():
    parent_config = build_parent_config(args.config_base)
    transformer_config = TransformerConfig(config_file=args.transformer_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_transformer(parent_config, transformer_config).to(device)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    with open(args.scaler, "rb") as f:
        y_scaler = pickle.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    original = run_single_evaluation(model, y_scaler, device, args.output_dir, f"{args.tag}.original", None)
    generalization = run_single_evaluation(model, y_scaler, device, args.output_dir, f"{args.tag}.generalization", args.config_generalization)
    combined = {
        "tag": args.tag,
        "model_path": args.model,
        "transformer_config": args.transformer_config,
        "original": original,
        "generalization": generalization,
        "mean_avg_rmse": float((original["avg_rmse"] + generalization["avg_rmse"]) / 2.0),
        "max_avg_rmse": float(max(original["avg_rmse"], generalization["avg_rmse"])),
    }
    with open(os.path.join(args.output_dir, f"rmse_summary.{args.tag}.dual.json"), "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(json.dumps(combined, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the reconstructed legacy MHA transformer.")
    parser.add_argument("--config-base", default=os.path.join(PROJECT_ROOT, "configfile.ini"))
    parser.add_argument("--config-generalization", required=True)
    parser.add_argument("--transformer-config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--scaler", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()
    main()
