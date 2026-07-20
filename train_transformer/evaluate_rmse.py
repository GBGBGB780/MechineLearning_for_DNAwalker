# coding=utf-8
"""
evaluate_rmse.py — Transformer 模型的端到端真实指标评估
evaluate_rmse.py — End-to-end real-metric evaluation for the Transformer model.

最终目标不是参数 MSE，而是：预测参数 → 正向模拟 → 与实验曲线的 RMSE。
本脚本封装完整链路，输出该真实指标，供实验 A/B 对比。

用法 / Usage:
    python evaluate_rmse.py
    python evaluate_rmse.py --transformer-config experiments/exp_a.ini
"""

import argparse
import os
import sys

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for p in (_PARENT_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pysim  # noqa: E402
from exp_data_io import load_experimental_curves  # noqa: E402
from inference_transformer import TransformerPredictor  # noqa: E402


def load_and_smooth_experiment(config, data_path):
    """加载实验数据并按预测流程做插值 + SG 平滑，返回 (3,T)。"""
    sim_total_time = config.get_sim_total_time()
    num_time_points = config.get_num_time_points()
    axis = np.linspace(0, sim_total_time, num_time_points)

    t, fam, tye, cy5 = load_experimental_curves(data_path)
    curves = []
    for y in (fam, tye, cy5):
        f = interp1d(t, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
        curves.append(f(axis))

    sg_w, sg_p = config.get_sg_window(), config.get_sg_polyorder()
    try:
        curves = [savgol_filter(c, sg_w, sg_p) for c in curves]
    except Exception:
        pass
    return np.stack(curves, axis=0)


def raw_experiment_curves(config, data_path):
    """加载实验数据并插值到模拟时间轴（不平滑），用于 RMSE 比较的真值。"""
    sim_total_time = config.get_sim_total_time()
    t, fam, tye, cy5 = load_experimental_curves(data_path)
    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)
    out = []
    for y in (fam, tye, cy5):
        out.append(np.interp(sim_time, t, y, left=y[0], right=y[-1]))
    return sim_time, np.stack(out, axis=0)


def evaluate(parent_override=None, transformer_override=None, model_path=None,
             ensemble=0, noise_std=0.005, verbose=True):
    """返回 dict: rmse_fam/tye/cy5/mean 以及预测参数。"""
    predictor = TransformerPredictor(
        model_path=model_path,
        parent_config_override_file=parent_override,
        transformer_config_override_file=transformer_override,
    )
    pc = predictor.parent_config
    data_path = pc.get_experimental_data_path()

    X = load_and_smooth_experiment(pc, data_path)

    # 预测（可选 test-time ensemble：对归一化输入加噪取中位数）
    if ensemble and ensemble > 1:
        preds = [predictor.predict(X)[0]]
        rng = np.random.default_rng(0)
        for _ in range(ensemble - 1):
            preds.append(predictor.predict(X + rng.normal(0, noise_std, X.shape))[0])
        params = np.median(np.stack(preds, axis=0), axis=0)
    else:
        params = predictor.predict(X)[0]

    names = predictor.get_param_names()
    pdict = {pysim.PARAM_NAMES[i]: float(params[i]) for i in range(len(params))}

    # 正向模拟
    signals, dt_used = pysim.run_simulation(pdict)
    if dt_used < 0:
        if verbose:
            print("  [warn] 预测参数产生无效模拟 (dt<0)")
        return dict(rmse_mean=float('inf'), rmse_fam=float('inf'),
                    rmse_tye=float('inf'), rmse_cy5=float('inf'),
                    params=pdict, dt=dt_used)

    _, exp_curves = raw_experiment_curves(pc, data_path)
    rmse = np.sqrt(np.mean((signals - exp_curves) ** 2, axis=1))
    res = dict(rmse_fam=float(rmse[0]), rmse_tye=float(rmse[1]),
               rmse_cy5=float(rmse[2]), rmse_mean=float(rmse.mean()),
               params=pdict, dt=float(dt_used))

    if verbose:
        print("\n=== 真实指标：模拟 vs 实验 RMSE ===")
        for i, nm in enumerate(['FAM', 'TYE', 'CY5']):
            print(f"  {nm}: {rmse[i]:.4f}")
        print(f"  平均 RMSE: {rmse.mean():.4f}")
        print("  预测参数:")
        for k, v in pdict.items():
            print(f"    {k:<16} = {v:.6e}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--transformer-config', default=None)
    ap.add_argument('--model', default=None, help='模型 .pth 路径 (覆盖配置)')
    ap.add_argument('--ensemble', type=int, default=0, help='test-time ensemble 次数')
    ap.add_argument('--noise-std', type=float, default=0.005)
    args = ap.parse_args()
    evaluate(
        parent_override=os.path.abspath(args.config) if args.config else None,
        transformer_override=os.path.abspath(args.transformer_config) if args.transformer_config else None,
        model_path=args.model,
        ensemble=args.ensemble,
        noise_std=args.noise_std,
    )


if __name__ == "__main__":
    main()
