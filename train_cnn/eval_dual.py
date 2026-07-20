# coding=utf-8
"""
eval_dual.py — CNN 原始 + 泛化双数据集评估 (含物理精修 + 对比图)
eval_dual.py — CNN dual-dataset (original + generalization) evaluation with refinement + plots.

对每个数据集： CNN 预测 → 物理精修 → 正向模拟 → 计算 RMSE → 画"模拟 vs 实验"图。
本脚本独立运行于 train_cnn/ 文件夹，使用 CNN 自己的 NanorobotPredictor。
共享物理内核 pysim.py / refine.py / exp_data_io.py 位于根目录。

用法 / Usage (from train_cnn/):
    python eval_dual.py --ensemble 20
"""

import argparse
import json
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
from refine import refine  # noqa: E402
from exp_data_io import load_experimental_curves  # noqa: E402
from inference_cnn import NanorobotPredictor  # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, 'results')

DATASETS = {
    'original': os.path.join(_PARENT_DIR, 'results', 'Fig3a_fitting.xlsx'),
    'generalization': os.path.join(_PARENT_DIR, 'results', 'Fig3a_fitting_generalization.xlsx'),
}


def load_and_smooth_experiment(config, data_path):
    """加载实验数据 → 插值 + SG 平滑 → (3,T)，供模型输入。"""
    axis = np.linspace(0, config.get_sim_total_time(), config.get_num_time_points())
    t, fam, tye, cy5 = load_experimental_curves(data_path)
    curves = []
    for y in (fam, tye, cy5):
        f = interp1d(t, y, kind='linear', bounds_error=False, fill_value=(y[0], y[-1]))
        curves.append(f(axis))
    try:
        curves = [savgol_filter(c, config.get_sg_window(), config.get_sg_polyorder()) for c in curves]
    except Exception:
        pass
    return np.stack(curves, axis=0)


def raw_experiment_curves(data_path):
    """实验曲线插值到模拟时间轴 (不平滑)，作为 RMSE 真值。"""
    t, fam, tye, cy5 = load_experimental_curves(data_path)
    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)
    out = [np.interp(sim_time, t, y, left=y[0], right=y[-1]) for y in (fam, tye, cy5)]
    return sim_time, np.stack(out, axis=0)


def _rmse(pdict, exp_curves):
    sig, dt = pysim.run_simulation(pdict)
    if dt < 0:
        return float('inf'), None
    rmse = np.sqrt(np.mean((sig - exp_curves) ** 2, axis=1))
    return rmse, sig


def eval_one(predictor, config, ranges, data_path, ensemble, noise_std,
             maxiter, multistart, seed, refine_on=True):
    """对单个数据集评估，返回 (结果dict, sim_signals, exp_raw)。"""
    X = load_and_smooth_experiment(config, data_path)
    _, exp_curves = raw_experiment_curves(data_path)

    # DL 预测 (可选 ensemble)
    if ensemble and ensemble > 1:
        preds = [predictor.predict(X)[0]]
        rng = np.random.default_rng(seed)
        for _ in range(ensemble - 1):
            preds.append(predictor.predict(X + rng.normal(0, noise_std, X.shape))[0])
        params = np.median(np.stack(preds), axis=0)
    else:
        params = predictor.predict(X)[0]
    pdict = {pysim.PARAM_NAMES[i]: float(params[i]) for i in range(len(params))}

    dl_rmse, _ = _rmse(pdict, exp_curves)
    dl_avg = float(dl_rmse.mean()) if np.all(np.isfinite(dl_rmse)) else float('inf')

    best = pdict
    best_rmse = dl_avg
    if refine_on:
        rp, rr, _ = refine(pdict, exp_curves, ranges, method='Powell',
                           maxiter=maxiter, verbose=False)
        if rr < best_rmse:
            best, best_rmse = rp, rr
        rng = np.random.default_rng(seed + 1)
        for _ in range(multistart):
            jitter = {}
            for nm in pysim.PARAM_NAMES:
                lo, hi = ranges[nm]
                jitter[nm] = float(np.clip(pdict[nm] + rng.normal(0, 0.1) * (hi - lo), lo, hi))
            rp, rr, _ = refine(jitter, exp_curves, ranges, method='Powell',
                               maxiter=maxiter, verbose=False)
            if rr < best_rmse:
                best, best_rmse = rp, rr

    final_rmse, sig = _rmse(best, exp_curves)
    t_raw, fam, tye, cy5 = load_experimental_curves(data_path)
    return (dict(dl_avg=dl_avg,
                 fam_rmse=float(final_rmse[0]), tye_rmse=float(final_rmse[1]),
                 cy5_rmse=float(final_rmse[2]), avg_rmse=float(final_rmse.mean()),
                 params=best),
            sig, (t_raw, fam, tye, cy5))


def plot_dual(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    chan = ['FAM', 'TYE', 'CY5']
    rmse_keys = ['fam_rmse', 'tye_rmse', 'cy5_rmse']

    for col, name in enumerate(['original', 'generalization']):
        res, sig, (t_raw, fam, tye, cy5) = results[name]
        exp = [fam, tye, cy5]
        for row in range(3):
            ax = axes[row, col]
            ax.plot(sim_time, sig[row], 'b-', lw=2, label='Simulation (refined)')
            ax.scatter(t_raw, exp[row], s=6, c='gray', alpha=0.25, label='Experimental')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'{chan[row]} Signal')
            ax.set_title(f'{name} — {chan[row]}  (RMSE={res[rmse_keys[row]]:.4f})',
                         fontweight='bold', fontsize=11)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle('CNN + 物理精修：原始 vs 泛化数据集拟合',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120)
    print(f"对比图已保存: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None, help='configfile.ini 覆盖')
    ap.add_argument('--model', default=None, help='模型 .pth 路径')
    ap.add_argument('--ensemble', type=int, default=20)
    ap.add_argument('--noise-std', type=float, default=0.005)
    ap.add_argument('--maxiter', type=int, default=500)
    ap.add_argument('--multistart', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-refine', action='store_true', help='只看纯 DL 结果')
    ap.add_argument('--tag', default='cnn', help='输出文件标签')
    args = ap.parse_args()

    predictor = NanorobotPredictor(
        config_override_file=os.path.abspath(args.config) if args.config else None,
        model_path=args.model,
    )
    config = predictor.config
    name_map = dict(zip(config.get_trainable_param_names(), pysim.PARAM_NAMES))
    ranges = {name_map[k]: v for k, v in config.get_param_ranges().items()}

    results = {}
    summary = {}
    for name, path in DATASETS.items():
        print(f"\n===== 评估数据集: {name}  ({os.path.basename(path)}) =====")
        res, sig, exp_raw = eval_one(
            predictor, config, ranges, path, args.ensemble, args.noise_std,
            args.maxiter, args.multistart, args.seed, refine_on=not args.no_refine)
        results[name] = (res, sig, exp_raw)
        print(f"  DL 平均 RMSE: {res['dl_avg']:.4f}")
        print(f"  精修后: FAM={res['fam_rmse']:.4f} TYE={res['tye_rmse']:.4f} "
              f"CY5={res['cy5_rmse']:.4f} 平均={res['avg_rmse']:.4f}")
        summary[name] = {k: res[k] for k in ('dl_avg', 'fam_rmse', 'tye_rmse', 'cy5_rmse', 'avg_rmse')}

    o = summary['original']['avg_rmse']
    g = summary['generalization']['avg_rmse']
    summary['mean_avg_rmse'] = (o + g) / 2
    summary['max_avg_rmse'] = max(o, g)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f'rmse_dual.{args.tag}.json')
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nRMSE 摘要已保存: {json_path}")

    plot_dual(results, os.path.join(RESULTS_DIR, f'dual_fit.{args.tag}.png'))

    print("\n================ 双数据集汇总 (CNN) ================")
    print(f"  original       平均 RMSE: {o:.4f}")
    print(f"  generalization 平均 RMSE: {g:.4f}")
    print(f"  两者均值: {summary['mean_avg_rmse']:.4f}   最差: {summary['max_avg_rmse']:.4f}")
    print("  (对照原 baseline m03: original=0.0115, generalization=0.0338, 均值=0.0227)")


if __name__ == "__main__":
    main()
