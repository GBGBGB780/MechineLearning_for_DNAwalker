# coding=utf-8
"""
predict_refine.py — CNN 预测 + 物理精修 (生产预测流程)
predict_refine.py — CNN prediction + physics refinement (production pipeline).

完整链路 / Full pipeline:
  1. CNN 预测 (可选 test-time ensemble) 得到参数初值
  2. refine.py 局部优化，直接最小化曲线 RMSE
  3. 输出最终参数到 matlab_input_params.txt，并报告 RMSE

本脚本独立运行于 train_cnn/ 文件夹，使用 CNN 自己的 NanorobotPredictor。
共享物理内核 pysim.py / refine.py / exp_data_io.py 位于根目录。

用法 / Usage (from train_cnn/):
    python predict_refine.py --ensemble 20
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
from refine import refine  # noqa: E402
from exp_data_io import load_experimental_curves  # noqa: E402
from inference_cnn import NanorobotPredictor  # noqa: E402

OUT_FILE = os.path.join(_THIS_DIR, "matlab_input_params.txt")
LOWER_KEYS = ['e_b', 'e_b_azo_trans', 'e_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']


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


def write_params(pdict, path=OUT_FILE):
    with open(path, 'w') as f:
        for lk, pk in zip(LOWER_KEYS, pysim.PARAM_NAMES):
            f.write(f"{lk}={repr(float(pdict[pk]))}\n")
        f.write("END_OF_PARAMS=1\n")


def _rmse(pdict, exp_curves):
    sig, dt = pysim.run_simulation(pdict)
    if dt < 0:
        return float('inf')
    return float(np.sqrt(np.mean((sig - exp_curves) ** 2, axis=1)).mean())


def run(config_override=None, model_path=None, ensemble=0, noise_std=0.005,
        method='Powell', maxiter=400, multistart=8, seed=0):
    predictor = NanorobotPredictor(
        config_override_file=config_override,
        model_path=model_path,
    )
    config = predictor.config
    data_path = config.get_experimental_data_path()

    X = load_and_smooth_experiment(config, data_path)
    _, exp_curves = raw_experiment_curves(data_path)

    # 1. DL 预测 (可选 ensemble)
    if ensemble and ensemble > 1:
        preds = [predictor.predict(X)[0]]
        rng = np.random.default_rng(seed)
        for _ in range(ensemble - 1):
            preds.append(predictor.predict(X + rng.normal(0, noise_std, X.shape))[0])
        params = np.median(np.stack(preds), axis=0)
    else:
        params = predictor.predict(X)[0]

    pdict = {pysim.PARAM_NAMES[i]: float(params[i]) for i in range(len(params))}

    name_map = dict(zip(config.get_trainable_param_names(), pysim.PARAM_NAMES))
    ranges = {name_map[k]: v for k, v in config.get_param_ranges().items()}

    dl_rmse = _rmse(pdict, exp_curves)
    print(f"\n[1] DL 预测 RMSE: {dl_rmse:.4f}")

    # 2. 精修 (主起点 = DL 预测)
    best_params, best_rmse, _ = refine(pdict, exp_curves, ranges,
                                       method=method, maxiter=maxiter, verbose=True)

    # 2b. multi-start
    if multistart and multistart > 0:
        rng = np.random.default_rng(seed + 1)
        for s in range(multistart):
            jitter = {}
            for nm in pysim.PARAM_NAMES:
                lo, hi = ranges[nm]
                jitter[nm] = float(np.clip(pdict[nm] + rng.normal(0, 0.1) * (hi - lo), lo, hi))
            rp, rr, _ = refine(jitter, exp_curves, ranges, method=method,
                               maxiter=maxiter, verbose=False)
            if rr < best_rmse:
                best_rmse, best_params = rr, rp
                print(f"  [multistart {s+1}] 新最优 RMSE: {rr:.4f}")

    print(f"\n[2] 精修后最终 RMSE: {best_rmse:.4f}  (DL→最终 降低 "
          f"{(dl_rmse-best_rmse)/dl_rmse*100:.1f}%)")
    print("\n最终参数:")
    for k in pysim.PARAM_NAMES:
        print(f"  {k:<16} = {best_params[k]:.6e}")

    write_params(best_params)
    print(f"\n已写入: {OUT_FILE}")
    return best_params, best_rmse, dl_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--model', default=None)
    ap.add_argument('--ensemble', type=int, default=0)
    ap.add_argument('--noise-std', type=float, default=0.005)
    ap.add_argument('--method', default='Powell', choices=['Powell', 'Nelder-Mead'])
    ap.add_argument('--maxiter', type=int, default=400)
    ap.add_argument('--multistart', type=int, default=8)
    args = ap.parse_args()
    run(
        config_override=os.path.abspath(args.config) if args.config else None,
        model_path=args.model,
        ensemble=args.ensemble, noise_std=args.noise_std,
        method=args.method, maxiter=args.maxiter, multistart=args.multistart,
    )


if __name__ == "__main__":
    main()
