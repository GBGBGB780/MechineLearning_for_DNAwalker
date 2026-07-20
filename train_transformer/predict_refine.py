# coding=utf-8
"""
predict_refine.py — Transformer 预测 + 物理精修 (生产预测流程)
predict_refine.py — Transformer prediction + physics refinement (production pipeline).

完整链路 / Full pipeline:
  1. Transformer 预测 (可选 test-time ensemble) 得到参数初值
  2. refine.py 局部优化，直接最小化曲线 RMSE
  3. 输出最终参数到 matlab_input_params.txt，并报告 RMSE

逆问题中，直接优化曲线 RMSE 通常比纯 DL 预测好得多 (实测 -74%)。

用法 / Usage:
    python predict_refine.py
    python predict_refine.py --transformer-config config_transformer.exp.ini --ensemble 20
"""

import argparse
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for p in (_PARENT_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pysim  # noqa: E402
from refine import refine  # noqa: E402
from inference_transformer import TransformerPredictor  # noqa: E402
from evaluate_rmse import load_and_smooth_experiment, raw_experiment_curves  # noqa: E402

OUT_FILE = os.path.join(_THIS_DIR, "matlab_input_params.txt")
# matlab_input_params.txt 用小写键
LOWER_KEYS = ['e_b', 'e_b_azo_trans', 'e_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']


def write_params(pdict, path=OUT_FILE):
    with open(path, 'w') as f:
        for lk, pk in zip(LOWER_KEYS, pysim.PARAM_NAMES):
            f.write(f"{lk}={repr(float(pdict[pk]))}\n")
        f.write("END_OF_PARAMS=1\n")


def run(parent_override=None, transformer_override=None, model_path=None,
        ensemble=0, noise_std=0.005, method='Powell', maxiter=400,
        multistart=8, seed=0):
    predictor = TransformerPredictor(
        model_path=model_path,
        parent_config_override_file=parent_override,
        transformer_config_override_file=transformer_override,
    )
    pc = predictor.parent_config
    data_path = pc.get_experimental_data_path()

    X = load_and_smooth_experiment(pc, data_path)
    _, exp_curves = raw_experiment_curves(pc, data_path)

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

    # 参数物理边界
    name_map = dict(zip(pc.get_trainable_param_names(), pysim.PARAM_NAMES))
    ranges = {name_map[k]: v for k, v in pc.get_param_ranges().items()}

    dl_rmse = _rmse(pdict, exp_curves)
    print(f"\n[1] DL 预测 RMSE: {dl_rmse:.4f}")

    # 2. 精修 (主起点 = DL 预测)
    best_params, best_rmse, _ = refine(pdict, exp_curves, ranges,
                                       method=method, maxiter=maxiter, verbose=True)

    # 2b. 可选 multi-start：在 DL 初值附近多次扰动重启，取最优
    if multistart and multistart > 0:
        rng = np.random.default_rng(seed + 1)
        for s in range(multistart):
            jitter = {}
            for nm in pysim.PARAM_NAMES:
                lo, hi = ranges[nm]
                span = hi - lo
                jitter[nm] = float(np.clip(pdict[nm] + rng.normal(0, 0.1) * span, lo, hi))
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


def _rmse(pdict, exp_curves):
    sig, dt = pysim.run_simulation(pdict)
    if dt < 0:
        return float('inf')
    return float(np.sqrt(np.mean((sig - exp_curves) ** 2, axis=1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--transformer-config', default=None)
    ap.add_argument('--model', default=None)
    ap.add_argument('--ensemble', type=int, default=0)
    ap.add_argument('--noise-std', type=float, default=0.005)
    ap.add_argument('--method', default='Powell', choices=['Powell', 'Nelder-Mead'])
    ap.add_argument('--maxiter', type=int, default=400)
    ap.add_argument('--multistart', type=int, default=8)
    args = ap.parse_args()
    run(
        parent_override=os.path.abspath(args.config) if args.config else None,
        transformer_override=os.path.abspath(args.transformer_config) if args.transformer_config else None,
        model_path=args.model,
        ensemble=args.ensemble, noise_std=args.noise_std,
        method=args.method, maxiter=args.maxiter, multistart=args.multistart,
    )


if __name__ == "__main__":
    main()
