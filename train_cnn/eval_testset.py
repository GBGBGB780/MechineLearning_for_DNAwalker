# coding=utf-8
"""
eval_testset.py — CNN 在留出测试集上的预测质量评估 (统计稳健)
eval_testset.py — CNN prediction quality on the held-out test set (statistically robust).

目的：判定模型"预测得好不好"，必须在大量有真值的样本上测，而非 2 条实验曲线。
对每个测试样本： 预测参数 → 正向模拟 → 与该样本的真实曲线比 RMSE，再对全体样本平均。
**不做精修** —— 精修是模型无关的曲线拟合，会抹平模型差异。

测试集复现：与 data_loader.py 完全一致 (同 seed、同比例、同 train_test_split 顺序)，
仅用其中的 test 划分；用未归一化的原始曲线喂给 predictor (predictor 内部自做归一化)。

用法 / Usage (from train_cnn/):
    python eval_testset.py
    python eval_testset.py --max-samples 300   # 抽样加速
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
for p in (_PARENT_DIR, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import pysim  # noqa: E402
from inference_cnn import NanorobotPredictor  # noqa: E402

RESULTS_DIR = os.path.join(_THIS_DIR, 'results')


def get_test_split(config):
    """复现 data_loader 的拆分，返回 (X_test_raw(N,3,T), Y_test_raw(N,7))。

    关键：train_test_split 的划分只由 random_state 决定，与数据数值无关，
    因此对原始 (未归一化) X 用同 seed 拆分，得到的样本与训练时 test 集一致。
    """
    npz = config.get_dataset_file()
    with np.load(npz) as d:
        X = d['X'].astype(np.float32)   # 原始曲线 (未归一化)
        Y = d['Y'].astype(np.float64)   # 物理参数

    test_ratio = config.get_test_split_ratio()
    seed = config.get_random_seed()
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=seed)
    return X_test, Y_test


def evaluate(config_override=None, model_path=None, max_samples=0, seed=0, tag='cnn'):
    predictor = NanorobotPredictor(
        config_override_file=config_override, model_path=model_path)
    config = predictor.config

    X_test, Y_test = get_test_split(config)
    n_total = X_test.shape[0]

    # 可选抽样加速
    if max_samples and max_samples < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_samples, replace=False)
        X_test, Y_test = X_test[idx], Y_test[idx]
    n = X_test.shape[0]
    print(f"测试样本数 / Test samples: {n} (总 {n_total})")

    # 批量预测 (predictor 内部自做归一化 + 反 log)
    preds = predictor.predict(X_test)   # (n, 7) 物理空间

    # 逐样本：正向模拟预测参数 → 与该样本真实曲线比 RMSE
    rmse_list = []
    n_invalid = 0
    n_extreme = 0
    for i in range(n):
        pdict = {pysim.PARAM_NAMES[j]: float(preds[i, j]) for j in range(7)}
        sig, dt = pysim.run_simulation(pdict)
        if dt < 0 or not np.all(np.isfinite(sig)):
            n_invalid += 1
            continue
        # 物理合理性：荧光信号应在 ~[0, 1.5]，超过即灾难性预测
        if np.max(np.abs(sig)) > 5.0:
            n_extreme += 1
            continue
        true_curve = X_test[i]  # (3, T) 该样本真实曲线
        rmse = np.sqrt(np.mean((sig - true_curve) ** 2, axis=1))
        rmse_list.append(rmse)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n}")

    rmse_arr = np.array(rmse_list)  # (m, 3)
    per_sample_mean = rmse_arr.mean(axis=1)  # (m,)

    # 参数空间 MSE (在各自 y_scaler 的 scaled 空间，公平)
    names = config.get_trainable_param_names()
    Y_true = Y_test.copy()
    P = preds.copy()
    for pp in config.get_log_transform_params():
        if pp in names:
            k = names.index(pp)
            Y_true[:, k] = np.log10(Y_true[:, k] + config.get_log_epsilon())
            P[:, k] = np.log10(np.clip(P[:, k], 1e-30, None) + config.get_log_epsilon())
    ys = predictor.y_scaler
    param_mse = float(np.mean((ys.transform(P) - ys.transform(Y_true)) ** 2))

    summary = dict(
        model='CNN',
        n_samples=int(n),
        n_invalid=int(n_invalid),
        n_extreme=int(n_extreme),
        n_valid=int(len(rmse_list)),
        curve_rmse_mean=float(per_sample_mean.mean()),
        curve_rmse_median=float(np.median(per_sample_mean)),
        curve_rmse_std=float(per_sample_mean.std()),
        curve_rmse_p90=float(np.percentile(per_sample_mean, 90)),
        fam_rmse_mean=float(rmse_arr[:, 0].mean()),
        tye_rmse_mean=float(rmse_arr[:, 1].mean()),
        cy5_rmse_mean=float(rmse_arr[:, 2].mean()),
        param_mse_scaled=param_mse,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f'testset_eval.{tag}.json')
    with open(out, 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n================ 测试集预测质量 (CNN) ================")
    print(f"  样本数: {n}  (无效模拟 dt<0: {n_invalid}, 灾难性预测 |sig|>5: {n_extreme})")
    print(f"  有效样本: {len(rmse_list)} / {n}  ({len(rmse_list)/n*100:.1f}%)")
    print(f"  曲线重构 RMSE  均值: {summary['curve_rmse_mean']:.4f}")
    print(f"                中位数: {summary['curve_rmse_median']:.4f}")
    print(f"                标准差: {summary['curve_rmse_std']:.4f}")
    print(f"                P90:   {summary['curve_rmse_p90']:.4f}")
    print(f"  逐通道均值: FAM={summary['fam_rmse_mean']:.4f} "
          f"TYE={summary['tye_rmse_mean']:.4f} CY5={summary['cy5_rmse_mean']:.4f}")
    print(f"  参数 MSE (scaled): {param_mse:.5f}")
    print(f"  摘要已保存: {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=None)
    ap.add_argument('--model', default=None)
    ap.add_argument('--max-samples', type=int, default=0, help='抽样数 (0=全部)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='cnn')
    args = ap.parse_args()
    evaluate(
        config_override=os.path.abspath(args.config) if args.config else None,
        model_path=args.model, max_samples=args.max_samples, seed=args.seed, tag=args.tag)


if __name__ == "__main__":
    main()
