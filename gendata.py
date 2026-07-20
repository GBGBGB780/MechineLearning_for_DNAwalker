# coding=utf-8
"""
gendata.py — 训练数据集生成器 (gendata.m 的 Python 移植，无需 MATLAB)
gendata.py — Training dataset generator, Python port of gendata.m (no MATLAB required).

流程 / Pipeline:
  1. 用 Latin Hypercube Sampling 在 7 维参数空间采样 (scipy.stats.qmc)
  2. 多进程并行运行正向物理模拟 (pysim.run_simulation)
  3. 质量过滤：dt 阈值、NaN/Inf、信号活跃度
  4. 按"活跃度分箱"做不平衡采样，避免数据集被弱信号样本淹没
  5. 输出 results/<name>.npz，键为 X(N,3,7801) / Y(N,7) / parameter_names

用法 / Usage:
    python gendata.py                       # 正式：30000 样本
    python gendata.py --smoke               # 冒烟：20 样本
    python gendata.py --target 5000 --workers 8 --out results/my.npz
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pysim  # noqa: E402

# 参数范围 (与 configfile.ini [TRAINING_PARAMETER_RANGES] 及 gendata.m 同步)
MIN_VALS = np.array([-2.0, -2.0, -0.5, 0.01, 1e-7, 0.20, 0.01])
MAX_VALS = np.array([-0.5, -0.5, 0.2, 0.30, 1e-4, 0.90, 0.50])

# 活跃度分箱边界 (gendata.m bin_edges)
BIN_EDGES = np.array([0.01, 0.05, 0.1, 0.2, 0.4, 2.0])

WEAK_SIGNAL_FLOOR = 0.01   # 信号死寂阈值


def _lhs(n, dim, seed):
    """Latin Hypercube Sampling，返回 (n, dim) ∈ [0,1)。优先用 scipy.qmc。"""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(n)
    except Exception:
        # 回退：分层随机
        rng = np.random.default_rng(seed)
        out = np.zeros((n, dim))
        for j in range(dim):
            perm = rng.permutation(n)
            out[:, j] = (perm + rng.random(n)) / n
        return out


def _simulate_one(y_row):
    """单样本模拟 worker。返回 (signals(3,T), dt_used, max_change)。"""
    signals, dt_used = pysim.run_simulation(y_row)
    if dt_used < 0:
        return signals.astype(np.float32), dt_used, 0.0
    changes = signals.max(axis=1) - signals.min(axis=1)
    return signals.astype(np.float32), dt_used, float(changes.max())


def _validate(dt_used, max_change, min_dt_threshold):
    """单样本质量判定。返回 (is_valid, reason)。"""
    if dt_used <= 0:
        return False, 'timeout/invalid'
    if dt_used <= min_dt_threshold:
        return False, 'dt too small'
    if not np.isfinite(max_change):
        return False, 'NaN/Inf'
    if max_change <= WEAK_SIGNAL_FLOOR:
        return False, 'signal too weak'
    return True, ''


def generate(target, ratio, workers, out_path, min_dt_threshold,
             max_rounds, batch_size, seed=0):
    """主生成流程。"""
    dim = len(pysim.PARAM_NAMES)
    num_bins = len(BIN_EDGES) - 1
    max_per_bin = int(np.ceil(target / num_bins))
    bin_counts = np.zeros(num_bins, dtype=int)

    print("=" * 60)
    print(f"目标样本数: {target}  | workers: {workers}  | 输出: {out_path}")
    print(f"分箱: {BIN_EDGES.tolist()}  每箱上限: {max_per_bin}")
    print("=" * 60)

    X_keep, Y_keep = [], []
    total_generated = 0
    t_start = time.time()

    def run_batch(Y_batch):
        nonlocal total_generated
        n = len(Y_batch)
        results = [None] * n
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for i, r in enumerate(ex.map(_simulate_one, list(Y_batch), chunksize=8)):
                    results[i] = r
        else:
            for i, row in enumerate(Y_batch):
                results[i] = _simulate_one(row)

        n_ok = 0
        for row, (sig, dt_used, mc) in zip(Y_batch, results):
            total_generated += 1
            ok, _ = _validate(dt_used, mc, min_dt_threshold)
            if not ok:
                continue
            bin_idx = np.searchsorted(BIN_EDGES, mc, side='right') - 1
            if bin_idx < 0 or bin_idx >= num_bins:
                continue
            if bin_counts[bin_idx] >= max_per_bin:
                continue
            bin_counts[bin_idx] += 1
            X_keep.append(sig)
            Y_keep.append(np.asarray(row, dtype=np.float64))
            n_ok += 1
        return n_ok

    # ---- 阶段 1: 初始采样 ----
    n_init = int(round(target * ratio))
    print(f"\n[阶段1] 初始采样 {n_init} 个")
    lhs0 = _lhs(n_init, dim, seed)
    Y0 = MIN_VALS + (MAX_VALS - MIN_VALS) * lhs0
    for b in range(0, n_init, batch_size):
        Y_batch = Y0[b:b + batch_size]
        n_ok = run_batch(Y_batch)
        elapsed = time.time() - t_start
        print(f"  批次 {b//batch_size + 1}: +{n_ok} 合格 | 累计 {len(Y_keep)}/{target} "
              f"| 已生成 {total_generated} | {elapsed:.0f}s")
        if len(Y_keep) >= target:
            break

    # ---- 阶段 2: 补充采样 ----
    rnd = 1
    while len(Y_keep) < target and rnd <= max_rounds:
        needed = target - len(Y_keep)
        extra = int(np.ceil(needed * 4))
        print(f"\n[阶段2 轮{rnd}] 缺 {needed}，补采 {extra}")
        lhs_a = _lhs(total_generated + extra, dim, seed)[total_generated:]
        Ya = MIN_VALS + (MAX_VALS - MIN_VALS) * lhs_a
        for b in range(0, extra, batch_size):
            Y_batch = Ya[b:b + batch_size]
            n_ok = run_batch(Y_batch)
            elapsed = time.time() - t_start
            print(f"  批次 {b//batch_size + 1}: +{n_ok} | 累计 {len(Y_keep)}/{target} "
                  f"| 箱: {bin_counts.tolist()} | {elapsed:.0f}s")
            if len(Y_keep) >= target:
                break
        rnd += 1

    # ---- 保存 ----
    if not Y_keep:
        print("错误：没有任何合格样本，未保存。")
        return False

    X = np.stack(X_keep, axis=0).astype(np.float32)   # (N, 3, 7801)
    Y = np.stack(Y_keep, axis=0).astype(np.float64)   # (N, 7)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, X=X, Y=Y,
                        parameter_names=np.array(pysim.PARAM_NAMES))

    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"✓ 完成: {X.shape[0]} 样本, 耗时 {elapsed:.0f}s")
    print(f"  X={X.shape}  Y={Y.shape}")
    print(f"  分箱最终: {bin_counts.tolist()}")
    print(f"  保存至: {out_path}")
    print("=" * 60)
    return True


def main():
    ap = argparse.ArgumentParser(description="DNA Walker 训练数据生成 (Python)")
    ap.add_argument('--smoke', action='store_true', help='冒烟模式：20 样本')
    ap.add_argument('--target', type=int, default=None, help='目标合格样本数')
    ap.add_argument('--ratio', type=float, default=1.1, help='初始冗余比例')
    ap.add_argument('--workers', type=int, default=None, help='并行进程数')
    ap.add_argument('--out', type=str, default=None, help='输出 npz 路径')
    ap.add_argument('--min-dt', type=float, default=pysim.MIN_DT, help='dt 下限阈值')
    ap.add_argument('--max-rounds', type=int, default=500)
    ap.add_argument('--batch', type=int, default=2000, help='每批样本数')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        target = args.target or 20
        out = args.out or os.path.join('results', 'training_dataset_smoke.npz')
        batch = min(args.batch, 20)
        ratio = max(args.ratio, 2.0)
    else:
        target = args.target or 30000
        out = args.out or os.path.join('results', 'training_dataset.npz')
        batch = args.batch
        ratio = args.ratio

    workers = args.workers or max(1, (os.cpu_count() or 2))

    generate(target=target, ratio=ratio, workers=workers, out_path=out,
             min_dt_threshold=args.min_dt, max_rounds=args.max_rounds,
             batch_size=batch, seed=args.seed)


if __name__ == "__main__":
    main()
