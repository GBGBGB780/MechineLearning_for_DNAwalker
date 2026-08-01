# coding=utf-8
"""
dnawalker.data.generate — 训练数据集生成器 (源自归档 MATLAB 的 Python 实现，无需 MATLAB)
dnawalker.data.generate — Training dataset generator (no MATLAB required).

流程 / Pipeline:
  1. 用 Latin Hypercube Sampling 在 7 维参数空间采样 (scipy.stats.qmc)
  2. 多进程并行运行正向物理模拟 (pysim.run_simulation)
  3. 质量过滤：dt 阈值、NaN/Inf、信号活跃度
  4. 按"活跃度分箱"做不平衡采样，避免数据集被弱信号样本淹没
  5. 输出 artifacts/datasets/<name>.npz，键为 X(N,3,7801) / Y(N,7) / parameter_names

用法 / Usage:
    python -m dnawalker.data.generate
    python -m dnawalker.data.generate --smoke
    python -m dnawalker.data.generate --target 5000 --workers 8 \
        --out artifacts/datasets/my.npz
"""

import argparse
import contextlib
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from dnawalker.physics import simulator as pysim
from dnawalker.config import Config
from dnawalker.paths import ARTIFACTS_DIR
from dnawalker.shared.logging import get_logger
from dnawalker.shared.seeding import derive_batch_seed

logger = get_logger("gendata")
DEFAULT_TARGET = 10_000


def _load_runtime_config():
    """Pull parameter ranges and filters from ``configs/common.ini``.

    范围按 ``pysim.PARAM_NAMES`` 的顺序返回；配置中键名大小写不敏感
    (configparser 默认转小写，所以这里也按小写查找)。

    Returns:
        (MIN_VALS, MAX_VALS, BIN_EDGES, WEAK_SIGNAL_FLOOR) — numpy/scalar values
        in the canonical ``pysim.PARAM_NAMES`` order.

    Raises:
        ValueError: 若配置缺少 pysim 需要的任何参数范围。
    """
    cfg = Config()
    raw_ranges = cfg.get_param_ranges()
    # configparser 把键名转成小写，所以做大小写无关查找。
    lower_ranges = {k.lower(): v for k, v in raw_ranges.items()}

    missing = [n for n in pysim.PARAM_NAMES if n.lower() not in lower_ranges]
    if missing:
        raise ValueError(
            f"配置文件 [TRAINING_PARAMETER_RANGES] 缺少参数 / missing ranges: {missing}"
        )

    mins = np.array([lower_ranges[n.lower()][0] for n in pysim.PARAM_NAMES],
                    dtype=np.float64)
    maxs = np.array([lower_ranges[n.lower()][1] for n in pysim.PARAM_NAMES],
                    dtype=np.float64)

    # 分箱边界：把最后一条边扩展为 +inf，使最后一个 bin 成为上端开放区间。
    # 否则 searchsorted(edges, mc, side='right')-1 会把活跃度 mc >= 最后一条有限
    # 边界 (如 2.0) 的合格样本判成 bin_idx>=num_bins 而被静默丢弃。当前物理模型下
    # 活跃度上界约 ~1 (信号为守恒概率之和)，故此改动不改变现有分箱结果，仅在未来
    # 活跃度突破末边时避免静默丢样本。
    bin_edges = np.array(cfg.get_bin_edges(), dtype=np.float64)
    if bin_edges.size >= 2:
        bin_edges[-1] = np.inf
    return mins, maxs, bin_edges, cfg.get_weak_signal_floor()


MIN_VALS, MAX_VALS, BIN_EDGES, WEAK_SIGNAL_FLOOR = _load_runtime_config()


def _lhs(n, dim, seed):
    """Latin Hypercube Sampling，返回 (n, dim) ∈ [0,1)。优先用 scipy.qmc。

    回退到分层随机仅在 scipy.qmc 不可用时触发；其他异常（如内存不足、
    用户中断）按原样抛出，避免吞掉真正的错误。
    """
    try:
        from scipy.stats import qmc
    except ImportError:
        qmc = None

    if qmc is not None:
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        return sampler.random(n)

    rng = np.random.default_rng(seed)
    out = np.zeros((n, dim))
    for j in range(dim):
        perm = rng.permutation(n)
        out[:, j] = (perm + rng.random(n)) / n
    return out


def _simulate_one(y_row):
    """单样本模拟 worker。返回 (signals(3,T), dt_used, max_change)。

    参数导致的算术异常降级为「无效样本」(dt=-1)，由 _validate 正常过滤。
    MemoryError、KeyboardInterrupt 和编程错误必须向上抛出，避免把系统故障伪装成
    普通坏样本后继续生成不完整或不可诊断的数据。
    """
    try:
        signals, dt_used = pysim.run_simulation(y_row)
        if not np.isfinite(dt_used) or dt_used < 0:
            return signals.astype(np.float32), dt_used, 0.0
        changes = signals.max(axis=1) - signals.min(axis=1)
        return signals.astype(np.float32), dt_used, float(changes.max())
    except ArithmeticError:
        return np.zeros((3, pysim.NUM_RESULTS), dtype=np.float32), -1.0, 0.0


def _validate(dt_used, max_change, min_dt_threshold):
    """单样本质量判定。返回 (is_valid, reason)。"""
    if not np.isfinite(dt_used) or dt_used <= 0:
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
    """主生成流程；仅在恰好达到 ``target`` 时写出数据集。"""
    if isinstance(target, bool) or not isinstance(target, (int, np.integer)) or target <= 0:
        raise ValueError(f"target 必须是正整数 / must be a positive integer: {target!r}")
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"ratio 必须 > 0 / must be positive: {ratio!r}")
    if isinstance(workers, bool) or not isinstance(workers, (int, np.integer)) or workers <= 0:
        raise ValueError(f"workers 必须是正整数 / must be positive: {workers!r}")
    if (isinstance(max_rounds, bool)
            or not isinstance(max_rounds, (int, np.integer))
            or max_rounds < 0):
        raise ValueError(
            f"max_rounds 必须是非负整数 / must be non-negative: {max_rounds!r}"
        )
    if (isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size <= 0):
        raise ValueError(
            f"batch_size 必须是正整数 / must be positive: {batch_size!r}"
        )
    if (isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or not 0 <= int(seed) <= 2 ** 32 - 1):
        raise ValueError(
            "seed 必须是 32 位非负整数 / must be an integer in "
            f"[0, {2 ** 32 - 1}]: {seed!r}"
        )
    seed = int(seed)
    if not np.isfinite(min_dt_threshold) or min_dt_threshold < 0:
        raise ValueError(
            "min_dt_threshold 必须是有限非负数 / must be finite and "
            f"non-negative: {min_dt_threshold!r}"
        )

    dim = len(pysim.PARAM_NAMES)
    num_bins = len(BIN_EDGES) - 1
    if num_bins <= 0 or not np.all(np.diff(BIN_EDGES) > 0):
        raise ValueError(
            f"BIN_EDGES 必须严格递增且至少含两个边界 / invalid edges: {BIN_EDGES}"
        )
    bin_cap = int(np.ceil(target / num_bins))
    bin_counts = np.zeros(num_bins, dtype=int)

    logger.info("=" * 60)
    logger.info("目标样本数: %d  | workers: %d  | 输出: %s", target, workers, out_path)
    logger.info("分箱: %s  初始每箱上限: %d", BIN_EDGES.tolist(), bin_cap)
    logger.info("=" * 60)

    X_keep, Y_keep = [], []
    total_generated = 0
    t_start = time.time()

    # 进程池只创建一次并跨所有批次 (阶段1 + 阶段2) 复用。macOS 默认 spawn 下，
    # 每次 with ProcessPoolExecutor(...) 都会重启全部 worker 并重新 import gendata
    # (进而重跑模块级 _load_runtime_config 解析 ``configs/common.ini``) —— 按批新建会把
    # batches×workers 次进程启动/import 开销白白重复。传入的 executor 由 generate()
    # 统一持有并在结束时关闭。
    def run_batch(Y_batch, executor):
        nonlocal total_generated
        n = len(Y_batch)
        results = [None] * n
        if executor is not None:
            for i, r in enumerate(executor.map(_simulate_one, list(Y_batch), chunksize=8)):
                results[i] = r
        else:
            for i, row in enumerate(Y_batch):
                results[i] = _simulate_one(row)

        n_ok = 0
        n_rejected_by_cap = 0
        # The whole batch has already been simulated, even if the target is
        # reached while consuming its results.
        total_generated += n
        for row, (sig, dt_used, mc) in zip(Y_batch, results):
            if len(Y_keep) >= target:
                break
            ok, _ = _validate(dt_used, mc, min_dt_threshold)
            if not ok:
                continue
            bin_idx = np.searchsorted(BIN_EDGES, mc, side='right') - 1
            if bin_idx < 0 or bin_idx >= num_bins:
                continue
            if bin_counts[bin_idx] >= bin_cap:
                n_rejected_by_cap += 1
                continue
            bin_counts[bin_idx] += 1
            X_keep.append(sig)
            Y_keep.append(np.asarray(row, dtype=np.float64))
            n_ok += 1
        return n_ok, n_rejected_by_cap

    # 进程池只开一次并跨阶段1/阶段2 全部批次复用 (见 run_batch 说明)。workers<=1
    # 时不建池，run_batch 走串行回退 (executor=None)，与历史单进程路径逐位等价。
    pool_ctx = (ProcessPoolExecutor(max_workers=workers)
                if workers > 1 else contextlib.nullcontext(None))
    with pool_ctx as executor:
        # ---- 阶段 1: 初始采样 ----
        n_init = max(1, int(np.ceil(target * ratio)))
        logger.info("[阶段1] 初始采样 %d 个", n_init)
        lhs0 = _lhs(n_init, dim, seed)
        Y0 = MIN_VALS + (MAX_VALS - MIN_VALS) * lhs0
        for b in range(0, n_init, batch_size):
            Y_batch = Y0[b:b + batch_size]
            n_ok, _ = run_batch(Y_batch, executor)
            elapsed = time.time() - t_start
            logger.info("  批次 %d: +%d 合格 | 累计 %d/%d | 已生成 %d | %.0fs",
                        b // batch_size + 1, n_ok, len(Y_keep), target,
                        total_generated, elapsed)
            if len(Y_keep) >= target:
                break

        # ---- 阶段 2: 补充采样 ----
        # 补采至少使用一个有统计意义的小批次，避免只差 1 条时每轮仅抽 4 条、
        # 因短期未命中稀有 bin 就误判不可达。停滞按累计抽样数而非轮数计算。
        min_supplement = min(
            batch_size,
            max(20, int(np.ceil(target * 0.01))),
        )
        stall_draw_budget = max(40, min_supplement * 2)
        rnd = 1
        no_growth_draws = 0
        no_growth_capped = 0
        while len(Y_keep) < target and rnd <= max_rounds:
            kept_before = len(Y_keep)
            needed = target - len(Y_keep)
            extra = max(int(np.ceil(needed * 4)), min_supplement)
            logger.info(
                "[阶段2 轮%d] 缺 %d，补采 %d，当前每箱上限 %d",
                rnd, needed, extra, bin_cap,
            )
            # 每轮独立抽 extra 个全新样本：用从 seed 派生的每轮种子，而非重新生成
            # total_generated+extra 个再切掉前缀。旧做法既浪费（每轮 LHS 越滚越大），
            # 又破坏 LHS 空间填充性（LHS(N) 并非 LHS(M) 的前缀超集，切片后不再是 LHS）。
            # 每轮种子由基础 seed 确定性派生，故固定 seed 下整个数据集仍可复现。
            round_seed = derive_batch_seed(seed, rnd)
            lhs_a = _lhs(extra, dim, round_seed)
            Ya = MIN_VALS + (MAX_VALS - MIN_VALS) * lhs_a
            rejected_by_cap = 0
            for b in range(0, extra, batch_size):
                Y_batch = Ya[b:b + batch_size]
                n_ok, n_capped = run_batch(Y_batch, executor)
                rejected_by_cap += n_capped
                elapsed = time.time() - t_start
                logger.info("  批次 %d: +%d | 累计 %d/%d | 箱: %s | %.0fs",
                            b // batch_size + 1, n_ok, len(Y_keep), target,
                            bin_counts.tolist(), elapsed)
                if len(Y_keep) >= target:
                    break
            rnd += 1

            # A hard balance cap can itself cause a plateau when one rare bin
            # cannot be filled. Relax it gradually after enough no-growth draws;
            # balancing is best-effort, while exact target size is mandatory.
            if len(Y_keep) == kept_before:
                no_growth_draws += extra
                no_growth_capped += rejected_by_cap
                if no_growth_draws >= stall_draw_budget:
                    if no_growth_capped > 0 and bin_cap < target:
                        old_cap = bin_cap
                        bin_cap = min(target, bin_cap + 1)
                        no_growth_draws = 0
                        capped_before_relax = no_growth_capped
                        no_growth_capped = 0
                        logger.warning(
                            "累计 %d 次补采无增长，其中 %d 条被分箱上限拒绝；"
                            "每箱上限由 %d 放宽至 %d。",
                            stall_draw_budget, capped_before_relax, old_cap, bin_cap,
                        )
                    else:
                        logger.warning(
                            "累计 %d 次补采无新增有效样本，停止补采 "
                            "(累计 %d/%d，分箱 %s)。",
                            no_growth_draws, len(Y_keep), target,
                            bin_counts.tolist(),
                        )
                        break
            else:
                no_growth_draws = 0
                no_growth_capped = 0

    # ---- 保存 ----
    if len(Y_keep) != target:
        logger.error(
            "错误：目标未达成，仅生成 %d/%d 条；未写入部分数据集 %s。",
            len(Y_keep), target, out_path,
        )
        return False

    X = np.stack(X_keep, axis=0).astype(np.float32)   # (N, 3, 7801)
    Y = np.stack(Y_keep, axis=0).astype(np.float64)   # (N, 7)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tmp_path = f"{out_path}.tmp.npz"
    np.savez_compressed(
        tmp_path,
        X=X,
        Y=Y,
        parameter_names=np.array(pysim.PARAM_NAMES),
    )
    os.replace(tmp_path, out_path)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info("完成: %d 样本, 耗时 %.0fs", X.shape[0], elapsed)
    logger.info("  X=%s  Y=%s", X.shape, Y.shape)
    logger.info("  分箱最终: %s", bin_counts.tolist())
    logger.info("  保存至: %s", out_path)
    logger.info("=" * 60)
    return True


def main(argv=None):
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
    args = ap.parse_args(argv)

    dataset_dir = os.fspath(ARTIFACTS_DIR / "datasets")
    if args.smoke:
        target = args.target if args.target is not None else 20
        out = args.out or os.path.join(
            dataset_dir, 'training_dataset_smoke.npz'
        )
        batch = min(args.batch, 20)
        ratio = max(args.ratio, 2.0)
    else:
        target = args.target if args.target is not None else DEFAULT_TARGET
        out = args.out or os.path.join(dataset_dir, 'training_dataset.npz')
        batch = args.batch
        ratio = args.ratio

    workers = (
        args.workers
        if args.workers is not None
        else max(1, (os.cpu_count() or 2))
    )

    ok = generate(
        target=target,
        ratio=ratio,
        workers=workers,
        out_path=out,
        min_dt_threshold=args.min_dt,
        max_rounds=args.max_rounds,
        batch_size=batch,
        seed=args.seed,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
