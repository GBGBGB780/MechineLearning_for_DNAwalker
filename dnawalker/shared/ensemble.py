# coding=utf-8
"""
dnawalker.shared.ensemble — Test-time ensemble 预测 (单一事实来源)
dnawalker.shared.ensemble — Test-time ensemble prediction (single source of truth).

CNN 与 Transformer 的精修预测、实验评估和直接预测都会使用
「干净样本 + N-1 个含噪副本 → 各自预测 → 取中位数」的测试时集成。此前这段循环在
5+ 处逐字复制，且逐条串行 forward N 次。这里集中为一处，并批量化：把 N 个副本堆成
一个 batch 一次前向，GPU/MPS 上显著加速。

逐位等价 / Bit-for-bit equivalence:
  - 两个推理器的 ``predict`` 都原生接受 ``(B, 3, T)`` 批量输入，且归一化用共享的
    ``preprocessing.normalize_per_sample`` (逐样本、``axis=(1,2)``)，故批量归一化与
    逐条归一化逐样本独立、结果相同。
  - 噪声用 ``np.random.default_rng(seed)`` 一次性抽 ``(N-1, 3, T)``，与旧代码逐次抽
    ``(3, T)`` N-1 次产生**完全相同**的字节流 (已验证)，故集成数值不变。
"""

import numpy as np


def ensemble_predict(predictor, X, ensemble=0, noise_std=0.005, seed=0,
                     return_all=False):
    """测试时集成预测，返回单样本的物理参数向量 ``(7,)``。

    Args:
        predictor: 暴露 ``predict(X) -> (N, P)`` 的推理器 (CNN 或 Transformer)。
        X: 单条曲线 ``(3, T)``。
        ensemble: 集成规模。``<= 1`` 时退化为单次预测 (仅干净样本)。
        noise_std: 加在**原始曲线空间**的高斯噪声标准差。
        seed: 噪声 RNG 种子。
        return_all: 为 True 时额外返回全部副本预测 ``(K, P)`` 供诊断 (如集成标准差)。

    Returns:
        ``np.ndarray`` 物理参数向量 ``(P,)`` —— 集成时为各副本预测的逐参数中位数，
        否则为干净样本的单次预测。若 ``return_all=True``，返回
        ``(median (P,), all_preds (K, P))``，其中集成时 ``K == ensemble``、否则 ``K == 1``。
    """
    if (isinstance(ensemble, (bool, np.bool_))
            or not isinstance(ensemble, (int, np.integer))
            or ensemble < 0):
        raise ValueError(
            f"ensemble must be a non-negative integer, got {ensemble!r}"
        )
    try:
        noise_std = float(noise_std)
    except (TypeError, ValueError) as exc:
        raise ValueError("noise_std must be a finite non-negative number") from exc
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError(
            f"noise_std must be a finite non-negative number, got {noise_std!r}"
        )

    try:
        X = np.asarray(X, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("X must be a numeric curve array") from exc
    if X.ndim != 2 or X.shape[0] <= 0 or X.shape[1] <= 0:
        raise ValueError(f"X must have shape (C, T), got {X.shape}")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf")

    if ensemble <= 1:
        preds = predictor.predict(X)          # (1, P)
        expected_rows = 1
    else:
        # 干净样本 + (ensemble-1) 个含噪副本，堆成一个 batch 一次前向。
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, noise_std, (ensemble - 1,) + X.shape)
        batch = np.concatenate([X[np.newaxis, ...], X[np.newaxis, ...] + noise], axis=0)
        preds = predictor.predict(batch)      # (ensemble, P)
        expected_rows = ensemble

    preds = np.asarray(preds)
    if preds.ndim != 2 or preds.shape[0] != expected_rows or preds.shape[1] == 0:
        raise ValueError(
            "predictor.predict returned an invalid shape: "
            f"expected ({expected_rows}, P), got {preds.shape}"
        )
    if not np.all(np.isfinite(preds)):
        raise ValueError("predictor.predict returned NaN or Inf")

    median = np.median(preds, axis=0)
    return (median, preds) if return_all else median
