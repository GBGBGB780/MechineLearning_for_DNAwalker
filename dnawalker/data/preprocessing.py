# coding=utf-8
"""
dnawalker.data.preprocessing — 训练/推理共享的数据预处理
dnawalker.data.preprocessing — Shared training/inference data preprocessing.

CNN 训练数据适配器 (``dnawalker.cnn.data``) 与两个推理适配器
(``dnawalker.cnn.inference`` / ``dnawalker.transformer.inference``)
必须使用**逐位一致**的输入归一化 —— 否则训练分布与推理分布错配会静默降低预测
质量。此段「单样本联合通道 z-score」集中为单一事实来源，统一用 ``np.nan_to_num``
(与训练侧历史行为一致)，避免各处实现漂移。

All three call sites must normalize model inputs identically; this module is the
single source of truth so training and inference cannot drift apart.

The pre-split sample-retention mask also lives here. CNN training, Transformer
training, and held-out evaluation must retain rows in exactly the same order;
otherwise reconstructing a test split can overlap a model's training set.
"""

import numpy as np

# 防止除零的标准差下限，与历史三处实现完全一致 / std floor matching all sites.
_STD_EPS = 1e-8


def prepare_labels_and_sample_mask(
    curves,
    labels,
    parameter_names,
    *,
    log_transform_params,
    log_epsilon,
    amplitude_thresholds,
    safe_threshold,
    nan_replacement,
):
    """Apply training label transforms and build the shared pre-split row mask.

    Labels are converted to ``float32`` before log transforms, matching both
    model loaders. The returned mask combines optional per-channel amplitude
    filtering with finite/range checks on curves and transformed labels.

    Returns:
        ``(transformed_labels, amplitude_mask, retention_mask)`` where all masks
        index the original row order.
    """
    x = np.asarray(curves)
    y_raw = np.asarray(labels)
    names = list(parameter_names)
    if (x.ndim != 3 or x.shape[0] == 0
            or x.shape[1] == 0 or x.shape[2] == 0):
        raise ValueError(f"curves must have shape (N, C, T), got {x.shape}")
    if y_raw.ndim != 2 or y_raw.shape != (x.shape[0], len(names)):
        raise ValueError(
            "labels shape does not match curves/parameter names: "
            f"expected {(x.shape[0], len(names))}, got {y_raw.shape}"
        )

    try:
        log_epsilon = float(log_epsilon)
        safe_threshold = float(safe_threshold)
        nan_replacement = float(nan_replacement)
    except (TypeError, ValueError) as exc:
        raise ValueError("preprocessing thresholds must be finite numbers") from exc
    if not np.isfinite(log_epsilon) or log_epsilon <= 0:
        raise ValueError("log_epsilon must be finite and positive")
    if not np.isfinite(safe_threshold) or safe_threshold <= 0:
        raise ValueError("safe_threshold must be finite and positive")
    if (not np.isfinite(nan_replacement)
            or abs(nan_replacement) < safe_threshold):
        raise ValueError(
            "nan_replacement must be finite and outside the accepted range"
        )

    name_index = {}
    for index, name in enumerate(names):
        key = str(name).lower()
        if key in name_index:
            raise ValueError(f"duplicate parameter name: {name!r}")
        name_index[key] = index

    y_transformed = np.asarray(y_raw, dtype=np.float32).copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in log_transform_params:
            key = str(name).lower()
            if key not in name_index:
                raise ValueError(f"unknown log-transform parameter: {name!r}")
            index = name_index[key]
            y_transformed[:, index] = np.log10(
                y_transformed[:, index] + log_epsilon
            )

    amplitude_mask = np.ones(x.shape[0], dtype=bool)
    if amplitude_thresholds is not None:
        thresholds = list(amplitude_thresholds)
        for channel in range(min(x.shape[1], len(thresholds))):
            threshold = float(thresholds[channel])
            if not np.isfinite(threshold) or threshold < 0:
                raise ValueError(
                    "amplitude thresholds must be finite and non-negative"
                )
            amplitudes = (
                np.max(x[:, channel, :], axis=1)
                - np.min(x[:, channel, :], axis=1)
            )
            amplitude_mask &= amplitudes <= threshold

    x_flat = x.reshape(x.shape[0], -1)
    x_valid = np.all(
        np.isfinite(x_flat)
        & (x_flat < safe_threshold)
        & (x_flat > -safe_threshold),
        axis=1,
    )
    y_valid = np.all(
        np.isfinite(y_transformed)
        & (y_transformed < safe_threshold)
        & (y_transformed > -safe_threshold),
        axis=1,
    )
    return y_transformed, amplitude_mask, amplitude_mask & x_valid & y_valid


def normalize_per_sample(curves, *, copy=True):
    """逐样本、跨全部通道与时间点的联合 z-score 归一化。

    对形状 ``(N, C, T)`` 的输入，按样本 (在 ``axis=(1, 2)`` 上) 用 nan 安全的均值/
    标准差做 ``(x - mean) / (std + 1e-8)``，再用 :func:`numpy.nan_to_num` 把结果中的
    ``NaN`` 置 0 (并把 ``±inf`` 收敛为有限值)。

    dtype 处理：浮点输入沿用 dtype (``float32`` 输入 → ``float32`` 输出)；整数输入
    转为 ``float64``。默认复制输入。训练加载器可传 ``copy=False`` 原地归一化其私有
    工作数组，避免为完整数据集额外分配近 1 GiB 内存。

    Args:
        curves: 形状 ``(N, C, T)`` 的曲线数组 (N 个样本，C 通道，T 时间点)。
        copy: 是否先复制输入。仅当调用方拥有可写浮点数组时可设为 ``False``。

    Returns:
        np.ndarray: 归一化后的数组，形状与 dtype 均与输入一致。
    """
    x = np.asarray(curves)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    elif copy:
        x = x.copy()
    elif not x.flags.writeable:
        raise ValueError("copy=False requires a writable floating-point array")

    means = np.nanmean(x, axis=(1, 2), keepdims=True)
    stds = np.nanstd(x, axis=(1, 2), keepdims=True) + _STD_EPS
    np.subtract(x, means, out=x)
    np.divide(x, stds, out=x)
    return np.nan_to_num(x, nan=0.0, copy=False)
