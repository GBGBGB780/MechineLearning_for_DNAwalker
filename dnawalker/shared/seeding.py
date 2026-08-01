# coding=utf-8
"""Deterministic seed helpers shared by generation and model training.

:mod:`dnawalker.data.generate` and the validation experiments use this module
to derive reproducible, independent per-batch seeds. CNN and Transformer
trainers also use it as the single source of truth for Python, NumPy, and
PyTorch random-state initialization. Keeping the implementation in
``dnawalker.shared`` avoids dependencies between the core data pipeline and
experiment orchestration.
"""

from numbers import Integral

import numpy as np


def _require_uint32(value, name):
    """Validate values shared by NumPy's legacy and modern RNG APIs."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or not 0 <= int(value) <= 2 ** 32 - 1):
        raise ValueError(
            f"{name} must be an integer in [0, {2 ** 32 - 1}], got {value!r}"
        )
    return int(value)


def seed_everything(seed, deterministic_torch=False):
    """设定训练随机源 (权重初始化 / DataLoader shuffle / Dropout)。

    这是训练侧随机种子的单一事实来源：CNN 与 Transformer canonical 训练模块
    在加载数据（含 ``train_test_split``）与创建模型之前各调用一次，使
    ``configs/common.ini`` 的 ``[TRAINING] random_seed`` 真正控制训练的全部随机
    来源，而不仅是数据划分。多种子实验通过 override INI 驱动每次重训。

    固定种子不承诺跨 PyTorch 版本或 CUDA/MPS/CPU 后端逐位一致。它保证实验设计中的
    随机源受控；跨机器验证应复用同一 checkpoint，并记录软件版本和设备。

    ``torch`` 采用惰性 import：本模块也被纯数据管线 (``dnawalker.data.generate``) 复用，那里
    没有、也不应引入 torch 依赖；只有训练脚本调用本函数时才导入。

    Args:
        seed: 已校验的整数种子。
        deterministic_torch: 为 True 时额外请求 cuDNN 确定性算法
            (``torch.backends.cudnn.deterministic=True``, ``benchmark=False``)。
            默认 False——它会显著拖慢 CUDA 卷积，且对 MPS/CPU 无影响；仅在需要
            提高同一 CUDA 软件栈内的确定性时开启。

    Returns:
        int: 实际使用的种子 (回传方便日志记录)。
    """
    import random as _random

    seed = _require_uint32(seed, "seed")
    _random.seed(seed)
    np.random.seed(seed)

    import torch  # 惰性 import：数据管线复用本模块时不引入 torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)   # 无 CUDA 时为 no-op，安全
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed


def derive_batch_seed(base_seed, batch_index):
    """由基础种子派生出确定性的「每批次」整数种子。

    用 :class:`numpy.random.SeedSequence` 以 ``(base_seed, batch_index)`` 为键，
    保证 (a) 相邻批次得到互不相同、良好混合的种子；(b) 相同 ``base_seed`` 的两次
    运行派生出完全相同的批次种子序列 —— 这正是各处 redraw/补采循环在固定种子下
    可复现的基础。

    Args:
        base_seed: 已校验的基础整数种子。
        batch_index: 批次 / 轮次的从 0 开始的索引。

    Returns:
        int: 该批次的确定性 32-bit 种子。
    """
    base_seed = _require_uint32(base_seed, "base_seed")
    batch_index = _require_uint32(batch_index, "batch_index")
    seq = np.random.SeedSequence([base_seed, batch_index])
    return int(seq.generate_state(1)[0])
