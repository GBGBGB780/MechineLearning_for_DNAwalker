# coding=utf-8
"""
dnawalker.shared.device — 统一的 PyTorch 设备选择 / Unified PyTorch device selection.

CNN 与 Transformer 两个推理适配器 (``dnawalker.cnn.inference`` 与
``dnawalker.transformer.inference``) 共享同一份 CUDA > MPS > CPU 选择逻辑，
集中到此处保证两个模型选设备的行为永远一致。

Both inference adapters share this single CUDA > MPS > CPU device selection so
their behavior stays identical.
"""

import torch


def pick_device():
    """按 CUDA > MPS > CPU 优先级返回首个可用的 :class:`torch.device`。

    与两个推理器的历史逻辑逐字等价：优先 CUDA，其次 Apple Silicon 的 MPS，
    最后回退 CPU。

    Returns:
        torch.device: 首个可用设备。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
