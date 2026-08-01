# coding=utf-8
"""
dnawalker.shared.parameters — matlab_input_params.txt 的统一读写 (单一事实来源)
dnawalker.shared.parameters — Single source of truth for the params exchange file.

该文件是预测模块与验证模块之间的参数交换格式：每行 ``key=value``，键为小写的
物理参数名，末行 ``END_OF_PARAMS=1``。写侧 (各模型的 ``predict --refine``) 与读侧
(``dnawalker.verify``) 共用这一份键列表/映射，键名一律由
``pysim.PARAM_NAMES`` 派生，保证读写与物理内核永远一致。

This is the parameter-exchange format between the prediction scripts and the
verification script: one ``key=value`` per line (lowercase physical-parameter
names) terminated by ``END_OF_PARAMS=1``. Keys are derived from
``pysim.PARAM_NAMES`` so they can never drift from the physics kernel.
"""

import numpy as np
from pathlib import Path

from dnawalker.physics import simulator as pysim

# 文件中使用的小写键顺序，以及 小写键 -> pysim 精确大小写名 的映射。
# 均由 pysim.PARAM_NAMES 派生，确保与物理内核同步 (例如 'E_b' <-> 'e_b')。
LOWER_KEYS = [name.lower() for name in pysim.PARAM_NAMES]
KEY_MAP = {name.lower(): name for name in pysim.PARAM_NAMES}


def build_name_map(config_param_names):
    """把配置的 trainable 参数名映射到 ``pysim.PARAM_NAMES`` 的精确大小写形式。

    历史上各处用 ``dict(zip(config_names, pysim.PARAM_NAMES))`` **按位置**建映射，
    正确性完全依赖"配置顺序恰好等于 pysim 顺序"这一未校验的隐含约定 —— 一旦有人
    重排 ``[PHYSICAL_PARAMETERS]`` 或编辑 ``pysim.PARAM_NAMES``，每个预测参数会被
    静默错标且无任何报错。本函数改为**按名字 (大小写无关) 匹配**，对当前配置产生
    与位置 zip **完全相同**的映射 (configparser 把键转小写，如 'e_b' ↔ 'E_b')，但
    对重排稳健，并在名字集合不一致时**显式报错**而非静默错标。

    Args:
        config_param_names: ``config.get_trainable_param_names()`` 的返回值
            (通常为小写的参数名列表)。

    Returns:
        dict: ``{config_name: pysim_name}``，键为传入的原始名字，值为
        ``pysim.PARAM_NAMES`` 中对应的精确大小写名。

    Raises:
        ValueError: 若配置参数名 (大小写无关) 的集合与 ``pysim.PARAM_NAMES`` 不一致
            (数量不符、缺参数、有多余/拼错的参数)。
    """
    config_lower = [str(n).lower() for n in config_param_names]
    if sorted(config_lower) != sorted(KEY_MAP.keys()):
        raise ValueError(
            "配置的 trainable 参数与 pysim.PARAM_NAMES 不一致 / config trainable "
            f"params do not match pysim.PARAM_NAMES.\n  config: {list(config_param_names)}\n"
            f"  pysim : {pysim.PARAM_NAMES}"
        )
    return {orig: KEY_MAP[low] for orig, low in zip(config_param_names, config_lower)}


def vector_to_param_dict(values, parameter_names):
    """Map a one-dimensional prediction vector to canonical ``pysim`` names.

    Model outputs follow the configured trainable-parameter order, which may be
    different from ``pysim.PARAM_NAMES`` after an INI section reorder. Mapping
    by name prevents a valid vector from being silently assigned to the wrong
    physical parameters.
    """
    vector = np.asarray(values)
    names = list(parameter_names)
    if vector.ndim != 1 or vector.shape[0] != len(names):
        raise ValueError(
            "参数向量形状与名称数量不一致 / parameter vector/name mismatch: "
            f"shape={vector.shape}, names={len(names)}"
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError(
            f"参数向量包含 NaN/Inf / parameter vector is non-finite: {vector}"
        )
    name_map = build_name_map(names)
    return {
        name_map[name]: float(vector[index])
        for index, name in enumerate(names)
    }


def resolve_checkpoint_param_names(checkpoint, configured_param_names):
    """Resolve the output-column order recorded by a model checkpoint.

    New checkpoints store ``param_names`` alongside ``model_state``. That
    order is authoritative because both the output layer and y-scaler were
    fitted in it. Historical checkpoints without metadata are accepted only
    while the config still uses canonical ``pysim`` order.
    """
    configured = [_decode_parameter_name(name) for name in configured_param_names]
    build_name_map(configured)

    raw_names = (
        checkpoint.get("param_names")
        if isinstance(checkpoint, dict)
        else None
    )
    if raw_names is None:
        canonical = [name.lower() for name in pysim.PARAM_NAMES]
        if [name.lower() for name in configured] != canonical:
            raise ValueError(
                "旧 checkpoint 缺少 param_names，且配置顺序已改变；无法安全推断"
                "模型输出顺序 / legacy checkpoint has no parameter-order metadata"
            )
        return configured

    names = [
        _decode_parameter_name(name)
        for name in np.asarray(raw_names).reshape(-1)
    ]
    build_name_map(names)
    return names


def _decode_parameter_name(value):
    """Normalize a NumPy string/bytes scalar to a non-empty Python string."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    name = str(value).strip()
    if not name:
        raise ValueError("数据集 parameter_names 包含空名称 / contains an empty name")
    return name


def load_npz_dataset(
    path,
    expected_param_names,
    *,
    allow_legacy_canonical=False,
    x_dtype=np.float32,
    y_dtype=np.float32,
    expected_x_shape=None,
):
    """Load ``X``/``Y`` and align label columns by ``parameter_names``.

    New datasets must carry a ``parameter_names`` array.  Its names are matched
    case-insensitively and ``Y`` is reordered to ``expected_param_names``.
    This prevents a config-section reorder from silently assigning every label
    to the wrong physical parameter.

    Historical NPZ files did not always carry metadata.  A caller may opt into
    the documented historical assumption that such columns use
    ``pysim.PARAM_NAMES`` order by passing ``allow_legacy_canonical=True``.
    The fallback is deliberately explicit rather than silently guessing.
    """
    expected = [_decode_parameter_name(name) for name in expected_param_names]
    # Also verifies that the configured trainable set exactly matches pysim.
    build_name_map(expected)

    with np.load(path, allow_pickle=False) as dataset:
        X = np.asarray(dataset["X"], dtype=x_dtype)
        Y = np.asarray(dataset["Y"], dtype=y_dtype)
        if "parameter_names" in dataset.files:
            raw_names = np.asarray(dataset["parameter_names"]).reshape(-1)
            source = [_decode_parameter_name(name) for name in raw_names]
        elif allow_legacy_canonical:
            source = list(pysim.PARAM_NAMES)
        else:
            raise ValueError(
                "数据集缺少 parameter_names；拒绝猜测 Y 列顺序 / dataset is "
                "missing parameter_names; refusing to guess Y-column order"
            )

    if X.ndim != 3 or Y.ndim != 2:
        raise ValueError(
            f"数据集形状无效 / invalid dataset shapes: X={X.shape}, Y={Y.shape}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "X/Y 样本数不一致 / sample-count mismatch: "
            f"X={X.shape[0]}, Y={Y.shape[0]}"
        )
    if X.shape[0] == 0 or any(size <= 0 for size in X.shape[1:]):
        raise ValueError(
            f"数据集不能为空 / dataset dimensions must be non-empty: X={X.shape}"
        )
    if (expected_x_shape is not None
            and tuple(X.shape[1:]) != tuple(expected_x_shape)):
        raise ValueError(
            "X 曲线形状与配置不一致 / curve shape does not match config: "
            f"{X.shape[1:]} vs {tuple(expected_x_shape)}"
        )
    if Y.shape[1] != len(source):
        raise ValueError(
            "Y 列数与 parameter_names 不一致 / Y-column/name mismatch: "
            f"{Y.shape[1]} columns vs {len(source)} names"
        )

    source_lower = [name.lower() for name in source]
    expected_lower = [name.lower() for name in expected]
    if len(set(source_lower)) != len(source_lower):
        raise ValueError(
            f"数据集 parameter_names 含重复项 / duplicate names: {source}"
        )
    if set(source_lower) != set(expected_lower):
        raise ValueError(
            "数据集参数与配置不一致 / dataset parameters do not match config."
            f"\n  dataset: {source}\n  config : {expected}"
        )

    source_index = {name: idx for idx, name in enumerate(source_lower)}
    order = [source_index[name] for name in expected_lower]
    return X, Y[:, order], expected


def write_matlab_params(pdict, path):
    """将 7 个物理参数写入 ``matlab_input_params.txt``。

    行序与 ``pysim.PARAM_NAMES`` 一致，值用 ``repr(float(...))`` 写出以保留全
    精度 (与历史写侧逐字等价)，末行写 ``END_OF_PARAMS=1``。

    Args:
        pdict: 以 ``pysim.PARAM_NAMES`` 为键的参数 dict。
        path: 输出文件路径。
    """
    missing = [name for name in pysim.PARAM_NAMES if name not in pdict]
    if missing:
        raise KeyError(f"参数字典缺少必需参数 / missing parameters: {missing}")
    values = {name: float(pdict[name]) for name in pysim.PARAM_NAMES}
    nonfinite = [name for name, value in values.items() if not np.isfinite(value)]
    if nonfinite:
        raise ValueError(
            f"参数必须是有限数 / parameters must be finite: {nonfinite}"
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w') as f:
        for lk, pk in zip(LOWER_KEYS, pysim.PARAM_NAMES):
            f.write(f"{lk}={repr(values[pk])}\n")
        f.write("END_OF_PARAMS=1\n")


def read_matlab_params(path):
    """读取 ``key=value`` 形式的参数文件，返回以 ``pysim.PARAM_NAMES`` 为键的 dict。

    对键名做小写归一化，跳过空行、无 ``=`` 的行以及 ``END_OF_PARAMS`` 标记，
    非数值的值被忽略 (与历史读侧逐字等价)。缺少任一必需参数时抛出 ``KeyError``。

    Args:
        path: 参数文件路径。

    Returns:
        dict: ``{pysim 参数名: float}``，含全部 7 个参数。

    Raises:
        KeyError: 若文件缺少任一必需参数。
    """
    raw = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip().lower()
            if k == 'end_of_params':
                continue
            try:
                value = float(v.strip())
            except ValueError:
                pass
            else:
                if not np.isfinite(value):
                    raise ValueError(f"参数 {k} 不是有限数 / is not finite")
                raw[k] = value

    params = {}
    for src, dst in KEY_MAP.items():
        if src not in raw:
            raise KeyError(f"参数文件缺少必需参数: {src}")
        params[dst] = raw[src]
    return params
