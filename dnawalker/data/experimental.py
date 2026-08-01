# coding=utf-8
"""
dnawalker.data.experimental — 实验荧光数据 Excel 的健壮读取工具
dnawalker.data.experimental — Robust loader for experimental fluorescence Excel files.

Fig3a_fitting.xlsx 的实际结构与早期 predict.py 假设的不同：
  - 第一列是时间(分钟)，列名为 'Unnamed: 0' 而非 'Time'
  - 首行可能是 'Panel A' 之类的字符串标题，需作为非数值剔除
  - 信号列名为 'FAM/FAM T (+)' / 'TYE/TYE T (-)' / 'CY5/CY5 T (m)'

本函数对列名做模糊匹配 + 数值强制转换，兼容上述情况。
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def _find_col(columns, *keywords):
    """在列名中模糊查找包含任一关键字的列 (大小写不敏感)。"""
    for col in columns:
        name = str(col).lower()
        for kw in keywords:
            if kw in name:
                return col
    return None


def load_experimental_curves(data_path):
    """读取实验 Excel，返回 (time_min, fam, tye, cy5) 四个 1D float 数组。

    对列名做模糊匹配，对单元格做数值强制转换 (非数值→NaN)，剔除任一列
    为 NaN/Inf 的行，按时间稳定排序，并把重复时间点的信号取平均。无法得到
    至少两个不同时间点时抛出 ValueError。
    """
    df = pd.read_excel(data_path)
    cols = list(df.columns)
    if len(cols) < 4:
        raise ValueError(f"实验数据列数不足 (得到 {len(cols)} 列): {cols}")

    fam_col = _find_col(cols, 'fam')
    tye_col = _find_col(cols, 'tye')
    cy5_col = _find_col(cols, 'cy5')

    # 时间列：优先名字含 'time'，否则取第一列
    time_col = _find_col(cols, 'time')
    if time_col is None:
        time_col = cols[0]

    # 信号列缺失时，退回按位置取第 2/3/4 列
    if fam_col is None:
        fam_col = cols[1]
    if tye_col is None:
        tye_col = cols[2]
    if cy5_col is None:
        cy5_col = cols[3]

    def to_num(series):
        return pd.to_numeric(series, errors='coerce').values.astype(float)

    t = to_num(df[time_col])
    fam = to_num(df[fam_col])
    tye = to_num(df[tye_col])
    cy5 = to_num(df[cy5_col])

    mask = np.isfinite(t) & np.isfinite(fam) & np.isfinite(tye) & np.isfinite(cy5)
    if not mask.any():
        raise ValueError("实验数据没有完整的有限数值行，无法解析。")

    t, fam, tye, cy5 = (values[mask] for values in (t, fam, tye, cy5))
    order = np.argsort(t, kind="stable")
    t, fam, tye, cy5 = (values[order] for values in (t, fam, tye, cy5))

    unique_t, first_idx, counts = np.unique(
        t, return_index=True, return_counts=True
    )
    if unique_t.size < 2:
        raise ValueError("实验数据必须包含至少两个不同的有限时间点。")
    if unique_t.size != t.size:
        fam = np.add.reduceat(fam, first_idx) / counts
        tye = np.add.reduceat(tye, first_idx) / counts
        cy5 = np.add.reduceat(cy5, first_idx) / counts
        t = unique_t

    return t, fam, tye, cy5


def load_and_smooth_experimental_curves(
    config,
    data_path: str,
    verbose: bool = False,
    on_load_error: str = "return_none",
) -> Optional[np.ndarray]:
    """加载实验 Excel → 插值到标准时间轴 → SG 平滑 → 返回 (3, T) ndarray。

    CNN 与 Transformer 各模块"加载实验数据 + 插值 + SG 平滑"流程的统一实现，
    集中为单一事实来源以避免多处重复代码漂移。

    Args:
        config: 任意暴露 ``get_sim_total_time / get_num_time_points /
                get_sg_window / get_sg_polyorder`` 的对象。
        data_path: 实验数据 Excel 路径。
        verbose: True 时打印进度/警告 (predict.py 风格)；
                 False 时静默执行 (eval_*.py 风格)。
        on_load_error: 加载失败时的行为：
            ``"return_none"`` — 捕获 ``FileNotFoundError/ValueError/KeyError``
              并返回 ``None`` (predict.py 风格，用户用 ``if X is None: return``)。
            ``"raise"`` — 不 try-wrap，原始异常向上冒泡 (eval_*.py 风格)。

    Returns:
        (3, T) numpy 数组：[FAM, TYE, CY5] × num_time_points。
        ``on_load_error="return_none"`` 且加载失败时返回 ``None``。
        SG 平滑失败时回退到未平滑曲线 (ValueError) 并仅在 verbose=True 时打印警告。

    Note:
        损坏 xlsx (zipfile.BadZipFile) 故意不在 ``return_none`` 的 except 列表内
        — 让它向上冒泡以便诊断；生产路径下数据文件随仓库分发，损坏概率极低。
    """
    if on_load_error not in ("return_none", "raise"):
        raise ValueError(
            f"on_load_error 必须是 'return_none' 或 'raise'，得到 {on_load_error!r}"
        )

    if verbose:
        print(f"--- 加载实验数据 / Loading experimental data: {data_path} ---")

    sim_total_time = config.get_sim_total_time()
    num_time_points = config.get_num_time_points()
    standard_time_axis = np.linspace(0, sim_total_time, num_time_points)

    if on_load_error == "raise":
        # 无 try wrap：原始异常类型 (FileNotFoundError/ValueError/KeyError)
        # 向上冒泡，与 eval_*.py 的旧实现等价。
        exp_time, exp_fam, exp_tye, exp_cy5 = load_experimental_curves(data_path)
    else:
        try:
            exp_time, exp_fam, exp_tye, exp_cy5 = load_experimental_curves(data_path)
        except (FileNotFoundError, ValueError, KeyError) as e:
            if verbose:
                print(f"错误 / Error: {e}")
            return None

    # 线性插值 (load_experimental_curves 已剔除 NaN 行)
    funcs = [
        interp1d(exp_time, d, kind='linear', bounds_error=False,
                 fill_value=(d[0], d[-1]))
        for d in (exp_fam, exp_tye, exp_cy5)
    ]
    curves = [f(standard_time_axis) for f in funcs]

    # SG 平滑 — 失败时回退到未平滑
    sg_window = config.get_sg_window()
    sg_poly = config.get_sg_polyorder()
    try:
        curves = [savgol_filter(c, sg_window, sg_poly) for c in curves]
        if verbose:
            print(f"SG 平滑完成 / SG smoothing done (window={sg_window}, poly={sg_poly})")
    except ValueError as e:
        # savgol_filter 在 window > 信号长度 / poly >= window 时抛 ValueError
        if verbose:
            print(f"警告 / Warning: SG 平滑失败 / SG smoothing failed: {e}")

    return np.stack(curves, axis=0)


def raw_experiment_curves_on_sim_axis(
    data_path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """加载实验数据，插值到 ``pysim`` 仿真时间轴 (不平滑)。

    用作 RMSE 比较的真值参考：以 ``pysim.NUM_RESULTS`` 个点（按
    ``pysim.SAVE_INTERVAL_SEC`` 秒采样）等距覆盖 0..NUM_RESULTS-1 秒，端点
    外推用首尾值（np.interp 的 ``left/right`` 参数）。

    Args:
        data_path: 实验数据 Excel 路径。

    Returns:
        (sim_time_min, curves) — ``sim_time_min`` 是分钟单位的时间轴，
        ``curves`` 形状为 ``(3, pysim.NUM_RESULTS)``，行序 [FAM, TYE, CY5]。

    Raises:
        FileNotFoundError / ValueError / KeyError: 来自 ``load_experimental_curves``。
    """
    # 延迟导入：保持普通 Excel 加载路径不初始化物理模拟器。
    from dnawalker.physics import simulator as pysim

    t, fam, tye, cy5 = load_experimental_curves(data_path)
    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)
    out = [
        np.interp(sim_time, t, y, left=y[0], right=y[-1])
        for y in (fam, tye, cy5)
    ]
    return sim_time, np.stack(out, axis=0)
