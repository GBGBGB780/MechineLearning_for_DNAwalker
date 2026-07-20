# coding=utf-8
"""
exp_data_io.py — 实验荧光数据 Excel 的健壮读取工具
exp_data_io.py — Robust loader for experimental fluorescence Excel files.

Fig3a_fitting.xlsx 的实际结构与早期 predict.py 假设的不同：
  - 第一列是时间(分钟)，列名为 'Unnamed: 0' 而非 'Time'
  - 首行可能是 'Panel A' 之类的字符串标题，需作为非数值剔除
  - 信号列名为 'FAM/FAM T (+)' / 'TYE/TYE T (-)' / 'CY5/CY5 T (m)'

本函数对列名做模糊匹配 + 数值强制转换，兼容上述情况。
"""

import numpy as np
import pandas as pd


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

    对列名做模糊匹配，对单元格做数值强制转换 (非数值→NaN)，并剔除任一
    列为 NaN 的行。无法识别列时抛出 ValueError。
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

    mask = ~(np.isnan(t) | np.isnan(fam) | np.isnan(tye) | np.isnan(cy5))
    if not mask.any():
        raise ValueError("实验数据全部为非数值，无法解析。")
    return t[mask], fam[mask], tye[mask], cy5[mask]
