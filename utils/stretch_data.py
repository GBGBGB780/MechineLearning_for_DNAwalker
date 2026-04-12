# coding=utf-8
"""
stretch_data.py — 实验数据振幅拉伸工具
stretch_data.py — Experimental data amplitude stretching utility

将荧光曲线的振幅拉伸 N 倍（保持起始点不变），用于测试模型对不同信号幅度的鲁棒性。
Stretches fluorescence curve amplitudes by a factor of N (keeping start point fixed),
useful for testing model robustness to different signal magnitudes.

用法 / Usage:
    python stretch_data.py
"""

import pandas as pd
import os

# 文件路径 / File paths
INPUT_FILE = 'Fig3a_fitting.xlsx'
OUTPUT_FILE = 'Fig3a_fitting_stretched.xlsx'


def stretch_y_data(input_path, output_path, factor=2):
    """
    拉伸荧光曲线振幅：y_new = y_start + factor * (y_old - y_start)
    Stretch fluorescence curve amplitudes: y_new = y_start + factor * (y_old - y_start)

    Args:
        input_path:  输入 Excel 路径 / input Excel path
        output_path: 输出 Excel 路径 / output Excel path
        factor:      拉伸倍数 / stretch factor
    """
    df = pd.read_excel(input_path)
    print(f"读取文件 / Reading: {input_path}")
    print(f"列名 / Columns: {df.columns.tolist()}")

    columns_to_stretch = ['FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)']

    for col in columns_to_stretch:
        if col in df.columns:
            start_val = df[col].iloc[0]
            df[col] = start_val + factor * (df[col] - start_val)
            print(f"  {col}: 拉伸 {factor}x / stretched {factor}x")
        else:
            print(f"  警告 / Warning: 未找到列 '{col}' / column not found")

    df.to_excel(output_path, index=False)
    print(f"\n完成 / Done! 保存至 / Saved to: {output_path}")


def compare_files(f1, f2):
    """
    对比原始与拉伸后数据的首尾值。
    Compare start/end values between original and stretched data.
    """
    df1 = pd.read_excel(f1)
    df2 = pd.read_excel(f2)
    cols = ['FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)']

    print(f"\n{'Column':<20} | {'Original Start/End':<40} | {'New Start/End':<40}")
    print("-" * 110)
    for col in cols:
        if col in df1.columns and col in df2.columns:
            s1, e1 = df1[col].iloc[0], df1[col].iloc[-1]
            s2, e2 = df2[col].iloc[0], df2[col].iloc[-1]
            print(f"{col:<20} | {s1:.6f} / {e1:.6f} | {s2:.6f} / {e2:.6f}")


if __name__ == "__main__":
    if os.path.exists(INPUT_FILE):
        stretch_y_data(INPUT_FILE, OUTPUT_FILE)
        compare_files(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"错误 / Error: 找不到文件 / File not found: {INPUT_FILE}")
