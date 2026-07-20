# coding=utf-8
"""
verify.py — 正向物理验证 (verify.m 的 Python 移植，无需 MATLAB)
verify.py — Forward physics verification, Python port of verify.m (no MATLAB required).

流程 / Pipeline:
  1. 读取 matlab_input_params.txt 中预测出的 7 个物理参数
  2. 用 pysim 运行正向模拟，得到 FAM/TYE/CY5 三条信号曲线
  3. 读取实验 Excel，插值到模拟时间轴
  4. 计算逐通道 RMSE 并绘制"模拟 vs 实验"对比图 (保存为 png)

用法 / Usage:
    python verify.py train_transformer/matlab_input_params.txt
    python verify.py train_cnn/matlab_input_params.txt --exp results/Fig3a_fitting.xlsx
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import pysim  # noqa: E402

# matlab_input_params.txt 用小写键；映射到 pysim 的参数名
KEY_MAP = {
    'e_b': 'E_b',
    'e_b_azo_trans': 'E_b_azo_trans',
    'e_b_azo_cis': 'E_b_azo_cis',
    'k_mig': 'k_mig',
    'k0': 'k0',
    'drt_z': 'drt_z',
    'drt_s': 'drt_s',
}


def read_params(path):
    """读取 key=value 形式的参数文件，返回 pysim 参数 dict。"""
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
                raw[k] = float(v.strip())
            except ValueError:
                pass

    params = {}
    for src, dst in KEY_MAP.items():
        if src not in raw:
            raise KeyError(f"参数文件缺少必需参数: {src}")
        params[dst] = raw[src]
    return params


def read_experimental(xlsx_path):
    """读取实验 Excel，返回 (time_min, fam, tye, cy5)。

    Fig3a_fitting.xlsx 结构：第一列为时间(分钟)，2-4 列为 FAM/TYE/CY5，
    含一行 'Panel A' 字符串需跳过。
    """
    df = pd.read_excel(xlsx_path)
    cols = list(df.columns)
    # 时间列：第一列 (可能名为 'Unnamed: 0' 或 'Time')
    time_col = cols[0]
    fam_col, tye_col, cy5_col = cols[1], cols[2], cols[3]

    def to_num(series):
        return pd.to_numeric(series, errors='coerce').values

    t = to_num(df[time_col])
    fam = to_num(df[fam_col])
    tye = to_num(df[tye_col])
    cy5 = to_num(df[cy5_col])

    mask = ~(np.isnan(t) | np.isnan(fam) | np.isnan(tye) | np.isnan(cy5))
    return t[mask], fam[mask], tye[mask], cy5[mask]


def main():
    ap = argparse.ArgumentParser(description="DNA Walker 正向物理验证 (Python)")
    ap.add_argument('params', nargs='?',
                    default='train_transformer/matlab_input_params.txt',
                    help='matlab_input_params.txt 路径')
    ap.add_argument('--exp', default='results/Fig3a_fitting.xlsx',
                    help='实验数据 Excel 路径')
    ap.add_argument('--out', default=None, help='对比图输出 png 路径')
    ap.add_argument('--no-plot', action='store_true', help='只算 RMSE 不绘图')
    args = ap.parse_args()

    params = read_params(args.params)
    print("读取参数:")
    for k, v in params.items():
        print(f"  {k:<16} = {v:.6e}")

    print("\n运行正向模拟...")
    signals, dt_used = pysim.run_simulation(params)
    if dt_used < 0:
        print("错误：该参数组合产生无效模拟 (dt 过小或非有限)，无法验证。")
        sys.exit(1)
    print(f"  dt = {dt_used:.2e} s,  信号形状 = {signals.shape}")

    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)  # 分钟
    sim_fam, sim_tye, sim_cy5 = signals[0], signals[1], signals[2]

    exp_t, exp_fam, exp_tye, exp_cy5 = read_experimental(args.exp)
    print(f"实验数据点数: {len(exp_t)}  时间范围: [{exp_t.min():.2f}, {exp_t.max():.2f}] min")

    # 插值实验到模拟时间轴 (线性 + 端点外推)
    def interp(y):
        return np.interp(sim_time, exp_t, y, left=y[0], right=y[-1])

    exp_fam_i = interp(exp_fam)
    exp_tye_i = interp(exp_tye)
    exp_cy5_i = interp(exp_cy5)

    rmse_fam = float(np.sqrt(np.nanmean((sim_fam - exp_fam_i) ** 2)))
    rmse_tye = float(np.sqrt(np.nanmean((sim_tye - exp_tye_i) ** 2)))
    rmse_cy5 = float(np.sqrt(np.nanmean((sim_cy5 - exp_cy5_i) ** 2)))
    rmse_total = (rmse_fam + rmse_tye + rmse_cy5) / 3.0

    print("\n=== RMSE (模拟 vs 实验) ===")
    print(f"  FAM   : {rmse_fam:.4f}")
    print(f"  TYE   : {rmse_tye:.4f}")
    print(f"  CY5   : {rmse_cy5:.4f}")
    print(f"  平均  : {rmse_total:.4f}")

    if args.no_plot:
        return

    try:
        import matplotlib
        matplotlib.use('Agg')   # 无显示环境，保存为文件
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"未能导入 matplotlib，跳过绘图: {e}")
        return

    out = args.out or (os.path.splitext(args.params)[0] + '_verify.png')
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    panels = [
        ('FAM', sim_fam, exp_t, exp_fam, rmse_fam),
        ('TYE', sim_tye, exp_t, exp_tye, rmse_tye),
        ('CY5', sim_cy5, exp_t, exp_cy5, rmse_cy5),
    ]
    for ax, (name, sim_y, et, ey, rmse) in zip(axes, panels):
        ax.plot(sim_time, sim_y, 'r-', lw=2, label='Simulation')
        ax.scatter(et, ey, s=8, c='b', alpha=0.3, label='Experimental')
        ax.set_xlabel('Time (min)')
        ax.set_ylabel(f'{name} Signal')
        ax.set_title(f'{name}: Simulation vs Experimental  (RMSE={rmse:.4f})',
                     fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\n对比图已保存: {out}")


if __name__ == "__main__":
    main()
