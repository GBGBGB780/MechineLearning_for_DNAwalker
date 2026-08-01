# coding=utf-8
"""
dnawalker.verify — 正向物理验证 (源自归档 MATLAB verify.m 的 Python 实现)
dnawalker.verify — Forward physics verification (no MATLAB required).

流程 / Pipeline:
  1. 读取 matlab_input_params.txt 中预测出的 7 个物理参数
  2. 用 pysim 运行正向模拟，得到 FAM/TYE/CY5 三条信号曲线
  3. 读取实验 Excel，插值到模拟时间轴
  4. 计算逐通道 RMSE 并绘制"模拟 vs 实验"对比图 (保存为 png)

Usage:
    python -m dnawalker.verify
    python -m dnawalker.verify results/predictions/cnn/matlab_input_params.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np

from dnawalker.paths import DATA_DIR, PREDICTIONS_DIR, RESULTS_DIR
from dnawalker.physics import simulator as pysim
from dnawalker.shared.parameters import read_matlab_params
from dnawalker.data.experimental import load_experimental_curves


_DEFAULT_PARAMS = str(
    PREDICTIONS_DIR / "transformer" / "matlab_input_params.txt"
)
_DEFAULT_EXPERIMENT = str(DATA_DIR / "experimental" / "Fig3a_fitting.xlsx")
_DEFAULT_VERIFICATION_DIR = RESULTS_DIR / "evaluation" / "verification"


def _default_output_path(params_path):
    """Keep canonical verification figures inside the owning model directory."""
    params = Path(params_path).expanduser().resolve()
    try:
        relative = params.relative_to(PREDICTIONS_DIR.resolve())
    except ValueError:
        output_dir = _DEFAULT_VERIFICATION_DIR
    else:
        model = relative.parts[0] if len(relative.parts) > 1 else None
        if model in {"cnn", "transformer"}:
            run_subdir = Path(*relative.parts[1:-1])
            output_dir = RESULTS_DIR / "evaluation" / model / run_subdir
        else:
            output_dir = _DEFAULT_VERIFICATION_DIR
    return str(output_dir / f"{params.stem}_verify.png")


def read_experimental(xlsx_path):
    """读取实验 Excel，返回 (time_min, fam, tye, cy5)。

    复用 ``dnawalker.data.experimental.load_experimental_curves``：
    对列名做模糊匹配 (兼容 'Unnamed: 0' 时间列、'FAM/FAM T (+)' 等信号列名)，
    数值强制转换并剔除非数值行，避免本脚本与预测脚本各自维护一份易脆的列解析逻辑。
    """
    return load_experimental_curves(xlsx_path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="DNA Walker 正向物理验证 (Python)")
    ap.add_argument('params', nargs='?',
                    default=_DEFAULT_PARAMS,
                    help='matlab_input_params.txt 路径')
    ap.add_argument('--exp', default=_DEFAULT_EXPERIMENT,
                    help='实验数据 Excel 路径')
    ap.add_argument('--out', default=None, help='对比图输出 png 路径')
    ap.add_argument('--no-plot', action='store_true', help='只算 RMSE 不绘图')
    args = ap.parse_args(argv)

    params = read_matlab_params(args.params)
    print("读取参数:")
    for k, v in params.items():
        print(f"  {k:<16} = {v:.6e}")

    print("\n运行正向模拟...")
    signals, dt_used = pysim.run_simulation(params)
    signals = np.asarray(signals, dtype=np.float64)
    if (not np.isfinite(dt_used)
            or dt_used < 0
            or signals.shape != (3, pysim.NUM_RESULTS)
            or not np.all(np.isfinite(signals))):
        print("错误：该参数组合产生无效模拟 (dt 过小或非有限)，无法验证。")
        return 1
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
        return 0

    try:
        import matplotlib
        matplotlib.use('Agg')   # 无显示环境，保存为文件
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"未能导入 matplotlib，跳过绘图: {e}")
        return 0

    out = args.out
    if out is None:
        out = _default_output_path(args.params)
    output_parent = os.path.dirname(os.path.abspath(out))
    os.makedirs(output_parent, exist_ok=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
