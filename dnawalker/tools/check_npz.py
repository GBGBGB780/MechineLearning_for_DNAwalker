# coding=utf-8
"""
dnawalker.tools.check_npz — 检查 NPZ 数据集内容的诊断工具
dnawalker.tools.check_npz — Diagnostic tool for inspecting NPZ dataset contents

用法 / Usage:
    python -m dnawalker.tools.check_npz <npz_path> [sample_idx]
    python -m dnawalker.tools.check_npz artifacts/datasets/training_dataset.npz 100
"""

import argparse
import os

import numpy as np

from dnawalker.paths import ARTIFACTS_DIR


def check_npz(npz_path, sample_idx=0):
    """
    加载并检查 NPZ 数据集的结构和数据质量。
    Load and inspect the structure and data quality of an NPZ dataset.

    Args:
        npz_path:   NPZ 文件路径 / path to the NPZ file
        sample_idx: 要检查的样本索引 / sample index to inspect
    """
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            print(f"NPZ 文件内容 / NPZ contents: {data.files}")
            X = np.asarray(data['X'])
            Y = np.asarray(data['Y'])
            if 'parameter_names' in data.files:
                param_names = [
                    str(name) for name in np.asarray(
                        data['parameter_names']
                    ).reshape(-1)
                ]
            else:
                param_names = [
                    'E_b', 'E_b_azo_trans', 'E_b_azo_cis',
                    'k_mig', 'k0', 'drt_z', 'drt_s',
                ]
                print("警告 / Warning: parameter_names metadata is missing")

            if X.ndim != 3 or Y.ndim != 2:
                raise ValueError(
                    f"Expected X(N,C,T) and Y(N,P), got X={X.shape}, Y={Y.shape}"
                )
            if X.shape[0] == 0 or X.shape[0] != Y.shape[0]:
                raise ValueError(
                    "X/Y must contain the same non-zero sample count: "
                    f"X={X.shape}, Y={Y.shape}"
                )
            if Y.shape[1] != len(param_names):
                raise ValueError(
                    "Y column count does not match parameter_names: "
                    f"{Y.shape[1]} vs {len(param_names)}"
                )

            print(f"\nX (曲线/curves) shape: {X.shape}")
            print(f"Y (参数/params) shape: {Y.shape}")
            print(f"参数顺序 / Parameter order: {param_names}")

            # 校正索引范围 / Clamp index to valid range
            if sample_idx < 0 or sample_idx >= X.shape[0]:
                print(
                    f"警告 / Warning: sample_idx {sample_idx} 超出范围，"
                    "重置为 0 / out of range, reset to 0"
                )
                sample_idx = 0

            sample_X = X[sample_idx]
            sample_Y = Y[sample_idx]
            curve_labels = ['sim_fam', 'sim_tye', 'sim_cy5']

            # 显示 X 的前 10 个时间点 / Show first 10 time points
            print(
                f"\n--- 样本 {sample_idx} 的前 10 个时间点 / "
                f"First 10 time points of sample {sample_idx} ---"
            )
            for c in range(min(sample_X.shape[0], len(curve_labels))):
                print(f"  {curve_labels[c]:<15}: {sample_X[c, :10]}")

            # 显示 Y 参数值 / Show Y parameter values
            print(
                f"\n--- 样本 {sample_idx} 的参数值 / "
                f"Parameter values of sample {sample_idx} ---"
            )
            for i, val in enumerate(sample_Y):
                print(f"  {param_names[i]:<15}: {val:.6e}")

            # 数据质量检查 / Data quality check
            nan_x = np.isnan(X).any(axis=(1, 2)).sum()
            inf_x = np.isinf(X).any(axis=(1, 2)).sum()
            nan_y = np.isnan(Y).sum()
            inf_y = np.isinf(Y).sum()

            print("\n--- 数据质量 / Data quality ---")
            print(f"X: NaN 样本数={nan_x}, Inf 样本数={inf_x}")
            print(f"Y: NaN 个数={nan_y}, Inf 个数={inf_y}")
            return not any((nan_x, inf_x, nan_y, inf_y))
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"错误 / Error: 无法检查 '{npz_path}': {exc}")
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect an NPZ training dataset and report data quality."
    )
    parser.add_argument(
        "npz_path",
        nargs="?",
        default=os.fspath(
            ARTIFACTS_DIR / "datasets" / "training_dataset.npz"
        ),
        help=(
            "Dataset path (default: repository "
            "artifacts/datasets/training_dataset.npz)."
        ),
    )
    parser.add_argument(
        "sample_idx",
        nargs="?",
        type=int,
        default=0,
        help="Sample index to display (default: 0).",
    )
    args = parser.parse_args(argv)
    return 0 if check_npz(args.npz_path, args.sample_idx) else 1


if __name__ == "__main__":
    raise SystemExit(main())
