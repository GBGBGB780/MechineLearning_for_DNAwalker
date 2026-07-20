# coding=utf-8
"""
refine.py — 物理参数局部精修 (直接最小化曲线 RMSE)
refine.py — Local refinement of physical parameters by directly minimizing curve RMSE.

逆问题中，DL 预测的参数 MSE 小 ≠ 曲线 RMSE 小。本模块把 DL 预测作为初值，
用 pysim 正向模拟 + 局部优化 (Nelder-Mead / Powell) 直接最小化与实验曲线的 RMSE。
这是原项目 [PREDICTION] 节 nm_lower_bound/nm_upper_bound 暗示但从未在 Python 端实现的步骤。

注意：k0 跨数量级，在 log10 空间优化更稳定。
"""

import numpy as np
from scipy.optimize import minimize

import pysim

# 在 log10 空间优化的参数
LOG_PARAMS = {'k0'}


def _to_opt_space(params, ranges):
    """物理参数 dict → 优化向量 (k0 取 log10)。"""
    vec = []
    for name in pysim.PARAM_NAMES:
        v = params[name]
        if name in LOG_PARAMS:
            v = np.log10(max(v, 1e-12))
        vec.append(v)
    return np.array(vec, dtype=float)


def _from_opt_space(vec):
    """优化向量 → 物理参数 dict (k0 反 log10)。"""
    out = {}
    for i, name in enumerate(pysim.PARAM_NAMES):
        v = vec[i]
        if name in LOG_PARAMS:
            v = 10.0 ** v
        out[name] = float(v)
    return out


def _opt_bounds(ranges):
    """各参数在优化空间的 (lo, hi)。"""
    bounds = []
    for name in pysim.PARAM_NAMES:
        lo, hi = ranges[name]
        if name in LOG_PARAMS:
            lo, hi = np.log10(max(lo, 1e-12)), np.log10(max(hi, 1e-12))
        bounds.append((lo, hi))
    return bounds


def curve_rmse(params, exp_curves):
    """给定参数 dict，正向模拟并返回平均 RMSE (无效模拟返回 inf)。"""
    signals, dt = pysim.run_simulation(params)
    if dt < 0:
        return float('inf')
    rmse = np.sqrt(np.mean((signals - exp_curves) ** 2, axis=1))
    return float(rmse.mean())


def refine(init_params, exp_curves, ranges, method='Nelder-Mead', maxiter=400,
           verbose=True):
    """从初值 init_params 出发，最小化曲线 RMSE。

    Args:
        init_params: dict, DL 预测的物理参数 (pysim.PARAM_NAMES 键)
        exp_curves: (3, T) 实验曲线 (已插值到模拟时间轴)
        ranges: dict {name: (min,max)} 物理边界
        method: 'Nelder-Mead' 或 'Powell'
        maxiter: 最大迭代次数
    Returns:
        (refined_params dict, refined_rmse, init_rmse)
    """
    bounds = _opt_bounds(ranges)
    x0 = _to_opt_space(init_params, ranges)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    init_rmse = curve_rmse(init_params, exp_curves)

    def objective(x):
        xc = np.clip(x, lo, hi)
        penalty = np.sum(np.abs(x - xc)) * 10.0   # 越界软惩罚
        params = _from_opt_space(xc)
        return curve_rmse(params, exp_curves) + penalty

    opts = {'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False}
    if method == 'Powell':
        opts = {'maxiter': maxiter, 'xtol': 1e-5, 'ftol': 1e-6, 'disp': False}

    res = minimize(objective, x0, method=method,
                   bounds=list(zip(lo, hi)) if method == 'Powell' else None,
                   options=opts)

    x_final = np.clip(res.x, lo, hi)
    refined = _from_opt_space(x_final)
    refined_rmse = curve_rmse(refined, exp_curves)

    # 安全网：精修结果若反而变差，退回初值
    if refined_rmse > init_rmse:
        if verbose:
            print(f"  [refine] 精修未改善 ({refined_rmse:.4f} >= {init_rmse:.4f})，保留初值")
        return dict(init_params), init_rmse, init_rmse

    if verbose:
        print(f"  [refine] RMSE {init_rmse:.4f} -> {refined_rmse:.4f} "
              f"(降低 {(init_rmse-refined_rmse)/init_rmse*100:.1f}%)")
    return refined, refined_rmse, init_rmse
