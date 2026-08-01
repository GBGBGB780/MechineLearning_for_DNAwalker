# coding=utf-8
"""
dnawalker.physics.refinement — 物理参数局部精修 (直接最小化曲线 RMSE)
dnawalker.physics.refinement — Local refinement of physical parameters by minimizing curve RMSE.

逆问题中，DL 预测的参数 MSE 小 ≠ 曲线 RMSE 小。本模块把 DL 预测作为初值，
用 pysim 正向模拟 + 局部优化 (Nelder-Mead / Powell) 直接最小化与实验曲线的 RMSE。
这是原项目 [PREDICTION] 节 nm_lower_bound/nm_upper_bound 暗示但从未在 Python 端实现的步骤。

注意：k0 跨数量级，在 log10 空间优化更稳定。
"""

from numbers import Integral
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import minimize

from . import simulator as pysim
from dnawalker.shared.logging import get_logger

# Keep the historical logger namespace stable for legacy command output.
logger = get_logger("refine")

# 默认在 log10 空间优化的参数 — 与 configs/common.ini [DATA_PROCESSING]
# log_transform_params 的当前值同步（'k0' 跨数量级）。调用方可以通过
# refine(..., log_params=...) 显式覆盖，确保与训练侧 log 变换一致。
LOG_PARAMS = {'k0'}

Params = Dict[str, float]
Ranges = Dict[str, Tuple[float, float]]


def _normalize_log_params(log_params: Optional[Iterable[str]]) -> Set[str]:
    """允许传 None / list / set / tuple，统一为 set；None 使用默认 LOG_PARAMS。

    始终返回新 set 副本，避免调用方意外修改返回值污染模块级 LOG_PARAMS。
    并做大小写无关匹配：返回的 set 始终用 ``pysim.PARAM_NAMES`` 中的精确
    大小写形式，因为 ``configparser`` 会把 INI 键名转小写（如 'E_b' →
    'e_b'），而 ``_to_opt_space`` 用 ``name in log_set`` 检查时需要大小写
    一致。未知/无法匹配的名字被保留原样（仍可能在循环中被忽略）。
    """
    if log_params is None:
        return set(LOG_PARAMS)
    # 建立 lower → exact 映射用于规范化
    lower_to_exact = {n.lower(): n for n in pysim.PARAM_NAMES}
    result = set()
    for name in log_params:
        result.add(lower_to_exact.get(str(name).lower(), name))
    return result


def _to_opt_space(params: Params, ranges: Ranges,
                  log_params: Optional[Iterable[str]] = None) -> np.ndarray:
    """物理参数 dict → 优化向量 (log_params 中的参数取 log10)。"""
    log_set = _normalize_log_params(log_params)
    vec = []
    for name in pysim.PARAM_NAMES:
        try:
            v = float(params[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid initial parameter {name!r}"
            ) from exc
        if not np.isfinite(v):
            raise ValueError(f"Initial parameter {name!r} must be finite")
        if name in log_set:
            v = np.log10(max(v, 1e-12))
        vec.append(v)
    return np.array(vec, dtype=float)


def _from_opt_space(vec: np.ndarray,
                    log_params: Optional[Iterable[str]] = None) -> Params:
    """优化向量 → 物理参数 dict (log_params 中的参数反 log10)。"""
    log_set = _normalize_log_params(log_params)
    out: Params = {}
    for i, name in enumerate(pysim.PARAM_NAMES):
        v = vec[i]
        if name in log_set:
            v = 10.0 ** v
        out[name] = float(v)
    return out


def _opt_bounds(ranges: Ranges,
                log_params: Optional[Iterable[str]] = None) -> List[Tuple[float, float]]:
    """各参数在优化空间的 (lo, hi)。"""
    log_set = _normalize_log_params(log_params)
    bounds: List[Tuple[float, float]] = []
    for name in pysim.PARAM_NAMES:
        try:
            lo, hi = (float(value) for value in ranges[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid parameter range for {name!r}"
            ) from exc
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            raise ValueError(
                f"Parameter range for {name!r} must be finite with min < max"
            )
        if name in log_set:
            if lo <= 0:
                raise ValueError(
                    f"Log-space parameter {name!r} requires a positive range"
                )
            lo, hi = np.log10(lo), np.log10(hi)
        bounds.append((lo, hi))
    return bounds


def jitter_params(
    params: Params,
    ranges: Ranges,
    rng,
    scale: float = 0.1,
    log_params: Optional[Iterable[str]] = None,
) -> Params:
    """Draw one bounded multi-start perturbation around ``params``.

    Linear parameters are perturbed by ``scale * (hi - lo)`` in physical
    space. Log-transformed parameters are perturbed by the same fraction of
    their log10 range, matching the optimizer and training transforms.
    """
    try:
        scale = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("jitter scale must be a finite non-negative number") from exc
    if not np.isfinite(scale) or scale < 0:
        raise ValueError(
            f"jitter scale must be a finite non-negative number, got {scale!r}"
        )
    if not hasattr(rng, "normal"):
        raise TypeError("rng must provide a normal() method")

    log_set = _normalize_log_params(log_params)
    physical_bounds = _opt_bounds(ranges, log_params=[])
    opt_bounds = _opt_bounds(ranges, log_params=log_set)
    jittered: Params = {}

    for index, name in enumerate(pysim.PARAM_NAMES):
        try:
            center = float(params[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Missing or invalid initial parameter {name!r}"
            ) from exc
        if not np.isfinite(center):
            raise ValueError(f"Initial parameter {name!r} must be finite")

        physical_lo, physical_hi = physical_bounds[index]
        if name in log_set:
            lo, hi = opt_bounds[index]
            center_opt = np.log10(
                np.clip(center, physical_lo, physical_hi)
            )
            value_opt = center_opt + rng.normal(0.0, scale) * (hi - lo)
            jittered[name] = float(10.0 ** np.clip(value_opt, lo, hi))
        else:
            value = center + rng.normal(0.0, scale) * (
                physical_hi - physical_lo
            )
            jittered[name] = float(
                np.clip(value, physical_lo, physical_hi)
            )
    return jittered


def channel_curve_rmse(
    params: Params,
    exp_curves: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Simulate once and return per-channel RMSE plus signals.

    Invalid physical simulations return ``(None, None)``. Malformed target or
    simulator shapes raise because they indicate a programming/artifact error,
    not an ordinary invalid parameter set.
    """
    exp_curves = np.asarray(exp_curves, dtype=np.float64)
    if exp_curves.ndim != 2 or not np.all(np.isfinite(exp_curves)):
        raise ValueError(
            "exp_curves must be a finite 2-D array, got "
            f"shape={exp_curves.shape}"
        )

    signals, dt = pysim.run_simulation(params)
    signals = np.asarray(signals, dtype=np.float64)
    if signals.shape != exp_curves.shape:
        raise ValueError(
            "simulation/target curve shape mismatch: "
            f"{signals.shape} vs {exp_curves.shape}"
        )
    if not np.isfinite(dt) or dt < 0 or not np.all(np.isfinite(signals)):
        return None, None
    rmse = np.sqrt(np.mean((signals - exp_curves) ** 2, axis=1))
    if not np.all(np.isfinite(rmse)):
        return None, None
    return rmse, signals


def curve_rmse(params: Params, exp_curves: np.ndarray) -> float:
    """给定参数 dict，正向模拟并返回平均 RMSE (无效模拟返回 inf)。"""
    rmse, _ = channel_curve_rmse(params, exp_curves)
    if rmse is None:
        return float("inf")
    return float(rmse.mean())


def build_objective(init_params: Params, exp_curves: np.ndarray, ranges: Ranges,
                    method: str = 'Nelder-Mead', maxiter: int = 400,
                    log_params: Optional[Iterable[str]] = None,
                    on_eval=None):
    """构建精修用的共享 ``(objective, x0, lo, hi, options)``。

    把「优化空间设置 + 越界软惩罚 objective + 优化器选项」集中在一处，作为
    :func:`refine` 与 ``benchmark_initguess.refine_with_cost`` 的**单一事实来源**，
    避免后者手工复刻本函数的配置（method 选项、bounds、log10 空间、软惩罚）而随
    时间悄悄漂移。

    Args:
        init_params: 初值参数 dict (``pysim.PARAM_NAMES`` 键)。
        exp_curves: ``(3, T)`` 目标曲线。
        ranges: ``{name: (min, max)}`` 物理边界。
        method: ``'Nelder-Mead'`` 或 ``'Powell'``，决定 options 的容差键。
        maxiter: 最大迭代次数。
        log_params: 在 log10 空间优化的参数集合；``None`` 用模块默认
            :data:`LOG_PARAMS`。
        on_eval: 可选回调 ``on_eval(rmse: float)``，在每次目标评估时以**未叠加越界
            惩罚的原始** :func:`curve_rmse` 值调用 —— :func:`refine` 用它捕获
            ``init_rmse``，``refine_with_cost`` 用它跟踪达标成本。

    Returns:
        ``(objective, x0, lo, hi, options)``：``objective(x)`` 返回
        ``curve_rmse(clip(x)) + 越界软惩罚``；``x0`` 是 ``init_params`` 的优化空间
        向量；``lo``/``hi`` 是各维边界数组；``options`` 传给 ``scipy.optimize.minimize``。
    """
    if method not in {'Nelder-Mead', 'Powell'}:
        raise ValueError(
            f"Unsupported refinement method {method!r}; use 'Nelder-Mead' or 'Powell'"
        )
    if isinstance(maxiter, bool) or not isinstance(maxiter, Integral) or maxiter <= 0:
        raise ValueError(f"maxiter must be a positive integer, got {maxiter!r}")

    exp_curves = np.asarray(exp_curves, dtype=np.float64)
    if exp_curves.ndim != 2 or not np.all(np.isfinite(exp_curves)):
        raise ValueError(
            "exp_curves must be a finite 2-D array, got "
            f"shape={exp_curves.shape}"
        )

    log_set = _normalize_log_params(log_params)
    bounds = _opt_bounds(ranges, log_set)
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(_to_opt_space(init_params, ranges, log_set), lo, hi)

    def objective(x):
        xc = np.clip(x, lo, hi)
        penalty = np.sum(np.abs(x - xc)) * 10.0   # 越界软惩罚
        rmse = curve_rmse(_from_opt_space(xc, log_set), exp_curves)
        if on_eval is not None:
            on_eval(rmse)
        return rmse + penalty

    opts = {'maxiter': maxiter, 'xatol': 1e-5, 'fatol': 1e-6, 'disp': False}
    if method == 'Powell':
        opts = {'maxiter': maxiter, 'xtol': 1e-5, 'ftol': 1e-6, 'disp': False}

    return objective, x0, lo, hi, opts


def refine(init_params: Params, exp_curves: np.ndarray, ranges: Ranges,
           method: str = 'Nelder-Mead', maxiter: int = 400,
           verbose: bool = True,
           log_params: Optional[Iterable[str]] = None) -> Tuple[Params, float, float]:
    """从初值 init_params 出发，最小化曲线 RMSE。

    Args:
        init_params: dict, DL 预测的物理参数 (pysim.PARAM_NAMES 键)
        exp_curves: (3, T) 实验曲线 (已插值到模拟时间轴)
        ranges: dict {name: (min,max)} 物理边界
        method: 'Nelder-Mead' 或 'Powell'
        maxiter: 最大迭代次数
        log_params: 在 log10 空间优化的参数名集合；``None`` 使用模块默认
            :data:`LOG_PARAMS`（即 ``{'k0'}``）。生产路径推荐传
            ``config.get_log_transform_params()`` 让 refine 与训练侧的 log
            变换保持一致。
    Returns:
        (refined_params dict, refined_rmse, init_rmse)
    """
    # 把可能是 generator 的 log_params 落地成 set 一次，避免后续 objective
    # 反复迭代时第二次取到空集合（generator 已耗尽）。
    log_set = _normalize_log_params(log_params)

    # minimize 起步必先以 x0 调用 objective —— 顺手记录 init_rmse 而非独立
    # 再算一次 curve_rmse(init_params)，省 1 次 forward simulation。
    # 注：经过 _to_opt_space / _from_opt_space 往返后 log 参数有 1 ulp 误差，
    # 导致这里记录的 init_rmse 与旧代码 curve_rmse(init_params) 差 ~1e-12。
    # 但旧代码的 minimize 第一步评估同样走的是 round-trip 后的 x0，所以
    # 新口径其实更"内部一致"。
    _init_rmse_holder = []

    def _capture_init_rmse(rmse):
        if not _init_rmse_holder:
            _init_rmse_holder.append(rmse)

    # 优化空间设置 + objective + options 由共享的 build_objective 统一构建，
    # 与 benchmark_initguess.refine_with_cost 共用同一份配置（单一事实来源）。
    objective, x0, lo, hi, opts = build_objective(
        init_params, exp_curves, ranges, method=method, maxiter=maxiter,
        log_params=log_set, on_eval=_capture_init_rmse)
    bounded_init_params = _from_opt_space(x0, log_set)

    res = minimize(objective, x0, method=method,
                   bounds=list(zip(lo, hi)) if method == 'Powell' else None,
                   options=opts)

    init_rmse = _init_rmse_holder[0] if _init_rmse_holder else float('inf')

    x_final = np.clip(res.x, lo, hi)
    refined = _from_opt_space(x_final, log_set)
    refined_rmse = curve_rmse(refined, exp_curves)

    # 安全网：精修结果若反而变差，退回初值
    if refined_rmse > init_rmse:
        if verbose:
            logger.info("[refine] 精修未改善 (%.4f >= %.4f)，保留初值",
                        refined_rmse, init_rmse)
        return bounded_init_params, init_rmse, init_rmse

    if verbose:
        # 防御 init_rmse==0 的退化情况（理论上 refined_rmse 也为 0 时进入这里）。
        pct = 0.0 if init_rmse == 0 else (init_rmse - refined_rmse) / init_rmse * 100
        logger.info("[refine] RMSE %.4f -> %.4f (降低 %.1f%%)",
                    init_rmse, refined_rmse, pct)
    return refined, refined_rmse, init_rmse
