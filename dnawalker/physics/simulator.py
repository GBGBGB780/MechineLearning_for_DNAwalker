# coding=utf-8
"""
dnawalker.physics.simulator — DNA Walker 正向物理模拟器 (源自归档 MATLAB 的 Python 实现)
dnawalker.physics.simulator — Forward physics simulator for the light-controlled DNA Walker.

完整移植自 gendata.m 中的 ``run_dna_motor_simulation``：
  - 14 态马尔可夫跳转模型 (zipper/shear 双足 + 发夹颈链力学)
  - 每 10 分钟在可见光(trans)/紫外(cis)之间切换光照
  - 前向欧拉积分主方程，输出 FAM / TYE / CY5 三条荧光信号曲线

实现保留 MATLAB 版本的状态定义、转移方程、10 分钟光照切换和每秒采样协议。
由于每个恒定光照块内转移矩阵不变，这里用矩阵幂压缩重复欧拉子步，并用数值回归
测试约束结果。浮点运算顺序与 MATLAB 逐步循环不同，因此不宣称逐位相同。
"""

import configparser
import math
import os
import threading
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from dnawalker.paths import DEFAULT_CONFIG

ParamInput = Union[Mapping[str, float], Sequence[float], np.ndarray]

# ---------------------------------------------------------------------------
# Fixed physical parameters loaded from the single source of truth.
# ---------------------------------------------------------------------------
_CONFIG_PATH = os.fspath(DEFAULT_CONFIG)
_FIXED_PARAM_NAMES = (
    "kBT", "lp_s", "lc_s", "lc_d", "di_DNA", "dE_TYE",
    "p_unbind_track", "n_D1", "n_D2", "n_S1", "n_gray",
    "n_hairpin_1", "n_hairpin_2", "n_azo_1", "n_azo_2",
    "n_T_hairpin_1", "n_T_hairpin_2", "n_track_1", "n_track_2",
)
_INTEGER_FIXED_PARAMS = {
    "di_DNA", "n_D1", "n_D2", "n_S1", "n_gray",
    "n_hairpin_1", "n_hairpin_2", "n_azo_1", "n_azo_2",
    "n_T_hairpin_1", "n_T_hairpin_2", "n_track_1", "n_track_2",
}


def _load_runtime_constants(path=_CONFIG_PATH):
    parser = configparser.ConfigParser()
    parser.optionxform = str
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(f"Physics config not found: {path}")

    fixed = {}
    for name in _FIXED_PARAM_NAMES:
        getter = parser.getint if name in _INTEGER_FIXED_PARAMS else parser.getfloat
        fixed[name] = getter("PHYSICAL_PARAMETERS", name)

    nonfinite = [
        name for name, value in fixed.items()
        if not math.isfinite(value)
    ]
    if nonfinite:
        raise ValueError(
            f"Fixed physical parameters must be finite: {nonfinite}"
        )

    strictly_positive = {
        "kBT", "lp_s", "lc_s", "lc_d", "di_DNA",
        "n_D1", "n_D2", "n_S1", "n_hairpin_1", "n_hairpin_2",
        "n_track_1", "n_track_2",
    }
    invalid_positive = [
        name for name in strictly_positive if fixed[name] <= 0
    ]
    if invalid_positive:
        raise ValueError(
            "Fixed physical parameters must be positive: "
            f"{sorted(invalid_positive)}"
        )

    nonnegative = {
        "n_gray", "n_azo_1", "n_azo_2",
        "n_T_hairpin_1", "n_T_hairpin_2",
    }
    invalid_nonnegative = [
        name for name in nonnegative if fixed[name] < 0
    ]
    if invalid_nonnegative:
        raise ValueError(
            "Fixed physical parameters must be non-negative: "
            f"{sorted(invalid_nonnegative)}"
        )
    if not 0.0 <= fixed["p_unbind_track"] <= 1.0:
        raise ValueError(
            "p_unbind_track must be a probability in [0, 1]: "
            f"{fixed['p_unbind_track']}"
        )

    sim_minutes = parser.getint("TRAINING", "sim_duration_minutes")
    configured_total = parser.getfloat("NANOROBOT_MODELING", "sim_total_time")
    if sim_minutes <= 0:
        raise ValueError(
            f"sim_duration_minutes must be positive: {sim_minutes}"
        )
    if (not math.isfinite(configured_total)
            or not math.isclose(
                configured_total, sim_minutes, rel_tol=0.0, abs_tol=1e-12
            )):
        raise ValueError(
            "sim_total_time must match sim_duration_minutes: "
            f"{configured_total} != {sim_minutes}"
        )

    vis_minutes = parser.getfloat("NANOROBOT_MODELING", "cycle_duration_vis")
    uv_minutes = parser.getfloat("NANOROBOT_MODELING", "cycle_duration_uv")
    if (not math.isfinite(vis_minutes) or not math.isfinite(uv_minutes)
            or vis_minutes <= 0
            or not math.isclose(
                vis_minutes, uv_minutes, rel_tol=0.0, abs_tol=1e-12
            )):
        raise ValueError(
            "pysim currently requires equal positive visible/UV cycle durations"
        )
    if parser.getint("NANOROBOT_MODELING", "light_start_mode") != 0:
        raise ValueError("pysim currently supports light_start_mode=0 only")

    block_seconds_float = vis_minutes * 60.0
    block_seconds = int(round(block_seconds_float))
    if not math.isclose(
        block_seconds_float, block_seconds, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("cycle durations must resolve to whole seconds")
    return fixed, sim_minutes, block_seconds


FIXED_PARAMS, SIMU_TIME_MIN, BLOCK_SEC = _load_runtime_constants()

# Y 参数列顺序 (与 configs/common.ini / dataset 对齐)
PARAM_NAMES = ['E_b', 'E_b_azo_trans', 'E_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']

MIN_DT = 1.2e-5            # dt 下限：低于此值方程过刚，样本作废
P_TOTAL = 0.945           # 初始有效态总概率
SAVE_INTERVAL_SEC = 1     # 每秒采样一次
NUM_RESULTS = SIMU_TIME_MIN * 60 + 1   # 7801 个时间点 (含 t=0)


def _state_min(num_track, foot_t, foot_c, e_azo_t, e_azo_c, fp):
    """对应 gendata.m 中 state 3-6 的发夹颈链能量最小化循环。

    Args:
        num_track: 颈链拉伸的轨道距离分子 (各态不同)
        foot_t/foot_c: 该态的双足结合能项 (trans / cis)
        e_azo_t/e_azo_c: 偶氮苯结合能 (trans / cis)
        fp: 固定参数字典
    Returns:
        (E_min_t, f_min_t, E_min_c, f_min_c)
    """
    lp_s, lc_s, kBT = fp['lp_s'], fp['lc_s'], fp['kBT']
    n_h1, n_h2 = fp['n_hairpin_1'], fp['n_hairpin_2']
    nT1, nT2 = fp['n_T_hairpin_1'], fp['n_T_hairpin_2']

    E_min_t, f_min_t = 1000.0, 0.0
    E_min_c, f_min_c = 1000.0, 0.0

    for n_open in range(1, n_h1 + n_h2 + 1):
        if n_open < n_h1:
            n_chain = n_open
        elif n_open < n_h1 + n_h2:
            n_chain = n_open + nT1
        else:
            n_chain = n_open + nT1 + nT2

        x = num_track / (n_chain * 2 * lc_s)
        if x < 1:
            E_neck = 2 * ((n_chain * 2 * lc_s) / lp_s) * x ** 2 * (3 - 2 * x) / 4 / (1 - x)
            f_state = 2 * kBT / lp_s * (x - 0.25 + (1 - x) ** -2 / 4)
        else:
            E_neck = 1000.0
            f_state = 1000.0

        E_state_t = E_neck + foot_t - 2 * n_open * e_azo_t
        E_state_c = E_neck + foot_c - 2 * n_open * e_azo_c

        if E_min_t > E_state_t:
            E_min_t, f_min_t = E_state_t, f_state
        if E_min_c > E_state_c:
            E_min_c, f_min_c = E_state_c, f_state

    return E_min_t, f_min_t, E_min_c, f_min_c


def _compute_config_energies(E_b, e_azo_t, e_azo_c, fp):
    """计算 14 态的构型自由能 E_config 与颈链力 f_config (trans / cis)。

    返回长度 15 的数组 (1-indexed，下标 0 弃用)，与 MATLAB 下标一致。
    """
    lc_s, lc_d = fp['lc_s'], fp['lc_d']
    n_D1, n_D2 = fp['n_D1'], fp['n_D2']
    n_gray = fp['n_gray']
    n_t1, n_t2 = fp['n_track_1'], fp['n_track_2']
    dE_TYE = fp['dE_TYE']

    # --- shear foot 结合能 (取最小) ---
    E_shear_foot = 100.0
    for n_det in range(0, n_D2 + 1):
        E_b_shear = E_b * (n_D1 + n_D2 - n_det)
        x = (n_t1 * lc_d) / (lc_s * (2 * n_det + n_D1))
        if x < 1:
            E_shear = E_b_shear + (lc_s * (2 * n_det + n_D1)) * x ** 2 * (3 - 2 * x) / 4 / (1 - x)
        else:
            E_shear = 1000.0
        if E_shear_foot > E_shear:
            E_shear_foot = E_shear
    E_zipper_foot = E_b * (n_D1 + n_D2)

    # --- state 3-6 颈链能量 ---
    num_34 = (n_t1 + n_t2 - 2 * n_gray) * lc_d
    num_5 = (n_t2 - 2 * n_gray) * lc_d
    num_6 = (2 * n_t1 + n_t2 - 2 * n_gray) * lc_d

    E3t, f3t, E3c, f3c = _state_min(num_34, 2 * E_zipper_foot, 2 * E_zipper_foot, e_azo_t, e_azo_c, fp)
    E4t, f4t, E4c, f4c = _state_min(num_34, 2 * E_shear_foot, 2 * E_shear_foot, e_azo_t, e_azo_c, fp)
    E5t, f5t, E5c, f5c = _state_min(num_5, E_zipper_foot + E_shear_foot, E_zipper_foot + E_shear_foot, e_azo_t, e_azo_c, fp)
    E6t, f6t, E6c, f6c = _state_min(num_6, E_zipper_foot + E_shear_foot, E_zipper_foot + E_shear_foot, e_azo_t, e_azo_c, fp)

    # 6 态 -> 14 态映射 (1-indexed)
    Et = np.zeros(15)
    Ec = np.zeros(15)
    ft = np.zeros(15)
    fc = np.zeros(15)

    Et6 = [None, E_zipper_foot, E_shear_foot, E3t, E4t, E5t, E6t]
    Ec6 = [None, E_zipper_foot, E_shear_foot, E3c, E4c, E5c, E6c]
    ft6 = [None, 0.0, 0.0, f3t, f4t, f5t, f6t]
    fc6 = [None, 0.0, 0.0, f3c, f4c, f5c, f6c]

    mapping = [(1, 3, 1), (4, 6, 2), (7, 8, 3), (9, 10, 4), (11, 12, 5), (13, 14, 6)]
    for lo, hi, src in mapping:
        for s in range(lo, hi + 1):
            Et[s] = Et6[src]
            Ec[s] = Ec6[src]
            ft[s] = ft6[src]
            fc[s] = fc6[src]

    # TYE 解结合能修正 (奇数标记态)
    for s in (1, 4, 7, 9, 11, 13):
        Et[s] += dE_TYE
        Ec[s] += dE_TYE

    return Et, Ec, ft, fc


def _build_k_matrix(E, f, k0, k_mig, drt_z, drt_s, kBT):
    """构建 15x15 转移速率矩阵 (1-indexed)。k[i,j] = state i -> state j 的速率。"""
    k = np.zeros((15, 15))
    ex = math.exp

    def E_(i, j):
        return ex(E[i] - E[j])

    # single-single
    k[4, 1] = k_mig; k[5, 2] = k_mig; k[6, 3] = k_mig
    k[1, 4] = k[4, 1] * E_(1, 4)
    k[2, 5] = k[5, 2] * E_(2, 5)
    k[3, 6] = k[6, 3] * E_(3, 6)

    # single-double
    k[7, 1] = k0 * ex(f[7] * drt_z / kBT); k[11, 1] = k0 * ex(f[11] * drt_s / kBT)
    k[1, 7] = k[7, 1] * E_(1, 7); k[1, 11] = k[11, 1] * E_(1, 11)

    k[7, 2] = k0 * ex(f[7] * drt_z / kBT); k[8, 2] = k0 * ex(f[8] * drt_z / kBT)
    k[12, 2] = k0 * ex(f[12] * drt_s / kBT); k[13, 2] = k0 * ex(f[13] * drt_s / kBT)
    k[2, 7] = k[7, 2] * E_(2, 7); k[2, 8] = k[8, 2] * E_(2, 8)
    k[2, 12] = k[12, 2] * E_(2, 12); k[2, 13] = k[13, 2] * E_(2, 13)

    k[8, 3] = k0 * ex(f[8] * drt_z / kBT); k[14, 3] = k0 * ex(f[14] * drt_s / kBT)
    k[3, 8] = k[8, 3] * E_(3, 8); k[3, 14] = k[14, 3] * E_(3, 14)

    k[9, 4] = k0 * ex(f[9] * drt_s / kBT); k[13, 4] = k0 * ex(f[13] * drt_z / kBT)
    k[4, 9] = k[9, 4] * E_(4, 9); k[4, 13] = k[13, 4] * E_(4, 13)

    k[9, 5] = k0 * ex(f[9] * drt_s / kBT); k[10, 5] = k0 * ex(f[10] * drt_s / kBT)
    k[11, 5] = k0 * ex(f[11] * drt_z / kBT); k[14, 5] = k0 * ex(f[14] * drt_z / kBT)
    k[5, 9] = k[9, 5] * E_(5, 9); k[5, 10] = k[10, 5] * E_(5, 10)
    k[5, 11] = k[11, 5] * E_(5, 11); k[5, 14] = k[14, 5] * E_(5, 14)

    k[10, 6] = k0 * ex(f[10] * drt_s / kBT); k[12, 6] = k0 * ex(f[12] * drt_z / kBT)
    k[6, 10] = k[10, 6] * E_(6, 10); k[6, 12] = k[12, 6] * E_(6, 12)

    # double-double
    k[7, 11] = k_mig; k[13, 7] = k_mig
    k[11, 7] = k[7, 11] * E_(11, 7); k[7, 13] = k[13, 7] * E_(7, 13)

    k[8, 12] = k_mig; k[14, 8] = k_mig
    k[12, 8] = k[8, 12] * E_(12, 8); k[8, 14] = k[14, 8] * E_(8, 14)

    k[9, 11] = k_mig; k[13, 9] = k_mig
    k[11, 9] = k[9, 11] * E_(11, 9); k[9, 13] = k[13, 9] * E_(9, 13)

    k[10, 12] = k_mig; k[14, 10] = k_mig
    k[12, 10] = k[10, 12] * E_(12, 10); k[10, 14] = k[14, 10] * E_(10, 14)

    return k


def _build_R(k14, dt):
    """由 14x14 速率矩阵构建前向欧拉传播矩阵 R。

    R[i,j] = k[j,i]*dt  (i!=j);  R[i,i] = 1 - dt*sum_j k[i,j]
    """
    R = dt * k14.T.copy()
    np.fill_diagonal(R, 1.0 - dt * k14.sum(axis=1))
    return R


def _signals_from_p(P, p_unbind_track):
    """由概率向量批量计算三条信号。P 形状 (..., 14) (0-indexed 状态)。"""
    fam = P[..., 0] + P[..., 1] + P[..., 3] + P[..., 4] + P[..., 6] + P[..., 10] + P[..., 8] + P[..., 12]
    tye = P[..., 1] + P[..., 2] + P[..., 4] + P[..., 5] + P[..., 7] + P[..., 11] + P[..., 9] + P[..., 13]
    cy5 = P[..., 0] + P[..., 2] + P[..., 3] + P[..., 5] + p_unbind_track
    return fam, tye, cy5


def _is_vis(sec):
    """t=sec 秒时是否处于可见光(trans)阶段。块号 floor(sec/600) 为偶数 -> 可见光。"""
    return ((sec // BLOCK_SEC) % 2) == 0


# ---------------------------------------------------------------------------
# Forward-Simulation 调用计数器 (Call_Counter)
# 平台无关的正向模拟调用计数，仅依赖 Python 标准库 (threading)。
# Platform-independent forward-simulation call counter; stdlib only.
#
# 作用域 / Scope: 此计数器是 **进程内** 全局 (per-process)。在多进程并行
# (例如 ``dnawalker.data.generate`` 的 ProcessPoolExecutor) 中，每个 worker 进程都有
# 独立的 ``_call_counter``，主进程读不到 worker 计数。需要跨进程总数时，
# 由调用方在 worker 返回后用 :func:`get_call_count` 汇总，或改用
# :class:`multiprocessing.Value`。
# ---------------------------------------------------------------------------
_call_counter = 0
_call_counter_lock = threading.Lock()


def _increment_call_count():
    """Increment the global Forward_Simulation counter by exactly 1 (thread-safe)."""
    global _call_counter
    with _call_counter_lock:
        _call_counter += 1


def get_call_count() -> int:
    """Return the current Forward_Simulation invocation count (non-negative int)."""
    with _call_counter_lock:
        return _call_counter


def reset_call_count() -> None:
    """Reset the Forward_Simulation invocation count to 0."""
    global _call_counter
    with _call_counter_lock:
        _call_counter = 0


def _run_simulation_impl(
    params: ParamInput,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """运行一次完整正向模拟。

    Args:
        params: dict 或长度 7 的序列，顺序为 PARAM_NAMES。
        fixed_params: 固定参数字典，默认 FIXED_PARAMS。
    Returns:
        (signals, dt_used)
        signals: (3, NUM_RESULTS) float64，行序 [FAM, TYE, CY5]。
        dt_used: 实际使用的 dt；无效样本返回 -1 且 signals 全零。
    """
    fp = FIXED_PARAMS if fixed_params is None else fixed_params
    if isinstance(params, Mapping):
        E_b = params['E_b']; e_azo_t = params['E_b_azo_trans']; e_azo_c = params['E_b_azo_cis']
        k_mig = params['k_mig']; k0 = params['k0']; drt_z = params['drt_z']; drt_s = params['drt_s']
    else:
        E_b, e_azo_t, e_azo_c, k_mig, k0, drt_z, drt_s = params

    kBT = fp['kBT']
    p_unbind = fp['p_unbind_track']
    zeros = np.zeros((3, NUM_RESULTS))

    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        Et, Ec, ft, fc = _compute_config_energies(E_b, e_azo_t, e_azo_c, fp)

        # 能量出现非有限值 -> 作废
        if not (np.all(np.isfinite(Et)) and np.all(np.isfinite(Ec))):
            return zeros, -1.0

        k_trans = _build_k_matrix(Et, ft, k0, k_mig, drt_z, drt_s, kBT)
        k_cis = _build_k_matrix(Ec, fc, k0, k_mig, drt_z, drt_s, kBT)

        kt14 = k_trans[1:15, 1:15]
        kc14 = k_cis[1:15, 1:15]

        max_rate = max(np.max(kt14), np.max(kc14))
        if not np.isfinite(max_rate) or max_rate <= 0:
            return zeros, -1.0

        mag = math.floor(math.log10(max_rate))
        dt = 10.0 ** (-mag - 1)

        if not np.isfinite(dt) or dt == 0 or dt < MIN_DT:
            return zeros, -1.0

        # 退化样本：dt>1 秒会偏离 1 秒采样网格。截断到 1 秒，使欧拉步与采样
        # 网格对齐（更精细的 dt 只会让积分更精确，不影响物理）。
        dt = min(dt, 1.0)

        R_vis = _build_R(kt14, dt)
        R_uv = _build_R(kc14, dt)

        if not (np.all(np.isfinite(R_vis)) and np.all(np.isfinite(R_uv))):
            return zeros, -1.0

        # 初始条件：仅 state 11,12 (0-indexed 10,11) 有占据
        p = np.zeros(14)
        p[10] = math.exp(-Ec[11] - 20)
        p[11] = math.exp(-Ec[12] - 20)
        pp = p[10] + p[11]
        if pp == 0:
            p[10] = p[11] = P_TOTAL / 2
        else:
            p[10] *= P_TOTAL / pp
            p[11] *= P_TOTAL / pp

        signals = np.zeros((3, NUM_RESULTS))

        def record(idx, pv):
            f_, t_, c_ = _signals_from_p(pv, p_unbind)
            signals[0, idx] = f_
            signals[1, idx] = t_
            signals[2, idx] = c_

        record(0, p)

        # ---- 分段批量传播 (核心加速) ----
        # 每个 10 分钟块内 per-second propagator 恒定，用特征分解一次性算出
        # 块内全部状态，避免 7800 次 Python 循环 matvec。
        sps = int(round(1.0 / dt))          # 每秒子步数 (10 的整数幂)
        M_vis = np.linalg.matrix_power(R_vis, sps)
        M_uv = np.linalg.matrix_power(R_uv, sps)

        # 预计算 R^(sps-1)：_propagate_all 在每个 10 分钟边界都要用，
        # 共 ~13 次。一次性算好传入，避免重复 matrix_power 调用。
        # （不直接由 M_vis = R_vis^sps 推出 R_vis^(sps-1)，因为 M_vis @ R_vis^-1
        # 需要矩阵求逆，相比 matrix_power 反而更慢且数值更不稳。）
        if sps > 1:
            R_vis_subm1 = np.linalg.matrix_power(R_vis, sps - 1)
            R_uv_subm1 = np.linalg.matrix_power(R_uv, sps - 1)
        else:
            R_vis_subm1 = None
            R_uv_subm1 = None

        ok = _propagate_all(signals, p, R_vis, R_uv, M_vis, M_uv,
                            sps, p_unbind, R_vis_subm1, R_uv_subm1)
        if not ok or not np.all(np.isfinite(signals)):
            return zeros, -1.0

    return signals, float(dt)


def run_simulation(
    params: ParamInput,
    fixed_params: Optional[Mapping[str, Any]] = None,
) -> Tuple[np.ndarray, float]:
    """Public entry point: count exactly once per call, then delegate.

    The increment is in a try/finally so the count rises by exactly 1 even if
    the implementation raises (Requirement 1.8) and on both valid (dt_used >= 0)
    and invalid (dt_used < 0) paths (Requirements 1.1, 1.2). The return value is
    the unmodified result of the original implementation (Requirement 1.4).
    """
    try:
        return _run_simulation_impl(params, fixed_params)
    finally:
        _increment_call_count()


def _batch_states(M, p0, n, eig=None):
    """用特征分解返回 M 作用 1..n 次于 p0 的状态序列 (n, 14)。

    M 为列随机转移矩阵，一般可对角化。失败时返回 None 由调用方回退。

    ``eig`` 可传入预计算的 ``(w, V) = np.linalg.eig(M)``。因为 M_vis/M_uv 在整次
    模拟中恒定，调用方 (``_propagate_all``) 对每个矩阵只需算一次特征分解即可跨
    全部常相位段复用。传入的 ``(w, V)`` 与此处现算的完全相同，故结果逐位不变。
    """
    try:
        w, V = eig if eig is not None else np.linalg.eig(M)
        c = np.linalg.solve(V, p0.astype(np.complex128))
    except np.linalg.LinAlgError:
        return None
    K = np.arange(1, n + 1)[:, None]        # (n,1)
    Wp = w[None, :] ** K                     # (n,14)
    P = (Wp * c[None, :]) @ V.T              # (n,14) complex
    P = P.real
    if not np.all(np.isfinite(P)):
        return None
    return P


def _iter_states(M, p0, n):
    """逐步 matvec 回退路径：返回状态序列 (n, 14)。"""
    out = np.empty((n, 14))
    p = p0
    for i in range(n):
        p = M @ p
        out[i] = p
    return out


def _propagate_all(signals, p, R_vis, R_uv, M_vis, M_uv, sps, p_unbind,
                   R_vis_subm1=None, R_uv_subm1=None):
    """填充 signals[:, 1:NUM_RESULTS]。返回 True/False (是否成功)。

    R_{vis,uv}_subm1 是预计算的 R^(sps-1)；调用方负责在 sps>1 时传入，避免
    在每个 10 分钟边界重复 matrix_power 调用（B4 优化）。sps==1 时不需要。
    """
    # M_vis / M_uv 在整次模拟中恒定，其特征分解对每个 10 分钟常相位段都相同 ——
    # 只算一次并跨段复用，避免每段 (共 ~13 段) 重复 np.linalg.eig。分解失败时置
    # None，_batch_states 会退回现算，最终由 _iter_states 兜底 (行为不变)。
    def _eig_or_none(M):
        try:
            return np.linalg.eig(M)
        except np.linalg.LinAlgError:
            return None
    eig_vis = _eig_or_none(M_vis)
    eig_uv = _eig_or_none(M_uv)

    sec = 1
    while sec < NUM_RESULTS:
        if sec % BLOCK_SEC == 0:
            # 10 分钟边界：单步复合矩阵 (前 sps-1 子步属旧块，末子步属新块)
            bulk_vis = (((sec // BLOCK_SEC) - 1) % 2) == 0
            last_vis = _is_vis(sec)
            M_last_sub = R_vis if last_vis else R_uv
            if sps > 1:
                M_bulk_subm1 = R_vis_subm1 if bulk_vis else R_uv_subm1
                Mb = M_last_sub @ M_bulk_subm1
            else:
                Mb = M_last_sub
            p = Mb @ p
            f, t, c = _signals_from_p(p, p_unbind)
            signals[0, sec] = f; signals[1, sec] = t; signals[2, sec] = c
            sec += 1
        else:
            # 常相位段：批量传播到下一个边界前
            next_boundary = ((sec // BLOCK_SEC) + 1) * BLOCK_SEC
            end = min(next_boundary - 1, NUM_RESULTS - 1)
            n = end - sec + 1
            vis = _is_vis(sec)
            M = M_vis if vis else M_uv
            P = _batch_states(M, p, n, eig=eig_vis if vis else eig_uv)
            if P is None:
                P = _iter_states(M, p, n)
            # 批量写信号 — 复用单点版本以保持口径一致 (P 形状 (n,14))
            fam, tye, cy5 = _signals_from_p(P, p_unbind)
            signals[0, sec:end + 1] = fam
            signals[1, sec:end + 1] = tye
            signals[2, sec:end + 1] = cy5
            p = P[-1]
            sec = end + 1
    return True
