# coding=gb2312
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error
import math


class NanorobotSolver:
    def __init__(self, initial_parameters, experimental_data_path_a):
        self.num_configs = 14  # 总共14种状态
        self.config_names = [f"State_{i}" for i in range(self.num_configs)]
        self.experimental_data_a = self._load_experimental_data(experimental_data_path_a)

        # 验证并设置初始参数
        if not isinstance(initial_parameters, dict):
            raise TypeError("initial_parameters must be a dictionary.")
        self.parameters = initial_parameters.copy()  # 使用副本以避免外部修改

        # *** NEW: 默认初始状态 P0，但可以被 set_matlab_initial_P 覆盖 ***
        self.initial_P_override = None

    @staticmethod
    def _safe_int(val, default=0, min_val=None, max_val=None):
        try:
            if val is None: return int(default)
            if isinstance(val, (float, np.floating)) and (np.isnan(val) or np.isinf(val)): return int(default)
            iv = int(val)
        except (ValueError, TypeError):
            try:
                iv = int(float(val))
            except (ValueError, TypeError):
                return int(default)
        if min_val is not None: iv = max(iv, min_val)
        if max_val is not None: iv = min(iv, max_val)
        return iv

    @staticmethod
    def _safe_float(val, default=0.0, min_val=None, max_val=None):
        try:
            if val is None:
                fv = float(default)
            else:
                fv = float(val)
                if np.isnan(fv) or np.isinf(fv): return float(default)
        except (ValueError, TypeError):
            return float(default)
        if min_val is not None: fv = max(fv, min_val)
        if max_val is not None: fv = min(fv, max_val)
        return fv

    @staticmethod
    def _sanitize_array(arr, nan_replacement=0.0, min_val=None, max_val=None):
        arr = np.array(arr, dtype=float)
        arr = np.nan_to_num(arr, nan=nan_replacement, posinf=max_val if max_val is not None else 1e12,
                            neginf=min_val if min_val is not None else -1e12)
        if min_val is not None or max_val is not None:
            arr = np.clip(arr, a_min=min_val, a_max=max_val)
        return arr

    def _load_experimental_data(self, path):
        """
        Loads experimental data from either a .csv or .xlsx file.
        """
        if path is None:  # FIX for data generation script
            return None

        try:
            if path.lower().endswith('.csv'):
                data = pd.read_csv(path)
            elif path.lower().endswith('.xlsx'):
                data = pd.read_excel(path)
            else:
                print(f"Error: Unsupported file format for '{path}'. Please use .csv or .xlsx.")
                return None

            if 'Time' not in data.columns and data.columns[0] != 'Time':
                data.rename(columns={data.columns[0]: 'Time'}, inplace=True)
            return data
        except FileNotFoundError:
            print(f"Error: The file was not found at path: '{path}'")
            return None
        except Exception as e:
            print(f"Error loading or processing data from '{path}': {e}")
            return None

    def set_parameters(self, params_dict):
        """
        Dynamically updates the model parameters for a simulation run.
        """
        if not isinstance(params_dict, dict):
            raise TypeError("parameters must be provided as a dictionary.")
        self.parameters = params_dict

    # *** NEW METHOD: 复制 MATLAB 的复杂初始条件逻辑 ***
    def set_matlab_initial_P(self):
        """
        根据 MATLAB 文件中的逻辑，
        初始化 P_config，用于与 MATLAB 对齐。
        """
        # 必须先计算自由能 E_config_c
        _, _, E_config_c, _ = self._calculate_free_energies()

        p_total = 0.945  # MATLAB 中的硬编码值
        p_config = np.zeros(self.num_configs)

        # P(11) 和 P(12) 对应 Python 索引 10 和 11
        # MATLAB: p_config(11:12) = exp(-E_config_c(11:12)-20);
        # Python 索引: [10, 11]

        # 确保 E_config_c 是 NumPy 数组以便切片
        E_config_c = np.array(E_config_c)

        p_config[10:12] = np.exp(-E_config_c[10:12] - 20)

        pp = np.sum(p_config)

        # MATLAB: p_config(11:12)=p_config(11:12)/pp*p_total;
        p_config[10:12] = p_config[10:12] / pp * p_total

        # 将结果存储到实例变量中
        self.initial_P_override = p_config

    def _calculate_free_energies(self):
        """
        Calculates the free energies and forces for all 14 configurations.
        NOTE: The core logic here is assumed to be correct based on previous revisions.
        """
        if self.parameters is None:
            raise ValueError("Parameters not set. Call set_parameters() first.")

        p = self.parameters
        kBT = self._safe_float(p.get("kBT", 4.14), min_val=1e-6)
        # ... (其余结构和固定参数加载不变) ...
        lp_s = self._safe_float(p.get("lp_s", 0.75), min_val=1e-6)
        lc_s = self._safe_float(p.get("lc_s", 0.7), min_val=1e-6)
        lc_d = self._safe_float(p.get("lc_d", 0.34), min_val=1e-6)
        E_b = self._safe_float(p.get('E_b', -1.2))
        E_b_azo_trans = self._safe_float(p.get('E_b_azo_trans', -1.0))
        E_b_azo_cis = self._safe_float(p.get('E_b_azo_cis', -0.1))

        n_D1 = self._safe_int(p.get('n_D1', 10), min_val=0)
        n_D2 = self._safe_int(p.get('n_D2', 10), min_val=0)
        n_gray = self._safe_int(p.get('n_gray', 10), min_val=0)
        n_hairpin_1 = self._safe_int(p.get('n_hairpin_1', 8), min_val=1)
        n_hairpin_2 = self._safe_int(p.get('n_hairpin_2', 8), min_val=1)
        n_T_hairpin_1 = self._safe_int(p.get('n_T_hairpin_1', 3), min_val=0)
        n_T_hairpin_2 = self._safe_int(p.get('n_T_hairpin_2', 2), min_val=0)
        n_track_1 = self._safe_int(p.get('n_track_1', 15), min_val=1)
        n_track_2 = self._safe_int(p.get('n_track_2', 55), min_val=1)
        dE_TYE = self._safe_float(p.get('dE_TYE', -1.55))

        E_config_t_base = np.zeros(6)
        E_config_c_base = np.zeros(6)
        f_config_t_base = np.zeros(6)
        f_config_c_base = np.zeros(6)

        E_shear_foot = 1000.0
        for i in range(n_D2 + 1):
            n_D2_detach = i
            E_b_shear = E_b * (n_D1 + n_D2 - n_D2_detach)
            denom = lc_s * (2 * n_D2_detach + n_D1)
            if abs(denom) < 1e-9:
                continue

            x = (n_track_1 * lc_d) / denom
            if 0 <= x < 1:
                try:
                    # Note: MATLAB uses /4/(1-x) which is equivalent to /(4*(1-x))
                    E_shear = E_b_shear + denom * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                except (ValueError, ZeroDivisionError):
                    E_shear = 1000.0
            else:
                E_shear = 1000.0

            if E_shear_foot > E_shear:
                E_shear_foot = E_shear

        E_zipper_foot = E_b * (n_D1 + n_D2)

        E_config_t_base[0] = E_zipper_foot
        E_config_t_base[1] = E_shear_foot
        E_config_c_base[0] = E_zipper_foot
        E_config_c_base[1] = E_shear_foot

        def calculate_double_feet_energy(track_distance, E_foot1, E_foot2):
            E_state_min_t, f_state_min_t = 1000.0, 0.0
            E_state_min_c, f_state_min_c = 1000.0, 0.0

            for i in range(1, n_hairpin_1 + n_hairpin_2 + 1):
                n_hairpin_open = i

                # Check MATLAB state 3-6 logic
                if n_hairpin_open < n_hairpin_1:
                    x_denominator = n_hairpin_open * 2 * lc_s
                    n_chain = n_hairpin_open
                elif n_hairpin_1 <= n_hairpin_open < n_hairpin_1 + n_hairpin_2:
                    x_denominator = (n_hairpin_open + n_T_hairpin_1) * 2 * lc_s
                    # NOTE: MATLAB code has n_chain=n_hairpin_open++n_T_hairpin_1;.
                    # We assume the intent was simple addition:
                    n_chain = n_hairpin_open + n_T_hairpin_1
                else:
                    x_denominator = (n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2) * 2 * lc_s
                    n_chain = n_hairpin_open + n_T_hairpin_1 + n_T_hairpin_2

                if abs(x_denominator) < 1e-9:
                    continue

                x = track_distance / x_denominator

                if 0 <= x < 1:
                    try:
                        E_neck = 2 * (n_chain * 2 * lc_s / lp_s) * x ** 2 * (3 - 2 * x) / (4 * (1 - x))
                        # Note: MATLAB has (1-x)^-2/4
                        f_state = 2 * kBT / lp_s * (x - 0.25 + 0.25 / ((1 - x) ** 2))
                    except (ValueError, ZeroDivisionError):
                        E_neck, f_state = 1000.0, 1000.0
                else:
                    E_neck, f_state = 1000.0, 1000.0

                E_state_t = E_neck + E_foot1 + E_foot2 - 2 * n_hairpin_open * E_b_azo_trans
                E_state_c = E_neck + E_foot1 + E_foot2 - 2 * n_hairpin_open * E_b_azo_cis

                if E_state_min_t > E_state_t:
                    E_state_min_t = E_state_t
                    f_state_min_t = f_state
                if E_state_min_c > E_state_c:
                    E_state_min_c = E_state_c
                    f_state_min_c = f_state

            return E_state_min_t, f_state_min_t, E_state_min_c, f_state_min_c

        track_dist_3 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[2], f_config_t_base[2], E_config_c_base[2], f_config_c_base[2] = calculate_double_feet_energy(
            track_dist_3, E_zipper_foot, E_zipper_foot)

        track_dist_4 = (n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[3], f_config_t_base[3], E_config_c_base[3], f_config_c_base[3] = calculate_double_feet_energy(
            track_dist_4, E_shear_foot, E_shear_foot)

        track_dist_5 = (n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[4], f_config_t_base[4], E_config_c_base[4], f_config_c_base[4] = calculate_double_feet_energy(
            track_dist_5, E_zipper_foot, E_shear_foot)

        track_dist_6 = (2 * n_track_1 + n_track_2 - 2 * n_gray) * lc_d
        E_config_t_base[5], f_config_t_base[5], E_config_c_base[5], f_config_c_base[5] = calculate_double_feet_energy(
            track_dist_6, E_zipper_foot, E_shear_foot)

        E_config_t_final = np.zeros(self.num_configs)
        E_config_c_final = np.zeros(self.num_configs)
        f_config_t_final = np.zeros(self.num_configs)
        f_config_c_final = np.zeros(self.num_configs)

        # Mapping logic (must match MATLAB index ranges)
        # MATLAB (1-based): [1:3]=1, [4:6]=2, [7:8]=3, [9:10]=4, [11:12]=5, [13:14]=6
        # Python (0-based): [0:3]=0, [3:6]=1, [6:8]=2, [8:10]=3, [10:12]=4, [12:14]=5
        map_indices = [
            (0, 3, 0), (3, 6, 1), (6, 8, 2),
            (8, 10, 3), (10, 12, 4), (12, 14, 5)
        ]

        for start, end, base_idx in map_indices:
            E_config_t_final[start:end] = E_config_t_base[base_idx]
            f_config_t_final[start:end] = f_config_t_base[base_idx]
            E_config_c_final[start:end] = E_config_c_base[base_idx]
            f_config_c_final[start:end] = f_config_c_base[base_idx]

        # 应用 dE_TYE 能量偏移
        # MATLAB 索引: 1, 4, 7, 9, 11, 13
        offset_indices = [0, 3, 6, 8, 10, 12]  # Python 索引 (0-based)
        E_config_t_final[offset_indices] += dE_TYE
        E_config_c_final[offset_indices] += dE_TYE

        # 清理最终结果
        E_config_t_final = self._sanitize_array(E_config_t_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        E_config_c_final = self._sanitize_array(E_config_c_final, nan_replacement=1e6, min_val=-1e6, max_val=1e6)
        f_config_t_final = self._sanitize_array(f_config_t_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)
        f_config_c_final = self._sanitize_array(f_config_c_final, nan_replacement=0.0, min_val=-1e6, max_val=1e6)

        return E_config_t_final, f_config_t_final, E_config_c_final, f_config_c_final

    def _calculate_transition_rates(self, E_config_t, f_config_t, E_config_c, f_config_c):
        """
        Calculates the 14x14 transition rate matrices.
        NOTE: This part relies heavily on direct MATLAB to Python index translation (1-based to 0-based).
        """
        if self.parameters is None:
            raise ValueError("Parameters not set.")

        p = self.parameters
        k0 = self._safe_float(p.get("k0", 0.000008), min_val=1e-12)
        k_mig = self._safe_float(p.get("k_mig", 0.05), min_val=0.0)
        drt_z = self._safe_float(p.get("drt_z", 0.5), min_val=1e-12)
        drt_s = self._safe_float(p.get("drt_s", 0.05), min_val=1e-12)
        kBT = self._safe_float(p.get("kBT", 4.14), min_val=1e-12)

        k_trans = np.zeros((self.num_configs, self.num_configs), dtype=np.float64)
        k_cis = np.zeros((self.num_configs, self.num_configs), dtype=np.float64)

        def safe_exp(val):
            # Clip is used to prevent numerical overflow/underflow, necessary in Python/C++ ODE solvers
            return math.exp(np.clip(val / kBT, -100, 100))

        try:
            # All rates are k[i, j] = transition FROM j TO i (MATLAB index i, j)

            # --- single-single transitions ---
            k_trans[3, 0] = k_mig;
            k_trans[4, 1] = k_mig;
            k_trans[5, 2] = k_mig;
            k_trans[0, 3] = k_trans[3, 0] * safe_exp(E_config_t[0] - E_config_t[3]);
            k_trans[1, 4] = k_trans[4, 1] * safe_exp(E_config_t[1] - E_config_t[4]);
            k_trans[2, 5] = k_trans[5, 2] * safe_exp(E_config_t[2] - E_config_t[5]);

            # --- single-double transitions (only k_trans is shown for brevity) ---
            k_trans[6, 0] = k0 * safe_exp(f_config_t[6] * drt_z);
            k_trans[10, 0] = k0 * safe_exp(f_config_t[10] * drt_s);
            k_trans[0, 6] = k_trans[6, 0] * safe_exp(E_config_t[0] - E_config_t[6]);
            k_trans[0, 10] = k_trans[10, 0] * safe_exp(E_config_t[0] - E_config_t[10]);

            k_trans[6, 1] = k0 * safe_exp(f_config_t[6] * drt_z);
            k_trans[7, 1] = k0 * safe_exp(f_config_t[7] * drt_z);
            k_trans[11, 1] = k0 * safe_exp(f_config_t[11] * drt_s);
            k_trans[12, 1] = k0 * safe_exp(f_config_t[12] * drt_s);
            k_trans[1, 6] = k_trans[6, 1] * safe_exp(E_config_t[1] - E_config_t[6]);
            k_trans[1, 7] = k_trans[7, 1] * safe_exp(E_config_t[1] - E_config_t[7]);
            k_trans[1, 11] = k_trans[11, 1] * safe_exp(E_config_t[1] - E_config_t[11]);
            k_trans[1, 12] = k_trans[12, 1] * safe_exp(E_config_t[1] - E_config_t[12]);

            k_trans[7, 2] = k0 * safe_exp(f_config_t[7] * drt_z);
            k_trans[13, 2] = k0 * safe_exp(f_config_t[13] * drt_s);
            k_trans[2, 7] = k_trans[7, 2] * safe_exp(E_config_t[2] - E_config_t[7]);
            k_trans[2, 13] = k_trans[13, 2] * safe_exp(E_config_t[2] - E_config_t[13]);

            k_trans[8, 3] = k0 * safe_exp(f_config_t[8] * drt_s);
            k_trans[12, 3] = k0 * safe_exp(f_config_t[12] * drt_z);
            k_trans[3, 8] = k_trans[8, 3] * safe_exp(E_config_t[3] - E_config_t[8]);
            k_trans[3, 12] = k_trans[12, 3] * safe_exp(E_config_t[3] - E_config_t[12]);

            k_trans[8, 4] = k0 * safe_exp(f_config_t[8] * drt_s);
            k_trans[9, 4] = k0 * safe_exp(f_config_t[9] * drt_s);
            k_trans[10, 4] = k0 * safe_exp(f_config_t[10] * drt_z);
            k_trans[13, 4] = k0 * safe_exp(f_config_t[13] * drt_z);
            k_trans[4, 8] = k_trans[8, 4] * safe_exp(E_config_t[4] - E_config_t[8]);
            k_trans[4, 9] = k_trans[9, 4] * safe_exp(E_config_t[4] - E_config_t[9]);
            k_trans[4, 10] = k_trans[10, 4] * safe_exp(E_config_t[4] - E_config_t[10]);
            k_trans[4, 13] = k_trans[13, 4] * safe_exp(E_config_t[4] - E_config_t[13]);

            k_trans[9, 5] = k0 * safe_exp(f_config_t[9] * drt_s);
            k_trans[11, 5] = k0 * safe_exp(f_config_t[11] * drt_z);
            k_trans[5, 9] = k_trans[9, 5] * safe_exp(E_config_t[5] - E_config_t[9]);
            k_trans[5, 11] = k_trans[11, 5] * safe_exp(E_config_t[5] - E_config_t[11]);

            # --- double-double transitions ---
            k_trans[6, 10] = k_mig;
            k_trans[12, 6] = k_mig;
            k_trans[10, 6] = k_trans[6, 10] * safe_exp(E_config_t[10] - E_config_t[6]);
            k_trans[6, 12] = k_trans[12, 6] * safe_exp(E_config_t[6] - E_config_t[12]);

            k_trans[7, 11] = k_mig;
            k_trans[13, 7] = k_mig;
            k_trans[11, 7] = k_trans[7, 11] * safe_exp(E_config_t[11] - E_config_t[7]);
            k_trans[7, 13] = k_trans[13, 7] * safe_exp(E_config_t[7] - E_config_t[13]);

            k_trans[8, 10] = k_mig;
            k_trans[12, 8] = k_mig;
            k_trans[10, 8] = k_trans[8, 10] * safe_exp(E_config_t[10] - E_config_t[8]);
            k_trans[8, 12] = k_trans[12, 8] * safe_exp(E_config_t[8] - E_config_t[12]);

            k_trans[9, 11] = k_mig;
            k_trans[13, 9] = k_mig;
            k_trans[11, 9] = k_trans[9, 11] * safe_exp(E_config_t[11] - E_config_t[9]);
            k_trans[9, 13] = k_trans[13, 9] * safe_exp(E_config_t[9] - E_config_t[13]);

            # --- The k_cis matrix must also be implemented with 1-based to 0-based index translation ---
            # (The logic for k_cis is identical to k_trans, using E_config_c)
            # ... (Full k_cis implementation is assumed/omitted for brevity)

            # Placeholder for k_cis implementation to keep the structure
            k_cis = self._calculate_k_cis(k_cis, safe_exp, E_config_c, f_config_c, k0, k_mig, drt_z, drt_s)


        except Exception as e:
            print(f"An error occurred during transition rate calculation: {e}")
            return np.zeros_like(k_trans), np.zeros_like(k_cis)

        return k_trans, k_cis

    def _calculate_k_cis(self, k_cis, safe_exp, E_config_c, f_config_c, k0, k_mig, drt_z, drt_s):
        # Full k_cis implementation for completeness (assuming safe_exp handles kBT scaling internally)

        # single-single
        k_cis[3, 0] = k_mig;
        k_cis[4, 1] = k_mig;
        k_cis[5, 2] = k_mig;
        k_cis[0, 3] = k_cis[3, 0] * safe_exp(E_config_c[0] - E_config_c[3]);
        k_cis[1, 4] = k_cis[4, 1] * safe_exp(E_config_c[1] - E_config_c[4]);
        k_cis[2, 5] = k_cis[5, 2] * safe_exp(E_config_c[2] - E_config_c[5]);

        # single-double
        k_cis[6, 0] = k0 * safe_exp(f_config_c[6] * drt_z);
        k_cis[10, 0] = k0 * safe_exp(f_config_c[10] * drt_s);
        k_cis[0, 6] = k_cis[6, 0] * safe_exp(E_config_c[0] - E_config_c[6]);
        k_cis[0, 10] = k_cis[10, 0] * safe_exp(E_config_c[0] - E_config_c[10]);

        k_cis[6, 1] = k0 * safe_exp(f_config_c[6] * drt_z);
        k_cis[7, 1] = k0 * safe_exp(f_config_c[7] * drt_z);
        k_cis[11, 1] = k0 * safe_exp(f_config_c[11] * drt_s);
        k_cis[12, 1] = k0 * safe_exp(f_config_c[12] * drt_s);
        k_cis[1, 6] = k_cis[6, 1] * safe_exp(E_config_c[1] - E_config_c[6]);
        k_cis[1, 7] = k_cis[7, 1] * safe_exp(E_config_c[1] - E_config_c[7]);
        k_cis[1, 11] = k_cis[11, 1] * safe_exp(E_config_c[1] - E_config_c[11]);
        k_cis[1, 12] = k_cis[12, 1] * safe_exp(E_config_c[1] - E_config_c[12]);

        k_cis[7, 2] = k0 * safe_exp(f_config_c[7] * drt_z);
        k_cis[13, 2] = k0 * safe_exp(f_config_c[13] * drt_s);
        k_cis[2, 7] = k_cis[7, 2] * safe_exp(E_config_c[2] - E_config_c[7]);
        k_cis[2, 13] = k_cis[13, 2] * safe_exp(E_config_c[2] - E_config_c[13]);

        k_cis[8, 3] = k0 * safe_exp(f_config_c[8] * drt_s);
        k_cis[12, 3] = k0 * safe_exp(f_config_c[12] * drt_z);
        k_cis[3, 8] = k_cis[8, 3] * safe_exp(E_config_c[3] - E_config_c[8]);
        k_cis[3, 12] = k_cis[12, 3] * safe_exp(E_config_c[3] - E_config_c[12]);

        k_cis[8, 4] = k0 * safe_exp(f_config_c[8] * drt_s);
        k_cis[9, 4] = k0 * safe_exp(f_config_c[9] * drt_s);
        k_cis[10, 4] = k0 * safe_exp(f_config_c[10] * drt_z);
        k_cis[13, 4] = k0 * safe_exp(f_config_c[13] * drt_z);
        k_cis[4, 8] = k_cis[8, 4] * safe_exp(E_config_c[4] - E_config_c[8]);
        k_cis[4, 9] = k_cis[9, 4] * safe_exp(E_config_c[4] - E_config_c[9]);
        k_cis[4, 10] = k_cis[10, 4] * safe_exp(E_config_c[4] - E_config_c[10]);
        k_cis[4, 13] = k_cis[13, 4] * safe_exp(E_config_c[4] - E_config_c[13]);

        k_cis[9, 5] = k0 * safe_exp(f_config_c[9] * drt_s);
        k_cis[11, 5] = k0 * safe_exp(f_config_c[11] * drt_z);
        k_cis[5, 9] = k_cis[9, 5] * safe_exp(E_config_c[5] - E_config_c[9]);
        k_cis[5, 11] = k_cis[11, 5] * safe_exp(E_config_c[5] - E_config_c[11]);

        # double-double
        k_cis[6, 10] = k_mig;
        k_cis[12, 6] = k_mig;
        k_cis[10, 6] = k_cis[6, 10] * safe_exp(E_config_c[10] - E_config_c[6]);
        k_cis[6, 12] = k_cis[12, 6] * safe_exp(E_config_c[6] - E_config_c[12]);

        k_cis[7, 11] = k_mig;
        k_cis[13, 7] = k_mig;
        k_cis[11, 7] = k_cis[7, 11] * safe_exp(E_config_c[11] - E_config_c[7]);
        k_cis[7, 13] = k_cis[13, 7] * safe_exp(E_config_c[7] - E_config_c[13]);

        k_cis[8, 10] = k_mig;
        k_cis[12, 8] = k_mig;
        k_cis[10, 8] = k_cis[8, 10] * safe_exp(E_config_c[10] - E_config_c[8]);
        k_cis[8, 12] = k_cis[12, 8] * safe_exp(E_config_c[8] - E_config_c[12]);

        k_cis[9, 11] = k_mig;
        k_cis[13, 9] = k_mig;
        k_cis[11, 9] = k_cis[9, 11] * safe_exp(E_config_c[11] - E_config_c[9]);
        k_cis[9, 13] = k_cis[13, 9] * safe_exp(E_config_c[9] - E_config_c[13]);

        return k_cis

    def _ode_system(self, t, P, k_matrix, k_photo, light_on):
        """
        Defines the system of ordinary differential equations (ODEs).
        """
        dP_dt = np.zeros(self.num_configs)
        for i in range(self.num_configs):
            sum_val = 0
            for j in range(self.num_configs):
                if i != j:
                    # k_matrix[j, i] * P[j] = flux FROM j TO i
                    # k_matrix[i, j] * P[i] = flux FROM i TO j
                    sum_val += k_matrix[j, i] * P[j] - k_matrix[i, j] * P[i]
            dP_dt[i] = sum_val

        if light_on:
            for i in range(self.num_configs):
                dP_dt[i] -= k_photo * P[i]

        return dP_dt

    def run_simulation(self, P0_default, total_sim_time, light_schedule):
        """
        Runs the full simulation based on a flexible light schedule.
        """
        E_config_t, f_config_t, E_config_c, f_config_c = self._calculate_free_energies()
        k_trans, k_cis = self._calculate_transition_rates(E_config_t, f_config_t, E_config_c, f_config_c)
        k_photo = self._safe_float(self.parameters.get('k_photo', 0.0))

        # *** MODIFIED: 使用 MATLAB 初始条件 (如果已设置) ***
        if self.initial_P_override is not None:
            current_P = self.initial_P_override
        else:
            current_P = np.array(P0_default, dtype=np.float64)

        current_time = 0.0
        all_times = [current_time]
        all_probs = [current_P]

        # ... (ODE simulation loop remains the same, using rtol/atol from previous discussions) ...
        for end_time, light_condition in light_schedule:
            if current_time >= total_sim_time:
                break

            segment_end_time = min(end_time, total_sim_time)

            if segment_end_time <= current_time:
                continue

            if light_condition.lower() == 'uv':
                k_matrix = k_cis
                is_light_on = True
            else:
                k_matrix = k_trans
                is_light_on = False

            sol = solve_ivp(
                lambda t, P: self._ode_system(t, P, k_matrix, k_photo, is_light_on),
                (current_time, segment_end_time),
                current_P,
                method='RK45', dense_output=True, rtol=1e-6, atol=1e-9  # Consider using rtol=1e-4, atol=1e-7 for speed
            )

            if sol.success and len(sol.t) > 1:
                all_times.append(sol.t[1:])
                all_probs.append(sol.y[:, 1:])
                current_time = sol.t[-1]
                current_P = sol.y[:, -1]

        # ... (Final assembly of DataFrame remains the same) ...
        if len(all_times) > 1:
            t_combined = np.concatenate([t if isinstance(t, np.ndarray) else [t] for t in all_times])
            P_combined = np.hstack([p if p.ndim == 2 else p.reshape(-1, 1) for p in all_probs])
        else:
            t_combined = np.array(all_times)
            P_combined = np.array(all_probs).T

        sim_df = pd.DataFrame(P_combined.T, columns=[f'P_{i}' for i in range(self.num_configs)])
        sim_df['Time'] = t_combined
        sim_df = sim_df[['Time'] + [f'P_{i}' for i in range(self.num_configs)]]

        return sim_df

    def evaluate_model(self, simulated_data_df, reward_flag=0):
        # ... (This function remains largely the same, focusing on NMSE calculation) ...
        # NOTE: This function's NMSE result will only align with MATLAB if both simulators use the exact same data points.
        if reward_flag == 0:
            if self.experimental_data_a is None:
                print("Error: Experimental dataset 'a' failed to load. Cannot calculate reward.")
                return -1000.0

            p_unbind_track = self._safe_float(self.parameters.get('p_unbind_track', 0.09507))

            datasets = {'a': self.experimental_data_a}
            total_nmse = 0
            num_signals = 0

            for name, exp_df in datasets.items():
                if exp_df is None:
                    print(f"Warning: Dataset '{name}' was not loaded. Skipping its evaluation.")
                    continue

                try:
                    exp_time = exp_df['Time'].values
                    exp_fam = exp_df['FAM/FAM T (+)'].values
                    exp_tye = exp_df['TYE/TYE T (-)'].values
                    exp_cy5 = exp_df['CY5/CY5 T (m)'].values

                    mask = ~np.isnan(exp_time) & ~np.isnan(exp_fam) & ~np.isnan(exp_tye) & ~np.isnan(exp_cy5)
                    exp_time, exp_fam, exp_tye, exp_cy5 = exp_time[mask], exp_fam[mask], exp_tye[mask], exp_cy5[mask]

                    if len(exp_time) == 0:
                        print(f"Warning: No valid data rows in dataset '{name}' after cleaning NaNs.")
                        continue

                    sim_time = simulated_data_df['Time'].values

                    # WARNING: The physical correctness of these summations is CRITICAL.
                    sim_fam = (simulated_data_df['P_0'] + simulated_data_df['P_1'] + simulated_data_df['P_3'] +
                               simulated_data_df['P_4'] + simulated_data_df['P_6'] + simulated_data_df['P_8'] +
                               simulated_data_df['P_10'] + simulated_data_df['P_12']).values
                    sim_tye = (simulated_data_df['P_1'] + simulated_data_df['P_2'] + simulated_data_df['P_4'] +
                               simulated_data_df['P_5'] + simulated_data_df['P_7'] + simulated_data_df['P_9'] +
                               simulated_data_df['P_11'] + simulated_data_df['P_13']).values
                    sim_cy5 = (simulated_data_df['P_0'] + simulated_data_df['P_2'] + simulated_data_df['P_3'] +
                               simulated_data_df['P_5']).values + p_unbind_track

                    interp_fam = interp1d(sim_time, self._sanitize_array(sim_fam), kind='linear',
                                          fill_value='extrapolate')(exp_time)
                    interp_tye = interp1d(sim_time, self._sanitize_array(sim_tye), kind='linear',
                                          fill_value='extrapolate')(exp_time)
                    interp_cy5 = interp1d(sim_time, self._sanitize_array(sim_cy5), kind='linear',
                                          fill_value='extrapolate')(exp_time)

                    mse_fam = mean_squared_error(exp_fam, interp_fam)
                    mse_tye = mean_squared_error(exp_tye, interp_tye)
                    mse_cy5 = mean_squared_error(exp_cy5, interp_cy5)

                    var_fam = np.var(exp_fam) + 1e-9
                    var_tye = np.var(exp_tye) + 1e-9
                    var_cy5 = np.var(exp_cy5) + 1e-9

                    nmse_fam = mse_fam / var_fam
                    nmse_tye = mse_tye / var_tye
                    nmse_cy5 = mse_cy5 / var_cy5

                    total_nmse += (nmse_fam + nmse_tye + nmse_cy5)
                    num_signals += 3

                except (KeyError, ValueError) as e:
                    print(f"Warning: Could not process dataset '{name}'. Check column names. Error: {e}")
                    return -1000.0
                except Exception as e:
                    print(f"An unexpected error occurred during evaluation of dataset '{name}': {e}")
                    return -1000.0

            if num_signals == 0:
                print("Error: No signals could be processed from any dataset.")
                return -1000.0

            average_nmse = total_nmse / num_signals
            reward = -average_nmse

            if not np.isfinite(reward):
                return -1000.0
            return float(reward)
        else:
            return -1000.0