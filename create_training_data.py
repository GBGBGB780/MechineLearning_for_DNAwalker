# coding=gb2312
import numpy as np
import configparser
import multiprocessing
from scipy.interpolate import interp1d
from scipy.stats import qmc
from tqdm import tqdm
import time
import os
from nanorobot_solver import NanorobotSolver

# --- 全局配置变量 ---
CONFIG = None
FIXED_PARAMS = {}
TRAINABLE_PARAMS_NAMES = []
PARAM_RANGES_MIN = []
PARAM_RANGES_MAX = []
LIGHT_SCHEDULE = []
INITIAL_P = None
SIM_TOTAL_TIME = 0.0
STANDARDIZED_TIME_AXIS = None
P_UNBIND_TRACK = 0.0


def load_config_globals():
    # 此函数将在主进程和所有子进程中独立运行一次
    global CONFIG, FIXED_PARAMS, TRAINABLE_PARAMS_NAMES, PARAM_RANGES_MIN, PARAM_RANGES_MAX
    global LIGHT_SCHEDULE, INITIAL_P, SIM_TOTAL_TIME, STANDARDIZED_TIME_AXIS
    global P_UNBIND_TRACK

    CONFIG = configparser.ConfigParser()
    CONFIG.read("configfile.ini", encoding="utf-8")

    # 1. 加载物理参数
    all_physical_params = CONFIG['PHYSICAL_PARAMETERS']
    temp_trainable_names = []
    for name, value in all_physical_params.items():
        if value.strip() == "":
            temp_trainable_names.append(name)
        else:
            FIXED_PARAMS[name] = float(value)

    # 确保参数顺序始终一致
    TRAINABLE_PARAMS_NAMES = sorted(temp_trainable_names)

    P_UNBIND_TRACK = float(CONFIG['PHYSICAL_PARAMETERS'].get('p_unbind_track', 0.09507))

    # 2. 加载参数范围 (分离min和max以用于LHS)
    if CONFIG.has_section('TRAINING_PARAMETER_RANGES'):
        default_min, default_max = [float(x.strip()) for x in
                                    CONFIG['TRAINING_PARAMETER_RANGES']['default_range'].split(',')]
        for name in TRAINABLE_PARAMS_NAMES:  # 使用排序后的列表
            value_str = CONFIG['TRAINING_PARAMETER_RANGES'].get(name)
            if value_str:
                min_val, max_val = [float(x.strip()) for x in value_str.split(',')]
            else:
                min_val, max_val = default_min, default_max
            PARAM_RANGES_MIN.append(min_val)
            PARAM_RANGES_MAX.append(max_val)

    # 3. 加载模拟设置
    SIM_TOTAL_TIME = float(CONFIG["NANOROBOT_MODELING"]["sim_total_time"])
    INITIAL_CONFIG_IDX = int(CONFIG["NANOROBOT_MODELING"]["initial_configuration_idx"])
    CYCLE_DURATION_VIS = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_vis"])
    CYCLE_DURATION_UV = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_uv"])
    LIGHT_START_MODE = int(CONFIG["NANOROBOT_MODELING"]["light_start_mode"])

    # 4. 创建光照计划
    LIGHT_SCHEDULE = []
    current_time = 0
    phases = [('visible', CYCLE_DURATION_VIS), ('uv', CYCLE_DURATION_UV)] if LIGHT_START_MODE == 0 else \
        [('uv', CYCLE_DURATION_UV), ('visible', CYCLE_DURATION_VIS)]

    if sum(p[1] for p in phases) > 0:
        while current_time < SIM_TOTAL_TIME:
            for light_type, duration in phases:
                if duration > 0 and current_time < SIM_TOTAL_TIME:
                    end_time = current_time + duration
                    LIGHT_SCHEDULE.append((end_time, light_type))
                    current_time = end_time

    # 5. 创建初始概率分布
    num_configs = 14  # 14种状态
    INITIAL_P = np.zeros(num_configs)
    INITIAL_P[INITIAL_CONFIG_IDX] = 1.0

    # 6. 创建标准时间轴
    num_time_points = int(CONFIG['DATA_GENERATION']['num_time_points'])
    STANDARDIZED_TIME_AXIS = np.linspace(0, SIM_TOTAL_TIME, num_time_points)

    # 在子进程中打印初始化信息
    if multiprocessing.current_process().name != "MainProcess":
        print(f"工作进程 {os.getpid()} 已初始化。")


def generate_sample(sample_data):
    """
    在单个CPU核心上运行，生成一个 (X, Y) 数据对。
    """
    sample_index, sampled_params_Y = sample_data

    try:
        trained_params = dict(zip(TRAINABLE_PARAMS_NAMES, sampled_params_Y))
        all_params = FIXED_PARAMS.copy()
        all_params.update(trained_params)

        # 2. 初始化模拟器 (每个子进程必须创建自己的实例)
        solver = NanorobotSolver(
            initial_parameters=all_params,
            experimental_data_path_a=None  # 我们不需要加载实验数据
        )

        # 3. 运行模拟
        sim_df = solver.run_simulation(INITIAL_P, SIM_TOTAL_TIME, LIGHT_SCHEDULE)

        if sim_df is None or sim_df.empty or 'Time' not in sim_df.columns:
            print(f"警告: 样本 {sample_index} 模拟失败，跳过。")
            return None

        # 4. 计算荧光曲线 (X)
        sim_time = sim_df['Time'].values
        # (确保列存在，以防模拟提前失败)
        sim_fam = (sim_df['P_0'] + sim_df['P_1'] + sim_df['P_3'] +
                   sim_df['P_4'] + sim_df['P_6'] + sim_df['P_8'] +
                   sim_df['P_10'] + sim_df['P_12']).values
        sim_tye = (sim_df['P_1'] + sim_df['P_2'] + sim_df['P_4'] +
                   sim_df['P_5'] + sim_df['P_7'] + sim_df['P_9'] +
                   sim_df['P_11'] + sim_df['P_13']).values
        sim_cy5 = (sim_df['P_0'] + sim_df['P_2'] + sim_df['P_3'] +
                   sim_df['P_5']).values + P_UNBIND_TRACK
        # 增加一个检查，防止插值失败
        if len(sim_time) < 2:
            print(f"警告: 样本 {sample_index} 模拟时间点不足，跳过。")
            return None

        # 5. 插值到标准时间轴
        interp_fam_func = interp1d(sim_time, sim_fam, kind='linear', fill_value='extrapolate')
        interp_tye_func = interp1d(sim_time, sim_tye, kind='linear', fill_value='extrapolate')
        interp_cy5_func = interp1d(sim_time, sim_cy5, kind='linear', fill_value='extrapolate')

        curve_fam = interp_fam_func(STANDARDIZED_TIME_AXIS)
        curve_tye = interp_tye_func(STANDARDIZED_TIME_AXIS)
        curve_cy5 = interp_cy5_func(STANDARDIZED_TIME_AXIS)

        # 清理NaN/Inf (如果插值或模拟产生非法值)
        curve_X = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)
        curve_X = np.nan_to_num(curve_X, nan=0.0, posinf=1e6, neginf=-1e6)

        return (sampled_params_Y, curve_X)

    except Exception as e:
        print(f"警告: 样本 {sample_index} 发生意外错误 '{e}'，跳过。")
        return None


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 在主进程中加载配置
    load_config_globals()
    print("--- 主进程配置加载完毕 ---")
    print(f"将要采样的 {len(TRAINABLE_PARAMS_NAMES)} 个参数: {TRAINABLE_PARAMS_NAMES}")
    print("----------------------")

    # 2. 从配置中获取生成参数
    NUM_SAMPLES = int(CONFIG['DATA_GENERATION']['num_samples'])
    N_THREADS = int(CONFIG['DATA_GENERATION'].get('n_threads', os.cpu_count()))
    OUTPUT_FILENAME = CONFIG['DATA_GENERATION']['output_filename']

    # 3. 使用 LHS 生成所有参数样本 (Y)
    print(f"正在使用拉丁超立方采样 (LHS) 生成 {NUM_SAMPLES} 组参数...")
    sampler = qmc.LatinHypercube(d=len(TRAINABLE_PARAMS_NAMES))
    unit_samples = sampler.random(n=NUM_SAMPLES)

    # 将 [0, 1] 范围的样本缩放到真实的物理参数范围
    scaled_samples = qmc.scale(unit_samples, PARAM_RANGES_MIN, PARAM_RANGES_MAX)

    # 将样本与索引配对，以便传递给 `generate_sample`
    tasks = list(enumerate(scaled_samples))
    print("参数生成完毕。")

    print(f"即将开始 {NUM_SAMPLES} 次模拟，使用 {N_THREADS} 个进程...")
    start_time = time.time()

    all_parameters_Y = []
    all_curves_X = []

    with multiprocessing.Pool(processes=N_THREADS, initializer=load_config_globals) as pool:
        # 使用 imap 来获取带进度条的迭代器
        results_iterator = pool.imap(generate_sample, tasks)

        # 使用 tqdm 显示进度条
        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="模拟进度"):
            if result is not None:
                params_Y, curve_X = result
                all_parameters_Y.append(params_Y)
                all_curves_X.append(curve_X)

    end_time = time.time()
    print(f"\n数据生成完毕。")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"成功生成 {len(all_parameters_Y)} / {NUM_SAMPLES} 个样本。")

    if not all_parameters_Y:
        print("错误：未能生成任何有效样本。请检查您的参数范围和模拟器设置。")
    else:
        # 5. 转换为 NumPy 数组
        X_data = np.array(all_curves_X, dtype=np.float32)
        Y_data = np.array(all_parameters_Y, dtype=np.float32)

        # 6. 保存到 .npz 文件
        np.savez_compressed(
            OUTPUT_FILENAME,
            X=X_data,  # 荧光曲线 (输入)
            Y=Y_data,  # 物理参数 (标签)
            parameter_names=np.array(TRAINABLE_PARAMS_NAMES)  # 存储参数名称
        )

        print(f"--- 数据集已保存到 {OUTPUT_FILENAME} ---")
        print(f"X (曲线) 形状: {X_data.shape}")  # (num_samples, 3, 100)
        print(f"Y (参数) 形状: {Y_data.shape}")  # (num_samples, 7)
        print(f"参数顺序: {TRAINABLE_PARAMS_NAMES}")
        print("---------------------------------")