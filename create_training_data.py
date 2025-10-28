# coding=gb2312
import numpy as np
import configparser
import multiprocessing
from scipy.interpolate import interp1d
from tqdm import tqdm
import time
from nanorobot_solver import NanorobotSolver


def load_config():
    """
    在主进程中加载一次配置。
    """
    global CONFIG, FIXED_PARAMS, TRAINABLE_PARAMS_NAMES, PARAM_RANGES
    global LIGHT_SCHEDULE, INITIAL_P, SIM_TOTAL_TIME, STANDARDIZED_TIME_AXIS
    global P_UNBIND_TRACK

    CONFIG = configparser.ConfigParser()
    CONFIG.read("configfile.ini", encoding="utf-8")

    # 1. 加载物理参数 
    all_physical_params = CONFIG['PHYSICAL_PARAMETERS']
    for name, value in all_physical_params.items():
        if value.strip() == "":
            TRAINABLE_PARAMS_NAMES.append(name)
        else:
            FIXED_PARAMS[name] = float(value)

    P_UNBIND_TRACK = float(CONFIG['PHYSICAL_PARAMETERS'].get('p_unbind_track', 0.09507))

    # 2. 加载参数范围 
    if CONFIG.has_section('TRAINING_PARAMETER_RANGES'):
        default_min, default_max = [float(x.strip()) for x in
                                    CONFIG['TRAINING_PARAMETER_RANGES']['default_range'].split(',')]
        for name in TRAINABLE_PARAMS_NAMES:
            value = CONFIG['TRAINING_PARAMETER_RANGES'].get(name)
            if value:
                min_val, max_val = [float(x.strip()) for x in value.split(',')]
                PARAM_RANGES[name] = (min_val, max_val)
            else:
                PARAM_RANGES[name] = (default_min, default_max)

    # 3. 加载模拟设置 
    SIM_TOTAL_TIME = float(CONFIG["NANOROBOT_MODELING"]["sim_total_time"])
    INITIAL_CONFIG_IDX = int(CONFIG["NANOROBOT_MODELING"]["initial_configuration_idx"])
    CYCLE_DURATION_VIS = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_vis"])
    CYCLE_DURATION_UV = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_uv"])
    LIGHT_START_MODE = int(CONFIG["NANOROBOT_MODELING"]["light_start_mode"])

    # 4. 创建光照计划 
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

    print("--- 配置加载完毕 ---")
    print(f"将要采样的 {len(TRAINABLE_PARAMS_NAMES)} 个参数: {TRAINABLE_PARAMS_NAMES}")
    print(f"标准时间点: {num_time_points} 个")
    print("----------------------")


def generate_sample(sample_index):
    """
    在单个CPU核心上运行，生成一个 (X, Y) 数据对。
    Y = 物理参数 (7维向量)
    X = 荧光曲线 (3 x 100 维矩阵)
    """
    try:
        # 1. 采样参数 (Y)
        trained_params = {}
        for name in TRAINABLE_PARAMS_NAMES:
            min_val, max_val = PARAM_RANGES[name]
            trained_params[name] = np.random.uniform(min_val, max_val)

        all_params = FIXED_PARAMS.copy()
        all_params.update(trained_params)

        # 2. 初始化模拟器 (每个子进程必须创建自己的实例)
        solver = NanorobotSolver(
            initial_parameters=all_params,
            experimental_data_path_a=None
        )

        # 3. 运行模拟
        sim_df = solver.run_simulation(INITIAL_P, SIM_TOTAL_TIME, LIGHT_SCHEDULE)

        if sim_df is None or sim_df.empty:
            print(f"警告: 样本 {sample_index} 模拟失败，跳过。")
            return None

        # 4. 计算荧光曲线 (X)
        sim_time = sim_df['Time'].values
        sim_fam = (sim_df['P_0'] + sim_df['P_1'] + sim_df['P_3'] +
                   sim_df['P_4'] + sim_df['P_6'] + sim_df['P_8'] +
                   sim_df['P_10'] + sim_df['P_12']).values
        sim_tye = (sim_df['P_1'] + sim_df['P_2'] + sim_df['P_4'] +
                   sim_df['P_5'] + sim_df['P_7'] + sim_df['P_9'] +
                   sim_df['P_11'] + sim_df['P_13']).values
        sim_cy5 = (sim_df['P_0'] + sim_df['P_2'] + sim_df['P_3'] +
                   sim_df['P_5']).values + P_UNBIND_TRACK

        # 5. 插值到标准时间轴
        interp_fam_func = interp1d(sim_time, sim_fam, kind='linear', fill_value='extrapolate')
        interp_tye_func = interp1d(sim_time, sim_tye, kind='linear', fill_value='extrapolate')
        interp_cy5_func = interp1d(sim_time, sim_cy5, kind='linear', fill_value='extrapolate')

        curve_fam = interp_fam_func(STANDARDIZED_TIME_AXIS)
        curve_tye = interp_tye_func(STANDARDIZED_TIME_AXIS)
        curve_cy5 = interp_cy5_func(STANDARDIZED_TIME_AXIS)

        # 将三条曲线堆叠成一个 (3, num_time_points) 的矩阵
        curve_X = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)

        # 按 `TRAINABLE_PARAMS_NAMES` 的顺序返回参数值
        params_Y = [trained_params[name] for name in TRAINABLE_PARAMS_NAMES]
        return (params_Y, curve_X)

    except Exception as e:
        print(f"警告: 样本 {sample_index} 发生意外错误 '{e}'，跳过。")
        return None


# --- 主程序入口 ---
if __name__ == "__main__":
    # 1. 在主进程中加载配置
    load_config()

    # 2. 从配置中获取生成参数
    NUM_SAMPLES = int(CONFIG['DATA_GENERATION']['num_samples'])
    N_THREADS = int(CONFIG['DATA_GENERATION']['n_threads'])
    OUTPUT_FILENAME = CONFIG['DATA_GENERATION']['output_filename']

    print(f"即将开始生成 {NUM_SAMPLES} 个样本，使用 {N_THREADS} 个进程...")
    start_time = time.time()

    all_parameters_Y = []
    all_curves_X = []

    # 3. 使用多进程池并行生成数据
    with multiprocessing.Pool(processes=N_THREADS) as pool:
        # 使用 imap 来获取带进度条的迭代器
        results_iterator = pool.imap(generate_sample, range(NUM_SAMPLES))

        # 使用 tqdm 显示进度条
        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="生成样本"):
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
        # 4. 转换为 NumPy 数组
        X_data = np.array(all_curves_X, dtype=np.float32)
        Y_data = np.array(all_parameters_Y, dtype=np.float32)

        # 5. 保存到 .npz 文件
        # .npz 格式是一个压缩包，可以同时存储多个数组
        np.savez_compressed(
            OUTPUT_FILENAME,
            X=X_data,  # 荧光曲线 (输入)
            Y=Y_data,  # 物理参数 (标签)
            parameter_names=np.array(TRAINABLE_PARAMS_NAMES)
        )

        print(f"--- 数据集已保存到 {OUTPUT_FILENAME} ---")
        print(f"X (曲线) 形状: {X_data.shape}")  # (num_samples, 3, num_time_points)
        print(f"Y (参数) 形状: {Y_data.shape}")  # (num_samples, 7)
        print(f"参数顺序: {TRAINABLE_PARAMS_NAMES}")
        print("---------------------------------")