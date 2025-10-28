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

# --- ȫ�����ñ��� ---
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
    # �˺������������̺������ӽ����ж�������һ��
    global CONFIG, FIXED_PARAMS, TRAINABLE_PARAMS_NAMES, PARAM_RANGES_MIN, PARAM_RANGES_MAX
    global LIGHT_SCHEDULE, INITIAL_P, SIM_TOTAL_TIME, STANDARDIZED_TIME_AXIS
    global P_UNBIND_TRACK

    CONFIG = configparser.ConfigParser()
    CONFIG.read("configfile.ini", encoding="utf-8")

    # 1. �����������
    all_physical_params = CONFIG['PHYSICAL_PARAMETERS']
    temp_trainable_names = []
    for name, value in all_physical_params.items():
        if value.strip() == "":
            temp_trainable_names.append(name)
        else:
            FIXED_PARAMS[name] = float(value)

    # ȷ������˳��ʼ��һ��
    TRAINABLE_PARAMS_NAMES = sorted(temp_trainable_names)

    P_UNBIND_TRACK = float(CONFIG['PHYSICAL_PARAMETERS'].get('p_unbind_track', 0.09507))

    # 2. ���ز�����Χ (����min��max������LHS)
    if CONFIG.has_section('TRAINING_PARAMETER_RANGES'):
        default_min, default_max = [float(x.strip()) for x in
                                    CONFIG['TRAINING_PARAMETER_RANGES']['default_range'].split(',')]
        for name in TRAINABLE_PARAMS_NAMES:  # ʹ���������б�
            value_str = CONFIG['TRAINING_PARAMETER_RANGES'].get(name)
            if value_str:
                min_val, max_val = [float(x.strip()) for x in value_str.split(',')]
            else:
                min_val, max_val = default_min, default_max
            PARAM_RANGES_MIN.append(min_val)
            PARAM_RANGES_MAX.append(max_val)

    # 3. ����ģ������
    SIM_TOTAL_TIME = float(CONFIG["NANOROBOT_MODELING"]["sim_total_time"])
    INITIAL_CONFIG_IDX = int(CONFIG["NANOROBOT_MODELING"]["initial_configuration_idx"])
    CYCLE_DURATION_VIS = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_vis"])
    CYCLE_DURATION_UV = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_uv"])
    LIGHT_START_MODE = int(CONFIG["NANOROBOT_MODELING"]["light_start_mode"])

    # 4. �������ռƻ�
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

    # 5. ������ʼ���ʷֲ�
    num_configs = 14  # 14��״̬
    INITIAL_P = np.zeros(num_configs)
    INITIAL_P[INITIAL_CONFIG_IDX] = 1.0

    # 6. ������׼ʱ����
    num_time_points = int(CONFIG['DATA_GENERATION']['num_time_points'])
    STANDARDIZED_TIME_AXIS = np.linspace(0, SIM_TOTAL_TIME, num_time_points)

    # ���ӽ����д�ӡ��ʼ����Ϣ
    if multiprocessing.current_process().name != "MainProcess":
        print(f"�������� {os.getpid()} �ѳ�ʼ����")


def generate_sample(sample_data):
    """
    �ڵ���CPU���������У�����һ�� (X, Y) ���ݶԡ�
    """
    sample_index, sampled_params_Y = sample_data

    try:
        trained_params = dict(zip(TRAINABLE_PARAMS_NAMES, sampled_params_Y))
        all_params = FIXED_PARAMS.copy()
        all_params.update(trained_params)

        # 2. ��ʼ��ģ���� (ÿ���ӽ��̱��봴���Լ���ʵ��)
        solver = NanorobotSolver(
            initial_parameters=all_params,
            experimental_data_path_a=None  # ���ǲ���Ҫ����ʵ������
        )

        # 3. ����ģ��
        sim_df = solver.run_simulation(INITIAL_P, SIM_TOTAL_TIME, LIGHT_SCHEDULE)

        if sim_df is None or sim_df.empty or 'Time' not in sim_df.columns:
            print(f"����: ���� {sample_index} ģ��ʧ�ܣ�������")
            return None

        # 4. ����ӫ������ (X)
        sim_time = sim_df['Time'].values
        # (ȷ���д��ڣ��Է�ģ����ǰʧ��)
        sim_fam = (sim_df['P_0'] + sim_df['P_1'] + sim_df['P_3'] +
                   sim_df['P_4'] + sim_df['P_6'] + sim_df['P_8'] +
                   sim_df['P_10'] + sim_df['P_12']).values
        sim_tye = (sim_df['P_1'] + sim_df['P_2'] + sim_df['P_4'] +
                   sim_df['P_5'] + sim_df['P_7'] + sim_df['P_9'] +
                   sim_df['P_11'] + sim_df['P_13']).values
        sim_cy5 = (sim_df['P_0'] + sim_df['P_2'] + sim_df['P_3'] +
                   sim_df['P_5']).values + P_UNBIND_TRACK
        # ����һ����飬��ֹ��ֵʧ��
        if len(sim_time) < 2:
            print(f"����: ���� {sample_index} ģ��ʱ��㲻�㣬������")
            return None

        # 5. ��ֵ����׼ʱ����
        interp_fam_func = interp1d(sim_time, sim_fam, kind='linear', fill_value='extrapolate')
        interp_tye_func = interp1d(sim_time, sim_tye, kind='linear', fill_value='extrapolate')
        interp_cy5_func = interp1d(sim_time, sim_cy5, kind='linear', fill_value='extrapolate')

        curve_fam = interp_fam_func(STANDARDIZED_TIME_AXIS)
        curve_tye = interp_tye_func(STANDARDIZED_TIME_AXIS)
        curve_cy5 = interp_cy5_func(STANDARDIZED_TIME_AXIS)

        # ����NaN/Inf (�����ֵ��ģ������Ƿ�ֵ)
        curve_X = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)
        curve_X = np.nan_to_num(curve_X, nan=0.0, posinf=1e6, neginf=-1e6)

        return (sampled_params_Y, curve_X)

    except Exception as e:
        print(f"����: ���� {sample_index} ����������� '{e}'��������")
        return None


# --- ��������� ---
if __name__ == "__main__":
    # 1. ���������м�������
    load_config_globals()
    print("--- ���������ü������ ---")
    print(f"��Ҫ������ {len(TRAINABLE_PARAMS_NAMES)} ������: {TRAINABLE_PARAMS_NAMES}")
    print("----------------------")

    # 2. �������л�ȡ���ɲ���
    NUM_SAMPLES = int(CONFIG['DATA_GENERATION']['num_samples'])
    N_THREADS = int(CONFIG['DATA_GENERATION'].get('n_threads', os.cpu_count()))
    OUTPUT_FILENAME = CONFIG['DATA_GENERATION']['output_filename']

    # 3. ʹ�� LHS �������в������� (Y)
    print(f"����ʹ���������������� (LHS) ���� {NUM_SAMPLES} �����...")
    sampler = qmc.LatinHypercube(d=len(TRAINABLE_PARAMS_NAMES))
    unit_samples = sampler.random(n=NUM_SAMPLES)

    # �� [0, 1] ��Χ���������ŵ���ʵ�����������Χ
    scaled_samples = qmc.scale(unit_samples, PARAM_RANGES_MIN, PARAM_RANGES_MAX)

    # ��������������ԣ��Ա㴫�ݸ� `generate_sample`
    tasks = list(enumerate(scaled_samples))
    print("����������ϡ�")

    print(f"������ʼ {NUM_SAMPLES} ��ģ�⣬ʹ�� {N_THREADS} ������...")
    start_time = time.time()

    all_parameters_Y = []
    all_curves_X = []

    with multiprocessing.Pool(processes=N_THREADS, initializer=load_config_globals) as pool:
        # ʹ�� imap ����ȡ���������ĵ�����
        results_iterator = pool.imap(generate_sample, tasks)

        # ʹ�� tqdm ��ʾ������
        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="ģ�����"):
            if result is not None:
                params_Y, curve_X = result
                all_parameters_Y.append(params_Y)
                all_curves_X.append(curve_X)

    end_time = time.time()
    print(f"\n����������ϡ�")
    print(f"�ܺ�ʱ: {end_time - start_time:.2f} ��")
    print(f"�ɹ����� {len(all_parameters_Y)} / {NUM_SAMPLES} ��������")

    if not all_parameters_Y:
        print("����δ�������κ���Ч�������������Ĳ�����Χ��ģ�������á�")
    else:
        # 5. ת��Ϊ NumPy ����
        X_data = np.array(all_curves_X, dtype=np.float32)
        Y_data = np.array(all_parameters_Y, dtype=np.float32)

        # 6. ���浽 .npz �ļ�
        np.savez_compressed(
            OUTPUT_FILENAME,
            X=X_data,  # ӫ������ (����)
            Y=Y_data,  # ������� (��ǩ)
            parameter_names=np.array(TRAINABLE_PARAMS_NAMES)  # �洢��������
        )

        print(f"--- ���ݼ��ѱ��浽 {OUTPUT_FILENAME} ---")
        print(f"X (����) ��״: {X_data.shape}")  # (num_samples, 3, 100)
        print(f"Y (����) ��״: {Y_data.shape}")  # (num_samples, 7)
        print(f"����˳��: {TRAINABLE_PARAMS_NAMES}")
        print("---------------------------------")