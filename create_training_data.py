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
    ���������м���һ�����á�
    """
    global CONFIG, FIXED_PARAMS, TRAINABLE_PARAMS_NAMES, PARAM_RANGES
    global LIGHT_SCHEDULE, INITIAL_P, SIM_TOTAL_TIME, STANDARDIZED_TIME_AXIS
    global P_UNBIND_TRACK

    CONFIG = configparser.ConfigParser()
    CONFIG.read("configfile.ini", encoding="utf-8")

    # 1. ����������� 
    all_physical_params = CONFIG['PHYSICAL_PARAMETERS']
    for name, value in all_physical_params.items():
        if value.strip() == "":
            TRAINABLE_PARAMS_NAMES.append(name)
        else:
            FIXED_PARAMS[name] = float(value)

    P_UNBIND_TRACK = float(CONFIG['PHYSICAL_PARAMETERS'].get('p_unbind_track', 0.09507))

    # 2. ���ز�����Χ 
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

    # 3. ����ģ������ 
    SIM_TOTAL_TIME = float(CONFIG["NANOROBOT_MODELING"]["sim_total_time"])
    INITIAL_CONFIG_IDX = int(CONFIG["NANOROBOT_MODELING"]["initial_configuration_idx"])
    CYCLE_DURATION_VIS = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_vis"])
    CYCLE_DURATION_UV = float(CONFIG["NANOROBOT_MODELING"]["cycle_duration_uv"])
    LIGHT_START_MODE = int(CONFIG["NANOROBOT_MODELING"]["light_start_mode"])

    # 4. �������ռƻ� 
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

    print("--- ���ü������ ---")
    print(f"��Ҫ������ {len(TRAINABLE_PARAMS_NAMES)} ������: {TRAINABLE_PARAMS_NAMES}")
    print(f"��׼ʱ���: {num_time_points} ��")
    print("----------------------")


def generate_sample(sample_index):
    """
    �ڵ���CPU���������У�����һ�� (X, Y) ���ݶԡ�
    Y = ������� (7ά����)
    X = ӫ������ (3 x 100 ά����)
    """
    try:
        # 1. �������� (Y)
        trained_params = {}
        for name in TRAINABLE_PARAMS_NAMES:
            min_val, max_val = PARAM_RANGES[name]
            trained_params[name] = np.random.uniform(min_val, max_val)

        all_params = FIXED_PARAMS.copy()
        all_params.update(trained_params)

        # 2. ��ʼ��ģ���� (ÿ���ӽ��̱��봴���Լ���ʵ��)
        solver = NanorobotSolver(
            initial_parameters=all_params,
            experimental_data_path_a=None
        )

        # 3. ����ģ��
        sim_df = solver.run_simulation(INITIAL_P, SIM_TOTAL_TIME, LIGHT_SCHEDULE)

        if sim_df is None or sim_df.empty:
            print(f"����: ���� {sample_index} ģ��ʧ�ܣ�������")
            return None

        # 4. ����ӫ������ (X)
        sim_time = sim_df['Time'].values
        sim_fam = (sim_df['P_0'] + sim_df['P_1'] + sim_df['P_3'] +
                   sim_df['P_4'] + sim_df['P_6'] + sim_df['P_8'] +
                   sim_df['P_10'] + sim_df['P_12']).values
        sim_tye = (sim_df['P_1'] + sim_df['P_2'] + sim_df['P_4'] +
                   sim_df['P_5'] + sim_df['P_7'] + sim_df['P_9'] +
                   sim_df['P_11'] + sim_df['P_13']).values
        sim_cy5 = (sim_df['P_0'] + sim_df['P_2'] + sim_df['P_3'] +
                   sim_df['P_5']).values + P_UNBIND_TRACK

        # 5. ��ֵ����׼ʱ����
        interp_fam_func = interp1d(sim_time, sim_fam, kind='linear', fill_value='extrapolate')
        interp_tye_func = interp1d(sim_time, sim_tye, kind='linear', fill_value='extrapolate')
        interp_cy5_func = interp1d(sim_time, sim_cy5, kind='linear', fill_value='extrapolate')

        curve_fam = interp_fam_func(STANDARDIZED_TIME_AXIS)
        curve_tye = interp_tye_func(STANDARDIZED_TIME_AXIS)
        curve_cy5 = interp_cy5_func(STANDARDIZED_TIME_AXIS)

        # ���������߶ѵ���һ�� (3, num_time_points) �ľ���
        curve_X = np.stack([curve_fam, curve_tye, curve_cy5], axis=0)

        # �� `TRAINABLE_PARAMS_NAMES` ��˳�򷵻ز���ֵ
        params_Y = [trained_params[name] for name in TRAINABLE_PARAMS_NAMES]
        return (params_Y, curve_X)

    except Exception as e:
        print(f"����: ���� {sample_index} ����������� '{e}'��������")
        return None


# --- ��������� ---
if __name__ == "__main__":
    # 1. ���������м�������
    load_config()

    # 2. �������л�ȡ���ɲ���
    NUM_SAMPLES = int(CONFIG['DATA_GENERATION']['num_samples'])
    N_THREADS = int(CONFIG['DATA_GENERATION']['n_threads'])
    OUTPUT_FILENAME = CONFIG['DATA_GENERATION']['output_filename']

    print(f"������ʼ���� {NUM_SAMPLES} ��������ʹ�� {N_THREADS} ������...")
    start_time = time.time()

    all_parameters_Y = []
    all_curves_X = []

    # 3. ʹ�ö���̳ز�����������
    with multiprocessing.Pool(processes=N_THREADS) as pool:
        # ʹ�� imap ����ȡ���������ĵ�����
        results_iterator = pool.imap(generate_sample, range(NUM_SAMPLES))

        # ʹ�� tqdm ��ʾ������
        for result in tqdm(results_iterator, total=NUM_SAMPLES, desc="��������"):
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
        # 4. ת��Ϊ NumPy ����
        X_data = np.array(all_curves_X, dtype=np.float32)
        Y_data = np.array(all_parameters_Y, dtype=np.float32)

        # 5. ���浽 .npz �ļ�
        # .npz ��ʽ��һ��ѹ����������ͬʱ�洢�������
        np.savez_compressed(
            OUTPUT_FILENAME,
            X=X_data,  # ӫ������ (����)
            Y=Y_data,  # ������� (��ǩ)
            parameter_names=np.array(TRAINABLE_PARAMS_NAMES)
        )

        print(f"--- ���ݼ��ѱ��浽 {OUTPUT_FILENAME} ---")
        print(f"X (����) ��״: {X_data.shape}")  # (num_samples, 3, num_time_points)
        print(f"Y (����) ��״: {Y_data.shape}")  # (num_samples, 7)
        print(f"����˳��: {TRAINABLE_PARAMS_NAMES}")
        print("---------------------------------")