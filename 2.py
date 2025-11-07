# coding=utf-8
import numpy as np
import pandas as pd
import configparser
import sys
import os

# 导入 NanorobotSolver 类
try:
    from nanorobot_solver import NanorobotSolver
except ImportError:
    print("错误: 无法导入 nanorobot_solver.py。请确保文件存在。")
    sys.exit(1)


# --- 1. 配置加载与参数定义 ---

def load_and_define_params(config_path="configfile.ini"):
    """
    加载配置并确定固定参数和待训练参数的名称。
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 '{config_path}'。")
        sys.exit(1)

    config.read(config_path, encoding="utf-8")

    # --- 智能识别固定参数和待训练参数 ---
    fixed_params = {}
    trainable_params_names = []
    all_physical_params = config['PHYSICAL_PARAMETERS']
    for name, value in all_physical_params.items():
        if value.strip() == "":
            trainable_params_names.append(name)
        else:
            try:
                fixed_params[name] = float(value)
            except ValueError:
                # 忽略可能存在的注释行
                continue

    # --- 读取其他关键配置 ---
    try:
        SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
        INITIAL_CONFIG_IDX = int(config["NANOROBOT_MODELING"]["initial_configuration_idx"])
        CYCLE_DURATION_VIS = float(config["NANOROBOT_MODELING"]["cycle_duration_vis"])
        CYCLE_DURATION_UV = float(config["NANOROBOT_MODELING"]["cycle_duration_uv"])
        LIGHT_START_MODE = int(config["NANOROBOT_MODELING"]["light_start_mode"])
        P_UNBIND_TRACK = float(config["PHYSICAL_PARAMETERS"]["p_unbind_track"])
    except KeyError as e:
        print(f"错误: 配置文件中缺少关键的键: {e}。")
        sys.exit(1)

    return fixed_params, trainable_params_names, SIM_TOTAL_TIME, INITIAL_CONFIG_IDX, \
        CYCLE_DURATION_VIS, CYCLE_DURATION_UV, LIGHT_START_MODE, P_UNBIND_TRACK


# --- 2. 模拟运行与曲线计算 ---

def run_and_calculate_curves(solver, all_params, sim_total_time, initial_config_idx,
                             cycle_duration_vis, cycle_duration_uv, light_start_mode, p_unbind_track):
    solver.set_parameters(all_params)

    # --- 创建光照时间表 ---
    light_schedule = []
    current_time = 0
    phases = [('visible', cycle_duration_vis), ('uv', cycle_duration_uv)] if light_start_mode == 0 else [
        ('uv', cycle_duration_uv), ('visible', cycle_duration_vis)]

    if sum(p[1] for p in phases) > 0:
        while current_time < sim_total_time:
            for light_type, duration in phases:
                if duration > 0 and current_time < sim_total_time:
                    end_time = current_time + duration
                    light_schedule.append((end_time, light_type))
                    current_time = end_time

    # --- 运行模拟 ---
    initial_P = np.zeros(solver.num_configs)
    initial_P[initial_config_idx] = 1.0
    sim_df = solver.run_simulation(initial_P, sim_total_time, light_schedule)

    if sim_df is None or sim_df.empty:
        print("\n*** 警告: 模拟失败或返回空数据。请检查您设置的参数。***")
        return None

    # --- 计算 3 条荧光曲线 ---
    # 这一部分逻辑严格复刻了 nanorobot_solver.py 中的 evaluate_model
    try:
        # P_i 代表 NanorobotSolver 状态 i 的概率
        sim_df['sim_fam'] = (sim_df['P_0'] + sim_df['P_1'] + sim_df['P_3'] +
                             sim_df['P_4'] + sim_df['P_6'] + sim_df['P_8'] +
                             sim_df['P_10'] + sim_df['P_12'])
        sim_df['sim_tye'] = (sim_df['P_1'] + sim_df['P_2'] + sim_df['P_4'] +
                             sim_df['P_5'] + sim_df['P_7'] + sim_df['P_9'] +
                             sim_df['P_11'] + sim_df['P_13'])
        sim_df['sim_cy5'] = (sim_df['P_0'] + sim_df['P_2'] + sim_df['P_3'] +
                             sim_df['P_5']) + p_unbind_track
    except KeyError as e:
        print(f"警告: 模拟数据中缺少状态概率列。请确保 NanorobotSolver 返回了所有 P_0 到 P_13。错误: {e}")

    return sim_df


# --- 3. 主程序入口 ---

if __name__ == "__main__":
    # 加载配置
    fixed_params, trainable_params_names, SIM_TOTAL_TIME, INITIAL_CONFIG_IDX, \
        CYCLE_DURATION_VIS, CYCLE_DURATION_UV, LIGHT_START_MODE, P_UNBIND_TRACK = load_and_define_params()

    print("\n--- DNA 纳米机器人动力学模拟 ---")
    print(f"总模拟时长: {SIM_TOTAL_TIME} 秒")
    print(f"光照周期: VIS={CYCLE_DURATION_VIS}s, UV={CYCLE_DURATION_UV}s")
    print(f"---------------------------------\n")

    # --- 获取用户输入 ---
    user_trained_params = {}
    print(f"请输入 7 个待训练参数的值 (用空格分隔):")
    print(f"参数顺序: {trainable_params_names}")

    while True:
        try:
            # 提示用户输入参数
            input_values = input(">> ").strip().split()
            if len(input_values) != len(trainable_params_names):
                print(f"错误: 需要输入 {len(trainable_params_names)} 个值，您输入了 {len(input_values)} 个。请重新输入。")
                continue

            # 将输入转换为浮点数并赋值
            for name, value_str in zip(trainable_params_names, input_values):
                user_trained_params[name] = float(value_str)
            break
        except ValueError:
            print("错误: 输入值必须是有效的数字。请重新输入。")
        except EOFError:
            print("\n输入终止。退出程序。")
            sys.exit(0)

    # --- 合并参数并初始化 Solver ---
    all_params = fixed_params.copy()
    all_params.update(user_trained_params)

    # NanorobotSolver 实例化时需要 experimental_data_path_a 参数
    # 我们不需要实际数据，但需要一个路径占位符
    initial_solver = NanorobotSolver(
        initial_parameters=all_params,
        experimental_data_path_a="path/to/dummy/data.xlsx"  # 占位符
    )

    # --- 运行模拟 ---
    sim_results_df = run_and_calculate_curves(
        initial_solver, all_params, SIM_TOTAL_TIME, INITIAL_CONFIG_IDX,
        CYCLE_DURATION_VIS, CYCLE_DURATION_UV, LIGHT_START_MODE, P_UNBIND_TRACK
    )

    if sim_results_df is not None:
        # --- 结果展示 ---
        print("\n=============================================")
        print("--- 4. 模拟结果数据摘要 ---")
        print(f"总行数/时间点数: {len(sim_results_df)}")
        print(f"总模拟时长: {sim_results_df['Time'].iloc[-1]:.4f} 秒")
        print("=============================================")

        curve_cols = ['sim_fam', 'sim_tye', 'sim_cy5']

        # 1. 详细展示前 10 个点
        print("\n--- 5. 详细数据: 前 10 个时间点 ---")
        display_df = sim_results_df[['Time'] + curve_cols].head(10).style.format(
            {'Time': '{:.4f}', **{col: '{:.4f}' for col in curve_cols}}
        )
        print(display_df.to_string())

        # 2. 详细展示后 10 个点
        print("\n--- 6. 详细数据: 后 10 个时间点 ---")
        display_df = sim_results_df[['Time'] + curve_cols].tail(10).style.format(
            {'Time': '{:.4f}', **{col: '{:.4f}' for col in curve_cols}}
        )
        print(display_df.to_string())

        # 3. 显示所有列的名称
        print(f"\nDataFrame中包含的所有列: {sim_results_df.columns.tolist()}")
        print("\n**提示: sim_fam, sim_tye, sim_cy5 是 3 条荧光曲线。P_i 是 14 个状态的概率。**")