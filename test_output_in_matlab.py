# coding=utf-8
import numpy as np
import pandas as pd
import configparser
import subprocess
import os
import sys
import time

# --- 定义文件路径和常量 ---
CONFIG_PATH = "configfile.ini"
MATLAB_SCRIPT_NAME = "mechanics_and_kinetics_of_the_winding_DNA_motor_kinetics_10minvis_10minUV.m"
MATLAB_INPUT_FILE = "matlab_input_params.txt"
MATLAB_OUTPUT_FILE = "matlab_output_results.csv"


# --- 1. 配置加载与参数定义 ---

def load_and_define_params(config_path=CONFIG_PATH):
    """
    加载配置并确定待训练参数的名称。
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 '{config_path}'。")
        sys.exit(1)

    config.read(config_path, encoding="utf-8")

    # --- 智能识别待训练参数 ---
    trainable_params_names = []
    all_physical_params = config['PHYSICAL_PARAMETERS']
    for name, value in all_physical_params.items():
        if value.strip() == "":
            trainable_params_names.append(name)

    # 从配置文件中获取其他信息
    try:
        SIM_TOTAL_TIME = float(config["NANOROBOT_MODELING"]["sim_total_time"])
    except KeyError as e:
        print(f"错误: 配置文件中缺少关键的键: {e}。")
        sys.exit(1)

    return trainable_params_names, SIM_TOTAL_TIME


# --- 2. 写入输入文件 ---

def write_matlab_input(trainable_params_names, user_params):
    """
    将用户输入的参数写入一个简单的文本文件供 MATLAB 读取。
    """
    print(f"正在写入 MATLAB 输入文件: {MATLAB_INPUT_FILE}")
    with open(MATLAB_INPUT_FILE, 'w') as f:
        # 写入参数名和值
        for name in trainable_params_names:
            f.write(f"{name}={user_params[name]}\n")
        f.write("END_OF_PARAMS=1\n")
    print("输入文件写入完毕。")


# --- 3. 调用 MATLAB 脚本 ---

def run_matlab_script(script_name):
    """
    使用 subprocess 调用 MATLAB 命令行运行脚本。
    """
    print(f"\n--- 正在调用 MATLAB 运行脚本 {script_name} ---")

    # 清理旧的输出文件，确保我们读取的是新的结果
    if os.path.exists(MATLAB_OUTPUT_FILE):
        os.remove(MATLAB_OUTPUT_FILE)

    # MATLAB 命令：-r 执行命令 (run 和 exit)
    command = f"matlab -nodisplay -nosplash -r \"run('{script_name}'); exit\""

    try:
        # 捕获 MATLAB 的 stdout 和 stderr
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=300)  # 设置5分钟超时

        if result.returncode != 0:
            print(f"*** 错误: MATLAB 运行失败 (退出码: {result.returncode}) ***")
            print("\n--- MATLAB 错误输出 (Stderr) ---")
            print(result.stderr)
            print("-----------------------------------------")
            return False

        print("MATLAB 脚本运行成功。")
        return True

    except subprocess.TimeoutExpired:
        print("\n*** 错误: MATLAB 脚本运行超时 (超过 5 分钟)。***")
        return False
    except FileNotFoundError:
        print("\n*** 致命错误: 找不到 'matlab' 命令。请确保 MATLAB 已安装且在环境变量中。***")
        return False


# --- 4. 读取 MATLAB 结果 ---

def read_matlab_output():
    """
    从 MATLAB 生成的 CSV 文件中读取最终结果。
    """
    print(f"\n正在读取结果文件: {MATLAB_OUTPUT_FILE}")
    try:
        # 假设 MATLAB 输出是 Time, FAM, TYE, CY5
        sim_results_df = pd.read_csv(MATLAB_OUTPUT_FILE)

        # 调整列名以便后续展示
        sim_results_df = sim_results_df.rename(columns={'FAM': 'sim_fam', 'TYE': 'sim_tye', 'CY5': 'sim_cy5'})

        return sim_results_df

    except FileNotFoundError:
        print(f"错误: 找不到 MATLAB 输出文件 {MATLAB_OUTPUT_FILE}。请检查 MATLAB 脚本是否成功写入。")
        return None
    except Exception as e:
        print(f"读取 CSV 文件时发生错误: {e}")
        return None


# --- 5. 主程序入口 ---

if __name__ == "__main__":

    # 1. 加载配置
    trainable_params_names, SIM_TOTAL_TIME = load_and_define_params()

    # 2. 获取用户输入
    user_trained_params = {}
    print("\n--- DNA 纳米机器人 MATLAB 模拟 ---")
    print(f"请为您提供的 MATLAB 脚本 输入 7 个参数的值 (用空格分隔):")
    print(f"参数顺序: {trainable_params_names}")

    try:
        input_values = input(">> ").strip().split()
        if len(input_values) != len(trainable_params_names):
            print(f"错误: 需要输入 {len(trainable_params_names)} 个值。")
            sys.exit(1)

        for name, value_str in zip(trainable_params_names, input_values):
            user_trained_params[name] = float(value_str)
    except ValueError:
        print("错误: 输入值必须是有效的数字。")
        sys.exit(1)

    # 3. 准备和运行 MATLAB
    write_matlab_input(trainable_params_names, user_trained_params)

    if run_matlab_script(MATLAB_SCRIPT_NAME):

        # 4. 读取结果
        sim_results_df = read_matlab_output()

        if sim_results_df is not None:
            # 5. 结果展示
            print("\n=============================================")
            print("--- 6. 模拟结果数据摘要 (来自 MATLAB) ---")
            print(f"总行数/时间点数: {len(sim_results_df)}")
            print(f"总模拟时长: {sim_results_df['Time'].iloc[-1]:.4f} 秒")
            print("=============================================")

            curve_cols = ['sim_fam', 'sim_tye', 'sim_cy5']

            # 详细展示前 10 个点
            print("\n--- 7. 详细数据: 前 10 个时间点 ---")
            display_df_head = sim_results_df[['Time'] + curve_cols].head(10).style.format(
                {'Time': '{:.4f}', **{col: '{:.4f}' for col in curve_cols}}
            )
            print(display_df_head.to_string())

            # 详细展示后 10 个点
            print("\n--- 8. 详细数据: 后 10 个时间点 ---")
            display_df_tail = sim_results_df[['Time'] + curve_cols].tail(10).style.format(
                {'Time': '{:.4f}', **{col: '{:.4f}' for col in curve_cols}}
            )
            print(display_df_tail.to_string())

    # 6. 清理临时文件
    if os.path.exists(MATLAB_INPUT_FILE):
        os.remove(MATLAB_INPUT_FILE)

    # 保持 MATLAB OUTPUT 文件存在，以便您检查其内容
    # if os.path.exists(MATLAB_OUTPUT_FILE):
    #     os.remove(MATLAB_OUTPUT_FILE)

    print("\n程序结束。请检查 'matlab_output_results.csv' 文件。")