# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import subprocess

# Local imports
from predict import NanorobotPredictor, load_real_experimental_data
from config_loader import Config


# --- 1. 配置常量 (来自 test_output_in_matlab.py) ---
MATLAB_SCRIPT_NAME = "mechanics_and_kinetics_of_the_winding_DNA_motor_kinetics_10minvis_10minUV.m"
MATLAB_INPUT_FILE = "matlab_input_params.txt"
MATLAB_OUTPUT_FILE = "matlab_output_results.csv"

# --- 2. 写入输入文件 ---

def write_matlab_input(params_dict):
    """
    将用户输入的参数写入一个简单的文本文件供 MATLAB 读取。
    (Directly copied logic from test_output_in_matlab.py)
    """
    print(f"正在写入 MATLAB 输入文件: {MATLAB_INPUT_FILE}")
    try:
        with open(MATLAB_INPUT_FILE, 'w') as f:
            # 写入参数名和值
            print("\n--- 写入 MATLAB 的参数 (Formatted) ---")
            for name, value in params_dict.items():
                if name == 'k0':
                    # k0通常很小 (e-7)，且用户要求不要有e。使用 .10f 保留足够精度
                    line = f"{name}={value:.10f}"
                else:
                    # 其他参数保留 3 位小数，且不使用科学计数法
                    line = f"{name}={value:.3f}"
                
                f.write(line + "\n")
                print(line)
            f.write("END_OF_PARAMS=1\n")
        print("输入文件写入完毕。")
        return True
    except IOError as e:
        print(f"Error writing MATLAB input file: {e}")
        return False


# --- 3. 调用 MATLAB 脚本 ---

def run_matlab_script(script_name=MATLAB_SCRIPT_NAME):
    """
    使用 subprocess 调用 MATLAB 命令行运行脚本。
    (Directly copied logic from test_output_in_matlab.py)
    """
    print(f"\n--- 正在调用 MATLAB 运行脚本 {script_name} ---")

    # 清理旧的输出文件，确保我们读取的是新的结果
    if os.path.exists(MATLAB_OUTPUT_FILE):
        os.remove(MATLAB_OUTPUT_FILE)

    # MATLAB 命令：-r 执行命令 (run 和 exit)
    command = f"matlab -nodisplay -nosplash -r \"run('{script_name}'); exit\""

    try:
        # 捕获 MATLAB 的 stdout 和 stderr
        # 增大超时时间到 1200s (20min) 以防止复杂仿真超时
        print("该步骤耗时较长，耐心等待...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=1200)

        if result.returncode != 0:
            print(f"*** 错误: MATLAB 运行失败 (退出码: {result.returncode}) ***")
            print("\n--- MATLAB 错误输出 (Stderr) ---")
            print(result.stderr)
            print("-----------------------------------------")
            return False

        print("MATLAB 脚本运行成功。")
        return True

    except subprocess.TimeoutExpired:
        print("\n*** 错误: MATLAB 脚本运行超时 (超过 20 分钟)。***")
        return False
    except FileNotFoundError:
        print("\n*** 致命错误: 找不到 'matlab' 命令。请确保 MATLAB 已安装且在环境变量中。***")
        return False


# --- 4. 读取 MATLAB 结果 ---

def read_matlab_output():
    """
    从 MATLAB 生成的 CSV 文件中读取最终结果。
    (Directly copied logic from test_output_in_matlab.py)
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


def verify_and_plot():
    """
    Main verification function:
    1. Load Real Data from xlsx
    2. Predict Parameters
    3. Simulate (Resimulate) using Predicted Parameters for verification
    4. Plot Comparison
    """
    print("=== 开始闭环验证 (Verification Loop) ===")
    
    # --- 1. 初始化 ---
    try:
        predictor = NanorobotPredictor()
        # solver = MATLABSolver() # Removed
    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # --- 2. 加载实验数据 ---
    data_path = predictor.config.get_experimental_data_path()
    print(f"读取实验数据: {data_path}")
    
    # Returns (3, 7801) numpy array [FAM, TYE, CY5]
    real_curves_np = load_real_experimental_data(predictor.config, data_path)
    
    if real_curves_np is None:
        print("无法加载实验数据，终止。")
        return

    # --- 3. 预测参数 ---
    print("\n--- 执行预测 ---")
    predicted_params_values = predictor.predict(real_curves_np)[0] # Shape (7,)
    param_names = predictor.get_param_names()
    
    # 构建参数字典
    params_dict = {}
    print("\n预测得到的参数:")
    for name, val in zip(param_names, predicted_params_values):
        params_dict[name] = val
        print(f"  {name}: {val:.6e}")
        
    # --- 4. 运行 MATLAB 模拟 (Refitting) ---
    print("\n--- 运行 MATLAB 模拟 (Refitting) ---")
    
    # 使用新集成的函数调用逻辑
    sim_df = None
    if write_matlab_input(params_dict):
        if run_matlab_script():
             sim_df = read_matlab_output()
    
    if sim_df is None:
        print("MATLAB 模拟失败，无法生成对比图。")
        return

    # --- 5. 绘图对比 ---
    print("\n--- 正在生成对比图 ---")
    plot_comparison(predictor.config, real_curves_np, sim_df, params_dict)
    
    
def plot_comparison(config, real_curves_np, sim_df, params_dict):
    """
    绘制对比图: 实验数据(点/线) vs 模拟数据(线)
    """
    # 获取时间轴
    sim_time = sim_df['Time'].values
    sim_fam = sim_df['sim_fam'].values
    sim_tye = sim_df['sim_tye'].values
    sim_cy5 = sim_df['sim_cy5'].values
    
    # 真实数据的时间轴 (Based on config)
    seq_length = config.get_seq_length()
    sim_total_time = config.get_sim_total_time()
    real_time_axis = np.linspace(0, sim_total_time, seq_length)
    
    real_fam = real_curves_np[0]
    real_tye = real_curves_np[1]
    real_cy5 = real_curves_np[2]
    
    # 创建 Output 目录
    output_path = config.get_output_path()
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    save_file = os.path.join(output_path, "verification_comparison.png")
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # FAM
    axes[0].plot(real_time_axis, real_fam, 'k-', linewidth=1.5, label='Experimental (Input)')
    axes[0].plot(sim_time, sim_fam, 'r--', linewidth=2, label='Reconstructed (Sim)')
    axes[0].set_title('FAM (F+)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # TYE
    axes[1].plot(real_time_axis, real_tye, 'k-', linewidth=1.5, label='Experimental')
    axes[1].plot(sim_time, sim_tye, 'g--', linewidth=2, label='Reconstructed')
    axes[1].set_title('TYE (T-)')
    axes[1].set_xlabel('Time (s)')
    axes[1].grid(True, alpha=0.3)

    # CY5
    axes[2].plot(real_time_axis, real_cy5, 'k-', linewidth=1.5, label='Experimental')
    axes[2].plot(sim_time, sim_cy5, 'b--', linewidth=2, label='Reconstructed')
    axes[2].set_title('CY5 (M)')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    
    # Add parameter text
    param_txt = "\n".join([f"{k}: {v:.2e}" for k, v in params_dict.items()])
    fig.text(0.01, 0.5, param_txt, fontsize=10, family='monospace', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(save_file, dpi=300)
    print(f"对比图已保存至: {save_file}")
    # plt.show() # Optional

if __name__ == "__main__":
    verify_and_plot()