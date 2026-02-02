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


class MATLABSolver:
    """
    封装 MATLAB 模拟器的调用逻辑。
    负责将参数写入文件，调用 MATLAB 脚本，并读取结果 CSV。
    (直接集成在 verify.py 中)
    """
    
    def __init__(self, config_file='configfile.ini'):
        self.config = Config(config_file)
        self.input_file = "matlab_input_params.txt"
        self.output_file = "matlab_output_results.csv"
        # 这里的脚本名称可以硬编码或者放入 config
        self.script_name = "mechanics_and_kinetics_of_the_winding_DNA_motor_kinetics_10minvis_10minUV.m"
        
        # 确保 MATLAB 脚本存在
        if not os.path.exists(self.script_name):
            print(f"Warning: MATLAB script '{self.script_name}' not found in current directory.")

    def run(self, params_dict):
        """
        运行模拟。
        
        Args:
            params_dict: 包含物理参数名和值的字典
            
        Returns:
            pd.DataFrame: 模拟结果 (Time, sim_fam, sim_tye, sim_cy5)，如果失败返回 None
        """
        # 1. 写入输入文件
        if not self._write_input_file(params_dict):
            return None
            
        # 2. 调用 MATLAB
        if not self._call_matlab():
            return None
            
        # 3. 读取输出文件
        return self._read_output_file()

    def _write_input_file(self, params):
        """写入 matlab_input_params.txt"""
        try:
            with open(self.input_file, 'w') as f:
                for name, value in params.items():
                    f.write(f"{name}={value}\n")
                f.write("END_OF_PARAMS=1\n")
            return True
        except IOError as e:
            print(f"Error writing MATLAB input file: {e}")
            return False

    def _call_matlab(self):
        """使用 subprocess 调用 MATLAB"""
        # 清理旧结果
        if os.path.exists(self.output_file):
            try:
                os.remove(self.output_file)
            except OSError:
                pass

        print(f"--- Calling MATLAB script: {self.script_name} ---")
        command = f"matlab -nodisplay -nosplash -r \"run('{self.script_name}'); exit\""
        
        try:
            # 设置超时，防止死循环 (增加到 1200秒 = 20分钟)
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=1200)
            
            if result.returncode != 0:
                print(f"MATLAB execution failed with return code {result.returncode}")
                # print(result.stderr) 
                return False
                
            return True
        except subprocess.TimeoutExpired:
            print("MATLAB execution timed out.")
            return False
        except FileNotFoundError:
            print("MATLAB executable not found. Please ensure 'matlab' is in your PATH.")
            return False

    def _read_output_file(self):
        """读取 matlab_output_results.csv"""
        if not os.path.exists(self.output_file):
            print(f"Error: MATLAB output file '{self.output_file}' not found.")
            return None
            
        try:
            df = pd.read_csv(self.output_file)
            # 重命名列以匹配预期
            # 假设 MATLAB 输出是 Time, FAM, TYE, CY5
            # 我们将其映射为 sim_fam, sim_tye, sim_cy5
            rename_map = {'FAM': 'sim_fam', 'TYE': 'sim_tye', 'CY5': 'sim_cy5'}
            df = df.rename(columns=rename_map)
            return df
        except Exception as e:
            print(f"Error reading MATLAB output CSV: {e}")
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
        solver = MATLABSolver()
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
        
    # --- 4. 运行 MATLAB 模拟 ---
    print("\n--- 运行 MATLAB 模拟 (Refitting) ---")
    sim_df = solver.run(params_dict)
    
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