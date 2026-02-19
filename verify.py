# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import subprocess

# Local imports
from inference import NanorobotPredictor
from predict import load_real_experimental_data
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


# --- 3. 主程序入口 ---

def predict_and_save_params():
    """
    Main function:
    1. Load Real Data from xlsx
    2. Predict Parameters
    3. Save Parameters to txt file (Formatted)
    """
    print("=== 开始预测参数 (Prediction Only) ===")
    
    # --- 1. 初始化 ---
    try:
        predictor = NanorobotPredictor()
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
        
    # --- 4. 保存参数到 TXT 文件 ---
    if write_matlab_input(params_dict):
        print(f"\n成功! 参数已保存至 '{MATLAB_INPUT_FILE}'")
        print("您可以将此文件复制到安装有 MATLAB 的机器上运行模拟。")
    else:
        print("\n保存参数文件失败。")

if __name__ == "__main__":
    predict_and_save_params()