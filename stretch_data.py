import pandas as pd
import os

# 使用相对路径
input_file = 'Fig3a_fitting.xlsx'
output_file = 'Fig3a_fitting_stretched.xlsx'

def stretch_y_data(input_path, output_path):
    # 读取原始 Excel 文件
    df = pd.read_excel(input_path)
    
    # 打印原始信息
    print(f"读取文件: {input_path}")
    print("列名:", df.columns.tolist())
    
    # 定义需要拉长的列
    columns_to_stretch = ['FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)']
    
    # 对每一列进行变换: y_new = y_start + 2 * (y_old - y_start)
    for col in columns_to_stretch:
        if col in df.columns:
            start_val = df[col].iloc[0]
            print(f"正在处理 {col}: 起始值 = {start_val:.4f}, 结束值从 {df[col].iloc[-1]:.4f} 变为..", end="")
            df[col] = start_val + 2 * (df[col] - start_val)
            print(f" {df[col].iloc[-1]:.4f} (拉长2倍)")
        else:
            print(f"警告: 未找到列 '{col}'")
            
    # 保存为新的 Excel 文件
    df.to_excel(output_path, index=False)
    print(f"\n处理完成！新文件已保存至: {output_path}")

def compare_files(f1, f2):
    df1 = pd.read_excel(f1)
    df2 = pd.read_excel(f2)
    
    cols = ['FAM/FAM T (+)', 'TYE/TYE T (-)', 'CY5/CY5 T (m)']
    
    print(f"{'Column':<20} | {'Original Start/End':<40} | {'New Start/End':<40}")
    print("-" * 110)
    for col in cols:
        if col in df1.columns and col in df2.columns:
            s1, e1 = df1[col].iloc[0], df1[col].iloc[-1]
            s2, e2 = df2[col].iloc[0], df2[col].iloc[-1]
            print(f"{col:<20} | {s1:.6f} / {e1:.6f} | {s2:.6f} / {e2:.6f}")

if __name__ == "__main__":
    if os.path.exists(input_file):
        stretch_y_data(input_file, output_file)
    else:
        print(f"错误: 找不到文件 {input_file}")
    compare_files('Fig3a_fitting.xlsx', 'Fig3a_fitting_stretched.xlsx')
