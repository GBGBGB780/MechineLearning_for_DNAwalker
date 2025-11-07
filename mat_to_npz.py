import numpy as np
import sys
import os
import h5py  # 用于读取 MATLAB v7.3 文件

def load_mat_v73(mat_file_path):
    """
    使用 h5py 加载 MATLAB v7.3 文件，并处理列优先存储和 HDF5 结构。
    """
    data = {}
    try:
        with h5py.File(mat_file_path, 'r') as f:
            for key in f.keys():
                # 我们只关心常见的几个变量
                if key in ['X', 'Y', 'parameter_names']:
                    dset = f[key]
                    arr = dset[:]

                    if key == 'parameter_names':
                        # 尝试解析字符串数组（若存在）
                        try:
                            names = []
                            for ref in arr.flatten():
                                item = f[ref]
                                if item.dtype.kind == 'S':
                                    names.append(item[()].tobytes().decode('utf-16').strip('\x00'))
                                else:
                                    names.append(str(item[()]))
                            data[key] = np.array(names)
                        except Exception as e:
                            print(f"警告: 无法解析 parameter_names ({e})，将保存原始数据。")
                            data[key] = arr.T if arr.ndim > 1 else arr
                    else:
                        # 对 X, Y 等数值数组进行转置
                        if arr.ndim >= 2:
                            data[key] = arr.T
                        else:
                            data[key] = arr

    except Exception as e:
        print(f"错误: 使用 h5py 加载 .mat 文件失败。错误信息: {e}")
        return None

    return data


def convert_mat_to_npz(mat_file_path):
    """
    将 MATLAB .mat 文件转换为 NumPy .npz 压缩文件。
    支持文件中不包含 parameter_names 的情况。
    """

    if not os.path.exists(mat_file_path):
        print(f"错误: 输入文件未找到: {mat_file_path}")
        return

    print(f"正在加载 MATLAB v7.3 文件: {mat_file_path}...")
    mat_data = load_mat_v73(mat_file_path)

    if mat_data is None:
        return

    # 必需的最少变量
    required_vars = ['X', 'Y']
    for var_name in required_vars:
        if var_name not in mat_data:
            print(f"错误: .mat 文件中缺少必需变量 '{var_name}'。")
            print(f"文件中包含的变量有: {list(mat_data.keys())}")
            return

    X_data = mat_data['X']
    Y_data = mat_data['Y']
    param_names = mat_data.get('parameter_names', None)  # 如果不存在则返回 None

    base_name = os.path.splitext(mat_file_path)[0]
    npz_file_path = base_name + ".npz"

    print(f"正在保存数据到 NumPy 压缩文件: {npz_file_path}...")

    try:
        if param_names is not None:
            np.savez_compressed(
                npz_file_path,
                X=X_data,
                Y=Y_data,
                parameter_names=param_names
            )
        else:
            np.savez_compressed(
                npz_file_path,
                X=X_data,
                Y=Y_data
            )
        print("✅ 转换成功!")
        print(f"X 数据的形状: {X_data.shape}")
        print(f"Y 数据的形状: {Y_data.shape}")
        if param_names is not None:
            print(f"参数名称: {param_names}")
        else:
            print("提示: 此文件未包含 parameter_names。")
    except Exception as e:
        print(f"错误: 无法保存 .npz 文件。错误信息: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python mat_to_npz_converter.py <input_mat_file_path>")
        print("示例: python mat_to_npz_converter.py training_data_10000_samples.mat")
    else:
        mat_file = sys.argv[1]
        convert_mat_to_npz(mat_file)
