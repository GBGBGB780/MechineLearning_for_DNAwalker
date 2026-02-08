# coding=utf-8
import numpy as np
import sys
import os
import h5py

def load_mat_v73(mat_file_path):
    """
    Load MATLAB v7.3 file using h5py, handle column-major storage and HDF5 structure.
    """
    data = {}
    try:
        with h5py.File(mat_file_path, 'r') as f:
            for key in f.keys():
                # 修复: 查找 X_final, Y_final, param_names (gendata.m 保存的变量名)
                if key in ['X_final', 'Y_final', 'param_names']:
                    dset = f[key]
                    arr = dset[:]

                    if key == 'param_names':
                        try:
                            names = []
                            for ref in np.ravel(arr):
                                item = f[ref]
                                if item.dtype.kind == 'S':
                                    byte_str = item[()].tobytes()
                                    names.append(byte_str.decode('utf-16').replace('\x00', ''))
                                else:
                                    names.append(str(item[()]))
                            data[key] = np.array(names)
                        except Exception as e:
                            print(f"Warning: Failed to parse parameter_names ({e}), saving raw data.")
                            data[key] = arr.T if arr.ndim > 1 else arr
                    else:
                        # 确保是纯 numpy 数组，复制数据以断开与 h5py 的连接
                        if arr.ndim >= 2:
                            data[key] = np.array(arr.T, dtype=np.float64)
                        else:
                            data[key] = np.array(arr, dtype=np.float64)

    except Exception as e:
        print(f"Error: Failed to load .mat file with h5py. Error: {e}")
        return None

    return data


def convert_mat_to_npz(mat_file_path):
    """
    Convert MATLAB .mat file to NumPy .npz compressed file.
    Supports cases where parameter_names is not included.
    """

    if not os.path.exists(mat_file_path):
        print(f"Error: Input file not found: {mat_file_path}")
        return

    print(f"Loading MATLAB v7.3 file: {mat_file_path}...")
    mat_data = load_mat_v73(mat_file_path)

    if mat_data is None:
        return

    # 修复: 查找 X_final 和 Y_final (gendata.m 保存的变量名)
    required_vars = ['X_final', 'Y_final']
    for var_name in required_vars:
        if var_name not in mat_data:
            print(f"Error: Required variable '{var_name}' missing in .mat file.")
            print(f"Variables in file: {list(mat_data.keys())}")
            return

    # 修复: 使用正确的变量名获取数据
    X_data = mat_data['X_final']
    Y_data = mat_data['Y_final']
    param_names = mat_data.get('param_names', None)

    base_name = os.path.splitext(mat_file_path)[0]
    npz_file_path = base_name + ".npz"

    print(f"Saving data to NumPy compressed file: {npz_file_path}...")

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
        print("Conversion successful!")
        print(f"X data shape: {X_data.shape}")
        print(f"Y data shape: {Y_data.shape}")
        if param_names is not None:
            print(f"Parameter names: {param_names}")
        else:
            print("Note: This file does not contain parameter_names.")
    except Exception as e:
        print(f"Error: Failed to save .npz file. Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        convert_mat_to_npz(sys.argv[1])
    else:
        convert_mat_to_npz("training_dataset.mat")