# coding=utf-8
"""
mat_to_npz.py — 将 MATLAB v7.3 (.mat) 转换为 NumPy (.npz) 格式
mat_to_npz.py — Convert MATLAB v7.3 (.mat) files to NumPy (.npz) format

用法 / Usage:
    python mat_to_npz.py <mat_file_path>
    python mat_to_npz.py training_dataset.mat
"""

import numpy as np
import sys
import os
import h5py


def _decode_matlab_string_dataset(dataset):
    """Decode a MATLAB HDF5 char/string dataset into a Python string."""
    value = dataset[()]

    if isinstance(value, bytes):
        return value.decode('utf-8', errors='ignore').replace('\x00', '').strip()

    arr = np.asarray(value)

    if arr.dtype.kind == 'S':
        raw = arr.tobytes()
        for encoding in ('utf-16-le', 'utf-16', 'utf-8', 'latin-1'):
            try:
                text = raw.decode(encoding).replace('\x00', '').strip()
            except UnicodeDecodeError:
                continue
            if text:
                return text

    if arr.dtype.kind == 'U':
        return ''.join(arr.ravel().tolist()).replace('\x00', '').strip()

    if arr.dtype.kind in ('i', 'u'):
        chars = [chr(int(code)) for code in arr.ravel() if int(code) != 0]
        return ''.join(chars).strip()

    return str(value).replace('\x00', '').strip()


def load_mat_v73(mat_file_path):
    """
    加载 MATLAB v7.3 文件，处理列主序存储和 HDF5 结构。
    Load a MATLAB v7.3 file, handling column-major storage and HDF5 structure.

    Args:
        mat_file_path: .mat 文件路径 / path to the .mat file

    Returns:
        dict: 包含 X_final, Y_final, (可选)param_names 的字典 / dict with loaded arrays
    """
    data = {}
    try:
        with h5py.File(mat_file_path, 'r') as f:
            for key in f.keys():
                if key in ['X_final', 'Y_final', 'param_names']:
                    dset = f[key]
                    arr = dset[:]

                    if key == 'param_names':
                        # 解析字符串引用数组 / Parse string reference array
                        try:
                            names = []
                            for ref in np.ravel(arr):
                                item = f[ref]
                                names.append(_decode_matlab_string_dataset(item))
                            data[key] = np.array(names)
                        except Exception as e:
                            print(f"Warning: Failed to parse param_names ({e}), saving raw data.")
                            data[key] = arr.T if arr.ndim > 1 else arr
                    else:
                        # 转置以适应 Python 行主序 / Transpose for Python row-major order
                        if arr.ndim >= 2:
                            data[key] = np.array(arr.T, dtype=np.float64)
                        else:
                            data[key] = np.array(arr, dtype=np.float64)

    except Exception as e:
        print(f"Error: Failed to load .mat file: {e}")
        return None

    return data


def convert_mat_to_npz(mat_file_path):
    """
    将 .mat 文件转换为压缩 .npz 文件。
    Convert a .mat file to a compressed .npz file.

    Args:
        mat_file_path: 输入 .mat 文件路径 / input .mat file path
    """
    if not os.path.exists(mat_file_path):
        print(f"Error: Input file not found: {mat_file_path}")
        return

    print(f"Loading MATLAB v7.3 file: {mat_file_path}...")
    mat_data = load_mat_v73(mat_file_path)
    if mat_data is None:
        return

    # 检查必要变量 / Check required variables
    for var in ['X_final', 'Y_final']:
        if var not in mat_data:
            print(f"Error: Required variable '{var}' missing. Found: {list(mat_data.keys())}")
            return

    X_data = mat_data['X_final']
    Y_data = mat_data['Y_final']
    param_names = mat_data.get('param_names', None)

    npz_file_path = os.path.splitext(mat_file_path)[0] + ".npz"
    print(f"Saving to: {npz_file_path}...")

    try:
        save_dict = {'X': X_data, 'Y': Y_data}
        if param_names is not None:
            save_dict['parameter_names'] = param_names
        np.savez_compressed(npz_file_path, **save_dict)

        print("Conversion successful!")
        print(f"X shape: {X_data.shape}, Y shape: {Y_data.shape}")
        if param_names is not None:
            print(f"Parameter names: {param_names}")
    except Exception as e:
        print(f"Error: Failed to save .npz: {e}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "training_dataset.mat"
    convert_mat_to_npz(path)
