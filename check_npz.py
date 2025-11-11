import numpy as np

try:
    data = np.load("training_dataset.npz")
except FileNotFoundError:
    print("错误: 未找到 'training_dataset.npz' 文件。")
    print("请确保该文件与此 Python 脚本在同一目录下。")
    exit()

# NPZ 文件应包含 'X', 'Y', 和 'param_names'
print(f"NPZ 文件内容: {data.files}")

# 像访问字典一样，使用键来获取特定的数组
X = data['X']
Y = data['Y']

# 尝试加载 param_names，如果不存在则使用默认值
try:
    param_names = data['param_names']
except KeyError:
    print("警告: 未在 NPZ 文件中找到 'param_names'。将使用默认标签。")
    # 创建 7 个默认参数名称
    param_names = [f"Param_{i}" for i in range(Y.shape[1])]

# 打印 shape 来确认
# 我们期望 X 的 shape 是 [样本数, 100, 3]
# 我们期望 Y 的 shape 是 [样本数, 7]
print(f"\nX(曲线) shape: {X.shape}")
print(f"Y(参数) shape: {Y.shape}")

# --- 假设 X 的 shape 是 [samples, time, curves] ---
# 例如 [8, 100, 3]

sample_idx = 520
# 确保 sample_idx 在范围内
if sample_idx >= X.shape[0]:
    print(f"警告: 样本索引 {sample_idx} 超出范围, 重置为 0")
    sample_idx = 0

# ===================================================================
# 1. 显示 (X) 样本数据
# ===================================================================

# sample_X 的 shape 将是 [time, curves]，例如 [100, 3]
sample_X = X[sample_idx]

curve_labels = ['sim_fam', 'sim_tye', 'sim_cy5']
param_names = ['E_b', 'E_b_azo_trans', 'E_b_azo_cis', 'k_mig', 'k0', 'drt_z', 'drt_s']

# 获取曲线数 (应该是 3)
try:
    num_curves = sample_X.shape[1]
except IndexError:
    print(f"错误: X 矩阵的维度不正确。期望至少有3个维度 [样本, 时间, 曲线]")
    print(f"      但样本 {sample_idx} 的 shape 仅为: {sample_X.shape}")
    exit()

print(f"\n--- (X) 样本 {sample_idx} 的前 10 个时间点 (点 0-9) ---")
# 循环 3 条曲线
for curve_idx in range(num_curves):
    # 选择标签
    label = curve_labels[curve_idx] if curve_idx < len(curve_labels) else f"曲线 P_{curve_idx}"

    # 访问: sample_X[时间切片, 曲线索引]
    first_10_points = sample_X[:10, curve_idx]
    print(f"  {label:<15}: {first_10_points}")

print(f"\n--- (X) 样本 {sample_idx} 的后 10 个时间点 (点 90-99) ---")
# 循环 3 条曲线
for curve_idx in range(num_curves):
    label = curve_labels[curve_idx] if curve_idx < len(curve_labels) else f"曲线 P_{curve_idx}"

    # 访问: sample_X[时间切片, 曲线索引]
    last_10_points = sample_X[-10:, curve_idx]
    print(f"  {label:<15}: {last_10_points}")
# 1. 检查 NaN
nan_samples_mask = np.isnan(X).any(axis=(1, 2))
nan_count_x = np.sum(nan_samples_mask)

# 2. 检查 Inf
inf_samples_mask = np.isinf(X).any(axis=(1, 2))
inf_count_x = np.sum(inf_samples_mask)
print(f"X (曲线数据):")
print(f"  包含 NaN (非数字) 个数: {nan_count_x}")
print(f"  包含 Inf (无穷大) 个数: {inf_count_x}")
# ===================================================================
# 2. (新增) 显示 (Y) 样本数据
# ===================================================================

# sample_Y 的 shape 将是 [7,] (即 7 个参数)
sample_Y = Y[sample_idx]
num_params = sample_Y.shape[0]

print(f"\n--- (Y) 样本 {sample_idx} 的 7 个参数值 ---")

for param_idx in range(num_params):
    # 从 param_names 获取标签
    label = param_names[param_idx] if param_idx < len(param_names) else f"Param_{param_idx}"

    # 获取对应的参数值
    value = sample_Y[param_idx]

    print(f"  {label:<15}: {value:.6e}")  # 使用科学计数法以更好地显示 k0
nan_count_y = np.isnan(Y).sum()
inf_count_y = np.isinf(Y).sum()
print(f"Y (参数数据):")
print(f"  包含 NaN (非数字) 个数: {nan_count_y}")
print(f"  包含 Inf (无穷大) 个数: {inf_count_y}")