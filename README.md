# DNA Nanorobot Inverse Design & Verification System
# DNA 纳米机器人逆向设计与验证系统

[English Description follows Chinese / 下方为英文说明]

---

# 🇨🇳 中文说明 (Chinese Version)

## 1. 项目背景与简介 (Introduction)
本项目旨在解决 DNA 纳米机器人的**逆向设计问题**。
在实验中，我们通常只能观测到纳米机器人的荧光动力学曲线（FAM, TYE, CY5 信号随时间的变化），但难以直接获知其微观的物理与动力学参数（如能量势垒、迁移速率等）。传统的手动调参拟合方法效率极低且依赖经验。

本系统提供了一套自动化的解决方案：
1.  利用 **MATLAB** 建立精确的物理化学模拟模型，批量生成通过物理法则推导出的“参数-曲线”数据集。
2.  使用 **深度学习 (1D-CNN)** 训练一个逆向模型，学习“从荧光曲线推断微观参数”的映射关系。
3.  输入真实的实验数据，系统即可**自动预测**出最符合实验现象的这 7 个关键物理参数。
4.  最后通过 **MATLAB** 进行回带验证，画图确认预测结果的准确性。

## 2. 核心功能 (Key Features)
*   **物理模拟 (Simulation)**: 基于能量景观理论（Energy Landscape）和主方程（Master Equation）的动力学仿真。
*   **自动筛选 (Auto-Filtering)**: 在数据生成阶段自动剔除无效、死锁或不发生反应的样本（Line 68-70 in gendata.m）。
*   **深度推理 (Deep Inference)**: 使用一维卷积神经网络 (InverseCNN) 处理时序荧光信号，具有极高的推理速度（毫秒级）。
*   **闭环验证 (closed-loop Verification)**: 预测 -> 仿真 -> 对比，形成完整的证据链。

## 3. 预测参数说明 (Parameters)
模型预测的 7 个核心物理参数如下：

| 参数名 | 物理含义 | 典型范围 | 单位 |
| :--- | :--- | :--- | :--- |
| **`E_b`** | 结合能 (Binding Energy) | -2.0 ~ -0.5 | $k_BT$ |
| **`E_b_azo_trans`** | 反式偶氮苯结合能 (Binding Energy, Trans) | -2.0 ~ 0.5 | $k_BT$ |
| **`E_b_azo_cis`** | 顺式偶氮苯结合能 (Binding Energy, Cis) | -1.0 ~ 0.0 | $k_BT$ |
| **`k_mig`** | 迁移速率常数 (Migration Rate) | 0.01 ~ 1.0 | $s^{-1}$ |
| **`k0`** | 零力解离速率 (Leg Detachment Rate at Zero Force) | 1e-6 ~ 1e-4 | $s^{-1}$ |
| **`drt_z`** | 拉链式解离耦合距离 (Coupling Distance, Unzipping) | 0.0 ~ 1.0 | nm |
| **`drt_s`** | 剪切式解离耦合距离 (Coupling Distance, Shearing) | 0.0 ~ 1.0 | nm |

## 4. 文件结构说明 (Files)

### 核心脚本
*   **`gendata.m`** (Matlab): **[步骤 1]** 数据生成脚本。
    *   使用拉丁超立方采样 (LHS) 在指定范围内随机采样参数。
    *   调用并行计算池 (Parpool) 进行大规模仿真。
    *   结果保存为 `training_dataset.mat`。
*   **`train_mlp.py`** (Python): **[步骤 2]** 模型训练脚本。
    *   加载 .mat 数据，清洗、标准化。
    *   训练 CNN 模型，保存最佳权重到 `results/best_mlp_model.pth`。
*   **`verify.py`** (Python): **[步骤 3]** 推理与预测脚本。
    *   读取实验数据 Excel 文件。
    *   加载训练好的模型进行推理，输出预测参数。
    *   自动生成 `matlab_input_params.txt` 供验证使用。
*   **`verify.m`** (Matlab): **[步骤 4]** 验证脚本。
    *   读取 `matlab_input_params.txt`。
    *   再次运行高精度物理仿真。
    *   绘制“实验数据点”与“预测模拟曲线”的对比图。

### 配置文件与辅助
*   **`configfile.ini`**: 全局配置文件。所有参数范围、文件路径、网络结构参数均在此修改。
*   **`model.py`**: 定义 `InverseCNN` 神经网络结构。
*   **`inference.py`**: 定义 `NanorobotPredictor` 类，封装了复杂的预测逻辑。
*   **`utils.py`**: 数据预处理工具（对数变换、标准化、数据加载）。
*   **`predict.py`**: 辅助脚本，被 `verify.py` 调用以加载真实数据。

## 5. 详细使用教程 (Usage Tutorial)

### 准备工作 (Prerequisites)
1.  **MATLAB 环境**:
    *   安装 MATLAB R2021b 或更新版本。
    *   必须安装 **Parallel Computing Toolbox**（用于 `gendata.m` 并行加速）。
2.  **Python 环境**:
    *   建议使用 Anaconda 创建虚拟环境。
    *   安装依赖:
        ```bash
        pip install torch numpy pandas scikit-learn matplotlib scipy openpyxl
        ```
3.  **实验数据**:
    *   准备你的实验数据 Excel 文件（例如 `Fig3a_fitting.xlsx`）。
    *   确保文件包含表头：`Time`, `FAM/FAM T (+)`, `TYE/TYE T (-)`, `CY5/CY5 T (m)`。

---

### Step 1: 生成训练数据 (Data Generation)
**目的**: 让计算机通过大量模拟，学会“什么参数对应什么曲线”。

1.  打开 MATLAB。
2.  打开 `gendata.m`。
3.  (可选) 修改 `configfile.ini` 或脚本顶部的 `num_samples` (推荐 10000+)。
4.  运行脚本。
    *   *提示*: 首次运行会启动并行池（Parpool），可能需要几十秒。
    *   *输出*: 脚本运行结束后，当前目录下会生成 **`training_dataset.mat`**。

### Step 2: 训练模型 (Model Training)
**目的**: 训练神经网络。

1.  打开终端 (Terminal) 或 CMD。
2.  切换到项目目录。
3.  运行命令：
    ```bash
    python train_mlp.py
    ```
4.  **过程**:
    *   程序会自动划分训练集/验证集。
    *   你会看到 Loss（损失值）不断下降。
    *   训练完成后，模型文件会保存在 `results/` 文件夹下 (`best_mlp_model.pth`)，同时保存归一化器 (`x_scaler.pkl`, `y_scaler.pkl`)。

### Step 3: 预测参数 (Prediction)
**目的**: 让模型看一眼你的实验数据，猜出背后的参数。

1.  确保 `configfile.ini` 中的 `path_to_experimental_data_a` 指向你的 Excel 文件名。
2.  运行命令：
    ```bash
    python verify.py
    ```
3.  **输出**:
    *   屏幕上会打印出预测到的 7 个参数值。
    *   当前目录下会生成一个文本文件 **`matlab_input_params.txt`**。

### Step 4: 结果验证 (Verification)
**目的**: 眼见为实，用预测出的参数跑一次模拟，看能不能重现实验现象。

1.  打开 MATLAB。
2.  (如果是在不同电脑上操作) 将 `matlab_input_params.txt` 复制到 MATLAB 的当前工作目录。
3.  运行 `verify.m`。
4.  **结果**:
    *   MATLAB 会读取该 txt 文件中的参数。
    *   运行一次精细的模拟。
    *   **弹出窗口**: 显示三张图（FAM, TYE, CY5）。红/绿/蓝色实线是**预测结果**，散点是你的**原始实验数据**。
    *   如果两者重合度高，说明反推成功！

---

## 6. 常见问题 (Troubleshooting)

*   **Q: 运行 `gendata.m` 时提示 `Out of Memory`？**
    *   A: 请减小 `num_samples` 或者在代码中搜索 `batch_size` 并调小该值（默认 5000）。
*   **Q: Python 提示 `ModuleNotFoundError`？**
    *   A: 请检查是否漏装了库，通常是 `pip install torch pandas openpyxl`。
*   **Q: 验证时的曲线完全对不上？**
    *   A:
        1.  检查 Excel 里的时间单位是否是**分钟 (min)**，代码默认按分钟读取但按秒模拟。
        2.  检查 `configfile.ini` 里的 `sim_duration_minutes` 是否与实验时长一致。
        3.  重新生成更多数据（Step 1）并重新训练（Step 2）。

---
---

# 🇺🇸 English Guide (English Version)

## 1. Introduction
This project automates the **inverse design** of DNA nanorobots.
Experimentalists typically observe fluorescence kinetics (FAM, TYE, CY5 curves) but struggle to determine the underlying microscopic parameters (e.g., Binding Energy, Migration Rate).

This system solves this by:
1.  **Simulating** massive datasets of "Parameter-to-Curve" pairs using MATLAB.
2.  **Training** a Deep Learning model (1D-CNN) to learn the inverse mapping (Curve-to-Parameter).
3.  **Predicting** the 7 key physical parameters from your real experimental data.
4.  **Verifying** the result by running a simulation with the predicted parameters to see if it reproduces the experiment.

## 2. Key Features
*   **Physics-based Simulation**: Accurate kinetic modeling based on Energy Landscapes.
*   **Deep Inference**: Millisecond-level parameter prediction using InverseCNN.
*   **Automated Workflow**: From raw Excel data to verified simulation plots.

## 3. Predicted Parameters
The model predicts the following 7 parameters:

| Parameter | Meaning | Typical Range | Unit |
| :--- | :--- | :--- | :--- |
| **`E_b`** | Binding Energy | -2.0 ~ -0.5 | $k_BT$ |
| **`E_b_azo_trans`** | Binding Energy (Trans-Azo) | -2.0 ~ 0.5 | $k_BT$ |
| **`E_b_azo_cis`** | Binding Energy (Cis-Azo) | -1.0 ~ 0.0 | $k_BT$ |
| **`k_mig`** | Migration Rate | 0.01 ~ 1.0 | $s^{-1}$ |
| **`k0`** | Leg Detachment Rate | 1e-6 ~ 1e-4 | $s^{-1}$ |
| **`drt_z`** | Splitting Dist. (Unzipping) | 0.0 ~ 1.0 | nm |
| **`drt_s`** | Splitting Dist. (Shearing) | 0.0 ~ 1.0 | nm |

## 4. File Structure (Files)

### Core Scripts
*   **`gendata.m`** (Matlab): **[Step 1]** Data generation script.
    *   Uses Latin Hypercube Sampling (LHS) to sample parameters within specified ranges.
    *   Utilizes a Parallel Computing Pool (Parpool) for large-scale simulations.
    *   Saves results to `training_dataset.mat`.
*   **`train_mlp.py`** (Python): **[Step 2]** Model training script.
    *   Loads the .mat dataset, performs cleaning and standardization.
    *   Trains the CNN model and saves the best weights to `results/best_mlp_model.pth`.
*   **`verify.py`** (Python): **[Step 3]** Inference and prediction script.
    *   Reads the experimental data Excel file.
    *   Loads the trained model for inference and outputs the predicted parameters.
    *   Automatically generates `matlab_input_params.txt` for verification.
*   **`verify.m`** (Matlab): **[Step 4]** Verification script.
    *   Reads `matlab_input_params.txt`.
    *   Runs a high-precision physical simulation based on the predicted parameters.
    *   Plots a comparison graph between "Experimental Data Points" and the "Predicted Simulation Curve".

### Configuration & Helpers
*   **`configfile.ini`**: Global configuration file. All parameter ranges, file paths, and network architecture settings are modified here.
*   **`model.py`**: Defines the `InverseCNN` neural network architecture.
*   **`inference.py`**: Defines the `NanorobotPredictor` class, encapsulating complex prediction logic.
*   **`utils.py`**: Data preprocessing tools (Log transformation, Standardization, Data loading).
*   **`predict.py`**: Helper script called by `verify.py` to load real data.

## 5. Work Flow Details

### Prerequisites
*   **MATLAB**: R2021b+ with Parallel Computing Toolbox.
*   **Python**: 3.8+ (`pip install torch numpy pandas scikit-learn matplotlib scipy openpyxl`).

### Step-by-Step Guide

#### Step 1: Data Generation (`gendata.m`)
Open MATLAB and run `gendata.m`.
*   This uses Latin Hypercube Sampling (LHS) to explore the parameter space.
*   It runs simulations in parallel.
*   **Output**: `training_dataset.mat`.

#### Step 2: Model Training (`train_mlp.py`)
Run in terminal: `python train_mlp.py`
*   Loads the `.mat` dataset.
*   Trains the Neural Network.
*   **Output**: Saves model to `results/best_mlp_model.pth`.

#### Step 3: Parameter Prediction (`verify.py`)
Run in terminal: `python verify.py`
*   Input: Your experimental Excel file (set path in `configfile.ini`).
*   Inference: Predicts the 7 parameters.
*   **Output**: Generates **`matlab_input_params.txt`**.

#### Step 4: Verification (`verify.m`)
Open MATLAB and run `verify.m`.
*   It reads `matlab_input_params.txt`.
*   Runs a high-precision simulation using these parameters.
*   **Plot**: Overlays the simulation result (lines) on top of your experimental data (dots).
*   **Success**: If the lines match the dots, the prediction is accurate.

## 6. Troubleshooting
*   **Memory Errors in MATLAB**: Reduce `num_samples` or the batch size in `gendata.m`.
*   **Python Imports**: Ensure you are in the correct directory having `utils.py`, `model.py`, etc.
*   **Poor Fit**:
    1.  Check if your experimental time units match the simulation (minutes).
    2.  Increase `num_samples` in Step 1 and retrain.
