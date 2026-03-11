# train_transformer/  —  DNA Walker Transformer 训练模块

## 文件说明

| 文件 | 说明 |
|------|------|
| `config_transformer.ini` | Transformer 专用超参数配置 |
| `config_loader_transformer.py` | 配置加载器（读取本目录 INI + 上层 configfile.ini）|
| `dataset.py` | 数据加载（X 保持 3D 形状，不展平）|
| `model_transformer.py` | PatchTST-Inspired Transformer 模型定义 |
| `train_transformer.py` | 主训练脚本（AdamW + Cosine Warmup + Early Stopping）|

---

## 快速开始

**在项目根目录或 `train_transformer/` 目录下均可运行。**

### 1. 烟雾测试（验证流程，约 1 分钟）

```powershell
cd d:\阿里云盘共享\NUSproject\MachineLearning_for_DNAwalker\train_transformer
python train_transformer.py --smoke
```

### 2. 正式训练

```powershell
cd d:\阿里云盘共享\NUSproject\MachineLearning_for_DNAwalker\train_transformer
python train_transformer.py
```

训练完成后，最佳模型保存到 `results/best_transformer_model.pth`。

### 3. 验证模型架构（不需要数据）

```powershell
python model_transformer.py
```

---

## 模型架构

```
输入: (B, 3, 7801)   — 三条归一化荧光曲线
  ↓ Patch Embedding (patch=50, stride=25) → (B, 3, ~310, 256)
  ↓ 可学习位置编码
  ↓ 4× Temporal Self-Attention (通道独立，8 heads)
  ↓ 2× Cross-Channel Attention (FAM↔TYE↔CY5 交互)
  ↓ Global Average Pool → (B, 256)
  ↓ Linear(256→128) → GELU → Dropout → Linear(128→7) → Sigmoid
输出: (B, 7)   — 预测物理参数（Sigmoid Lock，对应 y ∈ [0.1, 0.9]）
```

参数量约 **4.5M**（与 CNN 的 7M 相近，但全局感受野更强）。

---

## 与 CNN 的对比

| | CNN (`train_mlp.py`) | Transformer (本模块) |
|-|---|---|
| 输入 | `(N, 23403)` 展平 | `(N, 3, 7801)` 3D |
| 长程依赖 | ❌ 局部感受野 | ✅ 全局自注意力 |
| 通道关系 | 隐式融合 | 显式 Cross-Channel Attention |
| 优化器 | Adam | AdamW |
| LR 调度 | ReduceLROnPlateau | Cosine Annealing + Warmup |

---

## 配置调整建议

编辑 `config_transformer.ini` 中的 `[TRANSFORMER]` 节：

- **显存不足**：`batch_size = 64`，`d_model = 128`
- **提升精度**：`n_layers = 6`，`d_model = 512`，`n_heads = 8`
- **加快收敛**：`warmup_ratio = 0.05`，`learning_rate = 2e-4`

---

## 数据集路径

数据集默认读取 `../results/training_dataset.npz`（上层目录的 results/）。
若路径不同，请修改 `config_transformer.ini` 中的 `[PATHS] dataset_file`。
