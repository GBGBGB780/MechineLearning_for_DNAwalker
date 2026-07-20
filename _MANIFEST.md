# Changed Files Manifest / 改动文件清单

相对原始仓库 [GBGBGB780/MechineLearning_for_DNAwalker](https://github.com/GBGBGB780/MechineLearning_for_DNAwalker)
的全部改动与新增。**目录结构与原项目一致**，可直接覆盖回原项目对应位置。

> 不含：训练模型 (`*.pth`)、归一化器 (`*.pkl`)、数据集 (`*.npz`)、日志 (`*.log`)、
> 预测输出 (`matlab_input_params.txt`)。这些为可复现产物，按需用脚本重新生成。

---

## A. 修改过的文件 (Modified, 相对原始 repo)

| 文件 | 改动摘要 |
|------|---------|
| `README.md` | 大改：结果、消融实验、纯 Python 流程、目录结构、模型对比 |
| `requirements.txt` | matplotlib 注释更新 |
| `train_cnn/model_cnn.py` | 新增 `_MPSCompatAdaptiveAvgPool1d`（MPS 不支持非整除自适应池化）|
| `train_cnn/train_mlp.py` | device 选择加 MPS 分支 |
| `train_cnn/inference_cnn.py` | device 选择加 MPS 分支 |
| `train_cnn/predict.py` | 修复实验 Excel 列名读取（改用 `exp_data_io`）|
| `train_transformer/train_transformer.py` | MPS 分支；修复 `non_blocking` 在 MPS 上致 loss=inf 的竞争 bug |
| `train_transformer/dataset.py` | `pin_memory` 仅 CUDA 启用 |
| `train_transformer/inference_transformer.py` | device 选择加 MPS 分支 |
| `train_transformer/predict.py` | 修复实验 Excel 列名读取 |
| `train_transformer/config_transformer.ini` | 标准配置 patch=50→100, batch 32→64, epoch 1000→300 |

## B. 新增的源文件 (Added)

**根目录 — 共享物理内核 (纯 Python，无需 MATLAB):**
- `pysim.py` — 14 态马尔可夫正向模拟器 (gendata.m 移植，加速 35×)
- `gendata.py` — 训练数据生成 (gendata.m / gendata_smoke.m 移植)
- `verify.py` — 正向验证 + RMSE + 对比图 (verify.m 移植)
- `refine.py` — 物理参数局部精修 (直接最小化曲线 RMSE)
- `exp_data_io.py` — 实验 Excel 健壮读取
- `run_smoke_test.sh` — 纯 Python 端到端冒烟测试
- `DNAwalker_project.ipynb` — 学术报告 notebook (更新到最新结果)

**train_cnn/ (自包含):**
- `predict_refine.py` — 预测 + 物理精修
- `eval_dual.py` — 原始+泛化双数据集评估 + 出图
- `eval_testset.py` — 1000 样本测试集预测质量
- `configfile.ablation_small.ini` — 容量消融配置 (~3.27M)

**train_transformer/ (自包含):**
- `predict_refine.py` — 预测 + 物理精修
- `eval_dual.py` — 双数据集评估 + 出图
- `eval_testset.py` — 1000 样本测试集预测质量
- `evaluate_rmse.py` — 单数据集真实指标评估
- `config_transformer.ablation_big.ini` — 容量消融配置 (~4.30M)

## C. 展示用结果 (Presentation outputs)

**结果图 (PNG):**
- `results/evidence_spectrum.png` — 训练曲线功率谱 (证明低频/平滑)
- `results/evidence_autocorr.png` — 自相关半衰期分布
- `train_cnn/results/dual_fit.cnn.png` — CNN 原始+泛化拟合图
- `train_transformer/results/dual_fit.standard.png` — Transformer 原始+泛化拟合图
- `train_transformer/matlab_input_params_verify.png` — 单组参数验证图

**结果 JSON:**
- `train_cnn/results/testset_eval.cnn.json` — CNN 标准 (4.38M) 测试集指标
- `train_cnn/results/testset_eval.cnn_small.json` — CNN 消融 (3.27M) 测试集指标
- `train_cnn/results/rmse_dual.cnn.json` — CNN 双数据集 RMSE
- `train_transformer/results/testset_eval.transformer.json` — Transformer 标准 (3.24M)
- `train_transformer/results/testset_eval.transformer_big.json` — Transformer 消融 (4.30M)
- `train_transformer/results/rmse_dual.standard.json` — Transformer 双数据集 RMSE

## D. 预测参数 (Predicted parameters)

精修后的最优物理参数 (`predict_refine.py` 输出，供 MATLAB/Python 正向验证)：
- `train_cnn/matlab_input_params.txt` — CNN 预测 + 精修 (original 数据集, RMSE 0.0078)
- `train_transformer/matlab_input_params.txt` — Transformer 预测 + 精修 (RMSE 0.0080)

> 注：这是预测输出，每次运行 `predict.py` / `predict_refine.py` 会被覆盖；
> 数值取决于模型与 ensemble 随机种子。
