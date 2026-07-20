# Changed Files Manifest (v2) / 改动文件清单 (v2)

相对 **v1 快照** 新增的「验证实验套件 (validation suite)」改动与新增。
**目录结构与原项目一致**，可直接覆盖回原项目对应位置。

> 本 v2 包聚焦于 v1 之后新增的**科学验证与可识别性分析**工作：
> 合成参数恢复、Fisher/profile 可识别性分析、DL 暖启动消融、多种子重训，
> 以及属性测试套件 (Hypothesis) 和更新后的研究报告。
>
> 不含：训练模型 (`*.pth`)、归一化器 (`*.pkl`)、数据集 (`*.npz`)、日志 (`*.log`)、
> 预测输出 (`matlab_input_params.txt`)。这些为可复现产物，按需用脚本重新生成。
> v1 已包含的文件 (`gendata.py`, `refine.py`, `verify.py`, 各 `model_*.py`、
> `train_*.py`、`predict*.py` 等) 未重复打包，本包仅含 **v1 之后的增量**。

---

## A. 修改过的文件 (Modified, 相对 v1)

| 文件 | 改动摘要 |
|------|---------|
| `README.md` | 新增验证实验结果章节：参数恢复、Fisher/profile 可识别性、DL 暖启动消融、多种子重训 (双语，含复现命令) |
| `pysim.py` | 新增正向模拟调用计数器：函数体重命名为 `_run_simulation_impl`，外层 try/finally 包裹计数 (线程安全，仅用标准库)。用于暖启动消融的「到达目标所需前向模拟次数」度量 |
| `train_cnn/eval_testset.py` | 暴露 `test_seed` 参数，支持多种子重训复现 |
| `train_transformer/eval_testset.py` | 暴露 `test_seed` 参数，支持多种子重训复现 |
| `AIS5281 Research Report - Tse Kam Pui.docx` | 全面更新：MATLAB→pysim、模型架构修正 (PatchTST+MHA+物理精修)、结论改为 CNN 胜出、新增 §4e (可识别性/暖启动/多种子) 与 §4f (为何 CNN 优于 Transformer)、新增图 8b/8c/9-13 |

## B. 新增的源文件 (Added)

**根目录 — 验证实验套件 (纯 Python):**
- `validation_common.py` — 共享工具：随机种子守卫、拉丁超立方采样 (LHS)、统计指标 (R²/RMSE/bootstrap CI)、预测器工厂 (path-insert + 顶层 import)、JSON 写出
- `validate_recovery.py` — 合成参数恢复实验：LHS 采样真值 → 正向模拟 → DL+精修反演 → 逐参数 R²/散点图
- `benchmark_initguess.py` — DL vs 随机/参考/LHS-multistart 暖启动消融：以「到达目标所需前向模拟次数」为度量
- `multiseed_retrain.py` — 多种子重训：跨种子统计 CNN/Transformer 测试 RMSE 均值±标准差与 CV；含 `--models` 子集标志与 `merge_results()` 支持 CNN/Transformer 并行运行
- `analyze_identifiability.py` — Fisher 信息矩阵 (条件数/特征谱) 与 profile likelihood 可识别性分析；出敏感性图

**tests/ — 属性测试套件 (pytest + Hypothesis):**
- `conftest.py` — 共享 fixture 与路径配置
- `test_validation_common.py` — `validation_common` 工具单元/属性测试
- `test_validate_recovery.py` — 恢复实验测试
- `test_benchmark_initguess.py` — 暖启动消融测试
- `test_multiseed_retrain.py` — 多种子重训测试
- `test_pysim_counter.py` — pysim 调用计数器线程安全/正确性测试
- `test_computation_only.py` — 纯计算路径测试

## C. 展示用结果 (Presentation outputs)

**结果图 (PNG) — 根目录 results/validation/:**
- `results/validation/identifiability_sensitivity.png` — Fisher/profile 可识别性敏感性图 (ASCII-only)
- `results/validation/recovery_pair_k0_kmig.png` — k0 / k_mig 参数对恢复联合图 (ASCII-only)
- `results/validation/multiseed_compare.png` — CNN vs Transformer 多种子 RMSE 对比 (合并)
- `results/validation/ms_cnn/multiseed_compare.png` — CNN 多种子对比
- `results/validation/ms_transformer/multiseed_compare.png` — Transformer 多种子对比

**结果图 (PNG) — train_cnn/results/validation/:**
- `recovery_scatter_{k0,k_mig,drt_s,drt_z,E_b,E_b_azo_trans,E_b_azo_cis}.png` — 7 个参数逐一恢复散点图 (真值 vs 反演)
- `warmstart_cnn.png` — DL vs 随机/参考/LHS 暖启动收敛对比图

**结果 JSON:**
- `results/validation/identifiability_metrics.json` — FIM 条件数 (≈1.3×10¹⁵)、特征谱、profile likelihood 谷宽
- `results/validation/multiseed_metrics.json` — 合并多种子统计
- `results/validation/ms_cnn/multiseed_metrics.json` — CNN 多种子 (0.0196±0.0007, CV 3.5%, 5 seeds)
- `results/validation/ms_transformer/multiseed_metrics.json` — Transformer 多种子 (0.0395±0.0036, CV 9.1%, 4 seeds)
- `train_cnn/results/validation/recovery_metrics.json` — 逐参数恢复 R² (drt_z 0.55, k0 0.52, drt_s 0.42; E_b/azo/k_mig 较差)
- `train_cnn/results/validation/warmstart_cnn.json` — 暖启动消融中位前向调用数 (DL=1 vs random=96, reference=141, lhs_multistart=185)

## D. 关键结果摘要 (Key results summary)

| 实验 | 结论 |
|------|------|
| **参数恢复** (CNN, 200 样本) | 参数依赖的可识别性：drt_z/k0/drt_s 中等可恢复 (R² 0.42–0.55)，E_b/azo/k_mig 较差。证明问题本质病态 |
| **可识别性** (Fisher + profile) | FIM 条件数 ≈1.3×10¹⁵，特征值跨 ~15 个数量级；k0 谷最陡 (0.148)，drt_s/E_b_azo_trans 最平 |
| **DL 暖启动消融** (50 目标) | DL 中位到达目标仅需 1 次前向模拟，vs 随机 96 / 参考 141 / LHS-multistart 185 → 约降低 2 个数量级 |
| **多种子重训** | CNN 0.0196±0.0007 (CV 3.5%) vs Transformer 0.0395±0.0036 (CV 9.1%)，区间不重叠 → CNN 更优且更稳 |
| **测试集** (1000 样本) | CNN 0.0188 均值 / 0.1% 失败 vs Transformer 0.0465 / 3.2% 失败 |

> 复现命令见 `README.md` 对应章节。所有图均为纯 ASCII / 全英文 (无 Unicode tofu)。
