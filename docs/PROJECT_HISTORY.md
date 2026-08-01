# Project History

# 项目历程

Last reconstructed / 最后重建：**2026-07-30**

This document records how the DNA Walker inverse-design project developed from
its earliest recoverable form to the final local scientific closeout. It
connects the scientific work, implementation changes, training lineages,
validation decisions, repository reorganization, and publication cleanup that
are otherwise distributed across Git, the report, the Notebook, Kiro records,
artifact manifests, and archived experiments.

本文记录 DNA Walker 逆向设计项目从目前可追溯的最早形态，到本地科学研究正式收尾
为止的发展过程。它将原本分散在 Git、报告、Notebook、Kiro 记录、制品清单与历史
实验中的科学工作、实现变更、训练谱系、验证决策、仓库重组和发布清理串联起来。

This is a history, not the current operating manual. For current commands and
claims, use the root [`README.md`](../README.md), and treat the curated JSON in
[`evidence/`](evidence/) as the machine-readable final evidence.

本文是一份历史记录，而不是当前操作手册。当前命令和结论以根目录
[`README.md`](../README.md) 为准；机器可读的最终证据以
[`evidence/`](evidence/) 中经过整理的 JSON 为准。

---

## Contents / 目录

- [1. How to Read This Record / 如何阅读本记录](#history-reading)
- [2. Final State in One Page / 最终状态概览](#history-final-state)
- [3. Timeline Summary / 时间线总览](#history-timeline)
- [4. Reconstructed Origin / 重建的项目起点](#history-origin)
- [5. First Git Baseline and Initial Claims / 首个 Git 基线与早期结论](#history-baseline)
- [6. Reliability and Review Passes / 可靠性修复与审查](#history-review)
- [7. Freeze Before Repository Reorganization / 仓库重组前冻结](#history-freeze)
- [8. Canonical Package Migration / 规范包迁移](#history-migration)
- [9. Provenance Reset and Current 10k Study / 来源规范重建与当前 10k 实验](#history-10k)
- [10. Locked 30k Nested Learning Curve / 锁定的 30k 嵌套学习曲线](#history-30k)
- [11. Experimental-Fit Robustness / 实验拟合稳定性](#history-fits)
- [12. Identifiability and Interpretation Boundary / 可辨识性与解释边界](#history-identifiability)
- [13. Retained, Withdrawn, or Archived Work / 保留、撤回与归档工作](#history-lifecycle)
- [14. Dataset and Model Lineages / 数据集与模型谱系](#history-lineages)
- [15. Engineering and Repository Work / 工程与仓库工作](#history-engineering)
- [16. Publication Cleanup and Documentation Recovery / 发布清理与文档恢复](#history-publication)
- [17. Decision Log / 关键决策记录](#history-decisions)
- [18. Commit Ledger / 提交简表](#history-commits)
- [19. Current Documentation Map / 当前文档地图](#history-doc-map)
- [20. Final Status and Remaining Work / 最终状态与剩余工作](#history-status)

---

<a id="history-reading"></a>
## 1. How to Read This Record / 如何阅读本记录

### 1.1 Evidence levels / 证据等级

The project did not begin with a continuous Git history. The first commit
already contained a mature simulator, two model families, validation scripts,
a report, and a Notebook. This document therefore distinguishes four evidence
levels instead of presenting every early event as equally certain.

本项目在最初阶段没有连续的 Git 历史。第一个提交已经包含成熟的模拟器、两类模型、
验证脚本、报告和 Notebook。因此，本文将记录分为四种证据等级，不会把所有早期事件
都写成同等确定的事实。

| Level / 等级 | Meaning / 含义 | Main sources / 主要来源 |
|---|---|---|
| **A: hash-bound / 哈希绑定** | Exact dataset, split, checkpoint, scaler, result, or commit identity is recorded. / 数据集、划分、checkpoint、scaler、结果或提交具有精确身份记录。 | Git commits, SHA-256 manifests, current-schema checkpoints, curated JSON |
| **B: versioned record / 版本化记录** | The work is present in Git, but may predate the final artifact-provenance contract. / 工作已进入 Git，但可能早于最终制品来源规范。 | Git history beginning at `9f7bf3e` |
| **C: reconstructed / 重建记录** | The sequence is inferred from the dated report, Notebook, source layout, or archive names; exact intermediate commits are unavailable. / 根据带日期的报告、Notebook、源码布局或归档名称重建，缺少逐步提交。 | Local report, Notebook, `archive/` |
| **D: historical only / 仅供历史参考** | The files explain prior work but are not valid current scientific evidence. / 文件用于解释旧工作，但不能作为当前科学证据。 | Legacy checkpoints, old result summaries, withdrawn experiments |

### 1.2 Known limits / 已知限制

- The local report title page is dated **2026-04-19**, but the file was revised
  through the final audit. This is the earliest recoverable date associated
  with the project, not an authenticated snapshot of its exact April state.
  本地报告标题页日期为 **2026-04-19**，但该文件之后持续更新到最终审计。因此它能
  提供目前可恢复的最早项目标注日期，但不能视为经过认证的四月项目状态快照。
- The directory
  `archive/experiments/transformer/legacy_mha_smallrange_m03method_20260517/`
  carries a **2026-05-17** experiment label. This is useful historical context,
  not a Git-authenticated timestamp.
  目录
  `archive/experiments/transformer/legacy_mha_smallrange_m03method_20260517/`
  带有 **2026-05-17** 的实验标签。该标签可用于理解历史背景，但不是经过 Git
  认证的时间戳。
- Git history starts on **2026-07-06** with commit `9f7bf3e`, which imported
  79 files and about 22,000 source lines. Work before that point can only be
  reconstructed.
  Git 历史从 **2026-07-06** 的提交 `9f7bf3e` 开始；该提交一次性加入 79 个文件和
  约 22,000 行内容。更早的过程只能重建。
- [`ARTIFACTS.md`](ARTIFACTS.md) labels one compatibility check
  **2026-06-26**, while the referenced pre-reorganization Git checkpoint
  `5f122d8` has commit metadata dated **2026-07-17**. Both source records are
  preserved here; this document does not invent a correction for the timestamp
  discrepancy.
  [`ARTIFACTS.md`](ARTIFACTS.md) 将一次兼容性检查标记为
  **2026-06-26**，但其引用的重组前 Git checkpoint `5f122d8` 的提交元数据日期为
  **2026-07-17**。本文保留两项来源记录，不擅自推断哪个日期应被修改。
- Local `.kiro/` files were useful process records during reconstruction.
  Their durable decisions were incorporated here before the local files were
  removed, so the public history does not depend on agent state.
  本地 `.kiro/` 文件在历史重建时曾是有用的过程记录。其长期有效的决策已在删除本地
  文件前纳入本文，因此公开项目史不依赖代理状态。

---

<a id="history-final-state"></a>
## 2. Final State in One Page / 最终状态概览

The project addresses a seven-parameter inverse problem for a light-driven DNA
molecular walker. A 14-state physical simulator maps parameters to three
fluorescence trajectories, each containing 7,801 time points over 130 minutes.
CNN and Transformer models learn the inverse map from synthetic curves to
candidate parameters. Optional Powell refinement then minimizes curve RMSE
through the same forward simulator.

本项目研究光驱动 DNA 分子 Walker 的七参数逆问题。一个 14 状态物理模拟器将参数映射
为三条荧光轨迹；每条轨迹覆盖 130 分钟，共 7,801 个时间点。CNN 与 Transformer
从合成曲线学习到候选参数的逆映射；可选的 Powell 精修再通过同一个正向模拟器直接
最小化曲线 RMSE。

By the final local closeout, the repository contained:

本地收尾时，仓库已经具备：

- one canonical pure-Python package under `dnawalker/`;
  位于 `dnawalker/` 下的唯一纯 Python 规范实现；
- deterministic Latin-Hypercube candidate sampling, physical filtering,
  activity-bin quotas, and explicit dataset versioning;
  确定性的拉丁超立方候选采样、物理筛选、活跃度分桶配额和明确的数据集版本控制；
- CNN and Transformer training, inference, held-out evaluation, and
  experimental-curve evaluation;
  CNN 与 Transformer 的训练、推理、留出测试和实验曲线评估；
- model-agnostic physical refinement and forward verification;
  与模型架构无关的物理精修和正向验证；
- fixed-split, multi-seed, learning-curve, fit-robustness,
  identifiability, and signal-diagnostic workflows;
  固定划分、多种子、学习曲线、拟合稳定性、可辨识性与信号诊断流程；
- current-schema checkpoints that bind model seed, split seed, dataset hash,
  scaler hash, parameter order, epoch, and validation metric;
  绑定模型种子、划分种子、数据集哈希、scaler 哈希、参数顺序、epoch 与验证指标的
  当前格式 checkpoint；
- a current 401-test regression suite, curated public evidence, and an
  explicit publication boundary.
  当前 401 项回归测试、整理后的公开证据，以及明确的发布边界。

The final scientific conclusion is deliberately narrower than the early
project claims:

最终科学结论比项目早期结论更谨慎：

1. Both architectures improve when retained training volume increases.
   两种架构都会随着保留训练数据量增加而改善。
2. No controlled comparison establishes a stable CNN or Transformer
   advantage.
   没有任何受控对比能够证明 CNN 或 Transformer 具有稳定优势。
3. The 24k means are nearly tied, but five model seeds do not prove practical
   equivalence within the predeclared `+/-0.001` margin.
   24k 训练规模下两者均值近似持平，但五个模型种子不足以在预先声明的
   `+/-0.001` 范围内证明实际等价。
4. Experimental-fit variation remains sensitive to non-convex physical
   refinement, especially on the generalization trace.
   实验曲线拟合仍对非凸物理精修的起点敏感，泛化曲线尤其明显。
5. A low curve RMSE supports curve consistency, not unique recovery of all
   seven microscopic parameters.
   低曲线 RMSE 只能支持曲线一致性，不能证明七个微观参数被唯一恢复。

The scoped local study is complete. Additional training is not recommended for
the current question. Public binary-artifact release remains separate work.

既定范围内的本地研究已经完成。针对当前问题，不建议继续增加训练。公开二进制制品
发行仍属于独立工作。

---

<a id="history-timeline"></a>
## 3. Timeline Summary / 时间线总览

| Date or period / 日期或阶段 | Evidence / 证据 | Milestone / 里程碑 |
|---|---|---|
| Report date 2026-04-19 / 报告标注日期 2026-04-19 | C | The revised report preserves the earliest recoverable project date and records the scientific problem, mechanistic model, synthetic-data approach, and inverse-learning objective. / 修订后的报告保留了目前可恢复的最早项目日期，并记录科学问题、机理模型、合成数据方案与逆向学习目标。 |
| 2026-05-17 label / 标签 | C-D | Archived legacy Transformer MHA experiments explored targeted small-range datasets and several subset-selection strategies. / 归档的旧 Transformer MHA 实验探索了小范围数据与多种子集选择策略。 |
| Before 2026-07-06 / 2026-07-06 之前 | C-B | MATLAB generation/verification was ported into a Python-first workflow; CNN, Transformer, refinement, recovery, warm-start, speed, identifiability, and multi-seed code existed. / MATLAB 生成与验证流程被迁移到 Python 优先工作流；CNN、Transformer、精修、恢复、warm-start、速度、可辨识性和多种子代码已经存在。 |
| 2026-07-06 | B | The first Git baseline was created, followed by reproducibility, data-generation, evaluation, cost-accounting, ensemble, and FIM fixes. / 建立首个 Git 基线，并完成可复现性、数据生成、评估、成本计数、集成推理和 FIM 修复。 |
| 2026-07-11 | B | An engineering/scientific review fixed split leakage risk, dead configuration, Transformer preprocessing drift, parameter-order safety, duplicated evaluation, dependencies, and CI. / 一轮工程与科学审查修复了划分泄漏风险、无效配置、Transformer 预处理漂移、参数顺序安全、重复评估、依赖与 CI。 |
| 2026-07-17 | A-B | The pre-reorganization state was frozen at `5f122d8`; inventories, audit JSON, dependency locks, and migration constraints were recorded. / 在 `5f122d8` 冻结重组前状态，并记录文件清单、审计 JSON、依赖锁和迁移约束。 |
| 2026-07-19 | B | Active code moved into the canonical `dnawalker/` package while behavior-preservation checks remained in force. / 活跃代码迁入规范 `dnawalker/` 包，同时保持行为不变检查。 |
| 2026-07-21 | A-B | The compatibility facades were removed, configuration and paths were centralized, MATLAB was archived, and current-schema retraining safeguards became authoritative. / 删除兼容外壳，集中配置与路径，归档 MATLAB，并确立当前格式重训保障。 |
| 2026-07-21 to 2026-07-23 | A | A hardened 10k dataset and ten provenance-complete CNN/Transformer checkpoints were generated and evaluated with fixed `split_seed=42`. / 生成并评估经过加固的 10k 数据集与十个来源完整的 CNN/Transformer checkpoint，固定 `split_seed=42`。 |
| 2026-07-23 to 2026-07-29 | A | A locked 30k nested study completed 30/30 MPS runs across 8k, 16k, and 24k training subsets. / 锁定的 30k 嵌套实验完成 8k、16k、24k 三种规模下全部 30/30 次 MPS 训练。 |
| 2026-07-29 | A | Validation-selected 24k checkpoints completed ten experimental-fit robustness evaluations across refinement seeds 0-4. / 按验证集选出的 24k checkpoint 完成精修种子 0-4 下十次实验拟合稳定性评估。 |
| 2026-07-29 | A-B | The scientific closeout, curated evidence, sanitized public-source snapshot, publication boundary, and comprehensive bilingual README were completed. / 完成科学收尾、公开证据整理、清理后的源码快照、发布边界与完整双语 README。 |
| 2026-07-29 | B | After project history was consolidated at `2e280df`, nonpublication caches and raw outputs were deleted and withdrawn MATLAB/recovery/warm-start/speed/capacity source was removed from the streamlined release tree. / 在 `2e280df` 汇总项目史后，删除不发布的缓存与原始输出，并从精简发布树移除已撤回的 MATLAB、恢复、warm-start、速度与容量源码。 |

---

<a id="history-origin"></a>
## 4. Reconstructed Origin / 重建的项目起点

### 4.1 Scientific question / 科学问题

The starting point was not a generic machine-learning benchmark. It was a
mechanistic inverse problem: infer seven energetic and kinetic parameters from
three experimentally observable fluorescence channels produced by a
light-controlled DNA walker on a three-site track.

项目起点并不是通用机器学习基准，而是一个机理逆问题：根据光控 DNA Walker 在三位点
轨道上产生的三条可观测荧光通道，推断七个能量与动力学参数。

The seven inferred quantities are three binding-energy terms, one migration
rate, one zero-force dissociation rate, and two force-coupling distances. The
observable curves are indirect consequences of latent state occupancies, so
the inverse map is nonlinear and potentially non-unique.

七个待推断量包括三个结合能项、一个迁移速率、一个零力解离速率，以及两个力耦合
距离。可观测曲线只是隐状态占据率的间接结果，因此逆映射具有非线性，也可能并不唯一。

### 4.2 Forward model and MATLAB origin / 正向模型与 MATLAB 起源

The physical workflow originated in MATLAB scripts for dataset generation,
parameter optimization, and forward verification. These scripts encode a
14-state mechanical-kinetic model, illumination-dependent transitions, and the
conversion from state probabilities to FAM, TYE, and Cy5 signals.

物理流程最初来自用于数据生成、参数优化和正向验证的 MATLAB 脚本。这些脚本实现了
14 状态力学-动力学模型、光照相关转移，以及从状态概率到 FAM、TYE 和 Cy5 信号的
转换。

The simulator was then ported to Python so data generation, training,
inference, refinement, and verification could run without MATLAB. The final
implementation is `dnawalker/core/pysim.py`. The original scripts were first
isolated under `archive/matlab/` and later removed from the streamlined tree;
their last consolidated source snapshot is Git commit `2e280df`.

随后，模拟器被迁移到 Python，使数据生成、训练、推理、精修和验证都不再依赖
MATLAB。最终实现位于 `dnawalker/core/pysim.py`。原始脚本曾被隔离到
`archive/matlab/`，后来从精简发布树移除；最后一个汇总这些源码的 Git 快照是
`2e280df`。

### 4.3 Synthetic-data design / 合成数据设计

Candidate parameter vectors were sampled over configured ranges with Latin
Hypercube Sampling (LHS). Each candidate was passed through the forward
simulator. Invalid, numerically unsuitable, or insufficiently active curves
were rejected, and accepted samples were controlled with activity-bin quotas.

候选参数在配置范围内使用拉丁超立方采样（LHS）生成。每个候选参数都经过正向模拟器；
无效、数值不合适或活跃度不足的曲线会被拒绝，接受样本再通过活跃度分桶配额控制。

LHS makes the **candidate** one-dimensional marginals stratified. It does not
guarantee that the **retained** dataset remains uniform after physical
filtering and activity balancing. This distinction became important during the
later 30k audit.

LHS 使**候选样本**的一维边缘分布具有分层覆盖，但经过物理筛选与活跃度平衡后，
并不能保证**最终保留数据集**仍保持均匀。这一区分在后续 30k 审计中变得非常重要。

Each retained input has shape `(3, 7801)`. The target contains seven physical
parameters. `k0` is log-transformed, each sample's three channels are jointly
normalized, and target scalers are fitted only on the training partition.

每个保留输入的形状为 `(3, 7801)`，目标包含七个物理参数。`k0` 使用对数变换，
每个样本的三条通道进行联合归一化，目标 scaler 只在训练分区上拟合。

### 4.4 First inverse models / 第一批逆向模型

Two model families were developed:

项目开发了两类模型：

- `InverseCNN`: a four-stage 1D convolutional feature extractor followed by a
  bounded regression head;
  `InverseCNN`：四阶段一维卷积特征提取器加有界回归头；
- `DNAWalkerTransformer`: a patch-based temporal Transformer with
  cross-channel attention and a bounded regression head.
  `DNAWalkerTransformer`：基于时间 patch、带跨通道注意力和有界回归头的
  Transformer。

Both predict an initial seven-parameter vector. The project also developed a
model-agnostic Powell refinement stage, including log-space handling for `k0`
and multi-start search, to minimize curve RMSE through the forward simulator.

两者都先预测一个七参数初值。项目还开发了与模型架构无关的 Powell 精修阶段，包括
`k0` 的对数空间处理与多起点搜索，通过正向模拟器最小化曲线 RMSE。

The final formal configurations have the following exact sizes and analytical
forward costs:

最终正式配置具有以下精确规模与解析前向计算量：

| Model / 模型 | Trainable parameters / 可训练参数 | Raw FP32 weights / FP32 原始权重 | Analytical forward MAC/sample / 单样本解析前向 MAC |
|---|---:|---:|---:|
| CNN | `4,381,319` | about `16.7 MiB` | about `38,067,168` |
| Transformer | `3,243,271` | about `12.4 MiB` | about `780,223,360` |

The CNN stores more parameters because its flattened pooled feature map feeds
a `16,384 -> 256` dense layer with `4,194,304` weights. Transformer stores
fewer weights but repeatedly applies attention and feed-forward projections to
78 patches, producing about `20.5x` the CNN forward MAC count. These are
architecture calculations, not wall-clock measurements.

CNN 参数更多，主要因为池化后的展平特征进入 `16,384 -> 256` 全连接层，仅该层就有
`4,194,304` 个权重。Transformer 权重更少，但要对 78 个 patch 重复执行注意力与
前馈投影，解析前向 MAC 约为 CNN 的 `20.5x`。这些是架构计算值，不是墙钟实测值。

No current-lineage same-device timing record survived for a defensible
training- or raw-inference-speed ratio. The current 10k best epochs span
`243-268` for CNN and `189-227` for Transformer, but checkpoint epoch is not
elapsed time. For full prediction plus Powell refinement, repeated calls to
the physical simulator usually dominate the workflow. The project therefore
retains the RMSE conclusions while limiting speed claims to analytical
complexity.

当前谱系没有保留下来可用于公平比较的同设备计时记录，因此无法给出可信的训练或纯
网络推理实测倍数。当前 10k 最佳 epoch 范围为 CNN `243-268`、Transformer
`189-227`，但 checkpoint epoch 并不等于耗时。完整预测加 Powell 精修通常由反复
调用物理模拟器主导总耗时。因此项目保留 RMSE 结论，并将速度结论限定为解析复杂度。

### 4.5 Early Transformer exploration / 早期 Transformer 探索

The pre-cleanup local archive contained a recovered legacy
multi-head-attention baseline and a round of small-range experiments. The
variants used
orig-centered nearest-neighbor, original/generalization union, corridor, and
percentage-window subsets. They were attempts to improve experimental fitting
by changing the training distribution around selected regions.

清理前的本地归档中曾包含一条恢复出来的旧多头注意力基线，以及一轮小范围实验。
这些变体使用以原始曲线为中心的近邻子集、原始/泛化并集、走廊子集和百分比窗口
子集，目标是通过改变特定区域附近的训练分布改善实验拟合。

Those experiments predate fixed split membership and the final
dataset/checkpoint/scaler provenance contract. Their code and summaries explain
what was tried, but their numerical rankings are not part of the final claim
set.

这些实验早于固定划分成员关系和最终的数据集/checkpoint/scaler 来源规范。其代码和
摘要可以解释曾经尝试过什么，但数值排名不属于最终结论。

---

<a id="history-baseline"></a>
## 5. First Git Baseline and Initial Claims / 首个 Git 基线与早期结论

Commit `9f7bf3e` on 2026-07-06 is the first reproducible repository snapshot. It
already included:

2026-07-06 的提交 `9f7bf3e` 是第一个可复现的仓库快照，当时已经包括：

- the Python forward simulator and MATLAB references;
  Python 正向模拟器与 MATLAB 参考实现；
- LHS dataset generation and preprocessing;
  LHS 数据生成与预处理；
- CNN and Transformer training, prediction, refinement, and evaluation;
  CNN 与 Transformer 的训练、预测、精修和评估；
- synthetic recovery, warm-start, speed, multi-seed, and identifiability
  experiments;
  合成恢复、warm-start、速度、多种子与可辨识性实验；
- a bilingual README, report, Notebook, configuration, and initial tests.
  双语 README、报告、Notebook、配置和初始测试。

The baseline README made strong claims, including a CNN mean held-out curve
RMSE of `0.0188` versus Transformer `0.0465`, a large CNN advantage across
historical multi-seed runs, approximately two orders of magnitude warm-start
cost reduction, and capacity-based explanations for why CNN won.

基线 README 曾提出较强结论，包括 CNN 留出曲线平均 RMSE `0.0188`、Transformer
`0.0465`，旧多种子训练中的明显 CNN 优势、warm-start 约两个数量级的成本降低，
以及用容量消融解释 CNN 为何胜出。

These numbers were not fabricated, but they were produced before the final
lineage contract. The associated checkpoints did not consistently bind model
seed, split seed, dataset hash, scaler hash, and parameter order. Some
comparisons also used different effective training budgets or incomplete seed
sets. They were therefore later withdrawn from model selection and publication
claims.

这些数字并非虚构，但它们产生于最终谱系规范建立之前。相关 checkpoint 未能持续绑定
模型种子、划分种子、数据集哈希、scaler 哈希和参数顺序；部分对比还使用了不同的
有效训练预算或不完整种子集合。因此，这些结果后来退出模型选择和发表结论。

---

<a id="history-review"></a>
## 6. Reliability and Review Passes / 可靠性修复与审查

### 6.1 2026-07-06 hardening / 2026-07-06 加固

The commits immediately following the baseline addressed concrete correctness
and reproducibility defects:

基线后的连续提交修复了明确的正确性与可复现性问题：

- seeded Python, NumPy, and PyTorch training randomness, including weights,
  shuffling, and dropout (`64d630c`);
  固定 Python、NumPy 与 PyTorch 的训练随机性，包括权重、shuffle 与 dropout
  （`64d630c`）；
- opened the highest activity bin to positive infinity so valid
  high-activity samples were not silently dropped (`ef493df`);
  将最高活跃度分桶上界开放到正无穷，避免高活跃度有效样本被静默丢弃
  （`ef493df`）；
- handled data-generation worker failure and refill stalls (`f5d33a6`);
  处理数据生成 worker 失败和补样停滞（`f5d33a6`）；
- guarded held-out evaluation when no valid sample remained (`4f620da`);
  在留出评估没有有效样本时提供保护（`4f620da`）；
- corrected warm-start cost accounting so the initial guess was not counted
  twice (`be3c230`);
  修正 warm-start 成本计数，避免初值被重复计算（`be3c230`）；
- removed duplicated CNN prediction logic and batched test-time ensembles
  (`bdaa0bb`, `809c68a`);
  删除重复的 CNN 预测逻辑，并批量执行测试时集成（`bdaa0bb`、`809c68a`）；
- parallelized optional finite-difference Jacobian work and corrected the FIM
  condition-number calculation to use eigenvalue magnitude
  (`05c5d59`, `edd05ab`);
  为可选有限差分 Jacobian 增加并行，并将 FIM 条件数修正为使用特征值绝对值
  （`05c5d59`、`edd05ab`）。

### 6.2 2026-07-11 independent review / 2026-07-11 独立审查

The next review changed the project's evidentiary standard. It found and fixed:

下一轮审查提高了项目的证据标准，并发现、修复了：

- the risk that evaluation membership could depend on training randomness;
  评估集成员关系可能依赖训练随机性的风险；
- a configured `loss_weight_mode` that was not fully wired into behavior;
  配置存在但未完整接入行为的 `loss_weight_mode`；
- Transformer preprocessing drift relative to the shared intended contract;
  Transformer 预处理相对共享预期规范发生漂移；
- unsafe assumptions about physical-parameter ordering;
  对物理参数顺序的不安全假设；
- duplicated CNN/Transformer held-out evaluation implementations;
  CNN 与 Transformer 重复的留出评估实现；
- incomplete dependency metadata and the absence of automated CI.
  不完整的依赖元数据和自动化 CI 缺失。

The review added parameter-order validation, shared evaluation code,
physics-regression tests, `pyproject.toml`, GitHub Actions, and clearer
disclosure of unequal training budgets and Notebook-only illustrative logic.

审查增加了参数顺序验证、共享评估代码、物理回归测试、`pyproject.toml`、GitHub
Actions，并更清楚地披露训练预算不一致，以及 Notebook 中仅用于展示的逻辑。

This phase is why later documents separate "an implementation exists" from
"its historical checkpoint-dependent result is still citable."

从这一阶段开始，项目明确区分“功能已经实现”与“依赖旧 checkpoint 的历史结果仍可
引用”这两件事。

---

<a id="history-freeze"></a>
## 7. Freeze Before Repository Reorganization / 仓库重组前冻结

Commit `5f122d8` on 2026-07-17 created a deliberate preservation point before
structural work. The project recorded:

2026-07-17 的提交 `5f122d8` 在结构调整前建立了明确的保护点，并记录：

- source, configuration, data, result, and artifact inventories;
  源码、配置、数据、结果和制品清单；
- seven reviewed external/local artifact SHA-256 values;
  七个已审查外部/本地制品的 SHA-256；
- immutable CPU and Apple MPS legacy compatibility JSON;
  不可修改的 CPU 与 Apple MPS 旧版兼容性 JSON；
- dependency ranges, a reproducible lock, and exact tested Python 3.12
  constraints;
  依赖范围、可复现 lock，以及精确测试过的 Python 3.12 constraints；
- architecture and test checklists;
  架构与测试清单；
- migration rules forbidding mixed structural and numerical changes.
  禁止将结构变更与数值变更混在一起的迁移规则。

The governing invariant was that file movement must not change physics,
preprocessing, random-number order, split membership, metrics, output schemas,
or artifact bytes. Any unexplained regression was a stop condition.

核心约束是：文件移动不能改变物理计算、预处理、随机数顺序、划分成员、指标、输出
格式或制品字节。任何无法解释的回归都必须停止迁移。

The legacy compatibility check retained mean curve RMSE `0.025668` for CNN and
`0.021602` for Transformer over the reviewed 1,000-row test data. CPU/MPS
aggregate differences remained below `2e-7`. These values certify behavioral
preservation only, not model superiority.

旧版兼容性检查保留了 CNN `0.025668` 与 Transformer `0.021602` 的平均曲线 RMSE，
覆盖已审查的 1,000 行测试数据。CPU/MPS 汇总差异低于 `2e-7`。这些数值只证明行为
得到保留，不能用于证明模型优劣。

---

<a id="history-migration"></a>
## 8. Canonical Package Migration / 规范包迁移

### 8.1 2026-07-19 package move / 2026-07-19 包迁移

Commit `8d2c775` moved active implementations into explicit ownership layers:

提交 `8d2c775` 将活跃实现迁移到职责明确的分层中：

```text
dnawalker/
  config/
  core/
  data/
  experiments/
  inference/
  models/
  support/
  tools/
```

Experimental workbooks moved from the overloaded result area to
`data/experimental/`; audit records and formal documents moved under `docs/`;
shared prediction/refinement control flow moved into
`dnawalker/inference/workflows.py`.

实验工作簿从混杂的结果目录迁移到 `data/experimental/`；审计记录和正式文档迁移到
`docs/`；共享预测与精修控制流迁移到 `dnawalker/inference/workflows.py`。

Temporary root compatibility facades allowed migration checks to compare the
old and new import paths before the old paths were removed.

在删除旧路径之前，临时根目录兼容外壳使迁移检查能够对比新旧导入路径。

### 8.2 2026-07-21 canonical-only layout / 2026-07-21 唯一规范布局

Commit `3a5a494` completed "Plan C":

提交 `3a5a494` 完成了“方案 C”：

- removed root, `train_*`, `multiseed/`, and `utils/` Python facades;
  删除根目录、`train_*`、`multiseed/` 与 `utils/` 的 Python 兼容外壳；
- centralized INI files under `configs/`;
  将 INI 配置集中到 `configs/`；
- centralized project-root discovery in `dnawalker/paths.py`;
  将项目根目录发现集中到 `dnawalker/paths.py`；
- separated reusable support helpers from standalone command-line tools;
  将可复用支持工具与独立命令行工具分开；
- moved MATLAB into `archive/matlab/`;
  将 MATLAB 文件迁入 `archive/matlab/`；
- added canonical module and console-script entry points;
  增加规范模块入口与 console script；
- strengthened checkpoint, split, dataset, scaler, and parameter-order
  validation before retraining.
  在重训前加强 checkpoint、划分、数据集、scaler 和参数顺序验证。

This reorganization did **not** retrain a model or alter an existing artifact.
It changed ownership and import paths. The later numerical result changed
because new controlled datasets and checkpoints were created, not because
files moved between directories.

这次重组**没有**重训模型，也没有修改现有制品；它改变的是职责与导入路径。后续数值
结果之所以变化，是因为生成了新的受控数据集与 checkpoint，而不是因为文件在目录间
移动。

---

<a id="history-10k"></a>
## 9. Provenance Reset and Current 10k Study / 来源规范重建与当前 10k 实验

After the structural migration, the project stopped using legacy checkpoints
for model-selection claims. A hardened deterministic generator produced the
selected seed-42 dataset:

结构迁移后，项目停止使用旧 checkpoint 支撑模型选择结论。经过加固的确定性生成器
产生了选定的 seed-42 数据集：

```text
SHA-256:
557506e93079d1e5158aa30db7fd4a555234bb751e46049639b1fc827dca9b68
rows: 10,000
effective split: 8,000 train / 1,000 validation / 1,000 test
split_seed: 42
```

Repeated seed-42 generation produced the same dataset hash. A seed-0 run
produced a different hash, confirming both determinism within a seed and a real
dataset-version boundary across seeds.

重复执行 seed 42 得到相同数据集哈希；seed 0 得到不同哈希。这同时证明了同一种子下
的确定性，以及不同种子之间真实存在的数据集版本边界。

CNN and Transformer were each trained with model seeds 42-46 while keeping
`split_seed=42`. Ten checkpoints and their paired scalers passed the current
schema:

CNN 与 Transformer 分别使用模型种子 42-46 训练，同时固定 `split_seed=42`。
十个 checkpoint 及其配套 scaler 均通过当前格式要求：

- exact ordered physical-parameter names;
  精确、有序的物理参数名；
- model seed and split seed;
  模型种子与划分种子；
- dataset and scaler SHA-256;
  数据集与 scaler SHA-256；
- positive selected epoch and finite validation MSE;
  正的选中 epoch 与有限验证 MSE；
- finite model tensors.
  有限模型张量。

The held-out result committed in `5dd92a4` was:

提交 `5dd92a4` 固化的留出结果为：

| Architecture / 架构 | Mean curve RMSE +/- sample SD / 平均曲线 RMSE +/- 样本标准差 | Valid / invalid / extreme / 有效、无效、极端 |
|---|---:|---:|
| CNN | `0.02124865 +/- 0.00108720` | `5000 / 0 / 0` |
| Transformer | `0.02118662 +/- 0.00064572` | `4989 / 8 / 3` |

The paired Transformer-minus-CNN mean was `-0.00006203`, with exploratory
`p=0.915` and a 95% interval crossing zero. The correct conclusion was no
stable advantage, not "Transformer wins."

配对的 Transformer 减 CNN 均值为 `-0.00006203`；探索性 `p=0.915`，95% 区间
跨过零。正确结论是没有稳定优势，而不是“Transformer 胜出”。

### Why Reorganization Did Not Reverse the Result / 为什么目录重组没有推翻结果

The early CNN-favoring rows, the frozen legacy compatibility rows, and the
current 10k rows belong to different dataset/checkpoint lineages. The current
10k study changed the evidence in five controlled ways:

早期偏向 CNN 的结果、冻结的旧版兼容性结果与当前 10k 结果属于不同的数据集/checkpoint
谱系。当前 10k 实验通过五项受控改变更新了证据：

1. a newly generated, explicitly hashed dataset;
   新生成并明确哈希绑定的数据集；
2. fixed test membership across both architectures and all model seeds;
   两种架构与所有模型种子共享固定测试集成员；
3. complete seeds 42-46 for both architectures;
   两种架构都完整覆盖种子 42-46；
4. training-only target-scaler fitting and shared preprocessing rules;
   仅在训练集上拟合目标 scaler，并共享预处理规则；
5. checkpoints that bind every required provenance field.
   checkpoint 绑定全部必要来源字段。

The repository migration was checked against the old artifact hashes and
compatibility metrics. The controlled retraining, not the file movement,
produced the new comparison.

仓库迁移通过旧制品哈希和兼容性指标进行了检查。产生新对比结果的是受控重训，而不是
文件移动。

---

<a id="history-30k"></a>
## 10. Locked 30k Nested Learning Curve / 锁定的 30k 嵌套学习曲线

The near-tie in the 10k study raised a specific follow-up question: was the
earlier apparent CNN advantage driven by training volume, dataset lineage, or
seed variation? A larger test would only answer that question if the compared
training sizes used one master dataset, one fixed test set, and nested
membership.

10k 实验中的近似持平引出了明确的后续问题：早期看似明显的 CNN 优势，究竟来自训练
数据量、数据谱系，还是种子波动？只有在同一主数据集、同一固定测试集和嵌套成员关系
下比较不同训练规模，扩大实验才真正有解释力。

The locked design used:

锁定设计使用：

```text
master rows: 30,000
master SHA-256:
f19f0ae0c63a104ea23c96cc83ce5ce1e0d22fd22c5e1a498dfe6ff53418de06

split-manifest SHA-256:
a8bd1b7ae578bf48efc5ffc69e4372a432be132963be0ae142f3bffe5c88acd0

validation rows: 3,000 fixed
test rows:       3,000 fixed
train rows:      8,000 subset of 16,000 subset of 24,000
model seeds:     42, 43, 44, 45, 46
grid:            2 models x 3 sizes x 5 seeds = 30 runs
device:          Apple MPS
```

The generator still used LHS candidates and activity-bin quotas. The audit
confirmed exact five-bin activity balance and 30,000 unique retained parameter
rows, but also confirmed that retained parameter marginals were not uniformly
independent. The strongest retained association was
`corr(E_b, E_b_azo_cis)=0.43853`.

生成器继续使用 LHS 候选与活跃度分桶配额。审计确认五个活跃度分桶完全平衡，并且
30,000 行保留参数均唯一；同时也确认保留参数的边缘分布并非均匀独立。最强保留相关
为 `corr(E_b, E_b_azo_cis)=0.43853`。

All 30 training/evaluation runs completed. The predeclared contrast was
Transformer minus CNN, and practical equivalence required the complete 95%
interval to lie within `[-0.001, +0.001]`.

全部 30 次训练与评估完成。预先声明的对比量为 Transformer 减 CNN；只有完整 95%
区间落在 `[-0.001, +0.001]` 内，才判定实际等价。

| Train rows / 训练行数 | CNN mean +/- SD | Transformer mean +/- SD | Difference, 95% CI / 差值与 95% 区间 | Decision / 判定 |
|---:|---:|---:|---:|---|
| 8,000 | `0.01953252 +/- 0.00066847` | `0.02365253 +/- 0.00357211` | `+0.00412001 [-0.00080583, +0.00904584]` | Inconclusive / 不确定 |
| 16,000 | `0.01833638 +/- 0.00046472` | `0.02020585 +/- 0.00185304` | `+0.00186948 [-0.00074115, +0.00448010]` | Inconclusive / 不确定 |
| 24,000 | `0.01811960 +/- 0.00053594` | `0.01817831 +/- 0.00149517` | `+0.00005871 [-0.00225976, +0.00237718]` | Inconclusive / 不确定 |

The study established that both models improved with more retained training
data and that the observed gap contracted toward zero. At 24k, their means
were almost identical. Five seeds still did not make the interval narrow
enough to prove equivalence.

该实验确认两种模型都会随保留训练数据增加而改善，观察到的差距也逐渐收缩到零附近。
24k 时两者均值几乎相同，但五个种子仍不足以将区间缩窄到能够证明等价。

The 8k numeric rows survived, but temporary cleanup removed the five original
8k Transformer binaries and their checkpoint hashes. Those rows were recovered
from append-only session logs with a recorded source hash. They remain useful
for the discrepancy analysis, but the five checkpoints are not independently
reproducible. This limitation is retained in every final interpretation.

8k 数值记录得以保留，但临时清理删除了五个原始 8k Transformer 二进制文件及其
checkpoint 哈希。这些数值行从带来源哈希的追加式会话日志恢复，仍可用于差异分析，
但五个 checkpoint 无法独立复现。所有最终解释都保留这一限制。

Commit `94c2e3a` added the split implementation, training/evaluation
orchestration, a 427-line learning-curve test module, the locked protocol, and
the finalized result.

提交 `94c2e3a` 增加了划分实现、训练/评估编排、新增测试文件中的 427 行学习曲线
测试、锁定协议和最终结果。

---

<a id="history-fits"></a>
## 11. Experimental-Fit Robustness / 实验拟合稳定性

After held-out comparison, one 24k checkpoint per architecture was selected by
minimum validation MSE across model seeds 42-46, before inspecting
experimental-fit RMSE:

完成留出对比后，每种架构在查看实验拟合 RMSE 之前，仅按种子 42-46 中最低验证 MSE
选择一个 24k checkpoint：

| Architecture / 架构 | Selection / 选择 | Validation MSE / 验证 MSE |
|---|---|---:|
| CNN | 24k seed 43 | `0.0210643411` |
| Transformer | 24k seed 46 | `0.0210173298` |

Both checkpoints were evaluated against the original and generalization
experimental workbooks with identical settings: ensemble size 20, noise
standard deviation 0.005, Powell refinement, eight starts, 500 iterations, and
refinement RNG seeds 0-4. Ten MPS evaluations completed.

两个 checkpoint 使用完全相同的设置评估原始与泛化实验工作簿：集成大小 20、噪声
标准差 0.005、Powell 精修、八个起点、500 次迭代，以及精修随机种子 0-4。共完成
十次 MPS 评估。

| Architecture / 架构 | Original median (range) / 原始中位数（范围） | Generalization median (range) / 泛化中位数（范围） | Combined median (range) / 合并中位数（范围） |
|---|---:|---:|---:|
| CNN | `0.016290 (0.007779-0.020323)` | `0.016038 (0.015956-0.033571)` | `0.016303 (0.016082-0.020675)` |
| Transformer | `0.007806 (0.007796-0.008220)` | `0.019721 (0.016087-0.024852)` | `0.013767 (0.011946-0.016328)` |

These rows measure refinement-start sensitivity for one
validation-selected checkpoint per model. They are not a five-model-seed
architecture comparison. The remaining spread, particularly on the
generalization trace, showed that non-convex physical refinement was now a more
important unresolved source of variation than additional ordinary model
training.

这些结果衡量的是每种模型一个验证集选中 checkpoint 对精修起点的敏感性，而不是五个
模型种子的架构比较。剩余波动，特别是泛化曲线上的波动，说明非凸物理精修已经成为比
继续普通模型训练更重要的未解决变异来源。

Commit `70ef96a` added strict schema-v2 fit records, provenance validation,
cross-seed aggregation, English-only figures, tests, and synchronized
closeout documentation.

提交 `70ef96a` 增加了严格的 schema-v2 拟合记录、来源验证、跨种子汇总、全英文结果
图、测试和同步后的收尾文档。

---

<a id="history-identifiability"></a>
## 12. Identifiability and Interpretation Boundary / 可辨识性与解释边界

Identifiability analysis is checkpoint-independent because it differentiates
the forward simulator itself. Finite-difference Jacobians, Fisher information,
and profile scans showed a severely ill-conditioned inverse problem.

可辨识性分析直接针对正向模拟器求差分，因此不依赖 checkpoint。有限差分 Jacobian、
Fisher 信息和 profile 扫描显示该逆问题严重病态。

At the documented reference point:

在文档记录的参考点：

- FIM condition number is approximately `1.3e15`;
  FIM 条件数约为 `1.3e15`；
- eigenvalues span about 15 orders of magnitude;
  特征值横跨约 15 个数量级；
- the least-identifiable local direction is dominated by
  `E_b_azo_trans`;
  局部最不可辨识方向主要由 `E_b_azo_trans` 构成；
- `k0` has the sharpest conditional profile valley, while several other
  parameters have broad, flat valleys.
  `k0` 具有最陡的条件 profile 谷底，而多个其他参数呈现宽而平坦的谷底。

Consequently, a parameter vector that reproduces the curves is a
curve-consistent candidate, not proof of unique microscopic truth. Every
predicted parameter set must be interpreted with its regenerated curves.

因此，能够重现曲线的参数向量只能视为与曲线一致的候选解，不能证明其为唯一微观真实
参数。每个预测参数组合都必须结合其重新生成的曲线解释。

---

<a id="history-lifecycle"></a>
## 13. Retained, Withdrawn, or Archived Work / 保留、撤回与归档工作

### 13.1 Final/current evidence / 最终有效证据

- current-schema 10k five-seed fixed-split comparison;
  当前格式 10k 五种子固定划分对比；
- fixed 30k nested 8k/16k/24k learning curve;
  固定 30k 主数据集上的 8k/16k/24k 嵌套学习曲线；
- validation-selected experimental-fit robustness across refinement seeds;
  按验证集选择模型后的跨精修种子实验拟合稳定性；
- checkpoint-independent physics regression, identifiability, and signal
  evidence;
  不依赖 checkpoint 的物理回归、可辨识性与信号证据；
- immutable legacy CPU/MPS diagnostics, used only for compatibility.
  不可修改的旧版 CPU/MPS 诊断，仅用于兼容性。

### 13.2 Withdrawn from the Streamlined Source / 已从精简源码撤回

- synthetic parameter recovery;
  合成参数恢复；
- DL warm-start versus random/reference/LHS initialization;
  DL warm-start 与随机、参考值、LHS 初值比较；
- corrected speed and amortization benchmark;
  修正后的速度与摊销基准；
- CNN/Transformer capacity ablations.
  CNN/Transformer 容量消融。

Their historical numbers depend on checkpoints that do not satisfy the
selected final lineage, so the project did not rerun or cite them at closeout.
After documenting that decision, their implementations, dedicated tests, and
ablation configurations were removed from the streamlined release tree. They
remain recoverable at Git commit `2e280df`.

这些历史数值依赖不满足最终选定谱系要求的 checkpoint，因此项目在收尾时没有重跑或
引用这些数字。记录该决策后，相应实现、专用测试和消融配置已从精简发布树移除；
它们仍可从 Git 提交 `2e280df` 恢复。

### 13.3 Historical only / 仅供历史参考

- original MATLAB generation, optimization, and verification scripts;
  原始 MATLAB 生成、优化和验证脚本；
- recovered legacy Transformer MHA baseline;
  恢复的旧 Transformer MHA 基线；
- small-range targeted Transformer variants;
  小范围定向 Transformer 变体；
- incomplete-provenance default checkpoints and old experiment summaries.
  来源不完整的默认 checkpoint 与旧实验摘要。

The immutable compatibility diagnostics remain under `docs/audit/`. Historical
MATLAB and local experiment source is no longer present in the current tree
and remains recoverable from `2e280df`.

不可修改的兼容性诊断仍位于 `docs/audit/`。历史 MATLAB 与本地实验源码不再存在于
当前目录，可从 `2e280df` 恢复。

---

<a id="history-lineages"></a>
## 14. Dataset and Model Lineages / 数据集与模型谱系

| Lineage / 谱系 | Dataset identity / 数据集身份 | Models / 模型 | Valid use / 有效用途 |
|---|---|---|---|
| Legacy reviewed / 已审查旧版 | 10k SHA-256 `ada67b3f...e83602c` | Legacy CNN and Transformer with incomplete training provenance / 训练来源不完整的旧 CNN 与 Transformer | Compatibility diagnostics only / 仅兼容性诊断 |
| Hardened current 10k / 当前加固 10k | SHA-256 `557506e9...7dca9b68`, fixed `split_seed=42` | CNN and Transformer seeds 42-46, ten current-schema pairs / CNN 与 Transformer 种子 42-46，共十套当前格式制品 | Controlled five-seed held-out comparison / 受控五种子留出对比 |
| Nested 30k / 嵌套 30k | SHA-256 `f19f0ae0...418de06`; split manifest `a8bd1b7a...8acd0` | 30 runs over 8k/16k/24k, seeds 42-46 / 8k、16k、24k 与种子 42-46，共 30 次训练 | Training-volume study and selected 24k experimental fits / 数据量研究与选中 24k 实验拟合 |

A checkpoint filename alone never establishes lineage. The applicable dataset,
scaler, split manifest, parameter order, model seed, and selected epoch must be
validated together.

checkpoint 文件名本身不能证明谱系。必须共同验证适用的数据集、scaler、划分清单、
参数顺序、模型种子和选中 epoch。

---

<a id="history-engineering"></a>
## 15. Engineering and Repository Work / 工程与仓库工作

### 15.1 Reproducibility / 可复现性

The project added deterministic seed handling, fixed split membership,
training-only scaler fitting, strict JSON, finite-value checks, checkpoint
metadata, artifact SHA-256 validation, and count conservation for valid,
invalid, and extreme evaluation outcomes.

项目增加了确定性种子处理、固定划分成员、仅训练集 scaler 拟合、严格 JSON、有限值
检查、checkpoint 元数据、制品 SHA-256 验证，以及有效、无效和极端评估结果的数量
守恒。

### 15.2 Testing / 测试

The initial repository contained focused property and example tests for the
validation suite. Before source simplification, coverage peaked at 27 test
files and 473 passing local tests. Removing three dedicated test modules and
embedded checks for the withdrawn workflows produced the final streamlined
baseline of 24 test files and 387 passing tests. The retained suite covers
physics regression, configuration validation, generation, splits, inference,
artifact provenance, multi-seed orchestration, learning curves, fit
robustness, CLI entry points, and documentation evidence.

初始仓库包含针对验证套件的属性测试与示例测试。源码精简前，覆盖范围曾达到 27 个
测试文件和 473 项通过的本地测试。删除三个撤回工作流的专用测试模块及其嵌入检查后，
最终精简基线为 24 个测试文件和 387 项通过的测试。保留套件覆盖物理回归、配置验证、
数据生成、划分、推理、制品来源、多种子编排、学习曲线、拟合稳定性、CLI 入口与文档
证据。

The local engineering baseline also included:

本地工程基线还包括：

- full legacy held-out comparisons on CPU and Apple MPS;
  在 CPU 与 Apple MPS 上完成全部旧版留出对比；
- two deterministic forced-CPU smoke runs and one native-MPS smoke run;
  两次确定性的强制 CPU 冒烟运行和一次原生 MPS 冒烟运行；
- import and CLI checks from an unrelated working directory;
  从无关工作目录执行导入与 CLI 检查；
- static checks, `compileall`, dependency checks, shell syntax, workflow YAML,
  strict JSON/PNG, Notebook JSON, DOCX ZIP, and artifact hashes.
  静态检查、`compileall`、依赖检查、shell 语法、workflow YAML、严格 JSON/PNG、
  Notebook JSON、DOCX ZIP 与制品哈希。

Later model-first routing and output-ownership guards raised the suite to 398
tests. On 2026-07-30, one exact parameter-count regression test was added for
each architecture, producing a 400-test baseline. The formal one-command
application runner then added one help/portability check, producing the current
401-test baseline. The older 387 and 384 totals elsewhere in this history
remain dated milestones rather than claims about the current tree.

后续模型优先路由与输出归属保护将测试总数提高到 398。2026-07-30 又为两种架构各增加
一项精确参数量回归测试，形成 400 项测试基线；正式一键应用脚本随后增加一项帮助与
可移植性检查，形成当前 401 项测试基线。本文其他位置的 387 与 384 是对应当时版本的
历史节点，不代表当前目录。

### 15.3 Dependency records / 依赖记录

Three dependency files have distinct roles:

三个依赖文件承担不同职责：

- `requirements.txt`: human-maintained compatible ranges;
  `requirements.txt`：人工维护的兼容版本范围；
- `requirements-lock.txt`: reproducible Python 3.12 CPU resolution;
  `requirements-lock.txt`：可复现的 Python 3.12 CPU 依赖解析；
- `constraints-tested-py312.txt`: exact versions from the reviewed local
  environment.
  `constraints-tested-py312.txt`：已审查本地环境中的精确版本。

They are not accidental duplicates.

它们不是意外重复文件。

### 15.4 Artifact and result ownership / 制品与结果归属

The final ownership rule is:

最终归属规则为：

- `artifacts/`: datasets, split manifests, checkpoints, and scalers;
  `artifacts/`：数据集、划分清单、checkpoint 与 scaler；
- `results/`: generated metrics, plots, reports, predictions, and logs;
  `results/`：生成的指标、图片、报告、预测与日志；
- `docs/evidence/`: small, path-sanitized final evidence intended for
  publication;
  `docs/evidence/`：适合发布、经过路径清理的小型最终证据；
- model-specific result directories: model-owned outputs;
  模型专属结果目录：各模型拥有的输出；
- comparison directories: cross-model summaries only.
  对比目录：仅存放跨模型汇总。

Regenerable caches, duplicate result trees, interrupted runs, and `.DS_Store`
files were removed during closeout. Formal datasets, checkpoints, scalers,
split manifests, and experimental inputs with remaining provenance value were
retained for a separately licensed artifact release. Withdrawn source remains
available through Git rather than occupying the current release tree.

收尾期间删除了可再生缓存、重复结果树、中断运行和 `.DS_Store`。具有独立哈希或仍有
来源价值的正式数据集、checkpoint、scaler、划分清单与实验输入被保留，供单独授权的
制品发行使用。已撤回源码通过 Git 保留，不再占用当前发布树。

The later public application bundle made one narrow exception: byte-identical
copies of the validation-selected CNN seed-43 pair, Transformer seed-46 pair,
and their required split manifest are versioned under
`artifacts/application/` with `SHA256SUMS`. The 30k training dataset, other
model seeds, raw results, and experimental workbooks remain excluded.

后续公开应用制品包增加了一个严格限定的例外：按验证集选出的 CNN seed-43 配对、
Transformer seed-46 配对及其所需划分清单，以字节完全相同的副本和 `SHA256SUMS`
纳入 `artifacts/application/`。30k 训练数据集、其他模型种子、原始结果与实验
工作簿仍不发布。

---

<a id="history-publication"></a>
## 16. Publication Cleanup and Documentation Recovery / 发布清理与文档恢复

Commit `8cffb65` prepared a sanitized public-source snapshot:

提交 `8cffb65` 准备了经过清理的公开源码快照：

- local Kiro state, the editable private report, and experimental workbooks
  were removed from the tracked public tree;
  从公开跟踪树中移除本地 Kiro 状态、可编辑私人报告和实验工作簿；
- a publication-boundary document was added;
  增加发布边界文档；
- path-sanitized JSON and English figures were curated under
  `docs/evidence/`;
  在 `docs/evidence/` 中整理经过路径清理的 JSON 与英文图片；
- evidence checksums and a rebuild script were added;
  增加证据校验和与重建脚本；
- the existing private Git history was explicitly declared unsafe for direct
  publication because old commits contain local/private materials.
  明确说明现有私人 Git 历史包含本地/私人材料，不能直接公开。

The closeout rewrite also compressed the root README too aggressively. Commit
`b714a93` restored a comprehensive 950-line bilingual README with the final
results, complete workflow, dependency explanation, publication boundary, and
documentation links.

收尾改写曾过度压缩根 README。提交 `b714a93` 恢复了约 950 行的完整双语 README，
重新纳入最终结果、完整流程、依赖说明、发布边界与文档链接。

Other formal documents were audited for accidental shrinkage. Architecture,
artifact, learning-curve, Notebook, and report content remained stable or
expanded. The Kiro design/task files were intentionally condensed into final
process records, while detailed old versions remain recoverable from Git
history.

其他正式文档也接受了意外缩水审计。架构、制品、学习曲线、Notebook 与报告内容保持
稳定或有所增加。Kiro 设计/任务文件被有意压缩为最终过程记录，详细旧版本仍可从 Git
历史恢复。

### 16.1 Model-first clarity pass / 模型优先的结构清晰化

The final clarity pass reorganized the canonical package without changing the
scientific pipeline. CNN and Transformer became first-level packages, each
containing its configuration, data adapter, model, training, inference,
prediction, and evaluation responsibilities. Physics moved to
`dnawalker.physics`, shared cross-model behavior to `dnawalker.shared`, and
controlled experiments to `dnawalker.studies`. Prediction/refinement and the
two evaluation entry files were each consolidated behind explicit options or
subcommands.

最终清晰化整理在不改变科学流程的前提下重组了规范包。CNN 与 Transformer 成为一级
包，各自集中配置、数据适配、模型、训练、推理、预测和评估职责。物理实现迁入
`dnawalker.physics`，跨模型共享行为迁入 `dnawalker.shared`，受控研究迁入
`dnawalker.studies`。普通/精修预测及两类评估入口分别通过显式选项或子命令合并。

Configuration was split into `common.ini`, `cnn.ini`, `transformer.ini`, and
small profile/study overlays. A saved pre-move audit compared all 76 typed
zero-argument getters and found zero differences. The repository now publishes
one `dnawalker` command tree instead of 19 unrelated console-script names.
Large artifacts, workbooks, evidence JSON/PNG, model equations, preprocessing,
splits, and seeds were not changed.

配置被拆分为 `common.ini`、`cnn.ini`、`transformer.ini` 及小型 profile/研究覆盖层。
搬迁前保存的审计基线逐项比较了 76 个无参数类型化 getter，差异为零。仓库现在发布
一个 `dnawalker` 命令树，而不是 19 个分散的控制台命令。大型制品、实验工作簿、
证据 JSON/PNG、模型方程、预处理、数据划分与随机种子均未改变。

The bilingual README grew from 952 to more than 1,200 lines with a full system
architecture, dependency diagram, runtime/configuration flow, command matrix,
and commented repository tree. `ARCHITECTURE.md`, `FILE_INVENTORY.md`, the
documentation index, Notebook imports, and test records were synchronized to
the same final layout.

双语 README 从 952 行扩展到 1,200 行以上，补回完整系统架构、依赖图、运行与配置流、
命令矩阵和带注释目录树。`ARCHITECTURE.md`、`FILE_INVENTORY.md`、文档索引、
Notebook 导入和测试记录也同步到同一最终结构。

The resulting model-first suite collected and passed 384 tests with the one
known slow-test warning. The three-case reduction from the earlier 387-test
snapshot comes from replacing 19 separate console-script target checks with
one unified entry-point check plus explicit help checks for all 15 command
leaves; no scientific coverage category was removed.

最终模型优先测试套件收集并通过 384 项测试，仅保留一个已知慢测试 warning。相较此前
387 项快照减少 3 项，是因为 19 个分散控制台入口检查被替换为 1 个统一入口检查和
15 个命令叶节点帮助检查；没有删除任何科学覆盖类别。

### 16.2 Application-first documentation / 应用优先的文档定位

After the structural closeout, the root README was reframed around the actual
deliverable: predicting seven physical parameters from three experimental
fluorescence curves, followed by optional physics refinement and forward
verification. The detailed 10k/30k architecture comparisons, refinement
robustness, and identifiability boundaries moved into
`docs/MODEL_COMPARISON.md`; no evidence or result was discarded.

结构收尾后，根 README 重新围绕实际交付目标组织：从三条实验荧光曲线预测七个物理
参数，并可继续执行物理精修和正向验证。详细的 10k/30k 架构对比、精修稳定性和
可辨识性边界集中到 `docs/MODEL_COMPARISON.md`；没有删除任何证据或结果。

The current single-branch recommendation is the validation-selected 24k
Transformer plus physics refinement because it has the lowest observed
combined median across the two workbooks. CNN remains a complementary branch
because it has the lower generalization-workbook median. The documented future
application direction is therefore to refine both model candidates and select
by forward curve RMSE, while reporting the alternative and uncertainty.

当前单分支推荐为按验证集选出的 24k Transformer 加物理精修，因为它在两份工作簿上
具有最低的合并中位结果。CNN 在泛化工作簿上的中位结果更低，因此仍是互补分支。
文档记录的未来应用方向是：同时精修两种模型候选，并按正向曲线 RMSE 择优，同时报告
备选解与不确定性。

On 2026-07-30, both prediction CLIs gained an explicit `--exp` workbook
override and `scripts/run_application.sh` became the formal one-command entry
for a user workbook. It defaults to the selected Transformer, can run the CNN
alternative or both branches, preserves model-owned output directories, and is
explicitly separated from the tiny-model smoke workflow.

2026-07-30，两种预测 CLI 增加显式 `--exp` 工作簿覆盖参数，并新增
`scripts/run_application.sh` 作为自有工作簿的正式一键入口。脚本默认运行选定的
Transformer，也可运行 CNN 或两支，保持模型专属输出目录，并与微型模型冒烟流程明确
区分。

The same closeout then added the approximately 29 MiB selected inference bundle
to the public tree. Its configs resolve checkpoint, scaler, and split-manifest
paths without the 30k dataset, enabling application inference on a
user-supplied workbook while keeping training and held-out evaluation outside
the binary scope.

同一轮收尾随后把约 29 MiB 的选定推理制品包加入公开目录。配置无需 30k 数据集即可
解析 checkpoint、scaler 与划分清单，从而支持对用户工作簿执行应用推理；训练和留出
评估仍不属于该二进制范围。

---

<a id="history-decisions"></a>
## 17. Decision Log / 关键决策记录

| Decision / 决策 | Reason / 原因 | Consequence / 结果 |
|---|---|---|
| Make Python the canonical runtime / 以 Python 为规范运行时 | Remove MATLAB as an execution prerequisite while preserving the mechanistic model. / 在保留机理模型的同时消除 MATLAB 运行依赖。 | MATLAB is historical reference only. / MATLAB 仅作历史参考。 |
| Use LHS for candidates plus physical/activity filtering / 候选使用 LHS，再进行物理与活跃度筛选 | Cover configured ranges while retaining usable dynamic signals. / 覆盖配置范围并保留有效动态信号。 | Retained marginals must be audited; they are not assumed uniform. / 必须审计保留边缘分布，不能假设均匀。 |
| Separate `random_seed` from `split_seed` / 分离 `random_seed` 与 `split_seed` | Model stochasticity must not change test membership. / 模型随机性不能改变测试集成员。 | Architectures and seeds share fixed held-out rows. / 架构与种子共享固定留出样本。 |
| Fit scalers on training data only / scaler 只在训练集拟合 | Prevent evaluation leakage. / 防止评估泄漏。 | Every scaler is bound to its checkpoint and dataset. / 每个 scaler 与 checkpoint、数据集绑定。 |
| Treat datasets and checkpoints as versioned lineages / 将数据集与 checkpoint 视为版本谱系 | Corrected generation changes dataset bytes and invalidates old pairings. / 修正生成逻辑会改变数据字节，使旧配对失效。 | Old and new artifacts are never mixed. / 新旧制品不混用。 |
| Withdraw incomplete-provenance rankings / 撤回来源不完整的排名 | A numeric result without complete lineage cannot support model selection. / 缺少完整谱系的数值不能支撑模型选择。 | Legacy metrics remain compatibility diagnostics only. / 旧指标仅作兼容性诊断。 |
| Run a nested 30k learning curve / 运行嵌套 30k 学习曲线 | Isolate training volume within one dataset and split lineage. / 在同一数据与划分谱系内隔离训练量变量。 | Both models improve; the 24k point estimate is a near-tie. / 两种模型均改善，24k 点估计近似持平。 |
| Select fit checkpoints by validation MSE before viewing experimental fits / 查看实验拟合前按验证 MSE 选 checkpoint | Avoid choosing a checkpoint because it happens to fit the two experimental traces. / 避免因偶然拟合两条实验曲线而选模型。 | Fit robustness measures refinement sensitivity, not model-seed superiority. / 拟合稳定性衡量精修敏感性，而非模型种子优劣。 |
| Stop additional training for the scoped question / 针对既定问题停止追加训练 | Thirty nested runs and current fits answer the planned comparison; remaining variation is not mainly ordinary training shortage. / 30 次嵌套训练与当前拟合已回答计划问题，剩余变化并非主要来自普通训练不足。 | Local scientific closeout is complete. / 本地科学收尾完成。 |
| Separate local completion from cross-platform certification / 区分本地完成与跨平台认证 | External inputs and device-specific kernels need licensing and fresh validation. / 外部输入与设备相关内核需要许可和全新验证。 | The selected inference bundle is published; remaining P2 work blocks only a formal cross-platform claim. / 已发布选定推理包；剩余 P2 工作只阻止正式跨平台声明。 |
| Use model-first packages and one CLI / 使用模型优先包与单一 CLI | Readers need to find each model workflow without traversing generic namespace layers. / 读者应能直接找到每种模型的完整流程。 | `cnn/` and `transformer/` are self-contained; shared physics/data contracts remain centralized. / 两个模型目录自包含，共享物理与数据契约仍集中管理。 |
| Present the project as an inverse-prediction application / 将项目定位为逆向预测应用 | Model comparisons validate method choice but are not the primary deliverable. / 模型对比用于验证方法选择，但不是主要交付目标。 | README leads with curve-to-parameter prediction; detailed comparison lives in `MODEL_COMPARISON.md`; future deployment uses dual candidates and forward-RMSE selection. / README 以曲线到参数预测为主，详细对比进入独立文档，未来应用采用双候选与正向 RMSE 择优。 |
| Separate architecture complexity from measured speed / 区分架构复杂度与实测速度 | Parameter count, MAC, epoch count, and wall-clock latency answer different questions. / 参数量、MAC、epoch 数和墙钟延迟回答不同问题。 | Exact parameter counts and MAC estimates are published; unsupported current-artifact speedup claims are not. / 发布精确参数量和 MAC 估算，不发布缺少当前制品证据的加速倍数。 |

---

<a id="history-commits"></a>
## 18. Commit Ledger / 提交简表

The ledger below records substantive milestones through the model-first
clarity pass at `c0ee701`. Documentation-only commits that synchronize this
ledger are intentionally not listed within themselves; Git log is the
authoritative complete commit list.

下表记录截至模型优先清晰化提交 `c0ee701` 的实质里程碑。为避免自引用，仅同步本简表
的文档提交不在自身内容中列出；完整提交列表以 Git log 为准。

| Date / 日期 | Commit | Recorded change / 记录的变更 |
|---|---|---|
| 2026-07-06 | `9f7bf3e` | Imported the pre-optimization baseline. / 导入优化前基线。 |
| 2026-07-06 | `64d630c` | Seeded training randomness. / 固定训练随机性。 |
| 2026-07-06 | `ef493df` | Fixed the open upper activity bin. / 修复最高活跃度开放分桶。 |
| 2026-07-06 | `f5d33a6` | Hardened generator worker and refill failure handling. / 加固生成器 worker 与补样失败处理。 |
| 2026-07-06 | `4f620da` | Guarded evaluation with zero valid rows. / 保护零有效样本评估。 |
| 2026-07-06 | `be3c230` | Corrected initialization cost double-counting. / 修正初值成本重复计数。 |
| 2026-07-06 | `bdaa0bb` | Reused shared prediction and parameter I/O. / 复用共享预测与参数 I/O。 |
| 2026-07-06 | `809c68a` | Batched test-time ensembles. / 批量化测试时集成。 |
| 2026-07-06 | `05c5d59` | Added optional parallel Jacobian evaluation. / 增加可选并行 Jacobian。 |
| 2026-07-06 | `edd05ab` | Corrected FIM condition-number handling. / 修正 FIM 条件数处理。 |
| 2026-07-06 | `3155397` | Removed a dead Transformer evaluator variable. / 删除 Transformer 评估器无效变量。 |
| 2026-07-06 | `5a3590b` | Updated warm-start and speed documentation after cost correction. / 成本修正后更新 warm-start 与速度文档。 |
| 2026-07-11 | `e75f484` | Fixed review findings in splits, configuration, preprocessing, and parameter safety. / 修复划分、配置、预处理与参数安全审查问题。 |
| 2026-07-11 | `c115f59` | Tightened claims and loss-mode comparison. / 收紧结论并加固 loss 模式对比。 |
| 2026-07-11 | `7dfe905` | Shared held-out evaluation, CI, packaging, and dependencies. / 共享留出评估、CI、打包与依赖。 |
| 2026-07-11 | `f9cfe51` | Recorded the review-fix merge. / 记录审查修复合并。 |
| 2026-07-11 | `5279f7f` | Disclosed budget parity and Notebook illustrative boundaries. / 披露预算一致性与 Notebook 展示边界。 |
| 2026-07-17 | `5f122d8` | Froze the verified pre-reorganization state. / 冻结已验证的重组前状态。 |
| 2026-07-19 | `8d2c775` | Migrated implementations into the canonical package. / 将实现迁入规范包。 |
| 2026-07-21 | `3a5a494` | Removed compatibility facades and finalized retraining safeguards. / 删除兼容外壳并完成重训保障。 |
| 2026-07-23 | `5dd92a4` | Fixed multi-seed provenance and recorded current 10k evidence. / 修复多种子来源并记录当前 10k 证据。 |
| 2026-07-29 | `94c2e3a` | Completed the locked nested learning curve. / 完成锁定的嵌套学习曲线。 |
| 2026-07-29 | `70ef96a` | Completed fit robustness and local scientific closeout. / 完成拟合稳定性与本地科学收尾。 |
| 2026-07-29 | `8cffb65` | Prepared the sanitized public-source snapshot. / 准备清理后的公开源码快照。 |
| 2026-07-29 | `b714a93` | Restored the comprehensive bilingual README. / 恢复完整双语 README。 |
| 2026-07-29 | `2e280df` | Documented the complete project history before final source simplification. / 在最终源码精简前记录完整项目历程。 |
| 2026-07-29 | `c6a5b80` | Removed nonpublication files and withdrawn source from the streamlined release tree. / 从精简发布树删除不发布文件与已撤回源码。 |
| 2026-07-29 | `c0ee701` | Reorganized the model-first package and restored comprehensive architecture documentation. / 完成模型优先包重组并恢复完整架构文档。 |

The merge commit `f9cfe51` intentionally has no separate file diff; it records
the review branch integration.

合并提交 `f9cfe51` 本身没有独立文件差异；它记录了审查分支的整合。

---

<a id="history-doc-map"></a>
## 19. Current Documentation Map / 当前文档地图

| Question / 问题 | Authoritative document / 权威文档 |
|---|---|
| What is the project and how do I run it? / 项目是什么、如何运行？ | [`README.md`](../README.md) |
| How did the project get here? / 项目如何发展到现在？ | This document / 本文 |
| What owns each runtime responsibility? / 各运行时职责归谁？ | [`ARCHITECTURE.md`](ARCHITECTURE.md) |
| Which inverse method should the application use? / 应用应选择哪种逆向方法？ | [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) |
| How large and computationally expensive are the two models? / 两种模型的规模与计算成本如何？ | [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) |
| What does every stable path contain? / 每个稳定路径包含什么？ | [`FILE_INVENTORY.md`](FILE_INVENTORY.md) |
| Which dataset/checkpoint/scaler identities are trusted? / 哪些数据集、checkpoint、scaler 身份可信？ | [`ARTIFACTS.md`](ARTIFACTS.md) |
| How was the 30k comparison locked and interpreted? / 30k 对比如何锁定与解释？ | [`LEARNING_CURVE_PROTOCOL.md`](LEARNING_CURVE_PROTOCOL.md) |
| Which engineering and scientific gates passed? / 哪些工程与科学检查已通过？ | [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md) |
| What may be published? / 哪些内容可以发布？ | [`PUBLICATION.md`](PUBLICATION.md) |
| Where are the final machine-readable summaries and figures? / 最终机器可读摘要与图片在哪里？ | [`evidence/README.md`](evidence/README.md) |
| What do the old CPU/MPS values mean? / 旧 CPU/MPS 数值代表什么？ | [`audit/README.md`](audit/README.md) |

---

<a id="history-status"></a>
## 20. Final Status and Remaining Work / 最终状态与剩余工作

### Complete / 已完成

- scoped local scientific comparison;
  既定范围内的本地科学对比；
- current 10k five-seed lineage;
  当前 10k 五种子谱系；
- 30/30 nested learning-curve runs;
  30/30 次嵌套学习曲线训练；
- validation-selected experimental-fit robustness;
  按验证集选模后的实验拟合稳定性；
- local engineering, documentation, artifact, and evidence gates;
  本地工程、文档、制品和证据检查；
- exact model-size regression guards and documented compute estimates;
  精确模型规模回归保护与已记录的计算量估算；
- one-command prediction and verification for a user-supplied workbook;
  面向自有工作簿的一键预测与验证；
- hash-verified selected CNN/Transformer inference artifacts without the
  training dataset;
  不含训练数据集、经过哈希校验的选定 CNN/Transformer 推理制品；
- sanitized source snapshot and publication instructions.
  清理后的源码快照与发布说明。

### Not required for the local conclusion / 不影响本地结论

- fresh Linux CPU and CUDA validation;
  全新 Linux CPU 与 CUDA 验证；
- remote GitHub Actions execution;
  远程 GitHub Actions 执行；
- permanent external URLs for large datasets and non-selected artifacts;
  大型数据集与未选制品的永久外部地址；
- source and redistribution licenses for external experimental workbooks;
  外部实验工作簿的来源及再分发许可；
- any future revival of recovery, warm-start, speed, or capacity studies,
  which would require restoring or redesigning withdrawn source.
  未来若恢复参数回收、warm-start、速度或容量研究，需要恢复或重新设计已撤回源码。

These are portability, publication, or scope-extension tasks. They do not
change the final local answer: both models benefit from more retained training
data, neither has a proven stable advantage, and further ordinary training is
not justified for the question this project set out to answer.

这些属于可移植性、发布或范围扩展任务，不会改变最终本地答案：两种模型都受益于更多
保留训练数据；没有一种模型被证明具有稳定优势；针对项目最初设定的问题，没有充分
理由继续普通训练。
