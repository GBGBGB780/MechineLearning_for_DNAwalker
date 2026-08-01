# coding=utf-8
"""dnawalker.shared.evaluation — 留出测试集评估的共享核心 (CNN / Transformer 复用)。

两个模型的 ``evaluate`` 共享约 90% 的逻辑 (拆分复现、防泄漏清洗、逐样本正向
模拟 RMSE、参数 MSE、摘要/出图)，仅在 predictor 类、config 访问器、模型标签上不同。
把核心集中为单一事实来源，任何 bug 修复 (如防泄漏的 ``_clean_like_training``) 只改一处，
两个 ``evaluate`` 模块仅保留极薄的入口 (predictor 工厂 + 标签)，行为逐位不变。

Shared single source of truth for the held-out test-set evaluation used by both
model subsystems; the two model ``evaluate`` modules are thin wrappers.
"""

import math
import os
from numbers import Integral, Real

import numpy as np
from sklearn.model_selection import train_test_split

from dnawalker.physics import simulator as pysim
from dnawalker.shared.parameters import load_npz_dataset, vector_to_param_dict
from dnawalker.data.preprocessing import prepare_labels_and_sample_mask
from dnawalker.data.splits import configured_explicit_split
from dnawalker.shared.artifacts import require_matching_sha256, sha256_file
from dnawalker.studies.protocol import require_int, require_seed, write_json

_ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def training_sample_mask(X, Y, config):
    """Return the exact pre-split cleanup mask used by both trainers."""
    _, _, mask = prepare_labels_and_sample_mask(
        X,
        Y,
        config.get_trainable_param_names(),
        log_transform_params=config.get_log_transform_params(),
        log_epsilon=config.get_log_epsilon(),
        amplitude_thresholds=(
            config.get_amplitude_thresholds()
            if config.get_amplitude_filter_enabled()
            else None
        ),
        safe_threshold=config.get_safe_threshold(),
        nan_replacement=config.get_nan_replacement_value(),
    )
    return mask


def clean_like_training(X, Y, config):
    """复现训练侧 (data_loader / dataset) 拆分前的**样本保留掩码**。

    ``train_test_split`` 按索引划分，只有当被划分数组与训练时**行数、行序一致**才能
    复现训练的 test 集。训练侧在拆分**之前**做振幅过滤 + NaN/Inf/极端值清洗；若这里
    跳过，一旦训练丢过样本，重建的 "test 集" 会与训练集重叠 (数据泄漏)。故拆分前施加
    同一套过滤。

    保留掩码由 :func:`preprocessing.prepare_labels_and_sample_mask` 统一生成；
    标签会像训练侧一样先转 ``float32``、做 log 变换，再参与有限值/阈值判断。
    因而这不是近似复现，而是 CNN、Transformer 与评估共用的单一实现。

    Args:
        X: 原始曲线 ``(N, 3, T)``。
        Y: 物理参数 ``(N, 7)``。
        config: 暴露 amplitude/safe_threshold/nan_replacement 访问器的 config 对象
            (CNN 为 ``Config``，Transformer 传 ``parent_config``)。

    Returns:
        (X_clean, Y_clean) —— 行序与训练侧拆分前一致。
    """
    mask = training_sample_mask(X, Y, config)
    if int(mask.sum()) != X.shape[0]:
        print(f"[eval_testset] 清洗移除 {X.shape[0] - int(mask.sum())} 个样本 (与训练一致)，"
              f"再复现 test 划分。")
    return X[mask], Y[mask]


def get_test_split(config, test_seed=None):
    """复现训练侧的 test 拆分，返回 ``(X_test_raw(N,3,T), Y_test_raw(N,7))``。

    先施加与训练**相同**的过滤 (见 :func:`clean_like_training`) 再 ``train_test_split``，
    保证行序对齐、防止泄漏。``test_seed=None`` 时回退固定的
    ``config.get_split_seed()``；显式 ``test_seed`` 仅用于复现实验指定的固定划分。
    """
    npz = config.get_dataset_file()
    X, Y, _ = load_npz_dataset(
        npz,
        config.get_trainable_param_names(),
        allow_legacy_canonical=True,
        x_dtype=np.float32,
        y_dtype=np.float64,
        expected_x_shape=(config.get_num_curves(), config.get_seq_length()),
    )

    if test_seed is None:
        test_seed = config.get_split_seed()
    mask = training_sample_mask(X, Y, config)
    explicit_split = configured_explicit_split(
        config,
        npz,
        n_samples=X.shape[0],
        valid_mask=mask,
    )
    if explicit_split is not None:
        if int(test_seed) != explicit_split.split_seed:
            raise ValueError(
                "explicit split seed does not match requested test seed: "
                f"{explicit_split.split_seed} vs {test_seed}"
            )
        return X[explicit_split.test], Y[explicit_split.test]

    if int(mask.sum()) != X.shape[0]:
        print(
            f"[eval_testset] 清洗移除 {X.shape[0] - int(mask.sum())} 个样本 "
            "(与训练一致)，再复现 test 划分。"
        )
    X, Y = X[mask], Y[mask]
    test_ratio = config.get_test_split_ratio()
    _, X_test, _, Y_test = train_test_split(X, Y, test_size=test_ratio, random_state=test_seed)
    return X_test, Y_test


def evaluation_provenance(predictor, config):
    """Bind an evaluation result to its exact dataset/model/scaler files."""
    provenance = {
        "param_names": list(predictor.get_param_names()),
    }
    if hasattr(config, "get_split_seed"):
        provenance["split_seed"] = int(config.get_split_seed())
    if hasattr(predictor, "checkpoint_model_seed"):
        provenance["checkpoint_model_seed"] = (
            predictor.checkpoint_model_seed
        )
    if hasattr(predictor, "checkpoint_split_seed"):
        provenance["checkpoint_split_seed"] = (
            predictor.checkpoint_split_seed
        )
    if hasattr(predictor, "checkpoint_dataset_sha256"):
        provenance["checkpoint_dataset_sha256"] = (
            predictor.checkpoint_dataset_sha256
        )
    if hasattr(predictor, "checkpoint_y_scaler_sha256"):
        provenance["checkpoint_y_scaler_sha256"] = (
            predictor.checkpoint_y_scaler_sha256
        )
    if hasattr(predictor, "checkpoint_split_manifest_sha256"):
        provenance["checkpoint_split_manifest_sha256"] = (
            predictor.checkpoint_split_manifest_sha256
        )
    if hasattr(predictor, "checkpoint_train_subset_size"):
        provenance["checkpoint_train_subset_size"] = (
            predictor.checkpoint_train_subset_size
        )
    if hasattr(predictor, "checkpoint_epoch"):
        provenance["checkpoint_epoch"] = predictor.checkpoint_epoch
    if hasattr(predictor, "checkpoint_val_mse"):
        provenance["checkpoint_val_mse"] = (
            predictor.checkpoint_val_mse
        )
    if hasattr(predictor, "checkpoint_param_names_present"):
        provenance["checkpoint_param_names_present"] = bool(
            predictor.checkpoint_param_names_present
        )
    if hasattr(predictor, "device"):
        provenance["device"] = str(predictor.device)
    if hasattr(predictor, "inference_batch_size"):
        provenance["inference_batch_size"] = int(
            predictor.inference_batch_size
        )

    paths = {}
    if hasattr(config, "get_dataset_file"):
        paths["dataset"] = config.get_dataset_file()
    if hasattr(predictor, "model_path"):
        paths["checkpoint"] = predictor.model_path
    if hasattr(predictor, "y_scaler_path"):
        paths["y_scaler"] = predictor.y_scaler_path
    for name, path in paths.items():
        absolute = os.path.abspath(path)
        try:
            inside_repo = os.path.commonpath(
                [_ROOT_DIR, absolute]
            ) == _ROOT_DIR
        except ValueError:
            inside_repo = False
        provenance[f"{name}_path"] = (
            os.path.relpath(absolute, _ROOT_DIR)
            if inside_repo
            else absolute
        )
        expected = getattr(
            predictor, f"checkpoint_{name}_sha256", None
        )
        if name in {"dataset", "y_scaler"}:
            provenance[f"{name}_sha256"] = require_matching_sha256(
                absolute, expected, name
            )
        else:
            provenance[f"{name}_sha256"] = sha256_file(absolute)

    if hasattr(config, "get_split_manifest_file"):
        manifest_path = config.get_split_manifest_file()
        if manifest_path is not None:
            absolute = os.path.abspath(manifest_path)
            try:
                inside_repo = os.path.commonpath(
                    [_ROOT_DIR, absolute]
                ) == _ROOT_DIR
            except ValueError:
                inside_repo = False
            provenance["split_manifest_path"] = (
                os.path.relpath(absolute, _ROOT_DIR)
                if inside_repo
                else absolute
            )
            provenance["split_manifest_sha256"] = require_matching_sha256(
                absolute,
                getattr(
                    predictor,
                    "checkpoint_split_manifest_sha256",
                    None,
                ),
                "split_manifest",
            )
            provenance["train_subset_size"] = int(
                config.get_train_subset_size()
            )
    return provenance


def require_current_artifact_provenance(
        predictor, config, *, expected_model_seed=None,
        expected_split_seed=None):
    """Validate a publication checkpoint and bind it to actual artifacts.

    Legacy or partial checkpoints are rejected before model computation. The
    returned mapping includes actual dataset/checkpoint/scaler SHA-256 values,
    device, parameter order, and checkpoint training metadata.
    """
    values = {
        "model_seed": getattr(predictor, "checkpoint_model_seed", None),
        "split_seed": getattr(predictor, "checkpoint_split_seed", None),
        "dataset_sha256": getattr(
            predictor, "checkpoint_dataset_sha256", None
        ),
        "y_scaler_sha256": getattr(
            predictor, "checkpoint_y_scaler_sha256", None
        ),
        "epoch": getattr(predictor, "checkpoint_epoch", None),
        "val_mse": getattr(predictor, "checkpoint_val_mse", None),
    }
    missing = [name for name, value in values.items() if value is None]
    if not getattr(predictor, "checkpoint_param_names_present", False):
        missing.append("param_names")
    if missing:
        raise ValueError(
            "checkpoint lacks current provenance: "
            + ", ".join(sorted(missing))
        )

    for name in ("model_seed", "split_seed"):
        value = values[name]
        if (isinstance(value, bool)
                or not isinstance(value, Integral)
                or not 0 <= int(value) <= 2 ** 32 - 1):
            raise ValueError(
                f"checkpoint {name} must be a uint32 integer, got {value!r}"
            )
    if (isinstance(values["epoch"], bool)
            or not isinstance(values["epoch"], Integral)
            or int(values["epoch"]) <= 0):
        raise ValueError(
            "checkpoint epoch must be a positive integer, got "
            f"{values['epoch']!r}"
        )
    if (isinstance(values["val_mse"], bool)
            or not isinstance(values["val_mse"], Real)
            or not math.isfinite(float(values["val_mse"]))):
        raise ValueError(
            f"checkpoint val_mse must be finite, got {values['val_mse']!r}"
        )

    if (expected_model_seed is not None
            and int(values["model_seed"]) != int(expected_model_seed)):
        raise ValueError(
            "checkpoint model_seed does not match expected seed: "
            f"{values['model_seed']} vs {expected_model_seed}"
        )
    if (expected_split_seed is not None
            and int(values["split_seed"]) != int(expected_split_seed)):
        raise ValueError(
            "checkpoint split_seed does not match expected seed: "
            f"{values['split_seed']} vs {expected_split_seed}"
        )

    actual_names = [str(name).lower() for name in predictor.get_param_names()]
    expected_names = [str(name).lower() for name in pysim.PARAM_NAMES]
    if actual_names != expected_names:
        raise ValueError(
            "checkpoint parameter order does not match canonical order: "
            f"{actual_names} vs {expected_names}"
        )

    provenance = evaluation_provenance(predictor, config)
    provenance["release_schema"] = "current"
    return provenance


def run_evaluation(predictor, config, results_dir, model_label, tag,
                   max_samples=0, seed=0):
    """在留出测试集上评估 predictor 的预测质量，写出 JSON 摘要并返回。

    逐样本：预测参数 → ``pysim`` 正向模拟 → 与该样本真实曲线比 RMSE，对全体有效样本
    聚合 (均值/中位/标准差/P90 + 逐通道均值 + scaled 参数 MSE)。**不做精修**。与两个
    子系统此前各自的 ``evaluate`` 逐位等价，仅把 predictor/config/标签作为参数传入。

    Args:
        predictor: 暴露 ``predict(X)->(N,7)``、``y_scaler``、``get_trainable_param_names``
            等的推理器 (CNN 用 ``predictor.config``, Transformer 用 ``parent_config``)。
        config: 对应的 config 对象 (CNN=predictor.config, Transformer=parent_config)。
        results_dir: 摘要 JSON 输出目录。
        model_label: 摘要里的 ``model`` 字段 (``'CNN'`` / ``'Transformer'``)。
        tag: 输出文件名 ``testset_eval.<tag>.json`` 及打印用标签。
        max_samples: >0 时随机抽样加速 (确定性, 由 ``seed`` 控制)。
        seed: 抽样 RNG 种子。

    Returns:
        dict: 摘要 (含 0 有效样本时的诊断分支)。
    """
    max_samples = require_int(max_samples, "max_samples", minimum=0)
    seed = require_seed(seed, "run_evaluation")
    # Hash validation precedes dataset loading and model computation. A new
    # checkpoint must never be evaluated against a different dataset/scaler.
    provenance = evaluation_provenance(predictor, config)
    X_test, Y_test = get_test_split(config)
    n_total = X_test.shape[0]

    # 可选抽样加速
    if max_samples and max_samples < n_total:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=max_samples, replace=False)
        X_test, Y_test = X_test[idx], Y_test[idx]
    n = X_test.shape[0]
    print(f"测试样本数 / Test samples: {n} (总 {n_total})")

    # 批量预测 (predictor 内部自做归一化 + 反 log)
    preds = np.asarray(
        predictor.predict(X_test), dtype=np.float64
    )  # (n, 7) 物理空间
    config_names = config.get_trainable_param_names()
    prediction_names = predictor.get_param_names()
    if (preds.ndim != 2
            or preds.shape != (n, len(prediction_names))
            or not np.all(np.isfinite(preds))):
        raise ValueError(
            "预测输出无效 / invalid prediction shape or values: "
            f"expected {(n, len(prediction_names))}, got {preds.shape}"
        )
    # 逐样本：正向模拟预测参数 → 与该样本真实曲线比 RMSE
    rmse_list = []
    n_invalid = 0
    n_extreme = 0
    for i in range(n):
        pdict = vector_to_param_dict(preds[i], prediction_names)
        sig, dt = pysim.run_simulation(pdict)
        sig = np.asarray(sig, dtype=np.float64)
        true_curve = X_test[i]  # (3, T) 该样本真实曲线
        if (not np.isfinite(dt)
                or dt < 0
                or sig.shape != true_curve.shape
                or not np.all(np.isfinite(sig))):
            n_invalid += 1
            continue
        # 物理合理性：荧光信号应在 ~[0, 1.5]，超过即灾难性预测
        if np.max(np.abs(sig)) > 5.0:
            n_extreme += 1
            continue
        rmse = np.sqrt(np.mean((sig - true_curve) ** 2, axis=1))
        rmse_list.append(rmse)
        if (i + 1) % 200 == 0:
            print(f"  ...{i+1}/{n}")

    os.makedirs(results_dir, exist_ok=True)
    out = os.path.join(results_dir, f'testset_eval.{tag}.json')

    # 空结果保护：若所有样本都无效/非有限或灾难性 (|sig|>5)，rmse_list 为空，
    # 后续 mean/percentile/len/n*100 会抛 AxisError/ZeroDivision。这通常意味着
    # checkpoint 训练崩坏或数据集损坏——报告诊断信息并优雅返回。
    if not rmse_list:
        summary = dict(
            model=model_label, n_samples=int(n),
            n_invalid=int(n_invalid), n_extreme=int(n_extreme), n_valid=0,
            error='no valid samples (all invalid or catastrophic predictions)',
            provenance=provenance,
        )
        write_json(out, summary)
        print(f"\n[警告] 0 个有效样本 / no valid samples "
              f"(无效/非有限模拟: {n_invalid}, 灾难性预测 |sig|>5: {n_extreme})。"
              f"\n模型 checkpoint 可能训练崩坏或数据集损坏。摘要已保存: {out}")
        return summary

    rmse_arr = np.array(rmse_list)  # (m, 3)
    per_sample_mean = rmse_arr.mean(axis=1)  # (m,)

    # 参数空间 MSE (在各自 y_scaler 的 scaled 空间，公平)
    config_index = {
        name.lower(): index for index, name in enumerate(config_names)
    }
    try:
        y_order = [config_index[name.lower()] for name in prediction_names]
    except KeyError as exc:
        raise ValueError(
            "预测器与配置的参数集合不一致 / predictor/config parameter mismatch"
        ) from exc
    Y_true = Y_test[:, y_order].copy()
    P = preds.copy()
    for pp in config.get_log_transform_params():
        matching = [
            index for index, name in enumerate(prediction_names)
            if name.lower() == pp.lower()
        ]
        if matching:
            k = matching[0]
            Y_true[:, k] = np.log10(Y_true[:, k] + config.get_log_epsilon())
            P[:, k] = np.log10(np.clip(P[:, k], 1e-30, None) + config.get_log_epsilon())
    ys = predictor.y_scaler
    P_scaled = np.asarray(ys.transform(P), dtype=np.float64)
    Y_scaled = np.asarray(ys.transform(Y_true), dtype=np.float64)
    if (P_scaled.shape != P.shape
            or Y_scaled.shape != Y_true.shape
            or not np.all(np.isfinite(P_scaled))
            or not np.all(np.isfinite(Y_scaled))):
        raise ValueError("Y scaler produced an invalid shape or non-finite values")
    param_mse = float(np.mean((P_scaled - Y_scaled) ** 2))

    summary = dict(
        model=model_label,
        n_samples=int(n),
        n_invalid=int(n_invalid),
        n_extreme=int(n_extreme),
        n_valid=int(len(rmse_list)),
        curve_rmse_mean=float(per_sample_mean.mean()),
        curve_rmse_median=float(np.median(per_sample_mean)),
        curve_rmse_std=float(per_sample_mean.std()),
        curve_rmse_p90=float(np.percentile(per_sample_mean, 90)),
        fam_rmse_mean=float(rmse_arr[:, 0].mean()),
        tye_rmse_mean=float(rmse_arr[:, 1].mean()),
        cy5_rmse_mean=float(rmse_arr[:, 2].mean()),
        param_mse_scaled=param_mse,
        provenance=provenance,
    )

    write_json(out, summary)

    print(f"\n============ 测试集预测质量 ({model_label}) ============")
    print(f"  样本数: {n}  (无效/非有限模拟: {n_invalid}, "
          f"灾难性预测 |sig|>5: {n_extreme})")
    print(f"  有效样本: {len(rmse_list)} / {n}  ({len(rmse_list)/n*100:.1f}%)")
    print(f"  曲线重构 RMSE  均值: {summary['curve_rmse_mean']:.4f}")
    print(f"                中位数: {summary['curve_rmse_median']:.4f}")
    print(f"                标准差: {summary['curve_rmse_std']:.4f}")
    print(f"                P90:   {summary['curve_rmse_p90']:.4f}")
    print(f"  逐通道均值: FAM={summary['fam_rmse_mean']:.4f} "
          f"TYE={summary['tye_rmse_mean']:.4f} CY5={summary['cy5_rmse_mean']:.4f}")
    print(f"  参数 MSE (scaled): {param_mse:.5f}")
    print(f"  摘要已保存: {out}")
    return summary
