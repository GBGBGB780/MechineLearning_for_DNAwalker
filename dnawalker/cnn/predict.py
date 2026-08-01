# coding=utf-8
"""CNN experimental prediction, with optional physics refinement."""

import argparse
import os

import numpy as np

from dnawalker.paths import PREDICTIONS_DIR
from dnawalker.physics import simulator as pysim
from dnawalker.physics.refinement import curve_rmse, jitter_params, refine
from dnawalker.data.experimental import (
    load_and_smooth_experimental_curves,
    raw_experiment_curves_on_sim_axis,
)
from dnawalker.shared.ensemble import ensemble_predict
from dnawalker.shared.parameters import (
    build_name_map,
    vector_to_param_dict,
    write_matlab_params,
)
from dnawalker.studies.protocol import require_int
from .inference import NanorobotPredictor

MATLAB_INPUT_FILE = str(
    PREDICTIONS_DIR / "cnn" / "matlab_input_params.txt"
)


def load_real_experimental_data(config, data_path):
    """加载、插值并平滑真实实验数据 — 委托给共享实现。

    见 ``dnawalker.data.experimental.load_and_smooth_experimental_curves``，verbose=True 保留
    与原实现相同的诊断打印。
    """
    from dnawalker.data.experimental import load_and_smooth_experimental_curves
    return load_and_smooth_experimental_curves(config, data_path, verbose=True)


def predict_parameters(
    config_override=None,
    model_path=None,
    seed=42,
    ensemble=None,
    noise_std=None,
    output_path=MATLAB_INPUT_FILE,
    experimental_data_path=None,
):
    """
    完整预测流程。/ Full prediction pipeline.
    """
    # 初始化 / Initialize
    try:
        predictor = NanorobotPredictor(
            config_override_file=config_override,
            model_path=model_path,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"初始化失败 / Init failed: {e}")
        return 1

    data_path = (
        experimental_data_path
        or predictor.config.get_experimental_data_path()
    )

    # 加载数据 / Load data
    X_raw = load_real_experimental_data(predictor.config, data_path)
    if X_raw is None:
        return 1

    # DL 预测 (Test-Time Ensemble) / DL prediction
    # 归一化 + 反归一化 + 反 log 全部由 predictor.predict() 内部完成 (它复用共享的
    # preprocessing.normalize_per_sample，与训练侧逐位一致)；ensemble 逻辑与
    # 在原始曲线空间加噪后交给 predict()，再取中位数。
    print("\n--- 2. DL 模型预测 / DL prediction (Test-Time Ensemble) ---")
    N_ENS = (
        predictor.config.get_ensemble_size()
        if ensemble is None
        else require_int(ensemble, "ensemble", minimum=0)
    )
    noise_std = (
        predictor.config.get_ensemble_noise_std()
        if noise_std is None
        else noise_std
    )

    pred_real, all_preds = ensemble_predict(
        predictor, X_raw, ensemble=N_ENS, noise_std=noise_std, seed=seed,
        return_all=True)
    if all_preds.shape[0] > 1:
        print(f"  集成预测 / Ensemble (N={all_preds.shape[0]}): {pred_real}")
        print(f"  集成标准差 / Ensemble std: {np.round(np.std(all_preds, axis=0), 5)}")
    else:
        print(f"  单次预测 / Single prediction: {pred_real}")

    # 输出 / Output —— 按配置参数名映射到 pysim 的规范键名。
    param_names = predictor.get_param_names()
    params_dict = vector_to_param_dict(pred_real, param_names)
    write_matlab_params(params_dict, output_path)

    print("\n--- 3. 最终预测参数 / Final predicted parameters ---")
    print("=" * 35)
    for n, v in zip(param_names, pred_real):
        print(f"{n:<20}: {v:<15.6e}")
    print("=" * 35)
    print(f"\n参数已保存至 / Saved to '{output_path}'")
    print("运行 `python -m dnawalker.verify` 验证。")
    return 0


def run(
    config_override=None,
    model_path=None,
    ensemble=0,
    noise_std=0.005,
    method="Powell",
    maxiter=400,
    multistart=8,
    seed=0,
    output_path=MATLAB_INPUT_FILE,
    experimental_data_path=None,
):
    """Predict parameters and refine them against experimental curves."""
    multistart = require_int(multistart, "multistart", minimum=0)
    predictor = NanorobotPredictor(
        config_override_file=config_override,
        model_path=model_path,
    )
    config = predictor.config
    data_path = experimental_data_path or config.get_experimental_data_path()
    model_input = load_and_smooth_experimental_curves(
        config,
        data_path,
        verbose=False,
        on_load_error="raise",
    )
    _, experimental_curves = raw_experiment_curves_on_sim_axis(data_path)

    from dnawalker.shared.pipeline import run_prediction_refinement

    return run_prediction_refinement(
        predictor,
        config,
        model_input,
        experimental_curves,
        ensemble=ensemble,
        noise_std=noise_std,
        method=method,
        maxiter=maxiter,
        multistart=multistart,
        seed=seed,
        output_path=output_path,
        parameter_order_fn=lambda: pysim.PARAM_NAMES,
        ensemble_predict_fn=ensemble_predict,
        vector_to_param_dict_fn=vector_to_param_dict,
        build_name_map_fn=build_name_map,
        curve_rmse_fn=curve_rmse,
        refine_fn=refine,
        jitter_params_fn=jitter_params,
        write_params_fn=write_matlab_params,
        numpy_module=np,
    )


def main(argv=None):
    """Run the CNN prediction CLI."""
    parser = argparse.ArgumentParser(
        description="Run CNN prediction, optionally followed by refinement."
    )
    parser.add_argument(
        "--config",
        help="Optional override layered after common.ini and cnn.ini.",
    )
    parser.add_argument("--model", help="Optional checkpoint path.")
    parser.add_argument(
        "--exp",
        help=(
            "Experimental Excel workbook. Overrides the configured input "
            "for this prediction only."
        ),
    )
    parser.add_argument(
        "--out",
        default=MATLAB_INPUT_FILE,
        help=(
            "Predicted parameter file (default: repository "
            "results/predictions/cnn/matlab_input_params.txt)."
        ),
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine predicted parameters against the experimental curves.",
    )
    parser.add_argument("--ensemble", type=int)
    parser.add_argument("--noise-std", type=float)
    parser.add_argument(
        "--method",
        default="Powell",
        choices=["Powell", "Nelder-Mead"],
    )
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    config_override = os.path.abspath(args.config) if args.config else None
    experimental_data_path = os.path.abspath(args.exp) if args.exp else None
    if args.refine:
        run(
            config_override=config_override,
            model_path=args.model,
            experimental_data_path=experimental_data_path,
            ensemble=0 if args.ensemble is None else args.ensemble,
            noise_std=0.005 if args.noise_std is None else args.noise_std,
            method=args.method,
            maxiter=args.maxiter,
            multistart=args.multistart,
            seed=0 if args.seed is None else args.seed,
            output_path=args.out,
        )
        return 0
    return predict_parameters(
        config_override=config_override,
        model_path=args.model,
        experimental_data_path=experimental_data_path,
        ensemble=args.ensemble,
        noise_std=args.noise_std,
        seed=42 if args.seed is None else args.seed,
        output_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
