# coding=utf-8
"""Transformer experimental prediction, with optional physics refinement."""

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
from .inference import TransformerPredictor
MATLAB_INPUT_FILE = str(
    PREDICTIONS_DIR / "transformer" / "matlab_input_params.txt"
)


def load_real_experimental_data(config, data_path):
    """加载并预处理真实实验数据 — 委托给共享实现。

    见 ``dnawalker.data.experimental.load_and_smooth_experimental_curves``。
    """
    from dnawalker.data.experimental import load_and_smooth_experimental_curves
    return load_and_smooth_experimental_curves(config, data_path, verbose=True)


def predict_parameters(
    parent_config_override=None,
    transformer_config_override=None,
    model_path=None,
    ensemble=0,
    noise_std=0.005,
    seed=0,
    output_path=MATLAB_INPUT_FILE,
    experimental_data_path=None,
):
    # 1. 初始化预测器
    try:
        predictor = TransformerPredictor(
            model_path=model_path,
            parent_config_override_file=parent_config_override,
            transformer_config_override_file=transformer_config_override
        )
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"初始化预测器失败: {e}")
        return 1

    # 2. 读取实验数据路径
    data_file = (
        experimental_data_path
        or predictor.parent_config.get_experimental_data_path()
    )

    # 3. 加载并处理数据
    X_sample_raw = load_real_experimental_data(predictor.parent_config, data_file)
    if X_sample_raw is None:
        return 1

    # 4. 执行推理 (内含归一化和反对数变换)
    print("\n--- 2. 正在执行 Transformer 模型推理 ---")
    ensemble = require_int(ensemble, "ensemble", minimum=0)
    predicted_real = ensemble_predict(
        predictor,
        X_sample_raw,
        ensemble=ensemble,
        noise_std=noise_std,
        seed=seed,
    )

    # 5. 打印并保存结果。按配置参数名映射到 pysim 的规范键名，配置重排也安全。
    params_dict = vector_to_param_dict(
        predicted_real,
        predictor.get_param_names(),
    )

    print("\n--- 3. 预测完成 (Transformer Results) ---")
    print("=" * 45)
    for name in pysim.PARAM_NAMES:
        print(f"  {name:<20}: {params_dict[name]:<15.6e}")
    print("=" * 45)

    print(f"\n正在写入参数文件: {output_path}")
    write_matlab_params(params_dict, output_path)
    print(f"结果已保存至 {output_path}")
    print("运行 `python -m dnawalker.verify` 做正向物理验证。")
    return 0


def run(
    parent_override=None,
    transformer_override=None,
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
    predictor = TransformerPredictor(
        model_path=model_path,
        parent_config_override_file=parent_override,
        transformer_config_override_file=transformer_override,
    )
    config = predictor.parent_config
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
    """Run the Transformer prediction CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Transformer prediction, optionally followed by refinement."
        )
    )
    parser.add_argument(
        "--config",
        help="Optional shared override layered on configs/common.ini.",
    )
    parser.add_argument(
        "--transformer-config",
        help="Optional model override layered on configs/transformer.ini.",
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
            "results/predictions/transformer/matlab_input_params.txt)."
        ),
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Refine predicted parameters against the experimental curves.",
    )
    parser.add_argument("--ensemble", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.005)
    parser.add_argument(
        "--method",
        default="Powell",
        choices=["Powell", "Nelder-Mead"],
    )
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--multistart", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    parent_override = (
        os.path.abspath(args.config) if args.config else None
    )
    transformer_override = (
        os.path.abspath(args.transformer_config)
        if args.transformer_config
        else None
    )
    experimental_data_path = os.path.abspath(args.exp) if args.exp else None
    if args.refine:
        run(
            parent_override=parent_override,
            transformer_override=transformer_override,
            model_path=args.model,
            experimental_data_path=experimental_data_path,
            ensemble=args.ensemble,
            noise_std=args.noise_std,
            method=args.method,
            maxiter=args.maxiter,
            multistart=args.multistart,
            seed=args.seed,
            output_path=args.out,
        )
        return 0
    return predict_parameters(
        parent_config_override=parent_override,
        transformer_config_override=transformer_override,
        model_path=args.model,
        experimental_data_path=experimental_data_path,
        ensemble=args.ensemble,
        noise_std=args.noise_std,
        seed=args.seed,
        output_path=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
