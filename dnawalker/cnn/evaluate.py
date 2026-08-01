# coding=utf-8
"""CNN held-out and experimental-curve evaluation workflows."""

import argparse
import os

import numpy as np

from dnawalker.paths import DATA_DIR, REPO_ROOT, RESULTS_DIR as ROOT_RESULTS_DIR

from dnawalker.physics import simulator as pysim
from dnawalker.shared.ensemble import ensemble_predict
from dnawalker.physics.refinement import channel_curve_rmse, jitter_params, refine
from dnawalker.data.experimental import (
    load_experimental_curves,
    load_and_smooth_experimental_curves,
    raw_experiment_curves_on_sim_axis,
)
from .inference import NanorobotPredictor
from dnawalker.shared.parameters import build_name_map, vector_to_param_dict
from dnawalker.studies.protocol import require_int, write_json
from dnawalker.shared.evaluation import (
    evaluation_provenance,
    get_test_split as _get_test_split,
    require_current_artifact_provenance,
    run_evaluation,
)
from dnawalker.shared.artifacts import sha256_file

RESULTS_DIR = os.fspath(ROOT_RESULTS_DIR / "evaluation" / "cnn")

DATASETS = {
    "original": os.fspath(DATA_DIR / "experimental" / "Fig3a_fitting.xlsx"),
    "generalization": os.fspath(
        DATA_DIR / "experimental" / "Fig3a_fitting_generalization.xlsx"
    ),
}


def get_test_split(config, test_seed=None):
    """Return the canonical held-out split for multi-seed orchestration."""
    return _get_test_split(config, test_seed=test_seed)


def evaluate_testset(
    config_override=None,
    model_path=None,
    max_samples=0,
    seed=0,
    tag="cnn",
    results_dir=RESULTS_DIR,
    require_current_provenance=False,
):
    """Evaluate direct CNN predictions on the fixed synthetic test split."""
    predictor = NanorobotPredictor(
        config_override_file=config_override,
        model_path=model_path,
    )
    if require_current_provenance:
        require_current_artifact_provenance(predictor, predictor.config)
    return run_evaluation(
        predictor,
        predictor.config,
        results_dir,
        model_label="CNN",
        tag=tag,
        max_samples=max_samples,
        seed=seed,
    )


def load_and_smooth_experiment(config, data_path):
    """Load, interpolate, and smooth experimental curves for model input."""
    return load_and_smooth_experimental_curves(
        config, data_path, verbose=False, on_load_error='raise'
    )


def raw_experiment_curves(data_path):
    """Interpolate unsmoothed experimental curves to the simulation axis."""
    return raw_experiment_curves_on_sim_axis(data_path)


def _rmse(pdict, exp_curves):
    rmse, sig = channel_curve_rmse(pdict, exp_curves)
    if rmse is None:
        return float('inf'), None
    return rmse, sig


def eval_one(predictor, config, ranges, data_path, ensemble, noise_std,
             maxiter, multistart, seed, refine_on=True):
    """Evaluate one dataset and return metrics, simulation, and raw curves."""
    from dnawalker.shared.pipeline import evaluate_refined_dataset

    return evaluate_refined_dataset(
        predictor,
        config,
        ranges,
        ensemble=ensemble,
        noise_std=noise_std,
        maxiter=maxiter,
        multistart=multistart,
        seed=seed,
        refine_on=refine_on,
        load_model_input_fn=lambda: load_and_smooth_experiment(
            config, data_path
        ),
        load_interpolated_curves_fn=lambda: raw_experiment_curves(data_path),
        load_raw_curves_fn=lambda: load_experimental_curves(data_path),
        require_int_fn=require_int,
        ensemble_predict_fn=ensemble_predict,
        vector_to_param_dict_fn=vector_to_param_dict,
        rmse_fn=_rmse,
        refine_fn=refine,
        jitter_params_fn=jitter_params,
        numpy_module=np,
    )


def _build_metadata(args):
    """Return strict run settings and exact experimental-input identities."""
    return {
        "schema_version": 2,
        "experiment": "dual_experimental_curve_fit",
        "model": "cnn",
        "checkpoint_selection": args.checkpoint_selection,
        "evaluation_settings": {
            "ensemble": args.ensemble,
            "noise_std": args.noise_std,
            "refine_enabled": not args.no_refine,
            "refinement_method": "Powell",
            "maxiter": args.maxiter,
            "multistart": args.multistart,
            "seed": args.seed,
        },
        "experimental_inputs": {
            name: {
                "path": os.path.relpath(os.path.abspath(path), REPO_ROOT),
                "sha256": sha256_file(path),
            }
            for name, path in DATASETS.items()
        },
    }


def plot_dual(results, out_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    sim_time = np.arange(pysim.NUM_RESULTS) * (pysim.SAVE_INTERVAL_SEC / 60.0)
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    chan = ['FAM', 'TYE', 'Cy5']
    rmse_keys = ['fam_rmse', 'tye_rmse', 'cy5_rmse']

    for col, name in enumerate(['original', 'generalization']):
        res, sig, (t_raw, fam, tye, cy5) = results[name]
        exp = [fam, tye, cy5]
        for row in range(3):
            ax = axes[row, col]
            # Invalid final parameters yield no simulated signal.
            if sig is not None:
                ax.plot(sim_time, sig[row], 'b-', lw=2, label='Simulation (refined)')
            ax.scatter(t_raw, exp[row], s=6, c='gray', alpha=0.25, label='Experimental')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(f'{chan[row]} Signal')
            ax.set_title(f'{name} - {chan[row]}  (RMSE={res[rmse_keys[row]]:.4f})',
                         fontweight='bold', fontsize=11)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
    fig.suptitle('CNN + physics refinement: original vs. generalization',
                 fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Comparison figure saved: {out_path}")


def evaluate_experimental(args):
    """Evaluate original and generalization workbooks from parsed arguments."""
    predictor = NanorobotPredictor(
        config_override_file=os.path.abspath(args.config) if args.config else None,
        model_path=args.model,
    )
    config = predictor.config
    require_current = (
        args.require_current_provenance or bool(args.config or args.model)
    )
    provenance = (
        require_current_artifact_provenance(predictor, config)
        if require_current
        else evaluation_provenance(predictor, config)
    )
    name_map = build_name_map(config.get_trainable_param_names())
    ranges = {name_map[k]: v for k, v in config.get_param_ranges().items()}

    results = {}
    summary = _build_metadata(args)
    for name, path in DATASETS.items():
        print(f"\n===== Evaluating dataset: {name} ({os.path.basename(path)}) =====")
        res, sig, exp_raw = eval_one(
            predictor, config, ranges, path, args.ensemble, args.noise_std,
            args.maxiter, args.multistart, args.seed, refine_on=not args.no_refine)
        results[name] = (res, sig, exp_raw)
        print(f"  Direct-DL mean RMSE: {res['dl_avg']:.4f}")
        print(f"  Refined: FAM={res['fam_rmse']:.4f} TYE={res['tye_rmse']:.4f} "
              f"Cy5={res['cy5_rmse']:.4f} mean={res['avg_rmse']:.4f}")
        summary[name] = {k: res[k] for k in ('dl_avg', 'fam_rmse', 'tye_rmse', 'cy5_rmse', 'avg_rmse')}

    o = summary['original']['avg_rmse']
    g = summary['generalization']['avg_rmse']
    summary['mean_avg_rmse'] = (o + g) / 2
    summary['max_avg_rmse'] = max(o, g)
    summary['provenance'] = provenance

    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(
        args.results_dir, f'rmse_dual.{args.tag}.json'
    )
    write_json(json_path, summary)
    print(f"\nRMSE summary saved: {json_path}")

    plot_dual(
        results,
        os.path.join(args.results_dir, f'dual_fit.{args.tag}.png'),
    )

    print("\n================ Dual-dataset summary (CNN) ================")
    print(f"  original       mean RMSE: {o:.4f}")
    print(f"  generalization mean RMSE: {g:.4f}")
    print(
        f"  combined mean: {summary['mean_avg_rmse']:.4f}   "
        f"worst: {summary['max_avg_rmse']:.4f}"
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate the CNN model.")
    commands = parser.add_subparsers(dest="evaluation", required=True)

    testset = commands.add_parser(
        "testset",
        help="Evaluate the fixed synthetic held-out split.",
    )
    testset.add_argument("--config")
    testset.add_argument("--model")
    testset.add_argument("--max-samples", type=int, default=0)
    testset.add_argument("--seed", type=int, default=0)
    testset.add_argument("--tag", default="cnn")
    testset.add_argument("--results-dir", default=RESULTS_DIR)
    testset.add_argument("--require-current-provenance", action="store_true")

    experimental = commands.add_parser(
        "experimental",
        help="Evaluate original and generalization experimental curves.",
    )
    experimental.add_argument(
        "--config",
        help="Override layered after common.ini and cnn.ini.",
    )
    experimental.add_argument("--model", help="Checkpoint .pth path")
    experimental.add_argument("--ensemble", type=int, default=20)
    experimental.add_argument("--noise-std", type=float, default=0.005)
    experimental.add_argument("--maxiter", type=int, default=500)
    experimental.add_argument("--multistart", type=int, default=8)
    experimental.add_argument("--seed", type=int, default=0)
    experimental.add_argument("--no-refine", action="store_true")
    experimental.add_argument("--tag", default="cnn")
    experimental.add_argument("--results-dir", default=RESULTS_DIR)
    experimental.add_argument(
        "--checkpoint-selection",
        default="unspecified",
    )
    experimental.add_argument(
        "--require-current-provenance",
        action="store_true",
    )

    args = parser.parse_args(argv)
    if args.evaluation == "testset":
        evaluate_testset(
            config_override=(
                os.path.abspath(args.config) if args.config else None
            ),
            model_path=args.model,
            max_samples=args.max_samples,
            seed=args.seed,
            tag=args.tag,
            results_dir=args.results_dir,
            require_current_provenance=(
                args.require_current_provenance
                or bool(args.config or args.model)
            ),
        )
        return 0
    return evaluate_experimental(args)


if __name__ == "__main__":
    raise SystemExit(main())
