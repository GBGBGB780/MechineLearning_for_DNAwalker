"""Unified command line for the DNA Walker project."""

import argparse
import importlib
import sys


def _run_target(target, argv, prog):
    module_name, function_name = target.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    previous_prog = sys.argv[0]
    try:
        sys.argv[0] = prog
        result = function(argv)
    finally:
        sys.argv[0] = previous_prog
    return 0 if result is None else int(result)


def _add_leaf(subparsers, name, help_text, target):
    parser = subparsers.add_parser(name, help=help_text, add_help=False)
    parser.set_defaults(_command_target=target)
    return parser


def build_parser():
    """Build the lightweight command tree without importing model modules."""
    parser = argparse.ArgumentParser(
        prog="dnawalker",
        description="DNA Walker simulation, inverse models, and studies.",
    )
    commands = parser.add_subparsers(dest="command")

    for model in ("cnn", "transformer"):
        model_parser = commands.add_parser(
            model,
            help=f"{model.upper()} model workflows.",
        )
        model_commands = model_parser.add_subparsers(dest=f"{model}_command")
        _add_leaf(
            model_commands,
            "train",
            "Train the inverse model.",
            f"dnawalker.{model}.train:main",
        )
        _add_leaf(
            model_commands,
            "predict",
            "Predict parameters, optionally with physics refinement.",
            f"dnawalker.{model}.predict:main",
        )
        _add_leaf(
            model_commands,
            "evaluate",
            "Evaluate held-out or experimental curves.",
            f"dnawalker.{model}.evaluate:main",
        )

    data_parser = commands.add_parser(
        "data",
        help="Dataset generation and inspection tools.",
    )
    data_commands = data_parser.add_subparsers(dest="data_command")
    _add_leaf(
        data_commands,
        "generate",
        "Generate a balanced synthetic NPZ dataset.",
        "dnawalker.data.generate:main",
    )
    _add_leaf(
        data_commands,
        "inspect",
        "Inspect a canonical NPZ dataset.",
        "dnawalker.tools.check_npz:main",
    )
    _add_leaf(
        data_commands,
        "convert",
        "Convert a MATLAB/HDF5 dataset to NPZ.",
        "dnawalker.tools.mat_to_npz:main",
    )

    study_parser = commands.add_parser(
        "study",
        help="Controlled scientific validation studies.",
    )
    study_commands = study_parser.add_subparsers(dest="study_command")
    studies = (
        (
            "identifiability",
            "Run sensitivity and identifiability analysis.",
            "dnawalker.studies.identifiability:cli",
        ),
        (
            "signal",
            "Build signal spectrum and autocorrelation evidence.",
            "dnawalker.studies.signal_analysis:cli",
        ),
        (
            "multiseed",
            "Run or merge the controlled multi-seed comparison.",
            "dnawalker.studies.multiseed.runner:cli",
        ),
        (
            "learning-curve",
            "Run or merge the nested learning-curve study.",
            "dnawalker.studies.learning_curve:main",
        ),
        (
            "fit-robustness",
            "Aggregate experimental-fit refinement robustness.",
            "dnawalker.studies.fit_robustness:main",
        ),
    )
    for name, help_text, target in studies:
        _add_leaf(study_commands, name, help_text, target)

    _add_leaf(
        commands,
        "verify",
        "Forward-simulate a parameter file against experimental data.",
        "dnawalker.verify:main",
    )
    return parser


def main(argv=None):
    """Parse the command tree and delegate remaining arguments lazily."""
    parser = build_parser()
    args, remaining = parser.parse_known_args(
        sys.argv[1:] if argv is None else argv
    )
    target = getattr(args, "_command_target", None)
    if target is None:
        parser.print_help()
        return 0
    command_parts = ["dnawalker", args.command]
    nested = getattr(args, f"{args.command}_command", None)
    if nested:
        command_parts.append(nested)
    return _run_target(target, remaining, " ".join(command_parts))


if __name__ == "__main__":
    raise SystemExit(main())
