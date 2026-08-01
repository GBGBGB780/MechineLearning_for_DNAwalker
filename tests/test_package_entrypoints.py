"""Tests for canonical module and installed console-script entry points."""

import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from dnawalker.paths import PREDICTIONS_DIR, RESULTS_DIR
from dnawalker.verify import _default_output_path


_ROOT = Path(__file__).resolve().parents[1]
_EXPECTED_SCRIPTS = {"dnawalker"}


def _project_scripts():
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)["project"]["scripts"]


_SCRIPTS = _project_scripts()
_MODULES = tuple(dict.fromkeys(
    target.partition(":")[0] for target in _SCRIPTS.values()
))


@pytest.mark.parametrize("module_name", _MODULES)
def test_canonical_modules_support_help_from_arbitrary_cwd(
    module_name, tmp_path
):
    env = os.environ.copy()
    existing_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), existing_path) if value
    )

    result = subprocess.run(
        [sys.executable, "-m", module_name, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_pyproject_exposes_only_canonical_callable_scripts():
    assert set(_SCRIPTS) == _EXPECTED_SCRIPTS

    for script_name, target in _SCRIPTS.items():
        module_name, separator, attribute_name = target.partition(":")
        assert separator == ":", f"invalid target for {script_name}: {target}"
        assert module_name.startswith("dnawalker.")
        entrypoint = getattr(importlib.import_module(module_name), attribute_name)
        assert callable(entrypoint), f"entry point is not callable: {target}"


@pytest.mark.parametrize(
    "command",
    [
        ("cnn", "train"),
        ("cnn", "predict"),
        ("cnn", "evaluate"),
        ("transformer", "train"),
        ("transformer", "predict"),
        ("transformer", "evaluate"),
        ("data", "generate"),
        ("data", "inspect"),
        ("data", "convert"),
        ("study", "identifiability"),
        ("study", "signal"),
        ("study", "multiseed"),
        ("study", "learning-curve"),
        ("study", "fit-robustness"),
        ("verify",),
    ],
)
def test_unified_cli_leaf_help(command, tmp_path):
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(_ROOT), env.get("PYTHONPATH")) if value
    )
    result = subprocess.run(
        [sys.executable, "-m", "dnawalker", *command, "--help"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_verification_default_figure_is_model_owned(model):
    params = PREDICTIONS_DIR / model / "matlab_input_params.txt"

    assert Path(_default_output_path(params)) == (
        RESULTS_DIR
        / "evaluation"
        / model
        / "matlab_input_params_verify.png"
    )


@pytest.mark.parametrize("model", ["cnn", "transformer"])
def test_verification_preserves_model_run_subdirectory(model):
    params = (
        PREDICTIONS_DIR
        / model
        / "experiment_a"
        / "seed_42"
        / "params.txt"
    )

    assert Path(_default_output_path(params)) == (
        RESULTS_DIR
        / "evaluation"
        / model
        / "experiment_a"
        / "seed_42"
        / "params_verify.png"
    )


def test_verification_external_input_uses_generic_evaluation_dir(tmp_path):
    params = tmp_path / "external_params.txt"

    assert Path(_default_output_path(params)) == (
        RESULTS_DIR
        / "evaluation"
        / "verification"
        / "external_params_verify.png"
    )


@pytest.mark.parametrize("model", ["cnn", "transformer"])
@pytest.mark.parametrize("refine", [False, True])
def test_prediction_cli_forwards_explicit_input_and_output_paths(
    model, refine, tmp_path, monkeypatch
):
    module = importlib.import_module(f"dnawalker.{model}.predict")
    target_name = "run" if refine else "predict_parameters"
    captured = {}

    def fake_target(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(module, target_name, fake_target)
    output_path = tmp_path / model / "run_a" / "params.txt"
    experimental_path = tmp_path / "input.xlsx"
    argv = [
        "--out",
        str(output_path),
        "--exp",
        str(experimental_path),
    ]
    if refine:
        argv.append("--refine")

    assert module.main(argv) == 0
    assert captured["output_path"] == str(output_path)
    assert captured["experimental_data_path"] == str(experimental_path)


def test_application_script_help_from_arbitrary_cwd(tmp_path):
    script = _ROOT / "scripts" / "run_application.sh"

    result = subprocess.run(
        ["bash", str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
    assert "--exp" in result.stdout
