# coding=utf-8
"""Validation and optimizer-contract tests for identifiability analysis."""

from types import SimpleNamespace

import numpy as np
import pytest
import scipy.optimize

from dnawalker.studies import identifiability as analyze_identifiability
from dnawalker.physics import pysim, refine
from dnawalker.studies import protocol as validation_common


_RANGES = validation_common.load_configured_ranges()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rel_step": 0.0}, "rel_step"),
        ({"rel_step": float("nan")}, "rel_step"),
        ({"workers": 0}, "workers"),
        ({"workers": 1.5}, "workers"),
    ],
)
def test_compute_jacobian_rejects_invalid_numerics_before_simulation(
    kwargs, message
):
    with pytest.raises(ValueError, match=message):
        analyze_identifiability.compute_jacobian(
            analyze_identifiability.REFERENCE_PARAMETERS,
            _RANGES,
            **kwargs,
        )


@pytest.mark.parametrize("grid", [0, 1, 2.5, True])
def test_profile_requires_at_least_two_exact_grid_points(grid):
    with pytest.raises(ValueError, match="grid"):
        analyze_identifiability.profile_likelihood(_RANGES, grid=grid)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": 0},
        {"points": 1.5},
        {"workers": 0},
        {"rel_step": -1.0},
        {"do_profile": "false"},
        {"do_profile": True, "profile_grid": 1},
        {"do_profile": True, "profile_maxiter": 0},
    ],
)
def test_run_analysis_rejects_invalid_options_before_writing(kwargs, tmp_path):
    with pytest.raises(ValueError):
        analyze_identifiability.run_analysis(
            results_dir=str(tmp_path), **kwargs
        )
    assert not any(tmp_path.iterdir())


def test_fisher_analysis_validates_matrix_contract():
    with pytest.raises(ValueError, match="matrix"):
        analyze_identifiability.fisher_analysis(np.zeros((10, 6)))
    bad = np.zeros((10, len(pysim.PARAM_NAMES)))
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analyze_identifiability.fisher_analysis(bad)


def test_simulate_flat_rejects_every_nonfinite_timestep(monkeypatch):
    signals = np.zeros((3, pysim.NUM_RESULTS), dtype=np.float64)
    opt_vec = refine._to_opt_space(
        analyze_identifiability.REFERENCE_PARAMETERS, _RANGES
    )

    for bad_dt in (float("nan"), float("inf"), float("-inf")):
        monkeypatch.setattr(
            pysim,
            "run_simulation",
            lambda _params, dt=bad_dt: (signals, dt),
        )
        assert analyze_identifiability._simulate_flat(opt_vec) is None


def test_profile_rejects_every_nonfinite_reference_timestep(monkeypatch):
    signals = np.zeros((3, pysim.NUM_RESULTS), dtype=np.float64)

    for bad_dt in (float("nan"), float("inf"), float("-inf")):
        monkeypatch.setattr(
            pysim,
            "run_simulation",
            lambda _params, dt=bad_dt: (signals, dt),
        )
        with pytest.raises(RuntimeError, match="invalid simulation"):
            analyze_identifiability.profile_likelihood(_RANGES, grid=2)


def test_profile_refiner_explicitly_uses_nelder_mead(monkeypatch):
    captured = {}

    def fake_minimize(objective, x0, method, options):
        captured.update(method=method, options=options)
        objective(np.asarray(x0, dtype=np.float64))
        return SimpleNamespace(x=np.asarray(x0, dtype=np.float64))

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)
    monkeypatch.setattr(refine, "curve_rmse", lambda _params, _curves: 0.25)

    bounds = refine._opt_bounds(_RANGES)
    fixed = sum(bounds[0]) / 2.0
    rmse = analyze_identifiability._refine_fixing_one(
        0,
        fixed,
        np.zeros((3, pysim.NUM_RESULTS), dtype=np.float64),
        _RANGES,
        maxiter=17,
    )

    assert rmse == pytest.approx(0.25)
    assert captured["method"] == "Nelder-Mead"
    assert captured["options"]["maxiter"] == 17
