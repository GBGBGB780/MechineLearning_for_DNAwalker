import numpy as np
import pytest

from dnawalker.physics import pysim, refine


def test_refine_uses_canonical_physics_module():
    """The canonical refiner shares the canonical simulator module."""
    from dnawalker.physics import pysim as package_pysim
    from dnawalker.physics import refine as package_refine

    assert refine is package_refine
    assert refine.pysim is pysim is package_pysim
    for name in ("_to_opt_space", "_from_opt_space", "_opt_bounds"):
        assert callable(getattr(refine, name))


def _ranges():
    return {name: (0.1, 1.0) for name in pysim.PARAM_NAMES}


def _params():
    return {name: 0.5 for name in pysim.PARAM_NAMES}


def test_curve_rmse_rejects_nonfinite_target_before_simulation(monkeypatch):
    called = {"value": False}

    def _should_not_run(_params):
        called["value"] = True
        raise AssertionError("simulation should not run")

    monkeypatch.setattr(refine.pysim, "run_simulation", _should_not_run)
    curves = np.zeros((3, 4))
    curves[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        refine.curve_rmse(_params(), curves)
    assert called["value"] is False


def test_curve_rmse_rejects_shape_mismatch(monkeypatch):
    monkeypatch.setattr(
        refine.pysim,
        "run_simulation",
        lambda _params: (np.zeros((3, 5)), 1.0),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        refine.curve_rmse(_params(), np.zeros((3, 4)))


@pytest.mark.parametrize(
    "dt,signals",
    [
        (float("nan"), np.zeros((3, 4))),
        (1.0, np.full((3, 4), np.inf)),
    ],
)
def test_channel_curve_rmse_marks_nonfinite_simulation_invalid(
        monkeypatch, dt, signals):
    monkeypatch.setattr(
        refine.pysim,
        "run_simulation",
        lambda _params: (signals, dt),
    )

    rmse, returned_signals = refine.channel_curve_rmse(
        _params(), np.zeros((3, 4))
    )

    assert rmse is None
    assert returned_signals is None
    assert refine.curve_rmse(_params(), np.zeros((3, 4))) == float("inf")


@pytest.mark.parametrize("method", ["BFGS", "", "powell"])
def test_build_objective_rejects_unsupported_method(method):
    with pytest.raises(ValueError, match="Unsupported"):
        refine.build_objective(
            _params(), np.zeros((3, 4)), _ranges(), method=method
        )


@pytest.mark.parametrize("maxiter", [0, -1, 1.5, True])
def test_build_objective_requires_positive_integer_maxiter(maxiter):
    with pytest.raises(ValueError, match="maxiter"):
        refine.build_objective(
            _params(), np.zeros((3, 4)), _ranges(), maxiter=maxiter
        )


def test_build_objective_clips_initial_point_to_bounds():
    params = _params()
    params["E_b"] = 2.0
    _, x0, lo, hi, _ = refine.build_objective(
        params,
        np.zeros((3, 4)),
        _ranges(),
    )
    assert np.all(x0 >= lo)
    assert np.all(x0 <= hi)
    assert x0[pysim.PARAM_NAMES.index("E_b")] == pytest.approx(1.0)


def test_jitter_params_uses_log_space_for_log_parameters():
    ranges = _ranges()
    ranges["k0"] = (1e-8, 1e-4)
    params = _params()
    params["k0"] = 1e-6

    class _OneSigmaRng:
        @staticmethod
        def normal(_mean, _scale):
            return _scale

    jittered = refine.jitter_params(
        params,
        ranges,
        _OneSigmaRng(),
        scale=0.1,
        log_params={"k0"},
    )

    # One sigma is 10% of the four-decade range: 1e-6 -> 10^-5.6.
    assert jittered["k0"] == pytest.approx(10 ** -5.6)
    assert all(ranges[name][0] <= jittered[name] <= ranges[name][1]
               for name in pysim.PARAM_NAMES)


@pytest.mark.parametrize("scale", [-1, np.nan, np.inf, "bad"])
def test_jitter_params_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        refine.jitter_params(
            _params(),
            _ranges(),
            np.random.default_rng(0),
            scale=scale,
        )
