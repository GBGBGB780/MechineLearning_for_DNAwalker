# coding=utf-8
"""Golden-value & invariant regression tests for the physics kernel ``pysim``.

The rest of the suite exercises the Call_Counter wrapper (test_pysim_counter.py)
but never asserts that ``run_simulation`` produces the *correct* curves. This
file freezes the numerical behavior of the 14-state Markov forward simulator so
the matrix-power / eigendecomposition fast paths in ``pysim`` cannot silently
drift from the trusted reference implementation.

The golden numbers below were captured from the reviewed implementation for
three fixed Parameter_Sets, deliberately chosen to cover both propagation paths:
  - ``reference`` exercises the ``dt == 1.0`` path (one matrix-power per second);
  - ``case_b`` / ``case_c`` exercise the ``dt == 0.1`` sub-step path
    (``sps == 10``: per-second propagator via matrix_power + 10-minute boundary
    split), which is where the batched optimizations are most likely to drift.

If ``pysim`` is intentionally changed, regenerate these constants and review the
diff — a change here means the physics kernel's output moved.
"""

import configparser
import os
from types import MappingProxyType

import numpy as np
import pytest

from dnawalker.physics import pysim

# Fixed Parameter_Sets (keyed by pysim.PARAM_NAMES) and the golden per-channel
# statistics captured from the trusted implementation. Rows: [FAM, TYE, CY5].
GOLDEN = {
    "reference": {
        "params": dict(E_b=-1.2, E_b_azo_trans=-1.0, E_b_azo_cis=-0.1,
                       k_mig=0.05, k0=8e-6, drt_z=0.5, drt_s=0.05),
        "dt": 1.0,
        "mean": [7.432308722517e-01, 2.424854385231e-01, 1.262193322182e-01],
        "first": [7.795434765850e-01, 1.654565234150e-01, 9.507000000000e-02],
        "last": [7.002537116842e-01, 3.058383685527e-01, 1.494973038090e-01],
    },
    "case_b": {
        "params": dict(E_b=-1.0, E_b_azo_trans=-1.2, E_b_azo_cis=0.0,
                       k_mig=0.10, k0=2e-5, drt_z=0.6, drt_s=0.08),
        "dt": 0.1,
        "mean": [6.613855638200e-01, 3.683388783796e-01, 1.751845654659e-01],
        "first": [7.795434765850e-01, 1.654565234150e-01, 9.507000000000e-02],
        "last": [5.534397921716e-01, 5.124947497074e-01, 2.441325254065e-01],
    },
    "case_c": {
        "params": dict(E_b=-1.5, E_b_azo_trans=-0.8, E_b_azo_cis=-0.15,
                       k_mig=0.03, k0=1e-6, drt_z=0.35, drt_s=0.03),
        "dt": 0.1,
        "mean": [7.767898426641e-01, 1.687554181380e-01, 9.605126422196e-02],
        "first": [7.795434765850e-01, 1.654565234150e-01, 9.507000000000e-02],
        "last": [7.741699455388e-01, 1.720340991999e-01, 9.687346358309e-02],
    },
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_curves_match_reference(name):
    """run_simulation reproduces the frozen golden curves (both dt paths)."""
    spec = GOLDEN[name]
    signals, dt = pysim.run_simulation(spec["params"])

    assert dt == pytest.approx(spec["dt"], rel=1e-12), (
        f"{name}: dt path changed (got {dt}, expected {spec['dt']})")
    assert signals.shape == (3, pysim.NUM_RESULTS)
    assert np.all(np.isfinite(signals))

    for ci in range(3):
        assert signals[ci].mean() == pytest.approx(spec["mean"][ci], rel=1e-9), (
            f"{name}: channel {ci} mean drifted")
        assert signals[ci, 0] == pytest.approx(spec["first"][ci], rel=1e-9)
        assert signals[ci, -1] == pytest.approx(spec["last"][ci], rel=1e-9)


def test_cy5_initial_equals_unbind_floor():
    """At t=0 the CY5 signal equals p_unbind_track plus zero occupied CY5 states.

    The initial condition seeds only states 11/12 (0-indexed 10/11), none of
    which contribute to CY5, so CY5[0] must equal the p_unbind_track floor.
    """
    signals, dt = pysim.run_simulation(GOLDEN["reference"]["params"])
    assert dt > 0
    assert signals[2, 0] == pytest.approx(pysim.FIXED_PARAMS["p_unbind_track"], rel=1e-12)


def test_runtime_constants_are_loaded_from_canonical_config():
    """The simulator and canonical config must not carry divergent values."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "common.ini"
    )
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(path, encoding="utf-8")

    for name, value in pysim.FIXED_PARAMS.items():
        assert value == pytest.approx(
            parser.getfloat("PHYSICAL_PARAMETERS", name)
        )
    assert pysim.SIMU_TIME_MIN == parser.getint(
        "TRAINING", "sim_duration_minutes"
    )
    assert pysim.BLOCK_SEC == pytest.approx(
        parser.getfloat("NANOROBOT_MODELING", "cycle_duration_vis") * 60
    )


def _runtime_config_with_override(tmp_path, section, key, value):
    source = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "configs", "common.ini"
    )
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read(source, encoding="utf-8")
    parser.set(section, key, str(value))
    path = tmp_path / "invalid_runtime.ini"
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return path


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("PHYSICAL_PARAMETERS", "kBT", "nan", "finite"),
        ("PHYSICAL_PARAMETERS", "lp_s", 0, "positive"),
        ("PHYSICAL_PARAMETERS", "n_D1", 0, "positive"),
        ("PHYSICAL_PARAMETERS", "n_gray", -1, "non-negative"),
        ("PHYSICAL_PARAMETERS", "p_unbind_track", 1.1, "probability"),
        ("TRAINING", "sim_duration_minutes", 0, "positive"),
        ("NANOROBOT_MODELING", "cycle_duration_vis", "inf", "equal positive"),
    ],
)
def test_runtime_constants_reject_invalid_values(
    tmp_path, section, key, value, message
):
    path = _runtime_config_with_override(tmp_path, section, key, value)
    with pytest.raises(ValueError, match=message):
        pysim._load_runtime_constants(path)


def test_signals_within_physical_bounds():
    """Fluorescence signals stay in a physically plausible band for all cases."""
    for spec in GOLDEN.values():
        signals, dt = pysim.run_simulation(spec["params"])
        assert dt > 0
        # Signals are sums of probabilities (+ a small constant CY5 floor); they
        # must be well within [0, ~1.2], never negative or blowing up.
        assert signals.min() >= -1e-9
        assert signals.max() <= 1.2


def test_dict_and_sequence_inputs_agree():
    """Mapping implementations and PARAM_NAMES-ordered sequences agree."""
    params = GOLDEN["reference"]["params"]
    seq = [params[n] for n in pysim.PARAM_NAMES]
    sig_dict, dt_dict = pysim.run_simulation(params)
    sig_seq, dt_seq = pysim.run_simulation(seq)
    sig_mapping, dt_mapping = pysim.run_simulation(MappingProxyType(params))
    assert dt_dict == dt_seq
    assert np.array_equal(sig_dict, sig_seq)
    assert dt_mapping == dt_dict
    assert np.array_equal(sig_mapping, sig_dict)


def test_explicit_empty_fixed_params_does_not_fall_back_to_defaults():
    """An explicitly supplied mapping is authoritative, even when empty."""
    with pytest.raises(KeyError):
        pysim._run_simulation_impl(GOLDEN["reference"]["params"], fixed_params={})


def test_batched_matches_naive_iteration():
    """The batched eigendecomposition path agrees with naive per-step matvec.

    ``_batch_states`` (eigendecomposition) is the core optimization; this drives
    the same propagator through the ``_iter_states`` fallback and asserts the two
    agree, so a broken fast path cannot pass unnoticed.
    """
    params = GOLDEN["reference"]["params"]
    signals, dt = pysim.run_simulation(params)
    assert dt > 0

    # Reconstruct the per-second visible-light propagator M_vis exactly as the
    # implementation does, then compare batched vs. iterative state sequences.
    fp = pysim.FIXED_PARAMS
    Et, Ec, ft, fc = pysim._compute_config_energies(
        params["E_b"], params["E_b_azo_trans"], params["E_b_azo_cis"], fp)
    k_trans = pysim._build_k_matrix(
        Et, ft, params["k0"], params["k_mig"], params["drt_z"], params["drt_s"], fp["kBT"])
    kt14 = k_trans[1:15, 1:15]
    sps = int(round(1.0 / dt))
    R_vis = pysim._build_R(kt14, dt)
    M_vis = np.linalg.matrix_power(R_vis, sps)

    p0 = np.zeros(14)
    p0[10] = p0[11] = pysim.P_TOTAL / 2
    n = 50
    batched = pysim._batch_states(M_vis, p0, n)
    naive = pysim._iter_states(M_vis, p0, n)
    assert batched is not None, "eigendecomposition path returned None unexpectedly"
    assert np.allclose(batched, naive, rtol=1e-8, atol=1e-10)
