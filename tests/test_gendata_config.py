# coding=utf-8
"""Tests for config-driven runtime constants in data generation.

Guards the replacement of hard-coded parameter ranges, activity bins, and the
weak-signal floor with values read from ``configs/common.ini``. It verifies that
historical numerical defaults are preserved, lookup is case-insensitive, and
parameter order follows ``pysim.PARAM_NAMES``.
"""

import os

import numpy as np
import pytest

from dnawalker.data import generate as gendata
from dnawalker.physics import pysim

# Historical hardcoded values from the pre-refactor MATLAB/Python generators.
_HISTORICAL_MIN = np.array([-2.0, -2.0, -0.5, 0.01, 1e-7, 0.20, 0.01])
_HISTORICAL_MAX = np.array([-0.5, -0.5, 0.2, 0.30, 1e-4, 0.90, 0.50])
# configs/common.ini declares bin_edges = 0.01,0.05,0.1,0.2,0.4,2.0.
# replaces the final finite edge (2.0) with +inf so the top bin is an open
# interval and high-activity samples are never silently dropped; only the last
# edge changes, the interior edges are byte-for-byte historical.
_HISTORICAL_INTERIOR_EDGES = np.array([0.01, 0.05, 0.1, 0.2, 0.4])
_HISTORICAL_WEAK_FLOOR = 0.01


def test_module_constants_match_historical_hardcode():
    """Module-level MIN/MAX/FLOOR reproduce the literals; BIN interior unchanged
    with the top edge opened to +inf (see comment above)."""
    np.testing.assert_array_equal(gendata.MIN_VALS, _HISTORICAL_MIN)
    np.testing.assert_array_equal(gendata.MAX_VALS, _HISTORICAL_MAX)
    np.testing.assert_array_equal(gendata.BIN_EDGES[:-1], _HISTORICAL_INTERIOR_EDGES)
    assert np.isinf(gendata.BIN_EDGES[-1]) and gendata.BIN_EDGES[-1] > 0
    assert gendata.WEAK_SIGNAL_FLOOR == pytest.approx(_HISTORICAL_WEAK_FLOOR)


def test_min_max_indexed_by_pysim_param_names():
    """Column i of MIN_VALS must correspond to pysim.PARAM_NAMES[i]."""
    # E_b is index 0; configs/common.ini says [-2.0, -0.5].
    assert gendata.MIN_VALS[pysim.PARAM_NAMES.index('E_b')] == pytest.approx(-2.0)
    assert gendata.MAX_VALS[pysim.PARAM_NAMES.index('E_b')] == pytest.approx(-0.5)
    # k0 is index 4; configs/common.ini says [1e-7, 1e-4].
    assert gendata.MIN_VALS[pysim.PARAM_NAMES.index('k0')] == pytest.approx(1e-7)
    assert gendata.MAX_VALS[pysim.PARAM_NAMES.index('k0')] == pytest.approx(1e-4)


def test_min_below_max_elementwise():
    """Sanity: ranges are non-degenerate (min < max for every parameter)."""
    assert (gendata.MIN_VALS < gendata.MAX_VALS).all()


def test_load_runtime_config_is_idempotent():
    """Calling the loader twice yields identical numpy arrays."""
    a_min, a_max, a_bins, a_floor = gendata._load_runtime_config()
    b_min, b_max, b_bins, b_floor = gendata._load_runtime_config()
    np.testing.assert_array_equal(a_min, b_min)
    np.testing.assert_array_equal(a_max, b_max)
    np.testing.assert_array_equal(a_bins, b_bins)
    assert a_floor == b_floor


def test_load_runtime_config_handles_case_insensitive_lookup(monkeypatch):
    """configparser lowercases keys; loader must still resolve pysim 'E_b' style."""
    # _load_runtime_config builds a lowered dict internally; we just verify
    # the load succeeds — if case handling broke, missing-range ValueError
    # would have fired (and the module wouldn't have imported in the first
    # place). This test documents the invariant explicitly.
    mins, maxs, bins, floor = gendata._load_runtime_config()
    assert mins.shape == (len(pysim.PARAM_NAMES),)
    assert maxs.shape == (len(pysim.PARAM_NAMES),)
    assert np.isfinite(mins).all()
    assert np.isfinite(maxs).all()


def test_load_runtime_config_raises_when_pysim_param_missing(monkeypatch):
    """If get_param_ranges returns a dict that lacks a pysim param, raise.

    The production config loader fills missing ranges with default_range, so
    this never happens through configs/common.ini. The guard exists to protect
    against an upstream config_loader behavior change that would otherwise
    silently feed NaN/garbage into MIN_VALS.
    """
    from dnawalker.config import Config as RealConfig

    class _BrokenConfig(RealConfig):
        def get_param_ranges(self):
            # Drop k0 from the returned ranges — simulates an upstream regression.
            full = super().get_param_ranges()
            return {k: v for k, v in full.items() if k.lower() != "k0"}

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _broken_config_factory(_path=None):
        return _BrokenConfig(
            os.path.join(repo_root, "configs", "common.ini")
        )

    monkeypatch.setattr(gendata, "Config", _broken_config_factory)
    with pytest.raises(ValueError, match="missing ranges|缺少参数"):
        gendata._load_runtime_config()


def _patch_single_bin(monkeypatch):
    """Use one activity bin so tests focus on target-count semantics."""
    monkeypatch.setattr(gendata, "BIN_EDGES", np.array([0.0, np.inf]))


def _valid_result():
    return np.ones((3, 2), dtype=np.float32), 1.0, 0.2


def test_generate_writes_exact_target_without_batch_overshoot(tmp_path, monkeypatch):
    _patch_single_bin(monkeypatch)
    monkeypatch.setattr(gendata, "_simulate_one", lambda _row: _valid_result())
    out = tmp_path / "exact.npz"

    ok = gendata.generate(
        target=3,
        ratio=2.0,
        workers=1,
        out_path=str(out),
        min_dt_threshold=0.0,
        max_rounds=0,
        batch_size=10,
        seed=1,
    )

    assert ok is True
    with np.load(out) as data:
        assert data["X"].shape[0] == 3
        assert data["Y"].shape[0] == 3
        assert data["parameter_names"].tolist() == pysim.PARAM_NAMES


def test_generate_does_not_stall_on_tiny_shortfall(tmp_path, monkeypatch):
    """A valid sample after 20 misses must still fill a 19/20 dataset."""
    _patch_single_bin(monkeypatch)
    calls = {"n": 0}

    def _simulate(_row):
        calls["n"] += 1
        if calls["n"] <= 19 or calls["n"] == 40:
            return _valid_result()
        return np.zeros((3, 2), dtype=np.float32), -1.0, 0.0

    monkeypatch.setattr(gendata, "_simulate_one", _simulate)
    out = tmp_path / "filled.npz"

    ok = gendata.generate(
        target=20,
        ratio=1.0,
        workers=1,
        out_path=str(out),
        min_dt_threshold=0.0,
        max_rounds=2,
        batch_size=20,
        seed=2,
    )

    assert ok is True
    with np.load(out) as data:
        assert data["Y"].shape[0] == 20


def test_generate_accumulates_cap_rejections_before_relaxing(tmp_path, monkeypatch):
    """Cap evidence from earlier stalled rounds must not be forgotten."""
    monkeypatch.setattr(
        gendata, "BIN_EDGES", np.array([0.0, 0.5, np.inf])
    )
    calls = {"n": 0}

    def _simulate(_row):
        calls["n"] += 1
        if calls["n"] == 1:
            return np.ones((3, 2), dtype=np.float32), 1.0, 0.2
        if 3 <= calls["n"] <= 40:
            # Valid, but rejected because the first bin starts at its cap.
            return np.ones((3, 2), dtype=np.float32), 1.0, 0.2
        if calls["n"] in (41, 42):
            # The threshold-crossing round itself has no capped samples.
            return np.zeros((3, 2), dtype=np.float32), -1.0, 0.0
        if calls["n"] == 43:
            # Accepted only after cumulative cap evidence relaxes the cap.
            return np.ones((3, 2), dtype=np.float32), 1.0, 0.2
        return np.zeros((3, 2), dtype=np.float32), -1.0, 0.0

    monkeypatch.setattr(gendata, "_simulate_one", _simulate)
    out = tmp_path / "relaxed.npz"

    ok = gendata.generate(
        target=2,
        ratio=1.0,
        workers=1,
        out_path=str(out),
        min_dt_threshold=0.0,
        max_rounds=21,
        batch_size=2,
        seed=8,
    )

    assert ok is True
    with np.load(out) as data:
        assert data["Y"].shape[0] == 2


def test_generate_failure_never_writes_partial_dataset(tmp_path, monkeypatch):
    _patch_single_bin(monkeypatch)
    monkeypatch.setattr(
        gendata,
        "_simulate_one",
        lambda _row: (np.zeros((3, 2), dtype=np.float32), -1.0, 0.0),
    )
    out = tmp_path / "partial.npz"

    ok = gendata.generate(
        target=2,
        ratio=1.0,
        workers=1,
        out_path=str(out),
        min_dt_threshold=0.0,
        max_rounds=1,
        batch_size=2,
        seed=3,
    )

    assert ok is False
    assert not out.exists()


def test_generate_reuses_one_process_pool(tmp_path, monkeypatch):
    _patch_single_bin(monkeypatch)
    created = []

    class _FakePool:
        def __init__(self, max_workers):
            created.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        @staticmethod
        def map(func, rows, chunksize):
            assert chunksize == 8
            return map(func, rows)

    monkeypatch.setattr(gendata, "ProcessPoolExecutor", _FakePool)
    monkeypatch.setattr(gendata, "_simulate_one", lambda _row: _valid_result())

    assert gendata.generate(
        target=3,
        ratio=1.0,
        workers=2,
        out_path=str(tmp_path / "pooled.npz"),
        min_dt_threshold=0.0,
        max_rounds=0,
        batch_size=1,
        seed=4,
    )
    assert created == [2]


@pytest.mark.parametrize(
    "overrides",
    [
        {"target": 0},
        {"ratio": 0.0},
        {"workers": 0},
        {"max_rounds": -1},
        {"batch_size": 0},
        {"min_dt_threshold": float("nan")},
        {"seed": -1},
        {"seed": 1.5},
        {"seed": True},
        {"seed": 2 ** 32},
    ],
)
def test_generate_rejects_invalid_arguments(tmp_path, overrides):
    kwargs = {
        "target": 1,
        "ratio": 1.0,
        "workers": 1,
        "out_path": str(tmp_path / "invalid.npz"),
        "min_dt_threshold": 0.0,
        "max_rounds": 0,
        "batch_size": 1,
        "seed": 0,
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        gendata.generate(**kwargs)


def test_simulate_one_downgrades_arithmetic_error(monkeypatch):
    monkeypatch.setattr(
        gendata.pysim,
        "run_simulation",
        lambda _row: (_ for _ in ()).throw(OverflowError("numeric overflow")),
    )
    signals, dt, max_change = gendata._simulate_one(np.zeros(len(pysim.PARAM_NAMES)))
    assert signals.shape == (3, pysim.NUM_RESULTS)
    assert dt == -1.0
    assert max_change == 0.0


def test_simulate_one_rejects_every_nonfinite_timestep(monkeypatch):
    signals = np.ones((3, pysim.NUM_RESULTS), dtype=np.float64)

    for bad_dt in (float("nan"), float("inf"), float("-inf")):
        monkeypatch.setattr(
            gendata.pysim,
            "run_simulation",
            lambda _row, dt=bad_dt: (signals, dt),
        )
        _, dt, max_change = gendata._simulate_one(
            np.zeros(len(pysim.PARAM_NAMES))
        )
        assert not gendata._validate(dt, max_change, 0.0)[0]


@pytest.mark.parametrize("error", [MemoryError("oom"), RuntimeError("bug")])
def test_simulate_one_does_not_hide_system_or_programming_errors(monkeypatch, error):
    monkeypatch.setattr(
        gendata.pysim,
        "run_simulation",
        lambda _row: (_ for _ in ()).throw(error),
    )
    with pytest.raises(type(error), match=str(error)):
        gendata._simulate_one(np.zeros(len(pysim.PARAM_NAMES)))


def test_main_defaults_to_repository_dataset_directory(monkeypatch):
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(gendata, "generate", fake_generate)

    assert gendata.main(["--target", "1", "--workers", "1"]) == 0
    assert captured["out_path"] == os.fspath(
        gendata.ARTIFACTS_DIR / "datasets" / "training_dataset.npz"
    )
