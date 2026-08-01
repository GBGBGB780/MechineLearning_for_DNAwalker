"""Structural checks for the canonical forward-simulation counter."""

import sys
import threading

import numpy as np

from dnawalker.physics import pysim


def test_canonical_module_is_single_source_of_truth():
    from dnawalker.physics import pysim as package_pysim

    assert pysim is package_pysim
    assert pysim.PARAM_NAMES is package_pysim.PARAM_NAMES
    assert pysim._call_counter_lock is package_pysim._call_counter_lock


def test_counter_api_is_exposed():
    assert callable(pysim.get_call_count)
    assert callable(pysim.reset_call_count)
    assert callable(pysim._increment_call_count)


def test_reset_then_read_is_zero():
    pysim.reset_call_count()
    count = pysim.get_call_count()
    assert isinstance(count, int)
    assert count == 0


def test_increment_raises_count_by_one():
    pysim.reset_call_count()
    before = pysim.get_call_count()
    pysim._increment_call_count()
    assert pysim.get_call_count() - before == 1


def test_run_simulation_increments_count_by_one(monkeypatch):
    monkeypatch.setattr(
        pysim,
        "_run_simulation_impl",
        lambda _params, _fixed=None: (
            np.zeros((3, pysim.NUM_RESULTS)),
            1.0,
        ),
    )
    pysim.reset_call_count()
    before = pysim.get_call_count()
    pysim.run_simulation({})
    assert pysim.get_call_count() - before == 1


def test_lock_is_stdlib_threading_lock():
    lock = pysim._call_counter_lock
    assert type(lock) is type(threading.Lock())
    for attribute in ("acquire", "release", "__enter__", "__exit__"):
        assert hasattr(lock, attribute)
    with lock:
        pass


def test_counter_does_not_require_third_party_packages():
    third_party = ("numpy", "torch", "scipy")
    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name.split(".")[0] in third_party
    }
    for name in saved:
        del sys.modules[name]

    try:
        pysim.reset_call_count()
        pysim._increment_call_count()
        assert pysim.get_call_count() == 1
    finally:
        sys.modules.update(saved)


def test_counter_functions_defined_in_canonical_module():
    expected_module = "dnawalker.physics.simulator"
    assert pysim.get_call_count.__module__ == expected_module
    assert pysim.reset_call_count.__module__ == expected_module
    assert pysim._increment_call_count.__module__ == expected_module
