# coding=utf-8
"""Computation-only verification tests for the validation-experiments suite.

This module collects lightweight *smoke* checks (not Hypothesis property tests)
that verify structural/operational constraints the property tests cannot:

* The forward-simulation Call_Counter depends only on the Python standard
  library (Requirement 1.7) — see ``TestCounterStdlibOnly`` below.

Later tasks append further computation-only checks (e.g. the no-external-data
constraint, Requirements 5.1/5.2/5.3) to this same file. Each concern lives in
its own ``Test*`` class so additions stay isolated and easy to extend.

Run with::

    .venv/bin/python -m pytest tests/test_computation_only.py
"""

import threading

import pysim


# ---------------------------------------------------------------------------
# Requirement 1.7 — the Call_Counter uses only the Python standard library.
# ---------------------------------------------------------------------------
class TestCounterStdlibOnly:
    """Smoke checks that the call-counter API is stdlib-only and functional."""

    def test_counter_api_is_exposed(self):
        """pysim exposes the three counter functions as callables."""
        assert callable(pysim.get_call_count)
        assert callable(pysim.reset_call_count)
        assert callable(pysim._increment_call_count)

    def test_reset_then_read_is_zero(self):
        """reset_call_count() drives the count to a non-negative int 0."""
        pysim.reset_call_count()
        count = pysim.get_call_count()
        assert isinstance(count, int)
        assert count == 0

    def test_increment_raises_count_by_one(self):
        """_increment_call_count() raises the recorded count by exactly 1."""
        pysim.reset_call_count()
        before = pysim.get_call_count()
        pysim._increment_call_count()
        after = pysim.get_call_count()
        assert after - before == 1

    def test_run_simulation_increments_count_by_one(self):
        """A single run_simulation call raises the count by exactly 1.

        The counting contract is independent of the simulator numerics, so a
        fast stub stands in for the heavy ``_run_simulation_impl`` body. This
        keeps the smoke test cheap while still exercising the real public
        ``run_simulation`` entry point and its ``try/finally`` increment.
        """
        import numpy as np

        def _fast_stub(_params, _fixed_params=None):
            return np.zeros((3, pysim.NUM_RESULTS)), 1.0

        original = pysim._run_simulation_impl
        pysim._run_simulation_impl = _fast_stub
        try:
            pysim.reset_call_count()
            before = pysim.get_call_count()
            pysim.run_simulation({})
            after = pysim.get_call_count()
        finally:
            pysim._run_simulation_impl = original

        assert after - before == 1

    def test_lock_is_stdlib_threading_lock(self):
        """The counter is guarded by a stdlib ``threading.Lock``.

        ``threading.Lock`` is a factory (not a class), so an ``isinstance``
        check against it does not work. Instead assert the module-level lock
        exposes the stdlib lock protocol (``acquire``/``release`` and the
        context-manager methods) and behaves like a real lock — proving the
        synchronization primitive comes from the standard library ``threading``
        module rather than any third-party package.
        """
        lock = pysim._call_counter_lock

        # The lock must be the standard library lock type.
        assert type(lock) is type(threading.Lock())

        # And it must expose the stdlib lock protocol.
        for attr in ("acquire", "release", "__enter__", "__exit__"):
            assert hasattr(lock, attr)

        # It actually works as a context manager (acquire/release round-trips).
        with lock:
            pass

    def test_counter_does_not_require_third_party_packages(self):
        """The counting functions run without numpy/torch/scipy imported.

        The Call_Counter must be stdlib-only (Requirement 1.7), so the counting
        operations themselves must not depend on any third-party numerical
        package. This removes ``numpy``, ``torch`` and ``scipy`` from
        ``sys.modules`` for the duration of the call and confirms the counter
        API still executes correctly, then restores them.
        """
        import sys

        third_party = ("numpy", "torch", "scipy")
        saved = {name: sys.modules.get(name) for name in third_party}
        # Also stash any submodules (e.g. ``scipy.stats``) so removal is clean.
        saved_submodules = {
            name: mod
            for name, mod in list(sys.modules.items())
            if name.split(".")[0] in third_party
        }
        for name in saved_submodules:
            del sys.modules[name]

        try:
            pysim.reset_call_count()
            assert pysim.get_call_count() == 0
            pysim._increment_call_count()
            assert pysim.get_call_count() == 1
            pysim.reset_call_count()
            assert pysim.get_call_count() == 0
        finally:
            # Restore whatever was present before the test ran.
            for name, mod in saved_submodules.items():
                sys.modules[name] = mod
            for name, mod in saved.items():
                if mod is not None:
                    sys.modules[name] = mod

    def test_counter_functions_defined_in_pysim(self):
        """The counter functions are defined in the pysim module itself."""
        assert pysim.get_call_count.__module__ == "pysim"
        assert pysim.reset_call_count.__module__ == "pysim"
        assert pysim._increment_call_count.__module__ == "pysim"


# ---------------------------------------------------------------------------
# Requirements 5.1 / 5.2 / 5.3 — the Validation_Suite is computation-only:
# every ground-truth curve is generated by the DNA_Walker_Simulator (5.1), the
# suite needs no external/wet-lab data (5.2), and it reuses the existing pysim /
# refine / Inverse_Model kernel as its computational basis (5.3).
#
# These are lightweight smoke/import checks (NOT Hypothesis). They use the REAL
# ``pysim`` with a tiny sample count, so they are a little slower than the
# counter checks above but still fast.
#
# No-external-data verification strategy (documented choice)
# ----------------------------------------------------------
# Two complementary, robust checks are used so the constraint is covered from
# both directions:
#
#   1. **Loader-raiser (primary, behavioral).** The experimental-data loader
#      ``exp_data_io.load_experimental_curves`` is monkeypatched to raise the
#      instant it is invoked. A tiny ``generate_ground_truth`` /
#      ``make_target_curves`` run is then executed and must *succeed* — proving
#      the ground-truth curves are produced purely via ``pysim.run_simulation``
#      and that the wet-lab Excel loader is never touched (Req 5.1, 5.2). A
#      module-level "was it called?" flag makes an accidental call impossible to
#      miss even if some path swallowed the exception.
#
#   2. **Structural (supporting).** Importing the three experiment modules must
#      not pull ``exp_data_io`` (nor the Excel stack it needs, ``pandas``) into
#      ``sys.modules`` — i.e. none of the three modules imports the experimental
#      loader at all (Req 5.2).
# ---------------------------------------------------------------------------
class TestNoExternalData:
    """Computation-only smoke/import checks (Requirements 5.1, 5.2, 5.3)."""

    # The three experiment scripts plus their shared dependencies must all be
    # importable for the suite to run computation-only.
    _EXPERIMENT_MODULES = (
        "validate_recovery",
        "benchmark_initguess",
        "multiseed_retrain",
    )

    def test_experiment_scripts_import_cleanly(self):
        """The three experiment scripts (and their deps) import without error.

        ``import validate_recovery`` / ``benchmark_initguess`` /
        ``multiseed_retrain`` — together with ``validation_common`` and
        ``pysim`` — must succeed, since the whole suite is built on top of the
        existing computational kernel (Requirement 5.3).
        """
        import importlib

        import pysim
        import validation_common

        assert pysim is not None
        assert validation_common is not None

        for name in self._EXPERIMENT_MODULES:
            module = importlib.import_module(name)
            assert module is not None, f"failed to import {name!r}"

    def test_generate_ground_truth_uses_no_experimental_loader(self, monkeypatch):
        """generate_ground_truth produces curves via pysim, never the wet-lab loader.

        Patches ``exp_data_io.load_experimental_curves`` to raise on first call,
        then runs a tiny ``generate_ground_truth`` with the REAL ``pysim`` and a
        small draw budget. The run must succeed and retain a valid sample while
        the experimental loader is never invoked (Requirements 5.1, 5.2).
        """
        import exp_data_io
        import validate_recovery
        import validation_common

        called = {"hit": False}

        def _raiser(*_args, **_kwargs):
            called["hit"] = True
            raise AssertionError(
                "exp_data_io.load_experimental_curves must not be called by the "
                "computation-only recovery experiment (Requirements 5.1, 5.2)."
            )

        monkeypatch.setattr(exp_data_io, "load_experimental_curves", _raiser)

        ranges = validation_common.load_configured_ranges()
        # Tiny, fast run on the real simulator: 1 retained sample, small budget.
        Y_true, X_curves, n_excluded, exhausted = (
            validate_recovery.generate_ground_truth(1, ranges, seed=42,
                                                     max_total_draws=10)
        )

        # The experimental loader was never touched (Requirements 5.1, 5.2).
        assert called["hit"] is False
        # Ground truth came purely from pysim: one valid sample with the
        # simulator's (3, 7801) curve shape (Requirement 5.1).
        assert not exhausted
        assert Y_true.shape == (1, len(validation_common.PARAM_NAMES))
        assert X_curves.shape == (1, 3, pysim.NUM_RESULTS)

    def test_make_target_curves_uses_no_experimental_loader(self, monkeypatch):
        """make_target_curves produces curves via pysim, never the wet-lab loader.

        Same loader-raiser guard as the recovery check, applied to
        ``benchmark_initguess.make_target_curves``: a tiny run on the REAL
        ``pysim`` must succeed and yield a valid target curve set without ever
        invoking the experimental-data loader (Requirements 5.1, 5.2).
        """
        import benchmark_initguess
        import exp_data_io
        import validation_common

        called = {"hit": False}

        def _raiser(*_args, **_kwargs):
            called["hit"] = True
            raise AssertionError(
                "exp_data_io.load_experimental_curves must not be called by the "
                "computation-only warm-start experiment (Requirements 5.1, 5.2)."
            )

        monkeypatch.setattr(exp_data_io, "load_experimental_curves", _raiser)

        ranges = validation_common.load_configured_ranges()
        targets = benchmark_initguess.make_target_curves(1, ranges, seed=42)

        # The experimental loader was never touched (Requirements 5.1, 5.2).
        assert called["hit"] is False
        # Target curves came purely from pysim: one (truth_params, curves) pair
        # with the simulator's (3, 7801) curve shape (Requirement 5.1).
        assert len(targets) == 1
        truth_params, curves = targets[0]
        assert set(truth_params.keys()) == set(validation_common.PARAM_NAMES)
        assert curves.shape == (3, pysim.NUM_RESULTS)

    def test_experiment_modules_do_not_import_experimental_loader(self):
        """Importing the three scripts pulls in no wet-lab data stack (structural).

        Supporting structural check for Requirement 5.2: a fresh interpreter
        that imports only the three experiment modules must not have
        ``exp_data_io`` (nor ``pandas``, the Excel reader it depends on) in
        ``sys.modules`` — i.e. none of the modules imports the experimental
        loader, even indirectly. Run in a subprocess so the assertion is not
        contaminated by other tests in this session that may have imported the
        loader.
        """
        import os
        import subprocess
        import sys

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        code = (
            "import sys\n"
            "import validate_recovery, benchmark_initguess, multiseed_retrain\n"
            "assert 'exp_data_io' not in sys.modules, "
            "'exp_data_io was imported by the experiment modules'\n"
            "assert 'pandas' not in sys.modules, "
            "'pandas (Excel reader) was imported by the experiment modules'\n"
            "print('OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            "experiment modules transitively imported the experimental-data "
            f"loader.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
        assert "OK" in result.stdout

    def test_suite_reuses_existing_kernel(self):
        """The suite reuses pysim / refine / Inverse_Model as its basis (Req 5.3).

        Asserts the experiment modules build on the existing computational
        kernel rather than re-implementing it:

          * ``validate_recovery`` and ``benchmark_initguess`` reference the real
            ``pysim`` (with ``run_simulation``) and the real ``refine`` module
            (with ``refine`` / ``curve_rmse``);
          * ``validation_common.PARAM_NAMES`` *is* ``pysim.PARAM_NAMES`` (the
            shared seven-parameter contract); and
          * the predictor factory exists, exposing the Inverse_Model basis.
        """
        import benchmark_initguess
        import pysim
        import refine
        import validate_recovery
        import validation_common

        # Same physics kernel object, not a copy (Requirement 5.3).
        assert validate_recovery.pysim is pysim
        assert validate_recovery.refine is refine
        assert benchmark_initguess.pysim is pysim
        assert benchmark_initguess.refine is refine

        # The kernel entry points the experiments route through.
        assert callable(pysim.run_simulation)
        assert callable(refine.refine)
        assert callable(refine.curve_rmse)

        # The shared seven-parameter contract is pysim's, reused verbatim.
        assert validation_common.PARAM_NAMES is pysim.PARAM_NAMES

        # The Inverse_Model basis is reused via the shared predictor factory.
        assert callable(validation_common.make_predictor)
