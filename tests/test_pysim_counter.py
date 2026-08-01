# coding=utf-8
"""Property-based tests for the pysim Forward-Simulation Call_Counter.

Feature: validation-experiments
Property 1: Forward simulation increments the counter exactly once on every
termination path.

These tests exercise the real physics simulator (``pysim.run_simulation``) and
are therefore SLOW relative to ordinary unit tests. They use Hypothesis with
``@settings(max_examples=100)``.

The Call_Counter is *process-global*: every test reads ``get_call_count()``
immediately before and after a single ``run_simulation`` call and asserts the
delta is exactly 1, rather than relying on an absolute count.
"""

import threading
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from dnawalker.physics import pysim
from dnawalker.studies import protocol as vc

# Configured_Ranges for the seven trainable parameters, loaded once.
_RANGES = vc.load_configured_ranges()
_PARAM_NAMES = pysim.PARAM_NAMES


def _in_range_params():
    """Hypothesis strategy: a Parameter_Set drawn within Configured_Ranges.

    Produces a dict keyed by ``pysim.PARAM_NAMES`` whose every value lies inside
    that parameter's ``[min, max]`` Configured_Range. These draws mostly hit the
    valid (``dt_used >= 0``) termination path of the simulator.
    """
    return st.fixed_dictionaries(
        {
            name: st.floats(
                min_value=float(lo),
                max_value=float(hi),
                allow_nan=False,
                allow_infinity=False,
            )
            for name, (lo, hi) in _RANGES.items()
        }
    )


def _invalid_params():
    """Hypothesis strategy: in-range params forced onto the invalid path.

    Every other parameter is drawn within its Configured_Range, but ``k0`` is
    set to an extreme value (``10**4 .. 10**12``) that drives the integrator's
    ``dt`` below ``MIN_DT``, so the simulator returns ``dt_used < 0`` (the
    invalid termination path).
    """
    base = {
        name: st.floats(
            min_value=float(lo),
            max_value=float(hi),
            allow_nan=False,
            allow_infinity=False,
        )
        for name, (lo, hi) in _RANGES.items()
    }
    # Override k0 with a large positive exponent to force dt_used < 0.
    base["k0"] = st.floats(min_value=4.0, max_value=12.0).map(lambda e: 10.0 ** e)
    return st.fixed_dictionaries(base)


# Feature: validation-experiments, Property 1: Forward simulation increments the
# counter exactly once on every termination path
@settings(max_examples=100, deadline=None)
@given(params=_in_range_params())
def test_counter_increments_once_on_valid_path(params):
    """Valid path (dt_used >= 0): one call raises the count by exactly 1."""
    before = pysim.get_call_count()
    signals, dt_used = pysim.run_simulation(params)
    after = pysim.get_call_count()

    assert after - before == 1
    # In-range draws exercise the normal (non-raising) termination path: the
    # simulator returns a (3, 7801) array and a float dt_used.
    assert isinstance(signals, np.ndarray)
    assert signals.shape == (3, pysim.NUM_RESULTS)
    assert isinstance(dt_used, float)


# Feature: validation-experiments, Property 1: Forward simulation increments the
# counter exactly once on every termination path
@settings(max_examples=100, deadline=None)
@given(params=_invalid_params())
def test_counter_increments_once_on_invalid_path(params):
    """Invalid path (dt_used < 0): one call raises the count by exactly 1."""
    before = pysim.get_call_count()
    signals, dt_used = pysim.run_simulation(params)
    after = pysim.get_call_count()

    assert after - before == 1
    # Extreme k0 forces the invalid sample sentinel: dt_used < 0 with zeros.
    assert dt_used < 0
    assert signals.shape == (3, pysim.NUM_RESULTS)


# Feature: validation-experiments, Property 1: Forward simulation increments the
# counter exactly once on every termination path
@settings(max_examples=100, deadline=None)
@given(params=_in_range_params())
def test_counter_increments_once_when_impl_raises(params):
    """Exception path (stubbed impl raises): the exception propagates AND the
    count still rises by exactly 1 (the try/finally guarantee, Requirement 1.8).

    Uses ``mock.patch`` inside the test body (rather than the function-scoped
    ``monkeypatch`` fixture) so it composes cleanly with Hypothesis' repeated
    example generation.
    """

    def _boom(_params, _fixed_params=None):
        raise RuntimeError("forced failure for counter test")

    with mock.patch.object(pysim, "_run_simulation_impl", _boom):
        before = pysim.get_call_count()
        with pytest.raises(RuntimeError):
            pysim.run_simulation(params)
        after = pysim.get_call_count()

    assert after - before == 1


# Feature: validation-experiments, Property 1: Forward simulation increments the
# counter exactly once on every termination path
@settings(max_examples=100, deadline=None)
@given(_=st.none())
def test_counter_increments_once_on_malformed_params(_):
    """Exception path (malformed params): passing None raises inside the impl,
    the exception propagates, and the count still rises by exactly 1.
    """
    before = pysim.get_call_count()
    with pytest.raises(Exception):
        pysim.run_simulation(None)
    after = pysim.get_call_count()

    assert after - before == 1


def test_reset_call_count_baseline():
    """Sanity check: reset_call_count() zeroes the process-global counter."""
    pysim.reset_call_count()
    assert pysim.get_call_count() == 0


# Feature: validation-experiments, Property 2: Reset returns the counter to zero and counting resumes from zero
@settings(max_examples=100, deadline=None)
@given(
    n=st.integers(min_value=0, max_value=20),
    m=st.integers(min_value=0, max_value=20),
)
def test_reset_zeroes_counter_and_counting_resumes(n, m):
    """Property 2: Reset returns the counter to zero and counting resumes from zero.

    After ``N`` calls to ``run_simulation``, ``reset_call_count()`` drives the
    process-global Call_Counter to 0; a subsequent ``M`` calls bring it to
    exactly ``M``; and ``reset_call_count()`` is idempotent (calling it twice
    still leaves the count at 0).

    The counting contract is independent of the simulator's numerics, so a fast
    stub replaces ``_run_simulation_impl`` to keep 100 examples cheap while still
    exercising the real public ``run_simulation`` entry point (and its
    ``try/finally`` increment).
    """

    def _fast_stub(_params, _fixed_params=None):
        # Mimic the valid-path return shape without running the real simulator.
        return np.zeros((3, pysim.NUM_RESULTS)), 1.0

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_stub):
        # N calls move the (process-global) counter to some value; its exact
        # pre-reset value is irrelevant because reset zeroes it unconditionally.
        for _ in range(n):
            pysim.run_simulation({})

        # Reset returns the counter to zero regardless of the prior count.
        pysim.reset_call_count()
        assert pysim.get_call_count() == 0

        # Idempotent: a second reset still leaves the count at 0.
        pysim.reset_call_count()
        assert pysim.get_call_count() == 0

        # Counting resumes from zero: M subsequent calls => count == M exactly.
        for _ in range(m):
            pysim.run_simulation({})
        assert pysim.get_call_count() == m


# Feature: validation-experiments, Property 3: Counting is thread-safe with no lost increments
@settings(max_examples=100, deadline=None)
@given(
    t=st.integers(min_value=2, max_value=8),
    k=st.integers(min_value=1, max_value=50),
)
def test_counting_is_thread_safe_with_no_lost_increments(t, k):
    """Property 3: Counting is thread-safe with no lost increments.

    Spawn ``T`` worker threads that each call ``run_simulation`` ``K`` times
    after a single reset. Because every increment is guarded by
    ``threading.Lock``, no concurrent increment may be lost, so the final
    ``get_call_count()`` must equal ``T * K`` exactly.

    The thread-safety contract is independent of the simulator's numerics, so a
    FAST stub replaces ``_run_simulation_impl`` to keep the 100-example
    contention test cheap and reliable while still exercising the real public
    ``run_simulation`` entry point and its ``try/finally`` increment. A
    ``threading.Barrier`` releases all workers simultaneously to maximize
    contention on the lock.
    """

    def _fast_stub(_params, _fixed_params=None):
        # Mimic the valid-path return shape without running the real simulator.
        return np.zeros((3, pysim.NUM_RESULTS)), 1.0

    with mock.patch.object(pysim, "_run_simulation_impl", _fast_stub):
        # Reset the process-global counter before spawning the workers.
        pysim.reset_call_count()

        # Barrier ensures all T threads start hammering the counter together,
        # maximizing lock contention so any lost-increment bug is exposed.
        barrier = threading.Barrier(t)

        def _worker():
            barrier.wait()
            for _ in range(k):
                pysim.run_simulation({})

        threads = [threading.Thread(target=_worker) for _ in range(t)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # No increments lost under concurrency: count is exactly T * K.
        assert pysim.get_call_count() == t * k


def _extreme_params():
    """Hypothesis strategy: finite-but-extreme/out-of-range Parameter_Sets.

    Every parameter is drawn from a wide finite range (``[-1e6, 1e6]``) that
    overlaps the Configured_Ranges but also reaches far outside them. These
    draws push the simulator onto degenerate/invalid termination paths (and the
    occasional valid one), exercising Property 4 on inputs the experiments would
    never normally produce. Values stay finite (``allow_nan``/``allow_infinity``
    are ``False``) so the comparison targets the simulator's own handling rather
    than NaN/inf inputs.
    """
    return st.fixed_dictionaries(
        {
            name: st.floats(
                min_value=-1.0e6,
                max_value=1.0e6,
                allow_nan=False,
                allow_infinity=False,
            )
            for name in _PARAM_NAMES
        }
    )


def _call_capture(fn, params):
    """Call ``fn(params)`` and capture either its return value or its exception.

    Returns ``("ok", signals, dt_used)`` on a normal return, or
    ``("raised", exception_type_name, None)`` if the call raised. This lets the
    Property 4 test compare the *behavior* of the counter-instrumented
    ``run_simulation`` against the uncounted ``_run_simulation_impl`` on the same
    input, including the case where extreme inputs cause both to raise.
    """
    try:
        signals, dt_used = fn(params)
        return ("ok", signals, dt_used)
    except Exception as exc:  # noqa: BLE001 - we intentionally capture any error
        return ("raised", type(exc).__name__, None)


# Feature: validation-experiments, Property 4: The counter does not alter simulation outputs
@settings(max_examples=100, deadline=None)
@given(
    params=st.one_of(
        _in_range_params(),
        _invalid_params(),
        _extreme_params(),
    )
)
def test_counter_does_not_alter_simulation_outputs(params):
    """Property 4: The counter does not alter simulation outputs.

    For any Parameter_Set — drawn within Configured_Ranges, forced onto the
    invalid path, or finite-but-extreme/out-of-range — the ``(signals, dt_used)``
    returned by the counter-instrumented public ``pysim.run_simulation`` are
    IDENTICAL to those returned by directly calling the uncounted
    ``pysim._run_simulation_impl`` on the same input.

    This is the critical property that the ``try/finally`` counter wrapper is
    purely observational and does not change the simulator's numerics
    (Requirement 1.4):

      - ``signals`` are array-equal via ``np.array_equal`` (which also covers the
        all-zeros sentinel returned on the invalid ``dt_used < 0`` path);
      - ``dt_used`` is value-equal (``-1.0`` on the invalid path, ``float(dt)``
        on the valid path).

    The simulator is deterministic for a given input, so two independent calls on
    the same Parameter_Set must agree. If an extreme input makes the
    implementation raise, the wrapper must raise the same exception type (its
    ``finally`` increment never swallows or alters the error).
    """
    # Direct (uncounted) implementation call.
    impl_status, impl_signals, impl_dt = _call_capture(
        pysim._run_simulation_impl, params
    )
    # Counter-instrumented public entry point (delegates to the impl).
    wrap_status, wrap_signals, wrap_dt = _call_capture(
        pysim.run_simulation, params
    )

    # Both must agree on whether the call returned normally or raised.
    assert impl_status == wrap_status

    if impl_status == "raised":
        # The same exception type propagates through the wrapper unchanged;
        # the counter's finally-block increment does not alter error behavior.
        assert impl_signals == wrap_signals  # exception type names
        return

    # signals array-equal — handles the all-zeros invalid sample identically.
    assert np.array_equal(impl_signals, wrap_signals)
    # dt_used value-equal — identical sentinel/value on every path.
    assert impl_dt == wrap_dt
