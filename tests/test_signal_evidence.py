# coding=utf-8
"""Tests for reproducible signal-evidence calculations."""

import numpy as np
import pytest

from dnawalker.studies import signal_analysis as analyze_signal_evidence


def _sinusoid_curves(n_samples=2, n_time=256, cycles=8):
    time = np.arange(n_time, dtype=np.float64)
    signal = np.sin(2.0 * np.pi * cycles * time / n_time)
    return np.tile(signal, (n_samples, 3, 1))


def test_mean_power_spectrum_finds_known_sinusoid_frequency():
    curves = _sinusoid_curves()
    frequencies, power, cumulative, f95 = (
        analyze_signal_evidence.mean_power_spectrum(
            curves, sample_rate_hz=1.0, chunk_size=1
        )
    )
    expected = 8 / 256
    assert frequencies[np.argmax(power)] == pytest.approx(expected)
    assert f95 == pytest.approx(expected)
    assert cumulative[-1] == pytest.approx(1.0)


def test_autocorrelation_half_life_finds_all_curves():
    curves = _sinusoid_curves()
    half_lives, excluded_samples, excluded_channels = (
        analyze_signal_evidence.autocorrelation_half_lives(
            curves, threshold=0.5, chunk_size=1
        )
    )
    assert excluded_samples == 0
    assert excluded_channels == 0
    assert half_lives.shape == (curves.shape[0],)
    assert np.all((half_lives >= 5) & (half_lives <= 7))


def test_autocorrelation_excludes_zero_variance_curve():
    curves = _sinusoid_curves(n_samples=1)
    curves[0, 0] = 1.0
    half_lives, excluded_samples, excluded_channels = (
        analyze_signal_evidence.autocorrelation_half_lives(curves)
    )
    assert excluded_samples == 0
    assert excluded_channels == 1
    assert half_lives.size == 1


def test_open_npz_array_mmap_round_trips_without_pickle(tmp_path):
    expected = _sinusoid_curves(n_samples=1, n_time=32, cycles=2)
    path = tmp_path / "curves.npz"
    np.savez_compressed(path, X=expected)

    with analyze_signal_evidence.open_npz_array_mmap(path) as mapped:
        assert isinstance(mapped, np.memmap)
        np.testing.assert_array_equal(mapped, expected)
