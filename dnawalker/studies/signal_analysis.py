# coding=utf-8
"""Reproduce the training-curve spectrum and autocorrelation evidence figures.

The compressed NPZ member is extracted to a temporary ``.npy`` file and opened
with memory mapping. Calculations then run in chunks, so the 10,000-sample
dataset does not need to be expanded fully in RAM.
"""

import argparse
import contextlib
import os
import tempfile
import zipfile

import numpy as np

from dnawalker.paths import ARTIFACTS_DIR, RESULTS_DIR
from . import protocol as validation_common


_DEFAULT_DATASET = os.fspath(
    ARTIFACTS_DIR / "datasets" / "training_dataset.npz"
)
_DEFAULT_RESULTS_DIR = os.fspath(RESULTS_DIR / "validation" / "signal")


@contextlib.contextmanager
def open_npz_array_mmap(npz_path, key="X", temp_dir=None):
    """Yield one NPZ array as a read-only memory map."""
    member = f"{key}.npy"
    with tempfile.TemporaryDirectory(dir=temp_dir) as work_dir:
        with zipfile.ZipFile(npz_path) as archive:
            if member not in archive.namelist():
                raise KeyError(f"{npz_path!r} has no {member!r} member")
            archive.extract(member, path=work_dir)
        array = np.load(
            os.path.join(work_dir, member),
            mmap_mode="r",
            allow_pickle=False,
        )
        try:
            yield array
        finally:
            del array


def _validate_curves(curves):
    if curves.ndim != 3 or curves.shape[0] == 0 or curves.shape[1] != 3:
        raise ValueError(
            "curves must have shape (N, 3, T) with N,T > 0, got "
            f"{curves.shape}"
        )
    if curves.shape[2] < 2:
        raise ValueError("curves must contain at least two time points")


def mean_power_spectrum(curves, sample_rate_hz=1.0, chunk_size=64):
    """Return frequencies, mean one-sided FFT power, cumulative energy, and f95."""
    _validate_curves(curves)
    sample_rate_hz = validation_common.require_finite_real(
        sample_rate_hz,
        "sample_rate_hz",
        minimum=0.0,
        strict_minimum=True,
    )
    chunk_size = validation_common.require_int(
        chunk_size, "chunk_size", minimum=1
    )

    n_time = curves.shape[2]
    n_freq = n_time // 2 + 1
    power_sum = np.zeros(n_freq, dtype=np.float64)
    curve_count = 0

    for start in range(0, curves.shape[0], chunk_size):
        # Preserve the dataset dtype for centering. The reviewed evidence used
        # the stored float32 curves, so this also reproduces its tiny DC residue.
        batch = np.array(curves[start:start + chunk_size], copy=True)
        if not np.all(np.isfinite(batch)):
            raise ValueError(f"curves contain NaN/Inf in chunk starting at {start}")
        batch -= batch.mean(axis=-1, keepdims=True)
        spectrum = np.fft.rfft(batch, axis=-1)
        # Raw FFT power is reported in arbitrary units; only its relative
        # distribution is used for the cumulative-energy threshold.
        power = np.abs(spectrum) ** 2
        power_sum += power.sum(axis=(0, 1))
        curve_count += batch.shape[0] * batch.shape[1]

    mean_power = power_sum / curve_count
    total_power = float(mean_power.sum())
    if not np.isfinite(total_power) or total_power <= 0:
        raise ValueError("curves have no finite non-zero spectral energy")
    cumulative = np.cumsum(mean_power) / total_power
    frequencies = np.fft.rfftfreq(n_time, d=1.0 / sample_rate_hz)
    index_95 = min(int(np.searchsorted(cumulative, 0.95)), n_freq - 1)
    return frequencies, mean_power, cumulative, float(frequencies[index_95])


def autocorrelation_half_lives(curves, threshold=0.5, chunk_size=32):
    """Return per-sample median channel half-lives and exclusion counts."""
    _validate_curves(curves)
    threshold = validation_common.require_finite_real(
        threshold,
        "threshold",
        minimum=-1.0,
        maximum=1.0,
        strict_minimum=True,
        strict_maximum=True,
    )
    chunk_size = validation_common.require_int(
        chunk_size, "chunk_size", minimum=1
    )

    n_time = curves.shape[2]
    fft_size = 1 << (2 * n_time - 1).bit_length()
    half_lives = []
    excluded_samples = 0
    excluded_channels = 0

    for start in range(0, curves.shape[0], chunk_size):
        batch = np.asarray(
            curves[start:start + chunk_size], dtype=np.float64
        )
        if not np.all(np.isfinite(batch)):
            raise ValueError(f"curves contain NaN/Inf in chunk starting at {start}")
        flat = batch.reshape(-1, n_time)
        flat -= flat.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(flat, n=fft_size, axis=1)
        acf = np.fft.irfft(
            spectrum * spectrum.conj(), n=fft_size, axis=1
        )[:, :n_time]
        variance = acf[:, 0]
        valid_variance = np.isfinite(variance) & (variance > 0)
        normalized = np.full_like(acf, np.nan)
        normalized[valid_variance] = (
            acf[valid_variance] / variance[valid_variance, None]
        )
        crossed = normalized <= threshold
        has_crossing = valid_variance & np.any(crossed, axis=1)
        first = np.argmax(crossed, axis=1).astype(np.float64)
        first[~has_crossing] = np.nan
        first = first.reshape(batch.shape[0], batch.shape[1])
        excluded_channels += int(np.count_nonzero(~np.isfinite(first)))
        for row in first:
            valid = row[np.isfinite(row)]
            if valid.size:
                half_lives.append(float(np.median(valid)))
            else:
                excluded_samples += 1

    return (
        np.asarray(half_lives, dtype=np.float64),
        excluded_samples,
        excluded_channels,
    )


def _plot_spectrum(frequencies, power, cumulative, f95, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = frequencies <= 0.02
    fig, ax_power = plt.subplots(figsize=(10, 5.4))
    ax_energy = ax_power.twinx()
    power_line, = ax_power.plot(
        frequencies[mask],
        np.maximum(power[mask], np.finfo(np.float64).tiny),
        label="Mean one-sided FFT power",
    )
    cumulative_line, = ax_energy.plot(
        frequencies[mask],
        cumulative[mask],
        color="tab:orange",
        linewidth=2,
        label="Cumulative energy fraction",
    )
    f95_line = ax_power.axvline(
        f95,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"95% energy below {f95:.4f} Hz",
    )
    ax_energy.axhline(0.95, color="gray", linestyle=":", linewidth=1)
    ax_power.set_yscale("log")
    ax_power.set_xlabel("Frequency (Hz)")
    ax_power.set_ylabel("FFT power (a.u.)")
    ax_energy.set_ylabel("Cumulative energy fraction")
    ax_energy.set_ylim(0, 1.02)
    ax_power.set_title(
        "Power spectrum of the fluorescence curves "
        "(low-frequency dominated)"
    )
    lines = [power_line, cumulative_line, f95_line]
    ax_power.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_half_lives(half_lives, sequence_length, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    median = float(np.median(half_lives))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.hist(half_lives, bins=150, color="tab:blue", alpha=0.9)
    ax.axvline(
        median,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median = {median:.0f} s",
    )
    ax.axvline(
        sequence_length,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"Sequence length = {sequence_length} s",
    )
    ax.set_xlabel("Autocorrelation half-life (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Autocorrelation half-life across training curves")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_analysis(dataset_path, results_dir=_DEFAULT_RESULTS_DIR, chunk_size=32,
                 temp_dir=None):
    """Generate both evidence figures and a machine-readable metrics file."""
    chunk_size = validation_common.require_int(
        chunk_size, "chunk_size", minimum=1
    )
    os.makedirs(results_dir, exist_ok=True)

    with open_npz_array_mmap(dataset_path, "X", temp_dir=temp_dir) as curves:
        _validate_curves(curves)
        frequencies, power, cumulative, f95 = mean_power_spectrum(
            curves, chunk_size=chunk_size
        )
        half_lives, excluded_samples, excluded_channels = (
            autocorrelation_half_lives(
            curves, chunk_size=chunk_size
            )
        )
        if half_lives.size == 0:
            raise ValueError("no curve produced an autocorrelation crossing")

        spectrum_path = os.path.join(results_dir, "evidence_spectrum.png")
        autocorr_path = os.path.join(results_dir, "evidence_autocorr.png")
        _plot_spectrum(
            frequencies, power, cumulative, f95, spectrum_path
        )
        _plot_half_lives(half_lives, curves.shape[2], autocorr_path)

        metrics = {
            "experiment": "training_curve_signal_evidence",
            "dataset_path": os.path.normpath(dataset_path),
            "dataset_shape": [int(value) for value in curves.shape],
            "sample_rate_hz": 1.0,
            "spectrum_95pct_frequency_hz": f95,
            "autocorrelation_threshold": 0.5,
            "autocorrelation_half_life_median_seconds": float(
                np.median(half_lives)
            ),
            "autocorrelation_samples_used": int(half_lives.size),
            "autocorrelation_samples_excluded": int(excluded_samples),
            "autocorrelation_channels_excluded": int(excluded_channels),
            "algorithm": {
                "spectrum": "mean raw one-sided FFT power after per-curve centering",
                "autocorrelation": (
                    "FFT autocorrelation after per-channel centering; first lag "
                    "at or below threshold, then median across three channels "
                    "per training sample"
                ),
            },
        }

    validation_common.write_json(
        os.path.join(results_dir, "evidence_signal_metrics.json"),
        metrics,
    )
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce spectrum and autocorrelation evidence figures."
    )
    parser.add_argument(
        "--dataset",
        default=_DEFAULT_DATASET,
        help=(
            "NPZ dataset containing X (default: repository "
            "artifacts/datasets/training_dataset.npz)."
        ),
    )
    parser.add_argument(
        "--results-dir",
        default=_DEFAULT_RESULTS_DIR,
        help=(
            "Output directory for PNG/JSON artifacts (default: repository "
            "results/validation/signal)."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Number of samples processed per FFT chunk (default: 32).",
    )
    parser.add_argument(
        "--temp-dir",
        default=None,
        help="Optional directory with at least 1 GiB free for NPZ extraction.",
    )
    args = parser.parse_args(argv)
    metrics = run_analysis(
        args.dataset,
        results_dir=args.results_dir,
        chunk_size=args.chunk_size,
        temp_dir=args.temp_dir,
    )
    print(
        "Wrote evidence figures; "
        f"f95={metrics['spectrum_95pct_frequency_hz']:.6f} Hz, "
        "autocorrelation median="
        f"{metrics['autocorrelation_half_life_median_seconds']:.1f} s"
    )
    return metrics


def cli(argv=None):
    """Console-script wrapper returning a process exit status."""
    main(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
