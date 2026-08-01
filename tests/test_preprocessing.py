import numpy as np
import pytest

from dnawalker.data.preprocessing import (
    normalize_per_sample,
    prepare_labels_and_sample_mask,
)


def test_normalization_default_does_not_mutate_input():
    curves = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    original = curves.copy()

    normalized = normalize_per_sample(curves)

    np.testing.assert_array_equal(curves, original)
    assert normalized.dtype == curves.dtype
    np.testing.assert_allclose(
        normalized.mean(axis=(1, 2)),
        np.zeros(2),
        atol=1e-6,
    )


def test_in_place_normalization_matches_copying_path():
    curves = np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5)
    expected = normalize_per_sample(curves)
    working = curves.copy()

    actual = normalize_per_sample(working, copy=False)

    assert actual is working
    np.testing.assert_array_equal(actual, expected)


def test_in_place_normalization_requires_writable_float_array():
    curves = np.ones((1, 3, 4), dtype=np.float32)
    curves.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        normalize_per_sample(curves, copy=False)


def test_integer_input_is_promoted():
    curves = np.arange(12).reshape(1, 3, 4)
    normalized = normalize_per_sample(curves)
    assert normalized.dtype == np.float64


def test_training_mask_combines_log_amplitude_and_finite_checks():
    curves = np.zeros((5, 2, 3), dtype=np.float32)
    curves[1, 0, 1] = 2.0
    curves[2, 0, 0] = np.nan
    labels = np.array(
        [
            [1e-3, 1.0],
            [1e-3, 1.0],
            [1e-3, 1.0],
            [-1e-9, 1.0],
            [1e-3, 1e30],
        ],
        dtype=np.float64,
    )

    transformed, amplitude_mask, retention_mask = (
        prepare_labels_and_sample_mask(
            curves,
            labels,
            ["K0", "energy"],
            log_transform_params=["k0"],
            log_epsilon=1e-9,
            amplitude_thresholds=[1.0, 1.0],
            safe_threshold=1e20,
            nan_replacement=-1e30,
        )
    )

    assert transformed.dtype == np.float32
    assert transformed[0, 0] == pytest.approx(-3.0, abs=1e-5)
    np.testing.assert_array_equal(
        amplitude_mask,
        [True, False, False, True, True],
    )
    np.testing.assert_array_equal(
        retention_mask,
        [True, False, False, False, False],
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"log_transform_params": ["unknown"]},
        {"log_epsilon": 0.0},
        {"safe_threshold": np.inf},
        {"nan_replacement": 0.0},
        {"amplitude_thresholds": [-1.0]},
    ],
)
def test_training_mask_rejects_invalid_configuration(kwargs):
    options = {
        "log_transform_params": [],
        "log_epsilon": 1e-9,
        "amplitude_thresholds": None,
        "safe_threshold": 1e20,
        "nan_replacement": -1e30,
    }
    options.update(kwargs)

    with pytest.raises(ValueError):
        prepare_labels_and_sample_mask(
            np.zeros((2, 1, 3), dtype=np.float32),
            np.ones((2, 1), dtype=np.float64),
            ["k0"],
            **options,
        )
