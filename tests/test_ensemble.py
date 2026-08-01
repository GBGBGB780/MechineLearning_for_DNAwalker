import numpy as np
import pytest

from dnawalker.shared.ensemble import ensemble_predict


class _Predictor:
    def __init__(self, output=None):
        self.output = output
        self.last_input = None

    def predict(self, curves):
        self.last_input = np.asarray(curves)
        rows = 1 if self.last_input.ndim == 2 else self.last_input.shape[0]
        if self.output is not None:
            return self.output
        return np.arange(rows * 2, dtype=np.float64).reshape(rows, 2)


def test_single_prediction_preserves_predictor_contract():
    predictor = _Predictor()
    median, all_predictions = ensemble_predict(
        predictor,
        np.ones((3, 4)),
        ensemble=0,
        return_all=True,
    )
    assert predictor.last_input.shape == (3, 4)
    assert all_predictions.shape == (1, 2)
    np.testing.assert_array_equal(median, all_predictions[0])


def test_ensemble_batches_clean_and_noisy_samples():
    predictor = _Predictor()
    median, all_predictions = ensemble_predict(
        predictor,
        np.ones((3, 4)),
        ensemble=4,
        noise_std=0.01,
        seed=7,
        return_all=True,
    )
    assert predictor.last_input.shape == (4, 3, 4)
    np.testing.assert_array_equal(predictor.last_input[0], np.ones((3, 4)))
    assert all_predictions.shape == (4, 2)
    np.testing.assert_array_equal(median, np.median(all_predictions, axis=0))


@pytest.mark.parametrize("ensemble", [-1, 1.5, True, "3"])
def test_rejects_invalid_ensemble_size(ensemble):
    with pytest.raises(ValueError, match="ensemble"):
        ensemble_predict(_Predictor(), np.ones((3, 4)), ensemble=ensemble)


@pytest.mark.parametrize("noise_std", [-0.1, np.nan, np.inf, "bad"])
def test_rejects_invalid_noise_standard_deviation(noise_std):
    with pytest.raises(ValueError, match="noise_std"):
        ensemble_predict(
            _Predictor(),
            np.ones((3, 4)),
            ensemble=2,
            noise_std=noise_std,
        )


@pytest.mark.parametrize(
    "curves",
    [
        np.ones(4),
        np.ones((1, 3, 4)),
        np.empty((3, 0)),
        np.full((3, 4), np.nan),
    ],
)
def test_rejects_invalid_curve_input(curves):
    with pytest.raises(ValueError, match="shape|NaN|Inf"):
        ensemble_predict(_Predictor(), curves)


@pytest.mark.parametrize(
    "output",
    [
        np.ones(2),
        np.ones((2, 2)),
        np.empty((1, 0)),
        np.array([[np.nan, 0.0]]),
    ],
)
def test_rejects_invalid_predictor_output(output):
    with pytest.raises(ValueError, match="shape|NaN|Inf"):
        ensemble_predict(_Predictor(output), np.ones((3, 4)))
