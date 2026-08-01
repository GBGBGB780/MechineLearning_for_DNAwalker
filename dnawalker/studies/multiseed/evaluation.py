"""Validated outcome records for one multi-seed test-set evaluation."""

import math
from dataclasses import dataclass
from numbers import Integral, Real


@dataclass(frozen=True)
class EvalOutcome:
    """Metric and sample-accounting result for one evaluated checkpoint."""

    curve_rmse_mean: float | None
    n_test_samples: int
    n_valid: int
    n_invalid: int
    n_extreme: int
    error: str | None
    provenance: dict | None = None

    def __post_init__(self):
        counts = {
            "n_test_samples": self.n_test_samples,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "n_extreme": self.n_extreme,
        }
        for name, value in counts.items():
            if (isinstance(value, bool)
                    or not isinstance(value, Integral)
                    or value < 0):
                raise ValueError(
                    f"{name} must be a non-negative integer, got {value!r}"
                )
            object.__setattr__(self, name, int(value))

        accounted = self.n_valid + self.n_invalid + self.n_extreme
        if accounted != self.n_test_samples:
            raise ValueError(
                "evaluation sample counts do not sum to n_test_samples: "
                f"{accounted} vs {self.n_test_samples}"
            )

        metric = self.curve_rmse_mean
        if metric is not None:
            if (isinstance(metric, bool)
                    or not isinstance(metric, Real)
                    or not math.isfinite(float(metric))):
                raise ValueError(
                    "curve_rmse_mean must be finite or None, "
                    f"got {metric!r}"
                )
            if self.n_valid == 0:
                raise ValueError(
                    "a finite curve_rmse_mean requires at least one valid sample"
                )
            if self.error is not None:
                raise ValueError(
                    "a successful evaluation cannot carry an error"
                )
            object.__setattr__(self, "curve_rmse_mean", float(metric))
        elif not isinstance(self.error, str) or not self.error.strip():
            raise ValueError(
                "a failed evaluation must carry a non-empty error"
            )

    @property
    def ok(self):
        """Whether this evaluation produced a finite aggregate metric."""
        return self.curve_rmse_mean is not None

    def count_fields(self):
        """Return the JSON-ready sample-accounting fields."""
        return {
            "n_test_samples": self.n_test_samples,
            "n_valid": self.n_valid,
            "n_invalid": self.n_invalid,
            "n_extreme": self.n_extreme,
        }
