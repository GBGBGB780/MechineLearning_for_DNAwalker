"""Pure statistical helpers for the multi-seed experiment."""

from numbers import Real

import numpy as np


def aggregate(per_seed_values):
    """Aggregate successful ``(value, ok)`` pairs with sample standard deviation.

    Failed entries never contribute.  The mean is available with one successful
    seed; sample standard deviation requires at least two.  A successful entry
    must carry a finite real metric so invalid results cannot silently turn the
    aggregate into NaN/Inf.
    """
    successes = []
    for index, (value, ok) in enumerate(per_seed_values):
        if not ok:
            continue
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)):
            raise ValueError(
                f"successful metric at index {index} must be a finite real "
                f"number, got {value!r}"
            )
        numeric_value = float(value)
        if not np.isfinite(numeric_value):
            raise ValueError(
                f"successful metric at index {index} must be finite, "
                f"got {value!r}"
            )
        successes.append(numeric_value)

    n_success = len(successes)
    values = np.asarray(successes, dtype=float)

    mean = float(np.mean(values)) if n_success else None
    if n_success >= 2:
        std = float(np.std(values, ddof=1))
        std_available = True
        insufficient_seeds = False
    else:
        std = None
        std_available = False
        insufficient_seeds = True

    return {
        "mean": mean,
        "std": std,
        "std_available": std_available,
        "insufficient_seeds": insufficient_seeds,
        "n_success": n_success,
    }
