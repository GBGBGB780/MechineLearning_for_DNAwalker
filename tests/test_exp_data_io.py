import numpy as np
import pandas as pd
import pytest

from dnawalker.data import experimental as exp_data_io


def _patch_frame(monkeypatch, rows):
    frame = pd.DataFrame(
        rows,
        columns=["Time", "FAM signal", "TYE signal", "CY5 signal"],
    )
    monkeypatch.setattr(exp_data_io.pd, "read_excel", lambda _path: frame)


def test_loader_filters_nonfinite_sorts_and_averages_duplicate_times(monkeypatch):
    _patch_frame(
        monkeypatch,
        [
            [2.0, 20.0, 200.0, 2000.0],
            [1.0, 10.0, 100.0, 1000.0],
            [1.0, 14.0, 140.0, 1400.0],
            [np.inf, 99.0, 99.0, 99.0],
            [3.0, np.inf, 300.0, 3000.0],
            ["title", "bad", "bad", "bad"],
        ],
    )

    time, fam, tye, cy5 = exp_data_io.load_experimental_curves("unused.xlsx")

    np.testing.assert_array_equal(time, [1.0, 2.0])
    np.testing.assert_array_equal(fam, [12.0, 20.0])
    np.testing.assert_array_equal(tye, [120.0, 200.0])
    np.testing.assert_array_equal(cy5, [1200.0, 2000.0])


@pytest.mark.parametrize(
    "rows",
    [
        [["title", "bad", "bad", "bad"]],
        [[1.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]],
    ],
)
def test_loader_requires_two_unique_finite_time_points(monkeypatch, rows):
    _patch_frame(monkeypatch, rows)
    with pytest.raises(ValueError, match="有限|至少两个"):
        exp_data_io.load_experimental_curves("unused.xlsx")
