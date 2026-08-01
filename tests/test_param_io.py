# coding=utf-8
"""Dataset parameter-metadata regression tests."""

import numpy as np
import pytest

from dnawalker.physics import pysim
from dnawalker.shared.parameters import (
    load_npz_dataset,
    resolve_checkpoint_param_names,
    vector_to_param_dict,
)


def _arrays():
    X = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    Y = np.vstack([
        np.arange(len(pysim.PARAM_NAMES), dtype=np.float64),
        np.arange(len(pysim.PARAM_NAMES), dtype=np.float64) + 10,
    ])
    return X, Y


def test_npz_parameter_columns_are_reordered_by_name(tmp_path):
    X, Y = _arrays()
    path = tmp_path / "reordered.npz"
    np.savez(
        path,
        X=X,
        Y=Y,
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )
    expected = list(reversed([name.lower() for name in pysim.PARAM_NAMES]))

    loaded_x, loaded_y, names = load_npz_dataset(path, expected)

    np.testing.assert_array_equal(loaded_x, X)
    np.testing.assert_array_equal(loaded_y, Y[:, ::-1].astype(np.float32))
    assert names == expected


def test_npz_without_metadata_requires_explicit_legacy_opt_in(tmp_path):
    X, Y = _arrays()
    path = tmp_path / "legacy.npz"
    np.savez(path, X=X, Y=Y)

    with pytest.raises(ValueError, match="parameter_names"):
        load_npz_dataset(path, pysim.PARAM_NAMES)

    _, loaded_y, _ = load_npz_dataset(
        path, pysim.PARAM_NAMES, allow_legacy_canonical=True
    )
    np.testing.assert_array_equal(loaded_y, Y.astype(np.float32))


@pytest.mark.parametrize(
    "names",
    [
        list(pysim.PARAM_NAMES[:-1]) + [pysim.PARAM_NAMES[0]],
        list(pysim.PARAM_NAMES[:-1]) + ["not_a_parameter"],
    ],
)
def test_npz_rejects_duplicate_or_unknown_parameter_names(tmp_path, names):
    X, Y = _arrays()
    path = tmp_path / "bad_names.npz"
    np.savez(path, X=X, Y=Y, parameter_names=np.asarray(names))

    with pytest.raises(ValueError, match="duplicate|不一致|do not match"):
        load_npz_dataset(path, pysim.PARAM_NAMES)


def test_npz_rejects_shape_and_metadata_mismatch(tmp_path):
    X, Y = _arrays()
    path = tmp_path / "bad_shape.npz"
    np.savez(
        path,
        X=X,
        Y=Y[:, :-1],
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )

    with pytest.raises(ValueError, match="mismatch|不一致"):
        load_npz_dataset(path, pysim.PARAM_NAMES)


def test_prediction_vector_is_mapped_by_parameter_name():
    reordered = list(reversed([name.lower() for name in pysim.PARAM_NAMES]))
    values = np.arange(len(reordered), dtype=np.float64)

    mapped = vector_to_param_dict(values, reordered)

    assert list(mapped) == list(reversed(pysim.PARAM_NAMES))
    for index, configured_name in enumerate(reordered):
        canonical = next(
            name for name in pysim.PARAM_NAMES
            if name.lower() == configured_name
        )
        assert mapped[canonical] == index


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((1, len(pysim.PARAM_NAMES))),
        np.zeros(len(pysim.PARAM_NAMES) - 1),
    ],
)
def test_prediction_vector_rejects_shape_mismatch(values):
    with pytest.raises(ValueError, match="mismatch|不一致"):
        vector_to_param_dict(values, pysim.PARAM_NAMES)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_prediction_vector_rejects_nonfinite_values(bad_value):
    values = np.zeros(len(pysim.PARAM_NAMES))
    values[2] = bad_value
    with pytest.raises(ValueError, match="NaN|Inf|non-finite"):
        vector_to_param_dict(values, pysim.PARAM_NAMES)


def test_checkpoint_parameter_metadata_is_authoritative():
    checkpoint_names = list(reversed(pysim.PARAM_NAMES))
    configured_names = [name.lower() for name in pysim.PARAM_NAMES]
    assert resolve_checkpoint_param_names(
        {"param_names": checkpoint_names}, configured_names
    ) == checkpoint_names


def test_legacy_checkpoint_rejects_reordered_config():
    with pytest.raises(ValueError, match="legacy checkpoint|旧 checkpoint"):
        resolve_checkpoint_param_names(
            {"model_state": {}},
            list(reversed(pysim.PARAM_NAMES)),
        )


def test_npz_rejects_empty_dataset(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez(
        path,
        X=np.empty((0, 3, 4), dtype=np.float32),
        Y=np.empty((0, len(pysim.PARAM_NAMES)), dtype=np.float64),
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )
    with pytest.raises(ValueError, match="empty|不能为空"):
        load_npz_dataset(path, pysim.PARAM_NAMES)


def test_npz_rejects_curve_shape_mismatch(tmp_path):
    X, Y = _arrays()
    path = tmp_path / "wrong_curve_shape.npz"
    np.savez(
        path,
        X=X,
        Y=Y,
        parameter_names=np.asarray(pysim.PARAM_NAMES),
    )
    with pytest.raises(ValueError, match="shape|形状"):
        load_npz_dataset(
            path,
            pysim.PARAM_NAMES,
            expected_x_shape=(3, 5),
        )
