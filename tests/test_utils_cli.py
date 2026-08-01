import numpy as np

from dnawalker.tools.check_npz import check_npz
from dnawalker.shared.artifacts import sha256_file
from dnawalker.tools.mat_to_npz import convert_mat_to_npz


def test_sha256_file_streams_exact_file_contents(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path, chunk_size=2) == (
        "ba7816bf8f01cfea414140de5dae2223"
        "b00361a396177a9cb410ff61f20015ad"
    )


def test_check_npz_reports_valid_dataset(tmp_path):
    path = tmp_path / "valid.npz"
    names = np.asarray(["a", "b"])
    np.savez(
        path,
        X=np.ones((2, 3, 4), dtype=np.float32),
        Y=np.ones((2, 2), dtype=np.float64),
        parameter_names=names,
    )
    assert check_npz(path) is True


def test_check_npz_rejects_missing_or_nonfinite_dataset(tmp_path):
    assert check_npz(tmp_path / "missing.npz") is False

    path = tmp_path / "nonfinite.npz"
    X = np.ones((1, 3, 4), dtype=np.float32)
    X[0, 0, 0] = np.nan
    np.savez(
        path,
        X=X,
        Y=np.ones((1, 1), dtype=np.float64),
        parameter_names=np.asarray(["a"]),
    )
    assert check_npz(path) is False


def test_mat_conversion_returns_failure_for_missing_input(tmp_path):
    assert convert_mat_to_npz(tmp_path / "missing.mat") is False


def test_mat_conversion_defaults_to_artifacts_and_writes_atomically(
    tmp_path, monkeypatch
):
    mat_path = tmp_path / "dataset.mat"
    mat_path.touch()
    dataset_dir = tmp_path / "artifacts" / "datasets"
    arrays = {
        "X_final": np.ones((2, 3, 4), dtype=np.float64),
        "Y_final": np.ones((2, 2), dtype=np.float64),
        "param_names": np.asarray(["a", "b"]),
    }
    monkeypatch.setattr(
        "dnawalker.tools.mat_to_npz.load_mat_v73",
        lambda _path: arrays,
    )
    monkeypatch.setattr(
        "dnawalker.tools.mat_to_npz._DEFAULT_DATASET_DIR",
        str(dataset_dir),
    )

    assert convert_mat_to_npz(mat_path) is True
    output = dataset_dir / "dataset.npz"
    assert output.exists()
    assert not (dataset_dir / "dataset.npz.tmp.npz").exists()
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["X"], arrays["X_final"])
        np.testing.assert_array_equal(data["Y"], arrays["Y_final"])
        np.testing.assert_array_equal(
            data["parameter_names"], arrays["param_names"]
        )


def test_mat_conversion_honors_explicit_output_path(tmp_path, monkeypatch):
    mat_path = tmp_path / "source" / "dataset.mat"
    mat_path.parent.mkdir()
    mat_path.touch()
    arrays = {
        "X_final": np.ones((1, 3, 4), dtype=np.float64),
        "Y_final": np.ones((1, 2), dtype=np.float64),
    }
    monkeypatch.setattr(
        "dnawalker.tools.mat_to_npz.load_mat_v73",
        lambda _path: arrays,
    )

    output = tmp_path / "artifacts" / "datasets" / "converted.npz"
    assert convert_mat_to_npz(mat_path, output) is True
    assert output.exists()
    assert not (output.parent / "converted.npz.tmp.npz").exists()
