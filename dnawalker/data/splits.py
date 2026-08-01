"""Deterministic activity-stratified splits for learning-curve studies."""

from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from dnawalker.shared.artifacts import sha256_file


MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExplicitSplit:
    """Validated row indices selected by one split manifest."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    train_size: int
    split_seed: int
    manifest_path: str
    manifest_sha256: str
    dataset_sha256: str


def activity_scores(curves):
    """Return the generator's max per-channel signal change for each row."""
    values = np.asarray(curves)
    if values.ndim != 3 or values.shape[0] == 0:
        raise ValueError(
            f"curves must have shape (N,C,T) with N > 0, got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("curves contain NaN or Inf")
    return np.ptp(values, axis=2).max(axis=1)


def activity_bin_ids(curves, bin_edges):
    """Assign rows to the same right-open activity bins as data generation."""
    edges = np.asarray(bin_edges, dtype=np.float64)
    if (
        edges.ndim != 1
        or edges.size < 2
        or np.any(np.isnan(edges))
        or np.any(np.diff(edges) <= 0)
    ):
        raise ValueError(
            "bin_edges must be a one-dimensional, strictly increasing array"
        )
    scores = activity_scores(curves)
    bins = np.searchsorted(edges, scores, side="right") - 1
    n_bins = edges.size - 1
    invalid = (bins < 0) | (bins >= n_bins)
    if np.any(invalid):
        raise ValueError(
            f"{int(invalid.sum())} rows fall outside the activity-bin edges"
        )
    return bins.astype(np.int64, copy=False), scores


def _balanced_count(total, n_bins, name):
    if isinstance(total, (bool, np.bool_)) or not isinstance(
        total, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer, got {total!r}")
    total = int(total)
    if total <= 0 or total % n_bins:
        raise ValueError(
            f"{name} must be positive and divisible by {n_bins}, got {total}"
        )
    return total // n_bins


def build_nested_activity_splits(
    curves,
    bin_edges,
    *,
    train_sizes=(8_000, 16_000, 24_000),
    val_size=3_000,
    test_size=3_000,
    split_seed=42,
):
    """Build balanced fixed test/validation sets and nested training subsets."""
    if isinstance(split_seed, bool) or not isinstance(
        split_seed, (int, np.integer)
    ) or not 0 <= int(split_seed) <= 2**32 - 1:
        raise ValueError(f"split_seed must be a uint32 integer, got {split_seed!r}")
    split_seed = int(split_seed)

    bins, scores = activity_bin_ids(curves, bin_edges)
    n_samples = int(bins.size)
    n_bins = int(np.asarray(bin_edges).size - 1)
    val_per_bin = _balanced_count(val_size, n_bins, "val_size")
    test_per_bin = _balanced_count(test_size, n_bins, "test_size")

    sizes = []
    for raw_size in train_sizes:
        _balanced_count(raw_size, n_bins, "train_size")
        size = int(raw_size)
        if size in sizes:
            raise ValueError(f"train_sizes must be distinct, got {train_sizes}")
        sizes.append(size)
    if sizes != sorted(sizes):
        raise ValueError(f"train_sizes must be increasing, got {sizes}")

    rng = np.random.default_rng(split_seed)
    test_by_bin = []
    val_by_bin = []
    pool_by_bin = []
    bin_counts = np.bincount(bins, minlength=n_bins)
    max_train_per_bin = sizes[-1] // n_bins
    required_per_bin = test_per_bin + val_per_bin + max_train_per_bin
    for bin_index in range(n_bins):
        members = np.flatnonzero(bins == bin_index)
        if members.size < required_per_bin:
            raise ValueError(
                f"activity bin {bin_index} has {members.size} rows; "
                f"{required_per_bin} are required"
            )
        shuffled = members[rng.permutation(members.size)]
        stop_test = test_per_bin
        stop_val = stop_test + val_per_bin
        test_by_bin.append(shuffled[:stop_test])
        val_by_bin.append(shuffled[stop_test:stop_val])
        pool_by_bin.append(shuffled[stop_val:])

    def combine_and_shuffle(parts):
        combined = np.concatenate(parts).astype(np.int64, copy=False)
        return combined[rng.permutation(combined.size)]

    test_indices = combine_and_shuffle(test_by_bin)
    val_indices = combine_and_shuffle(val_by_bin)
    train_indices = {}
    for size in sizes:
        per_bin = size // n_bins
        train_indices[size] = combine_and_shuffle(
            [pool[:per_bin] for pool in pool_by_bin]
        )

    if test_indices.size + val_indices.size + sizes[-1] > n_samples:
        raise ValueError("requested partitions exceed the dataset size")

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "n_samples": n_samples,
        "split_seed": split_seed,
        "bin_edges": np.asarray(bin_edges, dtype=np.float64),
        "activity_scores": np.asarray(scores, dtype=np.float64),
        "activity_bin_ids": bins,
        "activity_bin_counts": bin_counts.astype(np.int64),
        "train_sizes": np.asarray(sizes, dtype=np.int64),
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_indices": train_indices,
    }


def write_activity_split_manifest(
    dataset_path,
    output_path,
    bin_edges,
    *,
    train_sizes=(8_000, 16_000, 24_000),
    val_size=3_000,
    test_size=3_000,
    split_seed=42,
):
    """Create an atomic NPZ manifest bound to an exact master dataset hash."""
    dataset_path = str(Path(dataset_path).resolve())
    output_path = str(Path(output_path).resolve())
    with np.load(dataset_path, allow_pickle=False) as dataset:
        curves = np.asarray(dataset["X"], dtype=np.float32)
        labels = np.asarray(dataset["Y"])
        parameter_names = np.asarray(dataset["parameter_names"])
    if labels.ndim != 2 or labels.shape[0] != curves.shape[0]:
        raise ValueError(
            f"master dataset X/Y shapes do not align: {curves.shape}, {labels.shape}"
        )

    split = build_nested_activity_splits(
        curves,
        bin_edges,
        train_sizes=train_sizes,
        val_size=val_size,
        test_size=test_size,
        split_seed=split_seed,
    )
    payload = {
        "schema_version": np.asarray(split["schema_version"], dtype=np.int64),
        "dataset_path": np.asarray(dataset_path),
        "dataset_sha256": np.asarray(sha256_file(dataset_path)),
        "n_samples": np.asarray(split["n_samples"], dtype=np.int64),
        "split_seed": np.asarray(split["split_seed"], dtype=np.int64),
        "bin_edges": split["bin_edges"],
        "activity_scores": split["activity_scores"],
        "activity_bin_ids": split["activity_bin_ids"],
        "activity_bin_counts": split["activity_bin_counts"],
        "train_sizes": split["train_sizes"],
        "val_indices": split["val_indices"],
        "test_indices": split["test_indices"],
        "parameter_names": parameter_names,
    }
    payload.update({
        f"train_{size}": indices
        for size, indices in split["train_indices"].items()
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, output)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return output_path


def _scalar(dataset, name):
    if name not in dataset.files:
        raise ValueError(f"split manifest is missing {name!r}")
    value = np.asarray(dataset[name])
    if value.size != 1:
        raise ValueError(f"split manifest field {name!r} must be scalar")
    return value.reshape(-1)[0].item()


def _validated_indices(values, name, n_samples):
    raw = np.asarray(values)
    if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    indices = raw.astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(indices < 0) or np.any(indices >= n_samples):
        raise ValueError(f"{name} contains out-of-range row indices")
    if np.unique(indices).size != indices.size:
        raise ValueError(f"{name} contains duplicate row indices")
    return indices


def load_explicit_split(
    manifest_path,
    dataset_path,
    *,
    n_samples,
    split_seed,
    train_size,
    valid_mask=None,
):
    """Load and fully validate one configured train/validation/test partition."""
    manifest_path = str(Path(manifest_path).resolve())
    dataset_path = str(Path(dataset_path).resolve())
    actual_dataset_sha256 = sha256_file(dataset_path)
    with np.load(manifest_path, allow_pickle=False) as manifest:
        schema = _scalar(manifest, "schema_version")
        manifest_n = _scalar(manifest, "n_samples")
        manifest_seed = _scalar(manifest, "split_seed")
        expected_dataset_sha256 = str(_scalar(manifest, "dataset_sha256"))
        raw_train_sizes = np.asarray(manifest["train_sizes"])
        if (
            raw_train_sizes.ndim != 1
            or raw_train_sizes.size == 0
            or not np.issubdtype(raw_train_sizes.dtype, np.integer)
        ):
            raise ValueError(
                "split manifest train_sizes must be a non-empty integer array"
            )
        manifest_train_sizes = raw_train_sizes.astype(
            np.int64, copy=False
        ).tolist()
        if manifest_train_sizes != sorted(set(manifest_train_sizes)):
            raise ValueError(
                "split manifest train_sizes must be distinct and increasing"
            )
        if int(train_size) not in manifest_train_sizes:
            raise ValueError(
                f"split manifest does not define train_size={train_size}"
            )
        train_sets = {}
        previous = set()
        for size in manifest_train_sizes:
            train_key = f"train_{size}"
            if train_key not in manifest.files:
                raise ValueError(
                    f"split manifest is missing {train_key!r}"
                )
            indices = _validated_indices(
                manifest[train_key], train_key, int(n_samples)
            )
            if indices.size != size:
                raise ValueError(
                    f"split manifest {train_key} count mismatch: "
                    f"{indices.size} vs {size}"
                )
            current = set(indices.tolist())
            if not previous.issubset(current):
                raise ValueError(
                    "split manifest training subsets are not nested"
                )
            previous = current
            train_sets[size] = indices
        train = train_sets[int(train_size)]
        val = _validated_indices(
            manifest["val_indices"], "val_indices", int(n_samples)
        )
        test = _validated_indices(
            manifest["test_indices"], "test_indices", int(n_samples)
        )

    if schema != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported split manifest schema {schema}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    if manifest_n != int(n_samples):
        raise ValueError(
            f"split manifest sample count mismatch: {manifest_n} vs {n_samples}"
        )
    if manifest_seed != int(split_seed):
        raise ValueError(
            f"split manifest seed mismatch: {manifest_seed} vs {split_seed}"
        )
    if expected_dataset_sha256 != actual_dataset_sha256:
        raise ValueError(
            "split manifest dataset SHA-256 mismatch: "
            f"{expected_dataset_sha256} vs {actual_dataset_sha256}"
        )
    all_selected = np.concatenate((
        train_sets[manifest_train_sizes[-1]],
        val,
        test,
    ))
    if np.unique(all_selected).size != all_selected.size:
        raise ValueError("split manifest train/validation/test partitions overlap")
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != (int(n_samples),):
            raise ValueError(
                f"valid_mask shape mismatch: {mask.shape} vs {(int(n_samples),)}"
            )
        rejected = all_selected[~mask[all_selected]]
        if rejected.size:
            raise ValueError(
                f"{rejected.size} explicitly selected rows fail training cleanup"
            )

    return ExplicitSplit(
        train=train,
        val=val,
        test=test,
        train_size=int(train_size),
        split_seed=int(split_seed),
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        dataset_sha256=actual_dataset_sha256,
    )


def configured_explicit_split(
    config,
    dataset_path,
    *,
    n_samples,
    valid_mask=None,
):
    """Resolve an optional explicit split from the shared training config."""
    manifest_getter = getattr(config, "get_split_manifest_file", None)
    size_getter = getattr(config, "get_train_subset_size", None)
    if manifest_getter is None and size_getter is None:
        return None
    if manifest_getter is None or size_getter is None:
        raise ValueError(
            "config must expose both explicit-split accessors or neither"
        )
    manifest_path = manifest_getter()
    train_size = size_getter()
    if manifest_path is None and train_size is None:
        return None
    if manifest_path is None or train_size is None:
        raise ValueError(
            "split_manifest_file and train_subset_size must be configured together"
        )
    return load_explicit_split(
        manifest_path,
        dataset_path,
        n_samples=n_samples,
        split_seed=config.get_split_seed(),
        train_size=train_size,
        valid_mask=valid_mask,
    )
