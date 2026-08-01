# Artifact Inventory

Last verified: **2026-07-30**.

Git contains source, tests, configuration, documentation, historical audit
JSON, curated path-sanitized evidence, and the minimal selected inference
bundle under `artifacts/application/`. Experimental workbooks, generated
datasets, non-selected checkpoints/scalers, complete figures, metrics, and logs
remain external or local artifacts. Active code resolves these through stable
configuration paths while its canonical implementation lives under
`dnawalker/`.

## Formal architecture identity

Current checkpoints are produced by two exact canonical architectures:

| Architecture | Trainable parameters | Raw FP32 weights | Analytical forward MAC/sample |
|---|---:|---:|---:|
| CNN | `4,381,319` | about `16.7 MiB` | about `38,067,168` |
| Transformer | `3,243,271` | about `12.4 MiB` | about `780,223,360` |

The parameter counts are instantiated from the production configs and guarded
by tests. They identify architecture shape, but are not sufficient artifact
provenance: dataset, scaler, split, parameter order, seed, selected epoch, and
file hash must still match. The MAC values are analytical estimates and are
not checkpoint fields or measured latency evidence.

## Reviewed legacy set

[`artifacts.sha256`](../artifacts.sha256) defines the exact local inputs used by
the legacy compatibility diagnostic and the repository-reorganization checks:

| Path | SHA-256 | Status |
|---|---|---|
| `data/experimental/Fig3a_fitting.xlsx` | `0518081ce35d750ae017bb9ca7d615b221e1ea1e778f4b3bbc5fedabccb2a419` | Local experimental input; redistribution pending |
| `data/experimental/Fig3a_fitting_generalization.xlsx` | `2e35c5c8b71725c52a2e12e257080549f1ded49c348a48b25886f55b4a0bc0bd` | Local experimental input; redistribution pending |
| `artifacts/datasets/training_dataset.npz` | `ada67b3f24d01051c39238b5d18cbe82b8c1edd612466f19c958763b3e83602c` | Frozen 10,000-sample legacy dataset |
| `artifacts/models/cnn/best_mlp_model.pth` | `0a177378b44037ee8afae7b862d7f787ec8b77076633ec2781931f49a54a4f22` | Legacy checkpoint; current provenance absent |
| `artifacts/models/cnn/y_scaler.pkl` | `8be5935bb3f06c0cc83d748b9f85ec478dcffc5e30df1c98b286298646679ccb` | Legacy scaler paired by reviewed hash only |
| `artifacts/models/transformer/best_transformer_model.pth` | `d3e698103227cb7752eab3ca81f4ef3b3b2f2a470a662305cc1b00bebdf520a2` | Legacy checkpoint; provenance incomplete |
| `artifacts/models/transformer/transformer_y_scaler.pkl` | `63eb894c008146757f2be866a6902e74de76ea98694da38f39938dd17e3762ba` | Legacy scaler paired by reviewed hash only |

After restoring all seven external/local files, verify them on macOS with:

```bash
shasum -a 256 -c artifacts.sha256
```

Linux can use `sha256sum -c artifacts.sha256`. On Windows, compare
`Get-FileHash <path> -Algorithm SHA256` with the manifest.

The legacy CNN checkpoint does not provide the current provenance fields. The
legacy Transformer checkpoint lacks `model_seed`, `split_seed`,
`dataset_sha256`, and `y_scaler_sha256`. Historical result JSON was not bound to
a complete checkpoint/scaler training lineage. These files are therefore
suitable only for the compatibility diagnostics in [`audit/`](audit/), not for
model selection, recovery, warm-start, capacity, speed, experimental-fit, or
multi-seed claims.

## 2026-06-26 local reorganization verification

The source migration did not change any of the seven hashes above. The frozen
NPZ was checked as `X=(10000,3,7801)`, `Y=(10000,7)`, with finite arrays and the
canonical ordered parameter metadata. Real CNN and Transformer checkpoints and
scalers loaded from an unrelated working directory and produced finite `(2,7)`
predictions for the two reviewed experimental curves.

Both legacy models were reevaluated on the complete fixed 1,000-sample held-out
set on CPU and Apple MPS. The stored top-level metrics matched the immutable
`docs/audit/*.json` baselines. Maximum CPU↔MPS aggregate differences were
`1.573722864176008e-08` for CNN and `1.9303667182779538e-07` for Transformer,
below the declared `2e-7` compatibility tolerance.

Two forced-CPU smoke runs produced the same mean RMSE (`0.0498`) and identical
hashes for all seven outputs; a native MPS smoke run produced mean RMSE
`0.0562`. Smoke artifacts are temporary development outputs and are not added
to the reviewed legacy manifest.

This historical record is a local behavior-preservation check on an uncommitted
working tree based on Git SHA
`5f122d8273f2279bc0ef7720a7ffa366442f46bd`. It is not clean-SHA release
certification. The later current-schema retraining checkpoint is recorded
separately below.

## Dataset version boundary

The current generator is deterministic, but corrected replenishment and
bin-quota behavior does not reproduce the frozen legacy dataset byte-for-byte:

| Dataset run | Seed | SHA-256 |
|---|---:|---|
| Frozen reviewed dataset | historical/default | `ada67b3f24d01051c39238b5d18cbe82b8c1edd612466f19c958763b3e83602c` |
| Selected hardened release dataset | 42 | `557506e93079d1e5158aa30db7fd4a555234bb751e46049639b1fc827dca9b68` |
| Hardened generator, repeat 2 | 42 | `557506e93079d1e5158aa30db7fd4a555234bb751e46049639b1fc827dca9b68` |
| Hardened generator | 0 | `6ede3e512e2f861cdf8bde81fe1774d4ab280e472673f902b063e4dae58fe916` |

The repeated seed-42 outputs establish determinism for the current generator;
they do not make old and new datasets interchangeable. The seed-42 artifact was
selected for `retrain-3a5a494-ds557506e93079`; the main CNN/Transformer
checkpoints were retrained against it. Never pair it with a legacy checkpoint.

## 2026-07-23 current-schema retraining checkpoint

The local external-artifact namespace is
`artifacts/releases/retrain-3a5a494-ds557506e93079/`. Its selected dataset is
stored as `training_dataset.npz`, and model pairs are under `models/`. All
checkpoints below bind the selected dataset hash, fixed `split_seed=42`,
ordered parameter names, model seed, scaler hash, positive epoch, and finite
validation MSE. Every stored tensor was checked for finite values.

| Model | Seed | Checkpoint SHA-256 | Best epoch |
|---|---:|---|---:|
| CNN | 42 | `1d027ec71ed001fd8e7ed27ba7acc5dcfa3004d9792836bba1fd05a108064270` | 260 |
| CNN | 43 | `18b37e1db8965eb0af3dc77a275b4364d31078682b76ebd4bfab73c46777f374` | 263 |
| CNN | 44 | `c2c748419dcbd928a1766819158af5cc30ae940c47e74fc1cc83540c1c08e316` | 268 |
| CNN | 45 | `fa390a2f5128c97245a706974a18dd5a78ae1d4f6d4bfb54153f05f58fb3e0ec` | 243 |
| CNN | 46 | `85d0dcce7b8aa5758343a354007779fbd4057a32b5dac44764b1cbeb2fcc158a` | 259 |
| Transformer | 42 | `3243d64481143eb1f8f165f6d2fc394377f31d8a6a748da30e16501950f7c600` | 227 |
| Transformer | 43 | `1549d15ebc02878b082bc2015379621bcbe29fabe4101be67d5dd7ec34668ec5` | 199 |
| Transformer | 44 | `3e4f5ab0b0d36c38018284d35fcac3f8b41cac883bc99a3da1fe546ca34ea843` | 191 |
| Transformer | 45 | `5304500804303d4aed3737696a03cd4a22251879edb0dd40a7599eab9b3d0d55` | 189 |
| Transformer | 46 | `9e748de74072a65fd9d79d60a4ab2daa5e67a777ad6dd8d76a4c6ef46d8f5c0d` | 206 |

All 10 scalers have SHA-256
`e28a1057f6812a3bc5a35207c0530ae99ab132cfbd98e68d3a91692ebd965382`
and the checked contract
`MinMaxScaler(feature_range=(0.1, 0.9), n_features=7, n_samples=8000)`.

The strict MPS five-seed merge reports CNN curve RMSE
`0.02124864772861178 +/- 0.0010872001256886506` and Transformer
`0.0211866192226767 +/- 0.000645718249821754` (mean +/- sample SD).
CNN retained all 5,000 attempted samples; Transformer recorded
`4989 valid + 8 invalid + 3 extreme = 5000`. The merged JSON SHA-256 is
`231bf138701f79f1fe56ca3cf749110562d5ef0fa574d2515a1b97ab912114b6`;
the merged figure SHA-256 is
`17942d8528a738c7ad2bf9ea32a4b975a8262fcb584ea7daff3b351e9d6d4028`.
These generated files remain outside Git under
`results/releases/retrain-3a5a494-ds557506e93079/`.

## 2026-07-28 nested learning-curve artifacts

The fixed 30k study uses master dataset SHA-256
`f19f0ae0c63a104ea23c96cc83ce5ce1e0d22fd22c5e1a498dfe6ff53418de06`
and explicit split-manifest SHA-256
`a8bd1b7ae578bf48efc5ffc69e4372a432be132963be0ae142f3bffe5c88acd0`.
The manifest fixes 3,000 validation rows, 3,000 test rows, and nested
8k/16k/24k training rows with exact five-bin activity balance. Every learning-
curve checkpoint schema binds both hashes plus its training-subset size.

The 30-run Apple MPS grid produced these cross-size outputs:

| Output | SHA-256 |
|---|---|
| `learning_curve_metrics.json` | `32f95cced73d1ff57d81a9d9e3247e0a007fe26210b947df956ec02166548975` |
| `learning_curve_compare.png` | `c32dd11811d49ea1e7dda8ab80f24ced533e06e7e67990ee554acae9ec169345` |
| `learning_curve_report.md` | `42eb56a705739f924f1182e5fcac746136af268196f8e421bead44b30a016867` |

CNN/Transformer means are `0.01953252/0.02365253` at 8k,
`0.01833638/0.02020585` at 16k, and `0.01811960/0.01817831` at 24k. All
three paired decisions are inconclusive. The 24k point estimate is a near-tie,
but five seeds do not place its entire 95% interval inside the predeclared
`+/-0.001` equivalence region.

The permanent ignored local binary snapshot is under
`artifacts/studies/nested_learning_curve_30k/`. It contains the master dataset,
split manifest, all surviving 24k checkpoints, and a subset of 16k artifacts.
Numeric summaries, per-size results, and logs are separated under
`results/learning_curve/nested_30k/`. Temporary 8k artifacts and the 16k CNN
binaries were removed by macOS cleanup. Ten 8k numeric rows were recovered from
the append-only session log with source SHA-256
`5c0a99bfab667b917be6e91e5cb32794550bf6e2cc64001b567dd5fdcf07e520`.
Original checkpoint hashes are recorded for the five CNN rows; the five
Transformer rows explicitly carry sentinel hash status because their original
checkpoint SHA-256 values could not be recovered. Those five binaries are not
independently verifiable and must not be described as fully preserved artifacts.

## 2026-07-29 experimental-fit robustness

One checkpoint per architecture was selected by minimum validation MSE across
model seeds 42-46, before inspecting experimental-fit RMSE:

| Model | Selected checkpoint | Validation MSE | Checkpoint SHA-256 |
|---|---|---:|---|
| CNN | 24k seed 43 | `0.0210643411` | `1fc8b1903e9105ac5a9a93cf9643e6059ea5fb8f260ae62edadc0467073da31e` |
| Transformer | 24k seed 46 | `0.0210173298` | `8a5420d3e7e5d3053c389900d3c4a45f0ee080061f63c81d95bef22ff2279b7e` |

Both selected checkpoints use scaler SHA-256
`15ea872072c1c379fe81cf7cdf8eba463173fc8c3a9d144a9ca237705182ca8a`.
Ten Apple MPS evaluations used identical ensemble/noise/refinement settings and
refinement RNG seeds 0-4. Every schema-v2 JSON binds the experimental-input
hashes and complete permanent dataset/checkpoint/scaler/split provenance.

The aggregate JSON is
`results/evaluation/comparisons/train24k_refinement_robustness/fit_robustness.json`
with SHA-256
`3e2ca90eb303f048eaaab4d206a56e7b0390ed26f1e0315bc5efcba92828d880`.
The corresponding English figure has SHA-256
`643bf97d886018e071394eaeaeb396ba34b88c4b798139e2a063b1813530fe05`.

These artifacts quantify refinement-start sensitivity for one checkpoint per
architecture. They do not replace the five-model-seed held-out comparison.

## Published application bundle

The public application bundle copies the two validation-selected pairs and
their required manifest into stable inference paths:

| Path | SHA-256 |
|---|---|
| `artifacts/application/cnn/best_mlp_model.seed43.pth` | `1fc8b1903e9105ac5a9a93cf9643e6059ea5fb8f260ae62edadc0467073da31e` |
| `artifacts/application/cnn/y_scaler.seed43.pkl` | `15ea872072c1c379fe81cf7cdf8eba463173fc8c3a9d144a9ca237705182ca8a` |
| `artifacts/application/transformer/best_transformer_model.seed46.pth` | `8a5420d3e7e5d3053c389900d3c4a45f0ee080061f63c81d95bef22ff2279b7e` |
| `artifacts/application/transformer/transformer_y_scaler.seed46.pkl` | `15ea872072c1c379fe81cf7cdf8eba463173fc8c3a9d144a9ca237705182ca8a` |
| `artifacts/application/split_manifest.npz` | `a8bd1b7ae578bf48efc5ffc69e4372a432be132963be0ae142f3bffe5c88acd0` |

These five binaries total about 29 MiB. `artifacts/application/SHA256SUMS`
verifies their identity. The split manifest contains provenance, activity
summaries, and row membership, but no fluorescence-curve or target-parameter
matrix. The 30k dataset is deliberately absent.

On 2026-07-30, a clean Python 3.12 CPU staging copy containing this bundle but
no training dataset loaded both selected models and produced finite `(1, 7)`
predictions. This is a local packaging/inference check; it does not replace the
still-open fresh Linux/CUDA validation.

## Current-schema release contract

An inference release must distribute each checkpoint together with its paired
scaler and, for an explicit-split checkpoint, the matching split manifest.
A full held-out-evaluation or retraining release must additionally provide the
bound dataset. Current-schema checkpoints contain:

- `model_state`, positive `epoch`, and finite `val_mse`;
- ordered `param_names`;
- `model_seed` and fixed `split_seed`;
- `dataset_sha256` and `y_scaler_sha256`.

An explicit-split checkpoint additionally contains
`split_manifest_sha256` and positive `train_subset_size`. Loading such a
checkpoint without the matching configured manifest and subset size is an
error; this prevents an 8k, 16k, or 24k learning-curve checkpoint from being
silently evaluated under a different row-membership contract.

Predictor loading rejects a mismatched scaler. Held-out evaluation rejects a
dataset mismatch and records actual dataset, checkpoint, and scaler hashes,
device, split, and inference batch size in strict JSON. Multi-seed
`--skip-train` additionally requires matching seeds and parameter order before
the actual dataset/scaler bytes are verified.

The validation-selected application configurations resolve CNN and Transformer
artifacts under `artifacts/application/{cnn,transformer}/` and share
`artifacts/application/split_manifest.npz`. Training datasets remain external.
Paths do not depend on caller cwd. CNN inference casts curves to stored training
dtype (`float32`) before canonical shared normalization. Multi-seed JSON
accounts for each attempted test sample as valid, invalid simulation, or
catastrophic/extreme; these counts must sum to the fixed test-set size.

Scaler files use pickle and must only be loaded from a trusted release. SHA-256
verifies identity, not trust or authorship.

## Cross-machine verification

There are three distinct reproducibility levels:

1. **Artifact identity:** SHA-256 equality is exact on every operating system.
2. **Evaluation:** checkpoints use device-independent tensor state and explicit
   `map_location`. Metrics are compared within a declared tolerance because
   CPU, CUDA, and MPS kernels can differ slightly.
3. **Retraining:** fixed seeds control split membership and primary random
   sources, but PyTorch retraining is not guaranteed to be bit-identical across
   versions or hardware. Compare complete multi-seed distributions.

The 2026-06-26 local environment used Python 3.12.13, PyTorch 2.12.0,
NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.8.0, and pandas 3.0.3. Use
[`requirements-lock.txt`](../requirements-lock.txt) for the reproducible Python
3.12 CPU environment; [`constraints-tested-py312.txt`](../constraints-tested-py312.txt)
records the exact tested snapshot. CUDA validation requires an explicitly
resolved CUDA PyTorch build and a recorded environment manifest.

## Derived evidence

Spectrum and autocorrelation figures are generated from the hashed dataset:

```bash
.venv/bin/python -m dnawalker study signal \
    --dataset artifacts/datasets/training_dataset.npz \
    --results-dir results/validation/signal
```

The script extracts the compressed `X.npy` member to a temporary memory-mapped
file and processes it in FFT chunks. Ensure at least 1 GiB of temporary space,
or select it with `--temp-dir`.

Small final records intended for the public source snapshot are rebuilt from
the ignored local result tree with:

```bash
.venv/bin/python scripts/build_public_evidence.py
```

The generated `docs/evidence/` snapshot excludes raw logs, generated overrides,
machine-local paths, and private submission material.

## Publication blockers

The selected 10k lineage, 30k nested learning curve, and current-artifact
experimental fits are complete for the local project conclusion. Synthetic
recovery, warm-start, corrected speed, and capacity effects are explicitly
outside the final claim set; their implementations and local outputs were
removed from the streamlined release tree and no historical number is cited.
No current checkpoint release should advertise a wall-clock training or
inference speedup until a same-device controlled benchmark is added.

The public tree now includes the selected inference artifacts and their
downloadable hash manifest while excluding the large dataset. Cross-machine
certification still requires remote CI plus fresh Linux CPU/CUDA validation.
Full held-out evaluation or retraining additionally requires a downloadable
manifest and distribution path for the selected dataset. Reviewed experimental
figure reproduction remains blocked by workbook redistribution permission. See
[`PUBLICATION.md`](PUBLICATION.md).
