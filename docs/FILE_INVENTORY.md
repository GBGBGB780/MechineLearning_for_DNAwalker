# Repository File Inventory

Last synchronized: **2026-07-30**.

This document is the single path-by-path reference for the canonical DNA Walker
repository. [`ARCHITECTURE.md`](ARCHITECTURE.md) explains dependency and runtime
relationships, while [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) explains
method selection; this file explains what each file owns.

## Scope and lifecycle

Stable operational files and the minimal published inference bundle are listed
individually. Large external artifacts are documented by exact naming contract
rather than by every file. The raw `results/` tree was removed after its
selected, path-sanitized evidence was copied to `docs/evidence/`.

| Label | Meaning |
|---|---|
| **Source** | Versioned runtime implementation |
| **Config/input** | Versioned run definition or external-input contract |
| **Test/automation** | Versioned engineering gate |
| **Generated** | Re-creatable dataset, model, metric, plot, prediction, or log |
| **Historical** | Provenance only; never imported by the active runtime |
| **Process record** | Requirements/design/task history, not runtime truth |

Excluded from the public canonical inventory: `.git/`, `.venv/`,
`.pytest_cache/`, `.hypothesis/`, every `__pycache__/`, temporary
`build/`/`dist/`, local IDE or agent state, raw result trees, and external
experimental workbooks. The workbooks and large artifacts are described below
only to define the separate artifact-release contract. The explicit exception
is `artifacts/application/`, which is part of the public tree.

## Root files and automation

| Path | Class | Responsibility |
|---|---|---|
| `README.md` | Documentation | Application-first user entry point: inverse-prediction contract, recommended strategy, result figures, setup, commands, workflow, and release limitations. |
| `LICENSE` | Metadata | Repository license terms. |
| `pyproject.toml` | Metadata/automation | Python 3.12 package metadata, setuptools discovery, pytest settings, and the unified `dnawalker` console script. |
| `requirements.txt` | Config | Human-maintained compatible dependency ranges and optional dependency notes. |
| `requirements-lock.txt` | Config | Reproducible Python 3.12 CPU dependency resolution used for cross-machine setup. |
| `constraints-tested-py312.txt` | Config | Exact package versions from the reviewed local Python 3.12 environment. |
| `artifacts.sha256` | Evidence | Identity manifest for two external Excel inputs, one frozen dataset, two legacy checkpoints, and their paired Y scalers. |
| `.gitignore` | Metadata | Excludes environments, caches, datasets, non-selected models, generated results, and local experiment snapshots while allowing `artifacts/application/`. |
| `.gitattributes` | Metadata | Forces deterministic LF endings for INI configuration files. |
| `.github/workflows/tests.yml` | Test/automation | Ubuntu/Python 3.12 CI: dependency setup, pytest, static checks, shell syntax, JSON, Notebook, curated-evidence validation, and application-artifact hashes. |

Local editor/agent state is not a product file. The removed Kiro/task records
and other pre-cleanup source remain recoverable from Git commit `2e280df`, but
are excluded from a history-free public export.

## Canonical Python package

### Package boundaries and project paths

| Path | Responsibility |
|---|---|
| `dnawalker/__init__.py` | Declares the top-level package and its public package boundary. |
| `dnawalker/__main__.py` | Makes `python -m dnawalker` invoke the unified CLI. |
| `dnawalker/cli.py` | Defines the lazy `data`, `cnn`, `transformer`, `study`, and `verify` command tree. |
| `dnawalker/config.py` | Loads and validates common physics/data/split/prediction configuration and exposes typed shared getters. |
| `dnawalker/physics/__init__.py` | Declares the physics/refinement kernel and stable simulator/refinement aliases. |
| `dnawalker/data/__init__.py` | Declares the data generation and preprocessing package. |
| `dnawalker/cnn/__init__.py` | Declares the CNN implementation package. |
| `dnawalker/transformer/__init__.py` | Declares the Transformer implementation package. |
| `dnawalker/shared/__init__.py` | Declares architecture-neutral runtime services. |
| `dnawalker/studies/__init__.py` | Declares controlled scientific studies. |
| `dnawalker/studies/multiseed/__init__.py` | Declares multi-seed orchestration. |
| `dnawalker/tools/__init__.py` | Declares standalone data-inspection/conversion tools. |
| `dnawalker/paths.py` | Discovers and validates the checkout root, supports `DNAWALKER_PROJECT_ROOT`, and centralizes `configs/`, `data/`, `artifacts/`, and `results/` locations. |

### Configuration and physics kernel

| Path | Responsibility |
|---|---|
| `dnawalker/physics/simulator.py` | Canonical 14-state DNA Walker forward simulator, energy/rate construction, transition propagation, three-channel signal output, and process-local simulation counter. |
| `dnawalker/physics/refinement.py` | Maps physical and optimization spaces, handles log-space `k0`, bounds, jittered multi-starts, and Powell curve-RMSE refinement. |
| `dnawalker/verify.py` | Reads one parameter text file, runs the forward model, compares experimental channels, and writes RMSE plus a verification figure; model run subdirectories are mirrored under `results/evaluation/<model>/`, while external inputs use `results/evaluation/verification/`. |

### Data and shared inference

| Path | Responsibility |
|---|---|
| `dnawalker/data/experimental.py` | Reads experimental Excel columns, filters finite rows, merges duplicate times, interpolates to the simulation grid, and applies Savitzky-Golay smoothing. |
| `dnawalker/data/generate.py` | CLI for LHS sampling, parallel forward simulation, validity/activity-bin filtering, deterministic replenishment, and canonical NPZ writing. |
| `dnawalker/data/preprocessing.py` | Shared label cleaning/log transform/masks and the only per-sample joint-channel X normalization implementation. |
| `dnawalker/data/splits.py` | Creates and validates dataset-bound activity-stratified manifests with fixed validation/test membership and strictly nested training subsets. |
| `dnawalker/shared/ensemble.py` | Single or noisy test-time ensemble prediction and median aggregation. |
| `dnawalker/shared/parameters.py` | Owns canonical seven-parameter ordering, NPZ metadata validation/reordering, checkpoint parameter validation, and parameter text I/O. |
| `dnawalker/shared/pipeline.py` | Architecture-neutral prediction plus physical refinement and refined experimental-dataset evaluation workflows. |
| `dnawalker/shared/evaluation.py` | Reconstructs the cleaning/split contract, verifies artifact provenance, and computes held-out metrics with valid/invalid/extreme accounting. |

### CNN implementation

| Path | Responsibility |
|---|---|
| `dnawalker/cnn/config.py` | Layers `common.ini`, `cnn.ini`, and optional overrides; validates architecture, optimizer, scheduler, and CNN artifact paths. |
| `dnawalker/cnn/data.py` | Loads/cleans NPZ data, applies shared X normalization, supports deterministic random or explicit manifest splits, fits the training-only Y scaler, and builds flattened DataLoaders. |
| `dnawalker/cnn/model.py` | Defines the `4,381,319`-parameter `InverseCNN` convolutional feature extractor and bounded regression head. |
| `dnawalker/cnn/inference.py` | Loads and validates CNN checkpoint/Y-scaler pairs, checks seed/dataset/scaler/manifest/subset metadata, normalizes inputs, and exposes batched physical-parameter prediction. |
| `dnawalker/cnn/train.py` | CNN training: seeding, adaptive/unweighted MSE, Adam, ReduceLROnPlateau, early stopping, test metrics, and current-schema checkpoint writing. |
| `dnawalker/cnn/predict.py` | Direct experimental prediction and optional `--refine` multi-start physical refinement; both modes honor explicit `--exp`/`--out` and default to the CNN prediction namespace. |
| `dnawalker/cnn/evaluate.py` | `testset` and `experimental` evaluation subcommands with provenance, JSON, and PNG output. |

CNN X values use shared per-sample normalization; there is no fitted or saved X
scaler in the current pipeline.

### Transformer implementation

| Path | Responsibility |
|---|---|
| `dnawalker/transformer/config.py` | Loads common and Transformer layers and validates patch geometry, attention dimensions, optimizer/scheduler values, artifact paths, and parent compatibility. |
| `dnawalker/transformer/data.py` | Mirrors shared cleaning and random/explicit split contracts while preserving `(N, 3, 7801)` tensors; fits and saves the training-only Transformer Y scaler. |
| `dnawalker/transformer/model.py` | Defines the `3,243,271`-parameter patch embedding, temporal self-attention, cross-channel attention, and bounded inverse-regression head. |
| `dnawalker/transformer/inference.py` | Loads validated checkpoint/Y-scaler pairs, verifies optional manifest/subset provenance, maps devices safely, bounds inference batches, and predicts physical parameters. |
| `dnawalker/transformer/train.py` | Transformer training: seeding, AdamW, cosine warmup, weighted/unweighted MSE, early stopping, test metrics, and current-schema checkpoint writing. |
| `dnawalker/transformer/predict.py` | Direct experimental prediction and optional `--refine` physical refinement; both modes honor explicit `--exp`/`--out` and default to the Transformer prediction namespace. |
| `dnawalker/transformer/evaluate.py` | `testset` and `experimental` evaluation subcommands with provenance, JSON, and PNG output. |

### Validation and scientific experiments

| Path | Responsibility |
|---|---|
| `dnawalker/studies/protocol.py` | Shared seed validation, LHS generation, RMSE/statistics helpers, strict finite JSON writing, and study utilities. |
| `dnawalker/studies/identifiability.py` | Finite-difference Jacobian, Fisher information, sensitivity, and optional profile-likelihood analysis. |
| `dnawalker/studies/signal_analysis.py` | Chunked signal spectrum and autocorrelation evidence generation. |
| `dnawalker/studies/learning_curve.py` | Prepares the fixed 30k nested split, runs/merges the 8k/16k/24k grid, audits retained parameter coverage, and writes paired statistics plus JSON/PNG/Markdown summaries. |
| `dnawalker/studies/fit_robustness.py` | Validates per-refinement-seed experimental-fit JSON and writes cross-model stability JSON/PNG without treating one selected checkpoint as a model-seed comparison. |

### Multi-seed orchestration

| Path | Responsibility |
|---|---|
| `dnawalker/studies/multiseed/constants.py` | Model names, filenames, metric names, and result-schema constants. |
| `dnawalker/studies/multiseed/runtime.py` | Per-model run specifications, seed override files, training subprocess execution, checkpoint locations, and safe split-seed inspection. |
| `dnawalker/studies/multiseed/evaluation.py` | Validated evaluation outcome type, sample-count conservation, and provenance attachment. |
| `dnawalker/studies/multiseed/statistics.py` | Aggregates successful seed metrics and handles insufficient-seed statistics. |
| `dnawalker/studies/multiseed/reporting.py` | Validates part results, merges CNN/Transformer runs, and writes comparison reports/figures. |
| `dnawalker/studies/multiseed/runner.py` | Top-level train/evaluate sequencing across models and seeds with continue-on-failure records. |

### Shared support and tools

| Path | Responsibility |
|---|---|
| `dnawalker/shared/artifacts.py` | Streaming SHA-256, constant-time hash matching, and strict optional checkpoint seed/positive-integer metadata validation. |
| `dnawalker/shared/device.py` | Selects CUDA, Apple MPS, or CPU consistently. |
| `dnawalker/shared/logging.py` | Shared logging configuration. |
| `dnawalker/shared/seeding.py` | Validates uint32 seeds, initializes Python/NumPy/PyTorch random sources, and derives deterministic per-batch seeds. |
| `dnawalker/tools/check_npz.py` | CLI to inspect NPZ keys, shapes, parameter order, sample values, and NaN/Inf counts. |
| `dnawalker/tools/mat_to_npz.py` | Converts historical MATLAB/HDF5 datasets to canonical NPZ layout and metadata; its default destination is `artifacts/datasets/`. |

## Configuration files

INI overlays are applied after their parent configuration; an override contains
only values that intentionally differ.

| Path | Responsibility |
|---|---|
| `configs/common.ini` | Shared physical constants, seven parameter ranges/order, repository paths, 130-minute protocol, generation filters, preprocessing, prediction, `random_seed`, and fixed `split_seed`. |
| `configs/cnn.ini` | CNN architecture, optimizer/scheduler, training budget, artifact directory, checkpoint, and Y-scaler names. |
| `configs/transformer.ini` | Transformer architecture, optimizer/scheduler, training budget, dataset, artifact directory, checkpoint, and Y-scaler names. |
| `configs/profiles/smoke.ini` | One shared two-model smoke override: small dataset, two epochs, isolated artifact names, CNN channels, and Transformer width/depth. |
| `configs/studies/nested_30k/cnn_seed43.ini` | Permanent CNN override for the validation-selected 24k seed-43 checkpoint under `artifacts/application/cnn/`. |
| `configs/studies/nested_30k/transformer_seed46.ini` | Permanent shared/Transformer override for the validation-selected 24k seed-46 checkpoint under `artifacts/application/transformer/`. |

## Operational scripts

| Path | Responsibility |
|---|---|
| `scripts/run_application.sh` | One-command formal application wrapper: accepts a workbook, runs the selected 24k Transformer, CNN, or both with locked ensemble/refinement settings, and performs model-owned forward verification. |
| `scripts/run_smoke_test.sh` | Pure-Python end-to-end smoke: generate 20 samples, train both models with isolated `.smoke` artifacts, then write prediction and verification output into model-owned `smoke/` result directories. |
| `scripts/build_public_evidence.py` | Builds the tracked path-sanitized evidence snapshot from private local results and rejects known private markers. |
| `scripts/hpc/cnn.pbs.sh` | PBS GPU job wrapper for canonical CNN training with configurable project/Conda/CUDA settings. |
| `scripts/hpc/transformer.pbs.sh` | PBS GPU job wrapper for canonical Transformer training and Transformer config selection. |

## Unified command ownership

`pyproject.toml` publishes only `dnawalker = dnawalker.cli:main`. The CLI
imports a target lazily after selecting a leaf:

| Command family | Target owners |
|---|---|
| `dnawalker data` | `data.generate`, `tools.check_npz`, `tools.mat_to_npz` |
| `dnawalker cnn` | `cnn.train`, `cnn.predict`, `cnn.evaluate` |
| `dnawalker transformer` | `transformer.train`, `transformer.predict`, `transformer.evaluate` |
| `dnawalker study` | `studies.identifiability`, `signal_analysis`, `multiseed.runner`, `learning_curve`, `fit_robustness` |
| `dnawalker verify` | `dnawalker.verify` |

Model `predict` modules select direct or physics-refined prediction with
`--refine`. Model `evaluate` modules select `testset` or `experimental` with a
required subcommand. This keeps one file per user-visible model responsibility.

## External experimental input contract

| Path | Responsibility |
|---|---|
| `data/experimental/README.md` | User-workbook column and protocol contract, one-command application usage, expected filenames, reviewed SHA-256 values, and redistribution boundary. |
| `data/experimental/Fig3a_fitting.xlsx` | Local primary experimental curves; intentionally not distributed until source and license are confirmed. |
| `data/experimental/Fig3a_fitting_generalization.xlsx` | Local generalization curves; intentionally not distributed until source and license are confirmed. |

Both local workbooks are covered by `artifacts.sha256`. Code reads them through
`dnawalker.data.experimental`; they are not training datasets or public source
files.

## Test suite

`tests/conftest.py` makes the checkout importable when pytest is invoked by an
absolute test path. The 24 `test_*.py` files below form the current local
suite; exact per-file case counts are in [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md).

| Path | Primary coverage |
|---|---|
| `tests/test_audit_results.py` | Frozen compatibility-diagnostic JSON and CPU/MPS tolerance. |
| `tests/test_computation_only.py` | Canonical simulator module and process-local counter identity. |
| `tests/test_config_loader.py` | INI validation, parameter ranges, dimensions, seeds, path anchoring, and the exact canonical CNN parameter count. |
| `tests/test_ensemble.py` | Ensemble shape, finite values, randomness, and aggregation. |
| `tests/test_eval_testset_common.py` | Held-out cleaning, parameter reordering, and provenance. |
| `tests/test_exp_data_io.py` | Excel filtering, ordering, duplicate-time handling, interpolation, and smoothing. |
| `tests/test_fit_robustness.py` | Strict per-seed fit metadata, provenance invariants, aggregate statistics, and English summary-figure generation. |
| `tests/test_gendata_config.py` | Generator configuration, budgets, determinism, quotas, and write safety. |
| `tests/test_identifiability.py` | Jacobian/FIM/profile likelihood and invalid inputs. |
| `tests/test_inference_validation.py` | Predictor dtype/batching/output validation and checkpoint/scaler identity. |
| `tests/test_learning_curve.py` | Activity-balanced nested manifests, explicit held-out membership, split provenance, retained-distribution audit, and paired decision rules. |
| `tests/test_multiseed_modules.py` | Multi-seed module boundaries, count conservation, merge, and corrupt artifacts. |
| `tests/test_multiseed_retrain.py` | Fixed held-out membership and train/evaluate orchestration. |
| `tests/test_package_entrypoints.py` | Unified console-script target, module execution, every CLI leaf's help contract, explicit prediction input/output routing, and application-script help. |
| `tests/test_param_io.py` | Parameter order, NPZ metadata, checkpoint names, and text I/O. |
| `tests/test_preprocessing.py` | Shared normalization, dtype, masks, and in-place behavior. |
| `tests/test_pysim_counter.py` | Process-local forward-simulation counter. |
| `tests/test_pysim_physics.py` | Physics invariants and numerical regression points. |
| `tests/test_refine.py` | Bounds, objectives, log space, multi-start, and shared refinement API. |
| `tests/test_signal_evidence.py` | Chunked FFT and autocorrelation analysis. |
| `tests/test_training_splits.py` | Identical fixed `split_seed` membership in both model adapters. |
| `tests/test_transformer_config.py` | Transformer configuration validation, entry behavior, and the exact canonical Transformer parameter count. |
| `tests/test_utils_cli.py` | NPZ inspection and MAT conversion command-line utilities. |
| `tests/test_validation_common.py` | Seeds, LHS, statistics, and strict JSON. |

## Documentation and reviewed diagnostics

| Path | Responsibility |
|---|---|
| `docs/README.md` | Documentation landing page and navigation. |
| `docs/PROJECT_HISTORY.md` | Bilingual project history from reconstructed origins through the final scientific and publication closeout, with evidence levels, lineage changes, withdrawn claims, and the complete pre-history-document commit ledger. |
| `docs/ARCHITECTURE.md` | Dependency layers, runtime/artifact flow, stable interfaces, split/provenance/device boundaries, and change rules. |
| `docs/MODEL_COMPARISON.md` | Detailed method-selection evidence, exact model sizes, analytical compute/runtime boundaries, direct held-out comparisons, experimental application fits, recommended deployment strategy, and interpretation boundaries. |
| `docs/FILE_INVENTORY.md` | This path-by-path responsibility index. |
| `docs/ARTIFACTS.md` | Artifact identity, trust boundary, current checkpoint schema, reviewed hashes, and release blockers. |
| `docs/TEST_CHECKLIST.md` | Local regression baselines, model-size guards, current retraining evidence, and P0/P1/P2 release gates. |
| `docs/LEARNING_CURVE_PROTOCOL.md` | Predeclared 30k nested learning-curve inputs, partitions, paired decision rule, test checklist, and interpretation boundary. |
| `docs/PUBLICATION.md` | Public source, external artifact, private local-record, and clean-history export boundary. |
| `docs/evidence/` | Curated public JSON/PNG evidence plus `SHA256SUMS`; generated from ignored local results. |
| `docs/audit/README.md` | Interpretation boundary for frozen legacy compatibility diagnostics. |
| `docs/audit/heldout_cnn_cpu.json` | Frozen legacy CNN held-out diagnostic on CPU. |
| `docs/audit/heldout_cnn_mps.json` | Frozen legacy CNN held-out diagnostic on Apple MPS. |
| `docs/audit/heldout_transformer_cpu.json` | Frozen legacy Transformer held-out diagnostic on CPU. |
| `docs/audit/heldout_transformer_mps.json` | Frozen legacy Transformer held-out diagnostic on Apple MPS. |
| `docs/notebooks/DNAwalker_project.ipynb` | Redacted English project walkthrough synchronized with model size/compute, the 10k comparison, nested 30k learning curve, and final refinement-stability evidence. |

Paths embedded inside `docs/audit/*.json` are immutable historical provenance and
may name the pre-reorganization layout. Do not rewrite them into executable
current commands.

## Removed historical source

The streamlined release tree contains no `archive/` source. Historical MATLAB,
recovery, warm-start, speed, capacity-ablation, and legacy stretching
implementations remain recoverable from Git commit `2e280df`; they are not
current runtime or publication files.

## Application and generated artifacts

The selected application bundle is versioned; all other files in this section
are ignored and can be absent in a fresh clone. A filename is not evidence of
identity; use the appropriate SHA-256 manifest and checkpoint provenance.
Pickle scalers must be loaded only from a trusted project source.

### Published application bundle

| Path | Responsibility/status |
|---|---|
| `artifacts/application/README.md` | English inference scope, provenance, exclusion, integrity, and trust contract. |
| `artifacts/application/SHA256SUMS` | Exact hashes for the two selected checkpoints, their paired scalers, and the shared manifest. |
| `artifacts/application/split_manifest.npz` | Required dataset/split/subset provenance for both selected checkpoints; contains no curve or target matrix. |
| `artifacts/application/cnn/` | Validation-selected 24k seed-43 CNN checkpoint and paired scaler. |
| `artifacts/application/transformer/` | Validation-selected 24k seed-46 Transformer checkpoint and paired scaler. |

### Dataset names

| Path/pattern | Responsibility/status |
|---|---|
| `artifacts/datasets/training_dataset.npz` | Frozen reviewed legacy training dataset; one of the seven manifest entries. |
| `artifacts/releases/retrain-3a5a494-ds557506e93079/training_dataset.npz` | Selected hardened seed-42 dataset paired with the complete current 10k release namespace. |
| `artifacts/studies/nested_learning_curve_30k/dnawalker_training_30k_seed42.npz` | Permanent local master dataset for the nested 30k study. |
| `artifacts/studies/nested_learning_curve_30k/split_manifest.npz` | Original local dataset-bound fixed validation/test and nested-training row membership; byte-identical selected copy is published under `artifacts/application/`. |

Every canonical NPZ contains `X`, `Y`, and `parameter_names`. The byte-identical
second copy of the selected release dataset was removed after its hash had
established deterministic generation.

### CNN artifact names

| Pattern | Responsibility/status |
|---|---|
| `artifacts/models/cnn/best_mlp_model.pth` + `y_scaler.pkl` | Reviewed legacy default pair in `artifacts.sha256`; loadable for compatibility diagnostics but missing current lineage. |
| `artifacts/application/cnn/best_mlp_model.seed43.pth` + `y_scaler.seed43.pkl` | Published validation-selected 24k CNN inference pair. |
| `artifacts/releases/retrain-3a5a494-ds557506e93079/models/cnn/best_mlp_model.seed{N}.pth` + `y_scaler.seed{N}.pkl` | Complete current-schema release pairs for seeds 42–46. |
| `artifacts/studies/nested_learning_curve_30k/models/train_24000/cnn/` | Complete dataset/split-manifest/subset-bound CNN pairs for seeds 42–46. |

### Transformer artifact names

| Pattern | Responsibility/status |
|---|---|
| `artifacts/models/transformer/best_transformer_model.pth` + `transformer_y_scaler.pkl` | Reviewed legacy default pair in `artifacts.sha256`; compatibility diagnostic only. |
| `artifacts/application/transformer/best_transformer_model.seed46.pth` + `transformer_y_scaler.seed46.pkl` | Published validation-selected 24k Transformer inference pair. |
| `artifacts/releases/retrain-3a5a494-ds557506e93079/models/transformer/best_transformer_model.seed{N}.pth` + `transformer_y_scaler.seed{N}.pkl` | Complete current-schema release pairs for seeds 42–46. |
| `artifacts/studies/nested_learning_curve_30k/models/train_16000/transformer/` | Partial retained Transformer set: checkpoints 42–46 and scalers 43–46; not a complete distributable group. |
| `artifacts/studies/nested_learning_curve_30k/models/train_24000/transformer/` | Complete dataset/split-manifest/subset-bound Transformer pairs for seeds 42–46. |

Current-schema checkpoint fields are defined in
[`ARTIFACTS.md`](ARTIFACTS.md). A scaler is valid only when its SHA-256 matches
the checkpoint field.

## Curated results

The raw `results/` tree is intentionally absent from the streamlined working
tree. Its final publication-safe subset is tracked under `docs/evidence/`:

| Path | Meaning |
|---|---|
| `docs/evidence/current_10k_manifest.json` + `current_10k_comparison.png` | Controlled 10k five-seed comparison. |
| `docs/evidence/nested_learning_curve.json` + `nested_learning_curve.png` | Fixed 30k nested learning curve and retained-distribution audit. |
| `docs/evidence/fit_robustness.json` + `refinement_robustness.png` | Cross-model refinement-start robustness summary. |
| `docs/evidence/{cnn,transformer}_experimental_fit.png` | Selected English three-channel experimental-fit figures. |
| `docs/evidence/identifiability_*` | Checkpoint-independent sensitivity and identifiability evidence. |
| `docs/evidence/signal_*` | Signal spectrum and autocorrelation evidence. |
| `docs/evidence/SHA256SUMS` | Integrity manifest for the curated evidence files. |

Only strict finite JSON tied to actual dataset/checkpoint/scaler hashes may be
used to update release claims. `scripts/build_public_evidence.py` can rebuild
the curated subset after private raw results are restored or regenerated.

## Where to make a change

| Goal | Primary owner |
|---|---|
| Change physical simulation | `dnawalker/physics/simulator.py` plus physics tests |
| Change parameter refinement | `dnawalker/physics/refinement.py` plus refinement tests |
| Change parameter order/NPZ contract | `dnawalker/shared/parameters.py`; this creates a new lineage |
| Change shared preprocessing | `dnawalker/data/preprocessing.py`; retrain both models |
| Change CNN/Transformer architecture | The corresponding model package's `model.py` and config |
| Change split or seed policy | `dnawalker/config.py`, both data adapters, and multi-seed tests |
| Add a scientific study | `dnawalker/studies/`, strict JSON, tests, and documentation |
| Add a CLI leaf | `dnawalker/cli.py`, target `main`/`cli`, entry-point tests, and this inventory |
| Change artifact schema | Trainers, predictors, `shared/artifacts.py`, `ARTIFACTS.md`, and tests |

When adding, deleting, or moving a stable file, update this inventory and
[`ARCHITECTURE.md`](ARCHITECTURE.md) in the same change.
