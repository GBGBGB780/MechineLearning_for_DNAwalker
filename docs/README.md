# Project Documentation

Last synchronized: **2026-07-30**.

This directory supports the DNA Walker inverse-prediction application with
architecture, method-selection evidence, reproducibility, publication, release
gate, and legacy diagnostic documentation. The redacted project notebook and
curated public evidence are versioned. The editable course report and local
process records were removed from the streamlined tree.

## Documents

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — canonical `dnawalker/` dependency
  layers, primary inverse-prediction runtime, training/artifact flow,
  retraining lineage, exact model sizes, analytical execution cost, stable
  interfaces, and change rules.
- [`MODEL_COMPARISON.md`](MODEL_COMPARISON.md) — detailed CNN/Transformer
  parameter/compute comparison, RMSE evidence, training and inference timing
  boundaries, experimental application fits, current single-branch
  recommendation, dual-initializer future direction, and interpretation
  boundaries.
- [`FILE_INVENTORY.md`](FILE_INVENTORY.md) — path-by-path responsibilities for
  every stable source/config/test/script/document plus generated artifact and
  result naming contracts.
- [`../data/experimental/README.md`](../data/experimental/README.md) —
  user-workbook column contract, one-command application usage, reviewed input
  identities, and redistribution boundary.
- [`ARTIFACTS.md`](ARTIFACTS.md) — bundled application artifact identities,
  external dataset/checkpoint/scaler lineage, provenance requirements, trust
  boundary, and release blockers.
- [`TEST_CHECKLIST.md`](TEST_CHECKLIST.md) — the canonical-only test baseline
  plus exact parameter-count guards and P0/P1/P2 engineering, scientific, and
  portability gates.
- [`LEARNING_CURVE_PROTOCOL.md`](LEARNING_CURVE_PROTOCOL.md) — the locked 30k
  nested 8k/16k/24k CNN/Transformer comparison, fixed partitions, final
  statistics, retained-distribution audit, recovery caveat, and decision rule.
- [`PUBLICATION.md`](PUBLICATION.md) — explicit public source plus selected
  inference-artifact, external dataset, and local-only file boundary.
- [`evidence/README.md`](evidence/README.md) — curated path-sanitized JSON,
  figures, and checksums intended for public distribution.
- [`audit/README.md`](audit/README.md) — interpretation boundary for immutable
  legacy compatibility diagnostics. The JSON files in `audit/` are historical
  records and are not rewritten during repository reorganization.
- [`PROJECT_HISTORY.md`](PROJECT_HISTORY.md) — bilingual end-to-end history
  from the reconstructed MATLAB/early-model period through reliability fixes,
  canonical-package migration, current retraining, application-first
  documentation, and public-source closeout. It labels reconstructed,
  versioned, hash-bound, and withdrawn evidence separately.
- [`notebooks/DNAwalker_project.ipynb`](notebooks/DNAwalker_project.ipynb) —
  English project walkthrough notebook with model size/compute and private
  identifiers removed.

## Source and compatibility placement

Active implementations live under `dnawalker/`:

- `dnawalker/physics/`: forward physics and refinement.
- `dnawalker/data/`: generation, experimental input, and preprocessing.
- `dnawalker/cnn/` and `dnawalker/transformer/`: complete model workflows.
- `dnawalker/shared/`: ensemble, parameter I/O, provenance, and shared flows.
- `dnawalker/studies/`: controlled validation and multi-seed workflows.
- `dnawalker/tools/`: NPZ inspection and MAT conversion.
- `dnawalker/config.py`: shared typed configuration.
- `dnawalker/cli.py`: unified command tree.

There is no legacy root, `train_*`, `multiseed/`, or `utils/` Python entry
layer; code imports `dnawalker.*` and commands use the unified
`python -m dnawalker` or `dnawalker` command tree.

## Configuration placement

All configuration is centralized under `configs/` and resolved through
`dnawalker.paths`:

- `configs/common.ini`: shared physics, data, preprocessing, and split
  configuration.
- `configs/{cnn,transformer}.ini`: model-specific architecture and training.
- `configs/profiles/smoke.ini`: shared two-model smoke override.
- `configs/studies/nested_30k/`: permanent evaluation overlays
  for the validation-selected 24k CNN and Transformer checkpoints.

Default output/artifact locations are derived from `dnawalker.paths`, so no INI
relative path depends on the caller's working directory. Source checkouts are
auto-detected. A non-editable wheel must run from the checkout or set
`DNAWALKER_PROJECT_ROOT=/absolute/path/to/checkout`, because repository configs
and the selected model artifacts live in the checkout rather than inside the
wheel; training datasets remain external.
