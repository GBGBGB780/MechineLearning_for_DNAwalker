# Publication Boundary

Last reviewed: **2026-07-30**.

This project separates the public source repository, curated evidence, external
research artifacts, and private local records. A file being useful locally does
not make it appropriate for Git or public distribution.

## Publish in the source repository

| Content | Paths |
|---|---|
| Runtime source | `dnawalker/` |
| Reproducible configuration | `configs/` |
| Tests and CI | `tests/`, `.github/workflows/` |
| Operational tooling | `scripts/`, excluding explicitly ignored local scripts |
| Package and environment definitions | `pyproject.toml`, `requirements*.txt`, `constraints-tested-py312.txt` |
| License and user documentation | `LICENSE`, `README.md`, `docs/*.md` |
| Redacted project notebook | `docs/notebooks/DNAwalker_project.ipynb` |
| Curated path-sanitized evidence | `docs/evidence/` |
| Selected inference artifacts | `artifacts/application/` |

The curated evidence contains small final figures and machine-readable
summaries. Raw logs and generated run directories are not publication
artifacts.

`artifacts/application/` contains only the validation-selected CNN seed-43 and
Transformer seed-46 checkpoints, their paired Y scalers, the required split
manifest, an English scope/provenance README, and `SHA256SUMS`. The training
dataset is not included. This bundle supports prediction and physics
refinement, not retraining or held-out dataset evaluation.

The exact canonical parameter counts (`4,381,319` CNN and `3,243,271`
Transformer) and analytical forward MAC estimates may be published because
they are derived from versioned source/configuration and pinned by tests.
They must be labeled as architecture calculations. The source snapshot does
not contain a controlled current-lineage wall-clock benchmark, so it must not
claim a measured training- or inference-speed ratio.

## Publish separately after licensing or generation

The following remain outside the public application tree and require permanent
URLs or regeneration instructions, SHA-256 manifests, source statements, and
licenses where applicable:

- generated training datasets;
- non-selected split manifests;
- non-selected model checkpoints and fitted scalers;
- any third-party or derived experimental workbook.

These files remain in ignored `artifacts/` namespaces or
`data/experimental/` locally. The selected fitted scalers are an explicit
exception under `artifacts/application/`; pickle scalers must only be loaded
from a trusted release after SHA-256 verification.

## Excluded from the source snapshot

The following must not be included in a public source snapshot:

- `.venv/`, caches, build output, and editor/agent state;
- `.kiro/`, `.claude/`, and local session records;
- complete `results/` trees, especially logs and generated INI overrides;
- the editable course-report DOCX under `docs/report/`;
- `scripts/sync_closeout_deliverables.py`, which is specific to the private
  report workflow;
- experimental Excel workbooks until their redistribution rights and source
  statements are documented;
- every `artifacts/` dataset, non-selected checkpoint/scaler, and non-selected
  manifest outside the explicit `artifacts/application/` release bundle.

Raw results are excluded because many historical records contain machine-local
paths. The public summaries preserve metrics, hashes, limitations, and
interpretation without exposing those paths.

The streamlined working tree removes local agent state, the editable report,
raw result directories, old experiment snapshots, historical MATLAB sources,
and withdrawn recovery/warm-start/speed/capacity implementations. Exact
pre-cleanup source remains recoverable at Git commit `2e280df`. Formal
datasets, non-selected checkpoints/scalers/manifests, and experimental
workbooks remain local. The minimal selected inference bundle is the only
versioned exception.

## Evidence refresh

After restoring or regenerating the private result tree, rebuild the tracked
evidence snapshot:

```bash
python scripts/build_public_evidence.py
```

The command rejects known private path and student-ID markers and writes
`docs/evidence/SHA256SUMS`.

## History warning

Earlier local commits contain the private report, Kiro records, experimental
workbooks, and a student identifier. Removing them from the current tree does
not erase Git history. Do not publish the existing `.git` history as-is.

Create a history-free public repository from the sanitized committed snapshot,
for example:

```bash
git archive --format=tar.gz --output dnawalker-public-source.tar.gz HEAD
mkdir dnawalker-public
tar -xzf dnawalker-public-source.tar.gz -C dnawalker-public
cd dnawalker-public
git init
git add .
git commit -m "Initial public source release"
```

## Release status

The public tree now contains a hash-verifiable inference bundle and can run the
documented application against a user-supplied workbook without the training
dataset. Formal cross-machine certification remains open until remote CI and
fresh Linux CPU/CUDA validation are recorded. Reproducing the reviewed
experimental figures still requires a legally redistributable copy of the
original workbook; retraining or held-out evaluation requires the separately
generated 30k dataset.
