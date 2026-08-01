# Curated Public Evidence

Last synchronized: **2026-07-30**.

This directory contains the small, path-sanitized evidence intended for the
public source snapshot. It was generated from the private `results/` tree by
`scripts/build_public_evidence.py`; rebuilding it requires restoring or
regenerating those raw results.

Use [`../MODEL_COMPARISON.md`](../MODEL_COMPARISON.md) for the application
decision, statistical interpretation, and claim boundaries. The files here
are evidence inputs, not a standalone architecture-ranking conclusion.

Published machine-readable records:

- `current_10k_manifest.json`: current-schema 10k five-seed lineage;
- `nested_learning_curve.json`: sanitized 30k nested learning-curve record;
- `fit_robustness.json`: five-refinement-seed experimental-fit summary;
- `identifiability_metrics.json`: checkpoint-independent identifiability
  diagnostics.

Published figures:

- `current_10k_comparison.png`;
- `nested_learning_curve.png`;
- `cnn_experimental_fit.png`;
- `transformer_experimental_fit.png`;
- `refinement_robustness.png`;
- `identifiability_sensitivity.png`;
- `signal_spectrum.png`;
- `signal_autocorrelation.png`.

`SHA256SUMS` binds every generated evidence file. Training logs, generated
overrides, raw per-run directories, training datasets, checkpoints, and
scalers are not included in this evidence directory. The separately versioned
selected inference pairs live under `artifacts/application/`.

The published JSON supports RMSE, sample accounting, provenance,
identifiability, and refinement-stability claims. It does not contain a
current-lineage, same-device wall-clock benchmark. Exact model parameter counts
and analytical MAC estimates are derived from the versioned model/config
definitions, pinned by tests, and documented in
[`../MODEL_COMPARISON.md`](../MODEL_COMPARISON.md); they must not be described
as measured training or inference latency.
