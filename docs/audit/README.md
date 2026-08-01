# Audited Compatibility Results

Last revalidated: **2026-06-26**.

The four JSON files in this directory are immutable legacy compatibility
records. They use the exact dataset/checkpoint/scaler identities recorded in
each file, Python 3.12.13, and PyTorch 2.12.0. They are deliberately not
rewritten during repository reorganization.

During the 2026-06-26 behavior-preservation check, CNN and Transformer were
again evaluated over the complete fixed 1,000-sample held-out set on CPU and
Apple MPS. The regenerated top-level metrics matched the stored JSON values
item by item. Maximum CPU↔MPS aggregate differences were
`1.573722864176008e-08` for CNN and `1.9303667182779538e-07` for Transformer,
below the declared `2e-7` compatibility tolerance. Transformer inference was
bounded to batches of 64; CNN inference cast curves to `float32` before shared
normalization, matching training arithmetic.

They remain **compatibility diagnostics, not release model-selection evidence**.
The legacy CNN checkpoint lacks the current provenance fields. The legacy
Transformer checkpoint lacks `model_seed`, `split_seed`, `dataset_sha256`, and
`y_scaler_sha256`. Historical result JSON is therefore not bound to a complete
training lineage.

| Model | Mean RMSE | Median | P90 | Valid |
|---|---:|---:|---:|---:|
| CNN legacy artifact | 0.025668 | 0.015664 | 0.061484 | 1000/1000 |
| Transformer legacy artifact | 0.021602 | 0.011102 | 0.055305 | 998/1000 |

The apparent ordering is not a scientific comparison and must not be used for
model selection. The JSON files are retained so future readers can reproduce
the compatibility boundary and understand why earlier README/report tables were
withdrawn.

## Current architecture note

As of 2026-07-30, the canonical source instantiates a `4,381,319`-parameter CNN
and a `3,243,271`-parameter Transformer. Their analytical forward costs are
about `38.1M` and `780.2M` MAC per sample, respectively. Those source-derived
values do not validate a wall-clock speed claim for the legacy artifacts in
this directory. Current RMSE evidence, runtime boundaries, and the application
decision are maintained in
[`../MODEL_COMPARISON.md`](../MODEL_COMPARISON.md).

Publication requires one selected release dataset, current-schema retraining of
both architectures for seeds 42–46 with fixed `split_seed=42`, and reruns of all
checkpoint-dependent experiments. Each new checkpoint must record ordered
parameter names, model/split seeds, dataset/scaler hashes, selected epoch, and a
finite validation metric.
