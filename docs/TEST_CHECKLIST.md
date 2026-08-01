# Final Project Test Checklist

Last synchronized: **2026-07-30**

This checklist separates three gate classes:

- **P0 engineering gates:** required before merging the local closeout.
- **P1 scientific gates:** required before citing model or fit results.
- **P2 portability gates:** required for a public cross-machine release.

The local project closeout can complete after P0 and the scoped P1 claims pass.
P2 remains a separate future publication task.

## 1. Current 10k Five-Seed Lineage

- [x] Selected hardened seed-42 dataset:
      `artifacts/releases/retrain-3a5a494-ds557506e93079/training_dataset.npz`
      with SHA-256
      `557506e93079d1e5158aa30db7fd4a555234bb751e46049639b1fc827dca9b68`.
- [x] CNN and Transformer seeds 42-46 completed with fixed
      `split_seed=42`.
- [x] All 10 checkpoints contain ordered parameter names, model/split seeds,
      dataset/scaler hashes, a positive epoch, finite validation MSE, and
      finite model weights.
- [x] All 10 scalers satisfy
      `MinMaxScaler(feature_range=(0.1, 0.9), n_features=7, n_samples=8000)`.
- [x] Apple MPS held-out evaluation and strict merge completed:

| Architecture | Curve RMSE, mean +/- sample SD | Valid / invalid / extreme |
|---|---:|---:|
| CNN | `0.0212486477 +/- 0.0010872001` | `5000 / 0 / 0` |
| Transformer | `0.0211866192 +/- 0.0006457182` | `4989 / 8 / 3` |

- [x] The paired Transformer-minus-CNN mean is `-0.00006203`;
      exploratory `p=0.915`, with a 95% interval crossing zero.
- [x] Every attempted row is conserved.
- [x] Strict merged JSON SHA-256:
      `231bf138701f79f1fe56ca3cf749110562d5ef0fa574d2515a1b97ab912114b6`.

Conclusion: the controlled 10k study does not support a stable architecture
advantage.

## 2. Fixed 30k Nested Learning Curve

- [x] Master dataset contains 30,000 unique parameter rows.
- [x] Dataset SHA-256:
      `f19f0ae0c63a104ea23c96cc83ce5ce1e0d22fd22c5e1a498dfe6ff53418de06`.
- [x] Explicit split-manifest SHA-256:
      `a8bd1b7ae578bf48efc5ffc69e4372a432be132963be0ae142f3bffe5c88acd0`.
- [x] Validation and test each contain 3,000 fixed rows.
- [x] Training subsets are strictly nested at 8k, 16k, and 24k.
- [x] Master, validation, test, and every training subset have exact
      five-bin activity balance.
- [x] Train/validation/test membership is disjoint; the 24k split exhausts
      the 30k master.
- [x] CNN and Transformer seeds 42-46 completed: `30/30` MPS runs.
- [x] Every architecture/size group conserves `5 x 3000 = 15000` test rows.
- [x] The predeclared contrast is Transformer minus CNN with practical
      equivalence region `[-0.001, +0.001]`.

| Train | CNN mean +/- SD | Transformer mean +/- SD | Mean difference (95% CI) | Decision |
|---:|---:|---:|---:|---|
| 8k | `0.01953252 +/- 0.00066847` | `0.02365253 +/- 0.00357211` | `+0.00412001 [-0.00080583, +0.00904584]` | Inconclusive |
| 16k | `0.01833638 +/- 0.00046472` | `0.02020585 +/- 0.00185304` | `+0.00186948 [-0.00074115, +0.00448010]` | Inconclusive |
| 24k | `0.01811960 +/- 0.00053594` | `0.01817831 +/- 0.00149517` | `+0.00005871 [-0.00225976, +0.00237718]` | Inconclusive |

- [x] Both architectures improve with more training data.
- [x] The 24k means are a near-tie, but five seeds do not prove equivalence.
- [x] LHS is correctly described as candidate sampling. Filtering and
      activity quotas make retained marginals non-uniform.
- [x] Strongest retained correlation:
      `corr(E_b, E_b_azo_cis)=0.43853`.
- [x] Fixed-test representativeness: maximum normalized mean difference
      `0.007086`; maximum two-sample KS statistic `0.0225`.
- [x] Summary artifact SHA-256 values:

| Output | SHA-256 |
|---|---|
| `learning_curve_metrics.json` | `32f95cced73d1ff57d81a9d9e3247e0a007fe26210b947df956ec02166548975` |
| `learning_curve_compare.png` | `c32dd11811d49ea1e7dda8ab80f24ced533e06e7e67990ee554acae9ec169345` |
| `learning_curve_report.md` | `42eb56a705739f924f1182e5fcac746136af268196f8e421bead44b30a016867` |

- [x] Ten 8k numeric rows were recovered from append-only session-log
      evidence with source SHA-256
      `5c0a99bfab667b917be6e91e5cb32794550bf6e2cc64001b567dd5fdcf07e520`.
- [x] Five original 8k Transformer checkpoint hashes and binaries remain
      unavailable and are not described as independently reproducible.

## 3. Final Experimental-Fit Robustness

- [x] Checkpoint selection was declared independently of experimental fit:
      minimum validation MSE across model seeds 42-46.
- [x] Selected CNN: 24k seed 43, validation MSE `0.0210643411`.
- [x] Selected Transformer: 24k seed 46, validation MSE `0.0210173298`.
- [x] Checkpoint SHA-256:

| Architecture | SHA-256 |
|---|---|
| CNN | `1fc8b1903e9105ac5a9a93cf9643e6059ea5fb8f260ae62edadc0467073da31e` |
| Transformer | `8a5420d3e7e5d3053c389900d3c4a45f0ee080061f63c81d95bef22ff2279b7e` |

- [x] Shared scaler SHA-256:
      `15ea872072c1c379fe81cf7cdf8eba463173fc8c3a9d144a9ca237705182ca8a`.
- [x] Both models used identical settings: ensemble 20, noise standard
      deviation 0.005, Powell refinement, 8 starts, 500 iterations, and
      refinement RNG seeds 0-4.
- [x] Ten evaluations completed on Apple MPS.
- [x] Every per-seed JSON is schema v2 and binds experimental-input hashes,
      exact settings, model/checkpoint/scaler/dataset/split hashes, subset
      size, and device.
- [x] No final evaluation provenance path references `/private/tmp`.
- [x] All figures contain English-only titles, labels, and legends.

| Architecture | Original median (range) | Generalization median (range) | Combined median (range) |
|---|---:|---:|---:|
| CNN | `0.016290 (0.007779-0.020323)` | `0.016038 (0.015956-0.033571)` | `0.016303 (0.016082-0.020675)` |
| Transformer | `0.007806 (0.007796-0.008220)` | `0.019721 (0.016087-0.024852)` | `0.013767 (0.011946-0.016328)` |

- [x] Aggregate JSON SHA-256:
      `3e2ca90eb303f048eaaab4d206a56e7b0390ed26f1e0315bc5efcba92828d880`.
- [x] Aggregate PNG SHA-256:
      `643bf97d886018e071394eaeaeb396ba34b88c4b798139e2a063b1813530fe05`.
- [x] Interpretation is limited to refinement-start sensitivity for one
      validation-selected checkpoint per architecture.

Conclusion: further model training is not justified by the current evidence.
The remaining variation is primarily a non-convex refinement issue.

## 4. Artifact and Evidence Layout

- [x] Binary datasets, split manifests, checkpoints, and scalers are under
      `artifacts/`.
- [x] The minimal selected inference bundle is versioned under
      `artifacts/application/` with `SHA256SUMS`; it contains no training
      dataset.
- [x] Publication-safe metrics and figures are under `docs/evidence/`.
- [x] The selected CNN and Transformer experimental-fit figures are retained
      separately under `docs/evidence/`.
- [x] Only the aggregate cross-model robustness JSON and figure are published.
- [x] The nested study's final metrics and comparison figure are published.
- [x] The permanent 30k binary inputs are under
      `artifacts/studies/nested_learning_curve_30k/`.
- [x] Current 10k release evidence is represented by the curated manifest and
      comparison figure.
- [x] The complete raw `results/` tree was removed after curation; it is not
      part of the streamlined working tree.
- [x] Regenerable caches, `.DS_Store`, duplicate result trees, interrupted
      run directories, and withdrawn non-release outputs were removed.
- [x] Historical datasets/checkpoints with independent hashes or remaining
      references were retained rather than treated as caches.

## 5. Local Engineering Baseline

Reviewed environment:

- macOS 26.5.2
- Apple M4 Pro
- Python 3.12.13
- PyTorch 2.12.0
- NumPy 2.4.6
- Apple MPS available

Historical reorganization checks:

- [x] Frozen dataset shape `X=(10000,3,7801)`, `Y=(10000,7)`.
- [x] All arrays finite with canonical parameter order.
- [x] Legacy CPU/MPS aggregate differences remain below `2e-7`.
- [x] Two forced-CPU smoke runs produced identical hashes.
- [x] Native MPS smoke completed.
- [x] The seven reviewed `artifacts.sha256` entries remained unchanged.

Current architecture-size checks:

- [x] Canonical CNN trainable parameter count is exactly `4,381,319`.
- [x] Canonical Transformer trainable parameter count is exactly `3,243,271`.
- [x] The two counts are instantiated from the production configuration and
      pinned by regression tests.
- [x] Analytical forward costs are documented as approximately `38,067,168`
      MAC/sample for CNN and `780,223,360` MAC/sample for Transformer.
- [x] MAC values are labeled as analytical estimates, not measured latency.
- [x] No current-lineage wall-clock training or inference speed ratio is
      claimed without a same-device controlled benchmark.

## 6. Pytest Inventory

Final collected suite after the application-runner addition:

| Test file | Cases |
|---|---:|
| `test_audit_results.py` | 2 |
| `test_computation_only.py` | 8 |
| `test_config_loader.py` | 46 |
| `test_ensemble.py` | 18 |
| `test_eval_testset_common.py` | 7 |
| `test_exp_data_io.py` | 3 |
| `test_fit_robustness.py` | 4 |
| `test_gendata_config.py` | 26 |
| `test_identifiability.py` | 19 |
| `test_inference_validation.py` | 38 |
| `test_learning_curve.py` | 10 |
| `test_multiseed_modules.py` | 19 |
| `test_multiseed_retrain.py` | 35 |
| `test_package_entrypoints.py` | 27 |
| `test_param_io.py` | 15 |
| `test_preprocessing.py` | 10 |
| `test_pysim_counter.py` | 8 |
| `test_pysim_physics.py` | 16 |
| `test_refine.py` | 18 |
| `test_signal_evidence.py` | 4 |
| `test_training_splits.py` | 3 |
| `test_transformer_config.py` | 14 |
| `test_utils_cli.py` | 6 |
| `test_validation_common.py` | 45 |
| **Total** | **401** |

## 7. P0 Final Engineering Gates

### Source and tests

- [x] Full model-first local suite passes: **401 passed, 1 known warning**.

  ```bash
  .venv/bin/python -m pytest
  ```

- [x] Tracked and untracked Python pass pyflakes.
- [x] Compile check passes:

  ```bash
  .venv/bin/python -m compileall -q dnawalker tests
  ```

- [x] Dependency check passes:

  ```bash
  .venv/bin/python -m pip check
  ```

- [x] Shell syntax checks pass:

  ```bash
  bash -n scripts/run_application.sh scripts/run_smoke_test.sh \
    scripts/hpc/cnn.pbs.sh scripts/hpc/transformer.pbs.sh
  ```

- [x] Bundled application artifact hashes pass:

  ```bash
  (cd artifacts/application && shasum -a 256 -c SHA256SUMS)
  ```

- [x] A clean Python 3.12 CPU staging copy with no training dataset loads both
      bundled predictors and produces finite `(1, 7)` outputs.

### Structured deliverables

- [x] All tracked JSON files parse strictly without NaN/Infinity.
- [x] Notebook parses as JSON and contains English-only visible content.
- [x] All 8 curated evidence PNG files decode and contain nonblank pixels.
- [x] Artifact hashes match their manifests and documented values.
- [x] `git diff --check` passes.

### Commit

- [x] Final changes are committed.
- [x] Post-commit worktree is clean.

## 8. P1 Scientific Gate Decision

Included final claims:

- [x] Current 10k five-seed held-out comparison.
- [x] Fixed 30k nested learning curve.
- [x] Retained-distribution audit.
- [x] Checkpoint-independent identifiability and signal diagnostics.
- [x] Experimental-fit refinement robustness for one validation-selected
      checkpoint per architecture.

Explicitly outside the final claim set:

- [x] Synthetic parameter-recovery accuracy.
- [x] DL warm-start advantage.
- [x] Corrected wall-clock speedup.
- [x] Capacity-ablation effect.

Their historical numerical claims, implementations, tests, configurations,
and generated outputs were removed from the streamlined release tree. The
pre-cleanup source remains recoverable at Git commit `2e280df`.

## 9. P2 Public-Release Gates

- [x] Curated public evidence contains no known local path or student-ID
      markers.
- [x] Private report, Kiro records, raw result trees, and unlicensed Excel
      inputs are excluded from the current public-source snapshot.
- [x] Selected CNN/Transformer inference artifacts and their required split
      manifest are included and verified by `artifacts/application/SHA256SUMS`.
- [x] The streamlined source tree passes the complete local suite:
      **401 passed, 1 known warning**.
- [ ] Run P0 in a fresh Linux CPU environment using `requirements-lock.txt`.
- [ ] Run smoke and formal evaluation on CUDA; record driver, CUDA, PyTorch,
      and metric tolerances.
- [x] Verify bundled application artifacts by SHA-256.
- [ ] Publish CPU/MPS/CUDA environment manifests and metric comparisons.
- [ ] Provide permanent URLs and source/license statements for any separately
      distributed dataset, experimental workbook, or non-selected artifact.
- [ ] Confirm remote CI on the final pushed commit.

The bundled models support local inference, but the remaining P2 items block a
formal cross-platform reproducibility claim. Existing local Git history
contains files excluded from the sanitized snapshot and must not be pushed
as-is; use the history-free export procedure in `docs/PUBLICATION.md`.

## 10. Stop Conditions

Do not publish a scientific result if any required P0 check fails, a split
overlaps, provenance is missing or mismatched, JSON contains non-finite values,
an artifact hash changes unexpectedly, failed seeds are silently omitted, or a
documented number cannot be traced to machine-readable evidence.
