# 30k Nested Learning-Curve Protocol

Protocol locked: **2026-07-23**, before the full training grid completed.
Result synchronized: **2026-07-30**.

## Question

The legacy single-checkpoint diagnostic favored Transformer by about `0.004066`
curve RMSE, while the current 10k-dataset five-seed study found a paired mean
difference near zero. This experiment tests whether that discrepancy is
explained by training-set size, using one master dataset and fixed row
membership throughout.

## Immutable Inputs

- Master dataset:
  `artifacts/studies/nested_learning_curve_30k/dnawalker_training_30k_seed42.npz`
- Dataset SHA-256:
  `f19f0ae0c63a104ea23c96cc83ce5ce1e0d22fd22c5e1a498dfe6ff53418de06`
- Split seed: `42`
- Model seeds: `42, 43, 44, 45, 46`
- Split manifest:
  `artifacts/studies/nested_learning_curve_30k/split_manifest.npz`
- Split-manifest SHA-256:
  `a8bd1b7ae578bf48efc5ffc69e4372a432be132963be0ae142f3bffe5c88acd0`

Generated datasets, checkpoints, scalers, logs, and result JSON remain ignored
external artifacts. Binary inputs and surviving model artifacts are retained
under `artifacts/studies/nested_learning_curve_30k/`; numeric results are
separated under `results/learning_curve/nested_30k/`. Neither directory is
added to Git. The only publication exception is a byte-identical copy of the
validation-selected seed-43 CNN pair, seed-46 Transformer pair, and required
split manifest under `artifacts/application/`; the master dataset is excluded.

## Fixed Partitions

The generator's activity score is the maximum channel range
`max_c(max_t X[c,t] - min_t X[c,t])`. The same five right-open activity bins
used during generation are applied to the 30,000 retained rows.

| Partition | Total | Rows per activity bin |
|---|---:|---:|
| Master dataset | 30,000 | 6,000 |
| Fixed test | 3,000 | 600 |
| Fixed validation | 3,000 | 600 |
| Training 8k | 8,000 | 1,600 |
| Training 16k | 16,000 | 3,200 |
| Training 24k | 24,000 | 4,800 |

The validation and test membership is identical at every training size and for
both architectures. Training membership is strictly nested:
`train_8k` is a subset of `train_16k`, which is a subset of `train_24k`.
At 24k, train, validation, and test form a disjoint exhaustive partition of all
30,000 rows.

## Retained-Distribution Audit

The generator uses LHS for each candidate batch, then applies physical-validity,
weak-signal, and activity-quota selection. Therefore LHS describes candidate
generation; it does not mathematically guarantee one retained row per marginal
LHS stratum. The actual 30k artifact has 30,000 unique parameter rows and exact
`6000 x 5` activity counts, but strict 30k-stratum occupancy is not preserved
after selection.

The table normalizes every parameter to its configured linear range. The KS
statistic compares the retained marginal with a continuous uniform
distribution; with 30,000 rows, practical effect size matters more than the
very small p-values.

| Parameter | Normalized mean | KS D vs uniform | Max deviation across 20 equal bins |
|---|---:|---:|---:|
| `E_b` | 0.5448 | 0.0850 | 47.3% |
| `E_b_azo_trans` | 0.6172 | 0.1935 | 93.5% |
| `E_b_azo_cis` | 0.4515 | 0.0776 | 29.9% |
| `k_mig` | 0.5032 | 0.0062 | 4.1% |
| `k0` | 0.2614 | 0.3817 | 460.5% |
| `drt_z` | 0.3943 | 0.1621 | 75.7% |
| `drt_s` | 0.4831 | 0.0281 | 11.5% |

The strongest retained linear association is
`corr(E_b, E_b_azo_cis) = 0.4385`. This is expected conditional-selection
structure, not a failure of the LHS candidate sampler. In particular, `k0`
strongly shifts the activity bin: its normalized mean rises from `0.068` in
the lowest-activity bin to `0.587` in the highest.

The split itself does not introduce a material new shift. Relative to the 30k
master, the fixed 3k test set has maximum normalized mean difference `0.0071`
and maximum two-sample KS statistic `0.0225`; every train/validation/test
partition is exactly activity-balanced. The earlier current-schema 10k
artifact is also exactly activity-balanced (`2000 x 5`), and its seven
parameter marginals are close to the 30k master (maximum two-sample KS
`0.0118`). Thus the experiment is a fair learning-curve comparison on the
generator's retained target distribution, but not proof over a uniform
seven-dimensional box.

## Training Grid

- Architectures: current standard CNN and Transformer.
- Training sizes: `8k`, `16k`, `24k`.
- Seeds per architecture and size: five (`42` through `46`).
- Total training runs: `2 x 3 x 5 = 30`.
- Hyperparameters, model capacity, preprocessing, loss weighting, schedulers,
  epoch limits, and early stopping remain the current defaults.
- Device: Apple MPS.
- Every checkpoint must bind dataset hash, scaler hash, model/split seed,
  ordered parameter names, split-manifest hash, and training-subset size.

The architecture and configured training budgets were fixed as follows:

| Property | CNN | Transformer |
|---|---:|---:|
| Trainable parameters | `4,381,319` | `3,243,271` |
| Analytical forward MAC/sample | about `38.1M` | about `780.2M` |
| Batch size | `256` | `64` |
| Epoch cap | `2000` | `300` |
| Early-stopping patience | `200` | `40` |

The MAC values are analytical architecture estimates, not measured MPS
latency. Training logs were not retained as a complete, same-boundary timing
dataset, so the grid supports RMSE and sample-accounting conclusions but no
wall-clock speed ratio. Different epoch caps and selected checkpoint epochs
also prevent epoch count from serving as a time proxy.

The current 10k dataset is not treated as the 8k point because LHS membership
depends on the requested generation size. All three learning-curve points come
from the same 30k master.

## Evaluation And Decision Rule

Each checkpoint is evaluated on all 3,000 fixed test rows without refinement.
The primary metric is mean curve RMSE over valid predicted forward simulations.
Every row is also accounted for as valid, invalid, or catastrophic/extreme.

For each size, seeds are paired and the primary architecture contrast is:

```text
Transformer curve RMSE - CNN curve RMSE
```

- Negative values favor Transformer.
- Positive values favor CNN.
- Report the five paired values, paired mean, sample SD, two-sided paired
  t-test, and 95% t confidence interval.
- Predeclared practical-equivalence margin: `[-0.001, +0.001]` RMSE.
- CI fully inside the margin: practical equivalence.
- CI wholly below `-0.001`: meaningful Transformer advantage.
- CI wholly above `+0.001`: meaningful CNN advantage.
- Every other outcome: inconclusive.

The p-value is descriptive and is not used alone to declare equivalence.

## Test Checklist

- [x] Dataset hash, finite values, shapes, and 30,000 unique parameter rows.
- [x] Master activity bins are exactly `6000 x 5`.
- [x] Test and validation bins are exactly `600 x 5`.
- [x] Training bins are exactly balanced at 8k, 16k, and 24k.
- [x] Train/validation/test partitions are disjoint.
- [x] Training subsets are strictly nested.
- [x] All parameter rows are within configured bounds.
- [x] Retained parameter marginals and correlations are audited separately
      from candidate LHS and activity-bin balance.
- [x] Fixed test representativeness is checked against the 30k master.
- [x] Current 10k and 30k retained parameter marginals are compared.
- [x] Real 30k one-epoch CNN and Transformer MPS smoke runs.
- [x] Canonical parameter counts are pinned at `4,381,319` for CNN and
      `3,243,271` for Transformer.
- [x] Explicit split produces `8000/3000/3000` in both loaders.
- [x] Full local regression suite before the grid: 458 passed.
- [x] All 30 training runs completed successfully.
- [x] Every evaluation satisfies count conservation.
- [x] Strict two-model validation passes at all three sizes.
- [x] Paired statistics and the learning-curve figure were regenerated from JSON.
- [x] Results were compared with both legacy and current 10k evidence.
- [x] Documentation, Notebook, private report, and local task records were
      synchronized at project closeout.
- [x] Final full test/static/artifact gates pass on the resulting source tree.

## Results

The primary contrast is Transformer curve RMSE minus CNN curve RMSE. Positive
values favor CNN. The confidence intervals and decisions use the rule locked
above.

| Train | CNN mean +/- SD | Transformer mean +/- SD | Mean difference (95% CI) | p | Decision |
|---:|---:|---:|---:|---:|---|
| 8,000 | 0.01953252 +/- 0.00066847 | 0.02365253 +/- 0.00357211 | +0.00412001 [-0.00080583, +0.00904584] | 0.08094 | Inconclusive |
| 16,000 | 0.01833638 +/- 0.00046472 | 0.02020585 +/- 0.00185304 | +0.00186948 [-0.00074115, +0.00448010] | 0.1177 | Inconclusive |
| 24,000 | 0.01811960 +/- 0.00053594 | 0.01817831 +/- 0.00149517 | +0.00005871 [-0.00225976, +0.00237718] | 0.9473 | Inconclusive |

All 30 runs completed, and each architecture/size combination accounts for
`5 x 3000 = 15000` attempted test rows. The aggregate valid/invalid/extreme
counts are:

| Train | CNN | Transformer |
|---:|---:|---:|
| 8,000 | 14998 / 2 / 0 | 14978 / 15 / 7 |
| 16,000 | 14994 / 0 / 6 | 14986 / 9 / 5 |
| 24,000 | 14989 / 3 / 8 | 14980 / 11 / 9 |

Both architectures improve as training data increase. The Transformer-minus-CNN
mean contracts from `+0.00412001` at 8k to `+0.00005871` at 24k. The 24k point
estimate is a near-tie and agrees in scale with the separate current 10k result
of `-0.00006203`, but its 95% interval extends beyond the predeclared
`[-0.001,+0.001]` region. Five seeds therefore do not establish practical
equivalence.

This result compares predictive error, not elapsed training or inference
speed. The lower CNN analytical MAC count supports a lower-compute expectation,
but no current-lineage wall-clock measurement was part of the locked decision
rule.

The legacy diagnostic gives `-0.00406635`, but it belongs to checkpoints and a
dataset without the current lineage. Repository restructuring preserved the
legacy artifact hashes and compatibility outputs; it did not change a controlled
training result. The apparent order changes across legacy, current 10k, and
nested 30k evidence because those are different dataset/checkpoint lineages.
Only the three nested points isolate training-set size within one lineage.

The retained-distribution audit limits the scope of the result. Candidate
parameters are sampled by LHS, then filtered by physical validity, signal, and
activity quota. The retained dataset is exactly activity-balanced and contains
30,000 unique parameter rows, but its final marginals are not uniform. The
strongest retained association is
`corr(E_b, E_b_azo_cis)=0.43853`. Relative to the master, the fixed test set has
maximum normalized mean difference `0.007086` and maximum two-sample KS
`0.0225`, so the split adds little shift within the retained distribution.

### Machine-readable outputs

| Output | SHA-256 |
|---|---|
| `learning_curve_metrics.json` | `32f95cced73d1ff57d81a9d9e3247e0a007fe26210b947df956ec02166548975` |
| `learning_curve_compare.png` | `c32dd11811d49ea1e7dda8ab80f24ced533e06e7e67990ee554acae9ec169345` |
| `learning_curve_report.md` | `42eb56a705739f924f1182e5fcac746136af268196f8e421bead44b30a016867` |

The numeric 8k rows, including counts, epochs, validation MSE, and the five CNN
checkpoint hashes, were recovered from append-only session-log lines 3582,
3601, 3602, and 5873 after temporary files were removed. The combined source
SHA-256 is
`5c0a99bfab667b917be6e91e5cb32794550bf6e2cc64001b567dd5fdcf07e520`.
The five original 8k Transformer checkpoint hashes could not be recovered; their
part records explicitly mark sentinel hashes. Those five checkpoint binaries
are not independently verifiable, even though their recorded evaluation rows
remain usable for this discrepancy analysis.

## Interpretation Boundary

This protocol can distinguish a stable architecture effect, practical
equivalence at the declared resolution, and unresolved seed variability. The
observed outcome is unresolved seed variability around a 24k near-tie, not a
proof of equivalence. It
does not prove universal superiority outside these parameter ranges, activity
selection rules, architectures, or optimization settings. Activity balancing
also does not imply uniform final marginal distributions for all seven physical
parameters.
