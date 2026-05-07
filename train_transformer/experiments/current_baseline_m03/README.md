# Current Baseline `m03`

This folder packages the current local baseline based on the recovered legacy MHA branch and the `m03_mha_weighted_base` checkpoint from the HPC.

Why this baseline:

- It is the strongest completed result on the recovered old MHA line.
- It preserves the older transformer behavior better than the later surrogate-heavy branches.
- It is now fully local: model, scaler, RMSE summaries, legacy architecture, train script, and eval script are all kept together here.

Observed RMSE from the shipped checkpoint:

- Original: `0.01054 / 0.01817 / 0.00570` (`avg=0.01147`)
- Generalization: `0.01630 / 0.07063 / 0.01458` (`avg=0.03384`)
- Combined: `mean_avg=0.02265`, `max_avg=0.03384`

Contents:

- `artifacts/best_transformer_model.current_baseline_m03.pth`
- `artifacts/transformer_y_scaler.current_baseline_m03.pkl`
- `artifacts/best_transformer_model.m03_mha_weighted_base.pth`
- `artifacts/transformer_y_scaler.m03_mha_weighted_base.pkl`
- `artifacts/rmse_summary.m03_mha_weighted_base.*.json`
- `model_transformer_mha_legacy.py`
- `train_legacy_mha.py`
- `evaluate_legacy_mha_rmse.py`
- `config_transformer.current_baseline.ini`

From the repo root:

Evaluate the shipped checkpoint:

```powershell
.\train_transformer\experiments\current_baseline_m03\run_current_baseline_eval.ps1
```

Reproduce the baseline training locally:

```powershell
.\train_transformer\experiments\current_baseline_m03\run_current_baseline_train.ps1
```

Notes:

- The eval script uses `configfile.ini` and `configfile.generalization.ini`.
- The legacy MHA model in this folder is self-contained, so this baseline does not depend on the rollback experiment directory.
- Local training reproduction expects `results/training_dataset.npz` at the repo root. That dataset is not bundled inside this baseline folder.
- `matlab_exit_code = 1` in the stored summaries is the known `verify.m` tail-end plotting issue; the RMSE values remain usable because `result_signal.txt` was generated successfully before that failure.
- The `current_baseline_m03` artifact names are just local aliases of the shipped `m03_mha_weighted_base` files, so later experiments can reference a stable baseline path.
