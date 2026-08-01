# External Experimental Inputs

## Workbook format

Application workbooks must contain at least four columns:

| Position when no keyword matches | Recommended column name | Content |
|---:|---|---|
| 1 | `Time` | Time in minutes |
| 2 | `FAM` | FAM fluorescence |
| 3 | `TYE` | TYE fluorescence |
| 4 | `Cy5` | Cy5 fluorescence |

Column matching is case-insensitive and searches for `time`, `fam`, `tye`, and
`cy5`. If a keyword is absent, the loader uses the positional fallback above.
Every signal must contain at least two distinct time points with complete,
finite numeric values. The loader removes incomplete rows, sorts time, averages
duplicate time points, and interpolates all channels onto the simulation axis.
The active model window is 0-130 minutes. A workbook should cover that window;
rows after 130 minutes are not used by the current model input.

The current trained models apply only to the same three-channel, 130-minute
illumination protocol, channel meanings, and configured physical parameter
ranges. A changed protocol or molecular design requires a new dataset lineage
and retraining.

## One-command application

Run the current recommended Transformer branch:

```bash
bash scripts/run_application.sh \
  --exp /absolute/path/to/curves.xlsx \
  --model transformer \
  --run-name experiment_01
```

Use `--model cnn` for the lower-compute branch or `--model both` to run and
verify both. In `both` mode, retain the candidate with the lower printed mean
forward RMSE. The script defaults to `.venv/bin/python`; override it with
`--python` or `DNAWALKER_PYTHON`.

The underlying model commands also accept `--exp FILE`, and forward
verification accepts the same workbook through `--exp FILE`. Parameter files
are written under `results/predictions/<model>/<run-name>/`; verification
figures are written under the corresponding
`results/evaluation/<model>/<run-name>/` directory.

## Reviewed local inputs

The evaluation commands expect two local workbooks in this directory:

| Filename | SHA-256 used for the final local study |
|---|---|
| `Fig3a_fitting.xlsx` | `0518081ce35d750ae017bb9ca7d615b221e1ea1e778f4b3bbc5fedabccb2a419` |
| `Fig3a_fitting_generalization.xlsx` | `2e35c5c8b71725c52a2e12e257080549f1ded49c348a48b25886f55b4a0bc0bd` |

The workbooks are intentionally excluded from the public source snapshot until
their source statements and redistribution rights are documented. This README
records expected identity only; it does not grant a license or provide a
download.

Place licensed copies at the paths above before running experimental prediction
or dual-dataset evaluation. Synthetic generation, unit tests, and the core
simulator do not require these workbooks.
