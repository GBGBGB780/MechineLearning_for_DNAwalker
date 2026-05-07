$ErrorActionPreference = "Stop"

$experimentsRoot = Split-Path -Parent $PSScriptRoot
$trainTransformerRoot = Split-Path -Parent $experimentsRoot
$repoRoot = Split-Path -Parent $trainTransformerRoot
$artifacts = Join-Path $PSScriptRoot "artifacts"
$baseConfig = Join-Path $repoRoot "configfile.ini"
$generalizationConfig = Join-Path $repoRoot "configfile.generalization.ini"
$modelPath = Join-Path $artifacts "best_transformer_model.current_baseline_m03.pth"
$scalerPath = Join-Path $artifacts "transformer_y_scaler.current_baseline_m03.pkl"

if (-not (Test-Path $baseConfig)) {
  throw "Missing config file: $baseConfig"
}
if (-not (Test-Path $generalizationConfig)) {
  throw "Missing config file: $generalizationConfig"
}
if (-not (Test-Path $modelPath)) {
  throw "Missing model file: $modelPath"
}
if (-not (Test-Path $scalerPath)) {
  throw "Missing scaler file: $scalerPath"
}

python (Join-Path $PSScriptRoot "evaluate_legacy_mha_rmse.py") `
  --config-base $baseConfig `
  --config-generalization $generalizationConfig `
  --transformer-config (Join-Path $PSScriptRoot "config_transformer.current_baseline.ini") `
  --model $modelPath `
  --scaler $scalerPath `
  --output-dir $artifacts `
  --tag "m03_mha_weighted_base_local_eval"
