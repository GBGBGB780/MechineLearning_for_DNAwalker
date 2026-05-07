$ErrorActionPreference = "Stop"

$experimentsRoot = Split-Path -Parent $PSScriptRoot
$trainTransformerRoot = Split-Path -Parent $experimentsRoot
$repoRoot = Split-Path -Parent $trainTransformerRoot
$datasetPath = Join-Path $repoRoot "results\\training_dataset.npz"

if (-not (Test-Path $datasetPath)) {
  throw "Missing dataset file: $datasetPath"
}

python (Join-Path $PSScriptRoot "train_legacy_mha.py") `
  --config-base (Join-Path $repoRoot "configfile.ini") `
  --transformer-config (Join-Path $PSScriptRoot "config_transformer.current_baseline.ini") `
  --adaptive-weight
