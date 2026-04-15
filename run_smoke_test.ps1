$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw "python is not available in PATH."
}

if (-not (Get-Command matlab -ErrorAction SilentlyContinue)) {
    throw "matlab is not available in PATH."
}

Write-Host "== MATLAB smoke dataset generation =="
matlab -batch "gendata_smoke"

Write-Host "== Convert MAT to NPZ =="
python .\utils\mat_to_npz.py .\results\training_dataset_smoke.mat

Write-Host "== CNN smoke training =="
python .\train_cnn\train_mlp.py --config .\configfile.smoke.ini

Write-Host "== Transformer smoke training =="
python .\train_transformer\train_transformer.py --config .\configfile.smoke.ini --transformer-config .\train_transformer\config_transformer.smoke.ini --smoke

Write-Host "Smoke test completed."
