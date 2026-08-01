#!/bin/bash
#PBS -N dna_transformer
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V

set -euo pipefail

# Submit from the repository root, or set PROJECT_DIR explicitly.
WORKDIR="${PROJECT_DIR:-${PBS_O_WORKDIR:-}}"
if [[ -z "${WORKDIR}" ]]; then
    WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "${WORKDIR}"
[[ -f pyproject.toml && -d dnawalker ]] || {
    echo "DNA Walker repository not found in ${WORKDIR}; set PROJECT_DIR to its root." >&2
    exit 2
}

if command -v module >/dev/null 2>&1; then
    module load "${CUDA_MODULE:-cuda/12.4}"
fi
if [[ -n "${CONDA_SH:-}" ]]; then
    source "${CONDA_SH}"
elif command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Conda not found; set CONDA_SH=/path/to/conda.sh." >&2
    exit 2
fi
conda activate "${CONDA_ENV:-dna_env}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
python -u -m dnawalker transformer train \
    --config "${DNA_CONFIG:-configs/common.ini}" \
    --transformer-config "${TRANSFORMER_CONFIG:-configs/transformer.ini}"
