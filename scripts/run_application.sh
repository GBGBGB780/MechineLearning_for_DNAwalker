#!/usr/bin/env bash
# One-command application runner for an experimental fluorescence workbook.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CALLER_CWD="$PWD"
PY="${DNAWALKER_PYTHON:-$ROOT/.venv/bin/python}"
EXP="$ROOT/data/experimental/Fig3a_fitting.xlsx"
MODEL="transformer"
RUN_NAME=""

usage() {
    cat <<'EOF'
Usage:
  bash scripts/run_application.sh [options]

Options:
  --exp FILE           Excel workbook with Time, FAM, TYE, and Cy5 columns.
  --model MODEL        transformer (default), cnn, or both.
  --run-name NAME      Output subdirectory name (default: workbook stem).
  --python FILE        Python interpreter (default: .venv/bin/python).
  -h, --help           Show this help.

Examples:
  bash scripts/run_application.sh --exp data/experimental/Fig3a_fitting.xlsx
  bash scripts/run_application.sh --exp /path/to/curves.xlsx --model both \
    --run-name experiment_01
EOF
}

require_value() {
    if [[ "$#" -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1." >&2
        usage >&2
        exit 2
    fi
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --exp)
            require_value "$@"
            EXP="$2"
            shift 2
            ;;
        --model)
            require_value "$@"
            MODEL="$2"
            shift 2
            ;;
        --run-name)
            require_value "$@"
            RUN_NAME="$2"
            shift 2
            ;;
        --python)
            require_value "$@"
            PY="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$MODEL" in
    transformer|cnn|both) ;;
    *)
        echo "--model must be transformer, cnn, or both; got: $MODEL" >&2
        exit 2
        ;;
esac

if [[ "$EXP" != /* ]]; then
    EXP="$CALLER_CWD/$EXP"
fi
if [[ "$PY" != /* ]]; then
    PY="$CALLER_CWD/$PY"
fi

if [[ ! -x "$PY" ]]; then
    echo "Python interpreter is not executable: $PY" >&2
    echo "Create .venv first or pass --python /absolute/path/to/python." >&2
    exit 2
fi
if [[ ! -f "$EXP" ]]; then
    echo "Experimental workbook not found: $EXP" >&2
    exit 2
fi

if [[ -z "$RUN_NAME" ]]; then
    WORKBOOK_NAME="$(basename "$EXP")"
    RUN_NAME="${WORKBOOK_NAME%.*}"
    RUN_NAME="$(printf '%s' "$RUN_NAME" | LC_ALL=C tr -c 'A-Za-z0-9._-' '_')"
fi
if [[ ! "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
    echo "Invalid --run-name: use only letters, numbers, dot, underscore, or hyphen." >&2
    exit 2
fi

cd "$ROOT"

run_transformer() {
    local params="$ROOT/results/predictions/transformer/$RUN_NAME/params.txt"
    echo "== Transformer prediction and physics refinement =="
    "$PY" -m dnawalker transformer predict --refine \
        --config configs/studies/nested_30k/transformer_seed46.ini \
        --transformer-config configs/studies/nested_30k/transformer_seed46.ini \
        --exp "$EXP" \
        --ensemble 20 \
        --noise-std 0.005 \
        --method Powell \
        --maxiter 500 \
        --multistart 8 \
        --seed 0 \
        --out "$params"
    "$PY" -m dnawalker verify "$params" --exp "$EXP"
}

run_cnn() {
    local params="$ROOT/results/predictions/cnn/$RUN_NAME/params.txt"
    echo "== CNN prediction and physics refinement =="
    "$PY" -m dnawalker cnn predict --refine \
        --config configs/studies/nested_30k/cnn_seed43.ini \
        --exp "$EXP" \
        --ensemble 20 \
        --noise-std 0.005 \
        --method Powell \
        --maxiter 500 \
        --multistart 8 \
        --seed 0 \
        --out "$params"
    "$PY" -m dnawalker verify "$params" --exp "$EXP"
}

case "$MODEL" in
    transformer)
        run_transformer
        ;;
    cnn)
        run_cnn
        ;;
    both)
        run_transformer
        run_cnn
        echo "Compare the two printed mean RMSE values; retain the lower-RMSE candidate."
        ;;
esac

echo "Application run complete: $RUN_NAME"
