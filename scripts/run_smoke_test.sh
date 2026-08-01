#!/usr/bin/env bash
# 端到端冒烟测试 (macOS/Linux，纯 Python，无需 MATLAB)
# End-to-end smoke test (macOS/Linux, pure Python, no MATLAB required).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "未找到虚拟环境解释器: $PY"
    echo "请先创建环境：uv venv --python 3.12 .venv && uv pip install -r requirements.txt"
    exit 1
fi

SMOKE_WORKERS="${SMOKE_WORKERS:-1}"
SMOKE_PARAMS="results/predictions/transformer/smoke/matlab_input_params.txt"
SMOKE_FIGURE="results/evaluation/transformer/smoke/matlab_input_params_verify.png"

# 1. 小数据集；2. CNN 冒烟训练；3. Transformer 冒烟训练；
# 4. Transformer 预测；5. 正向模拟验证。
echo "== 1. 生成冒烟数据集 (Python) =="
"$PY" -m dnawalker data generate \
    --smoke --workers "$SMOKE_WORKERS"

echo "== 2. CNN 冒烟训练 =="
"$PY" -m dnawalker cnn train --config configs/profiles/smoke.ini

echo "== 3. Transformer 冒烟训练 =="
"$PY" -m dnawalker transformer train \
    --config configs/profiles/smoke.ini \
    --transformer-config configs/profiles/smoke.ini \
    --smoke

echo "== 4. Transformer 预测 =="
"$PY" -m dnawalker transformer predict \
    --config configs/profiles/smoke.ini \
    --transformer-config configs/profiles/smoke.ini \
    --out "$SMOKE_PARAMS"

echo "== 5. 正向验证 (Python) =="
"$PY" -m dnawalker verify \
    "$SMOKE_PARAMS" \
    --out "$SMOKE_FIGURE"

echo "冒烟测试完成。"
