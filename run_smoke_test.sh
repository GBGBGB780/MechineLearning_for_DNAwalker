#!/usr/bin/env bash
# 端到端冒烟测试 (Mac/Linux 版，纯 Python，无需 MATLAB)
# End-to-end smoke test (Mac/Linux, pure Python, no MATLAB required).
#
# 步骤 / Steps:
#   1. gendata.py 生成 20 样本小数据集 (替代 MATLAB gendata_smoke.m)
#   2. CNN 冒烟训练 (2 epochs)
#   3. Transformer 冒烟训练 (2 epochs)
#   4. Transformer 预测 → matlab_input_params.txt
#   5. verify.py 正向验证 + RMSE (替代 MATLAB verify.m)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PY="$ROOT/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "未找到虚拟环境解释器: $PY"
    echo "请先创建环境：uv venv --python 3.12 .venv && uv pip install -r requirements.txt"
    exit 1
fi

echo "== 1. 生成冒烟数据集 (Python) =="
"$PY" gendata.py --smoke

echo "== 2. CNN 冒烟训练 =="
( cd train_cnn && "$PY" train_mlp.py --config ../configfile.smoke.ini )

echo "== 3. Transformer 冒烟训练 =="
( cd train_transformer && "$PY" train_transformer.py \
    --config ../configfile.smoke.ini \
    --transformer-config config_transformer.smoke.ini --smoke )

echo "== 4. Transformer 预测 =="
( cd train_transformer && "$PY" predict.py \
    --config ../configfile.smoke.ini \
    --transformer-config config_transformer.smoke.ini )

echo "== 5. 正向验证 (Python) =="
"$PY" verify.py train_transformer/matlab_input_params.txt

echo "冒烟测试完成。"
