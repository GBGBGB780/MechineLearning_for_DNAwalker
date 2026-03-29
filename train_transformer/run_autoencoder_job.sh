#!/bin/bash
#PBS -N dna_tf_encoder
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V

# =====================================================================
# Step 1: 训练纯 Transformer Encoder (参数预测)
#
# 说明:
#   - 使用 train_transformer.py 训练纯 Encoder (曲线 → 参数)
#   - 训练完成后模型保存在 results/best_transformer_model.pth
#   - 后续使用 run_decoder_phase2_job.sh 训练 Decoder + 数字孪生 Phase 2
#
# 工作流:
#   Step 1: qsub run_job.sh      
#   Step 2: qsub run_autoencoder_job.sh  ← 你在这里: 训练 Decoder + Phase 2
# =====================================================================

# 1. 切换目录
cd $PBS_O_WORKDIR

# 2. 加载模块
module load cuda/12.4

# 3. 激活 Conda 环境
source /home/svu/e1554355/miniconda3/etc/profile.d/conda.sh
conda activate dna_env

# 4. 诊断
export CUDA_VISIBLE_DEVICES=0
echo "=============================="
echo "  Job ID:    $PBS_JOBID"
echo "  Node:      $(hostname)"
echo "  Time:      $(date)"
echo "  Task:      Transformer Encoder 训练 (Step 1)"
echo "=============================="
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 5. 执行 Encoder 训练
echo ""
echo "开始训练 Transformer Encoder (纯参数预测模式)..."
echo ""
python -u train_transformer.py

# 6. 提示下一步
echo ""
echo "========================================"
echo "  Encoder 训练完成!"
echo "  模型保存在: results/best_transformer_model.pth"
echo ""
echo "  下一步: qsub run_decoder_phase2_job.sh"
echo "========================================"