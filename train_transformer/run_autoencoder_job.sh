#!/bin/bash
#PBS -N dna_tf_ae
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V

# =====================================================================
# DNA Walker Transformer Encoder-Decoder 训练作业
# 支持 HPC 断点续训: 作业自动从上次 checkpoint 恢复
#
# 用法:
#   首次提交:    qsub run_autoencoder_job.sh
#   断点续训:    qsub run_autoencoder_job.sh    (完全相同，自动恢复)
#
# 说明:
#   - 脚本会自动检测是否存在 checkpoint，有则恢复、无则从头开始
#   - 训练程序在 23h 时自动保存并退出（留 1h 余量给系统）
#   - 建议: 训练完一轮后 qsub 再次提交即可，直到全部完成
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
echo "=============================="
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# 5. 检查是否存在 checkpoint → 自动决定是否 --resume
CHECKPOINT_FILE="results/transformer_autoencoder_checkpoint.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "发现 checkpoint 文件，将从断点恢复训练..."
    echo ""
    python -u train_transformer_autoencoder.py --resume --max-hours 23
else
    echo ""
    echo "未发现 checkpoint，将从头开始训练..."
    echo ""
    python -u train_transformer_autoencoder.py --max-hours 23
fi

# 6. 检查训练是否完成（checkpoint 被清理 = 训练完成）
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo ""
    echo "训练已全部完成！"
else
    echo ""
    echo "本轮训练因时间限制暂停，请再次提交作业继续训练:"
    echo "   cd $(pwd) && qsub run_autoencoder_job.sh"
fi
