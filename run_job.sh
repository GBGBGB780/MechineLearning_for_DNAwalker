#!/bin/bash
#PBS -N dna_1dcnn
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V

# 1. 切换目录
cd $PBS_O_WORKDIR

# 2. 加载模块 (注意：Volta 节点通常匹配 CUDA 11 或 12)
module load cuda/12.4

# 3. 激活 Conda 环境
source /home/svu/e1554355/miniconda3/etc/profile.d/conda.sh
conda activate dna_env

# 4. 诊断：增加环境变量强制检测
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print('GPU 可用性:', torch.cuda.is_available()); print('显卡型号:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 5. 执行训练
python -u train_mlp.py
