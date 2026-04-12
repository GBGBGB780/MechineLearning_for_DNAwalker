#!/bin/bash
#PBS -N dna_1dcnn
#PBS -l select=1:ncpus=8:mem=50gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -q ais_gpu
#PBS -V

# 切换到脚本所在目录 / Change to script directory
cd $PBS_O_WORKDIR

# 加载 CUDA / Load CUDA
module load cuda/12.4

# 激活 Conda 环境 / Activate Conda environment
source /home/svu/e1554355/miniconda3/etc/profile.d/conda.sh
conda activate dna_env

# GPU 诊断 / GPU diagnostics
export CUDA_VISIBLE_DEVICES=0
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

# 执行训练 / Run training
python -u train_mlp.py
