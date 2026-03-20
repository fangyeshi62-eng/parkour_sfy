#!/bin/bash
#SBATCH -p gpu_4090
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
unset LD_LIBRARY_PATH
module load miniforge3/25.11.0-1
module load cuda/12.1

source /data/apps/miniforge3/25.11.0-1/etc/profile.d/conda.sh

# 3. 激活环境（现在 conda activate 命令可以被正常识别了）
conda activate /data/home/scxi678/run/parkour_env

#export LD_LIBRARY_PATH=/data/home/scxi678/run/parkour_env/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/data/home/scxi678/run/parkour_env/lib:$LD_LIBRARY_PATH
python -u train.py --task=ddog --headless