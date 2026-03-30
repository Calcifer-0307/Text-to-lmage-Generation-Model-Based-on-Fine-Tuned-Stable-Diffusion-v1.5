#!/bin/bash
#SBATCH --job-name=SD_Showcase
#SBATCH --output=logs/showcase_%j.out
#SBATCH --error=logs/showcase_%j.err
#SBATCH --partition=gpu-h100
#SBATCH --qos=qos-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

echo "Job started on $(hostname) at $(date)"

# 加载环境
source venv_hpc/bin/activate

# 运行生成脚本
python scripts/generate_showcase.py

echo "Job completed at $(date)"