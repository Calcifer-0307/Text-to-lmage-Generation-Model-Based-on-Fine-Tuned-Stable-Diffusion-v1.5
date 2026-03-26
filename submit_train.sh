#!/bin/bash

# ============================================================
# SLURM 资源申请配置
# ============================================================
#SBATCH --job-name=SD_LoRA_Train      # 任务名称
#SBATCH --partition=gpu-a100          # 分区名称（使用sinfo可获取HPC支持的分区名称）
#SBATCH --nodes=1                     # 申请 1 个节点
#SBATCH --ntasks-per-node=1           # 每个节点运行 1 个任务
#SBATCH --cpus-per-task=4             # 每个任务分配的 CPU 核心数
#SBATCH --gres=gpu:1                  # 申请 1 块 GPU（如果显存小，可以申请 A100）
#SBATCH --mem=4G                      # 申请 4GB 内存
#SBATCH --time=24:00:00               # 最大运行时间 12 小时
#SBATCH --output=logs/train_%j.out    # 标准输出日志 (%j 会替换为任务 ID)
#SBATCH --error=logs/train_%j.err     # 错误输出日志

# ============================================================
# 1. 环境加载
# ============================================================
# 加载 HPC 上的 Python 和 CUDA 模块 (根据你学校的模块设置修改)
module load python/3.9.6
module load cuda/12.1

echo "同步最新代码..."
git pull origin main  # 确保运行前代码是最新的

# 创建并激活虚拟环境 (如果尚未创建)
if [ ! -d "venv_hpc" ]; then
    python -m venv venv_hpc
fi
source venv_hpc/bin/activate

# 安装依赖 (仅在首次运行时需要，后续可以注释掉)
pip install --upgrade pip
pip install -r requirements.txt
pip install peft accelerate diffusers transformers datasets

# 创建必要的目录
mkdir -p logs
mkdir -p output
mkdir -p scripts/fine_tuning/checkpoint

# ============================================================
# 2. 阶段 1：数据预处理 (仅需运行一次)
# ============================================================
# 如果你已经在本地或 Colab 传好了预处理后的数据集，可以跳过此步
echo "开始数据预处理..."
python scripts/preprocessing/data_import.py
python scripts/preprocessing/data_correct.py --split train --out-csv data/lunara_corrected_labels.csv
python scripts/preprocessing/merge_labels.py \
  --csv-path "data/lunara_corrected_labels.csv" \
  --cache-dir "data/moonworks___lunara-aesthetic" \
  --output-dir "data/lunara_final_dataset_rev"

# ============================================================
# 3. 阶段 3：正式 LoRA 微调
# ============================================================
echo "开始 LoRA 微调训练..."
# 使用 accelerate 可以更好地利用单机多卡或单卡性能
python scripts/fine_tuning/train_lora.py

echo "✅ 任务完成！"