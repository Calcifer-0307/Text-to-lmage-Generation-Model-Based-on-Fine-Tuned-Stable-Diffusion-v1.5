#!/bin/bash
#SBATCH --job-name=SD_Backend
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                      
#SBATCH --time=04:00:00                
#SBATCH --output=logs/backend_%j.out

module load python/3.9.6
module load cuda/12.1

# 获取当前分配的计算节点主机名
NODE_NAME=$(hostname)

echo "====================================================================="
echo "🚀 Backend is starting on compute node: $NODE_NAME"
echo "👉 1. 请在本地电脑（Windows）打开一个新的终端"
echo "👉 2. 复制并运行以下命令建立 SSH 隧道映射端口："
echo "ssh -L 8001:${NODE_NAME}:8000 ruilingyuan@luhpc.ln.edu.hk"
echo "====================================================================="

# 进入项目目录
cd ~/prj_ruilingyuan/Text-to-lmage-Generation-Model-Based-on-Fine-Tuned-Stable-Diffusion-v1.5-main

# 创建并激活虚拟环境 (复用 train 的 venv_hpc)
if [ ! -d "venv_hpc" ]; then
    python -m venv venv_hpc
fi
source venv_hpc/bin/activate

# 确保安装了所需的依赖
pip install --upgrade pip
pip install -r requirements.txt peft>=0.17.0

# 启动后端服务
echo "Starting Uvicorn Server..."
PYTHONUNBUFFERED=1 uvicorn scripts.backend.app:app --host 0.0.0.0 --port 8000
