#!/bin/bash
#SBATCH --job-name=SD_Eval_Loop
#SBATCH --partition=gpu-a100
#SBATCH --qos=qos-normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=4G                      
#SBATCH --time=1:00:00                
#SBATCH --output=logs/eval_%j.out

module load python/3.9.6
module load cuda/12.1
source venv_hpc/bin/activate

# for loop in bash
for STEP in 0 500 1000 1500 2000 5000 10000 15000 20000 25000
do
    echo "-----------------------------------"
    echo "Start step $STEP evaluation..."
    python scripts/fine_tuning/eval2.py --seed 42 --step $STEP --prompt "Alpine village at twilight, snow-dusted rooftops, glowing amber windows, soft-focus telephoto lens, tranquil winter mood."
    echo "Finish Step $STEP。"
done

echo "✅ All evaluation tasks completed!"
